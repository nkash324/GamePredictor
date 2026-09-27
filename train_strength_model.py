#!/usr/bin/env python3
"""
train_strength_model.py  -  the champion strength model (a small neural network, PyTorch).

Built the way a game works:
  1. STRENGTH: one small network turns a champion's numbers (stats, abilities, class,
     effect flags like dash/shield, and how their usual items changed) and role into a
     single strength score. The same network scores all 10 champions, so what
     it learns about "+10% AP ratio on a mid mage" comes from EVERY game with every
     champion, not only the few champions that were changed.
     Besides the numbers themselves, the network sees how each number changed since the
     champion's first patch in the data (scaled so balance changes stand out).
     A small per-champion correction on top catches what the numbers miss (it has to
     earn its size from the games, so most champions stay near 0).
  2. LANES: a second small network scores each lane matchup (top vs top, ...).
  3. TEAM MIX: a third scores each team's overall composition (e.g. all-AD teams).
  4. Win chance = blue strength - red strength + lanes + team mix (+ blue-side / region /
     rank effect), turned into a probability.

Patch effect: take the previous patch's real games, give every champion its numbers on
the new patch, and see how each champion's predicted win chance moves - the same method
the game model uses, but here a nerf changes that champion's strength directly.

Testing is forward in time: each tested patch is predicted using only earlier patches.
By default only the latest 6 patches are tested (--test-last), to keep it quick.

Needs PyTorch. With an NVIDIA graphics card it trains on the GPU automatically.
  Install: see README_WINDOWS.md ("Champion strength model").

Usage:
  python train_strength_model.py                   test on the latest 6 patches + final model
  python train_strength_model.py --test-last 0     test on every patch (slower)
  python train_strength_model.py --predict 16.19   predictions for a patch (works before
                                                   any games are played on it)
  python train_strength_model.py --cpu             don't use the GPU
"""

import argparse
import copy
import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

import build_training_tables as btt

POSITIONS = btt.POSITIONS
SLOTS = [f"blue_{p}_champ" for p in POSITIONS] + [f"red_{p}_champ" for p in POSITIONS]
HIDDEN = 64            # size of the strength network
LANE_HIDDEN = 32
COMP_DIM = 8
DROPOUT = 0.1
LR = 2e-3
WEIGHT_DECAY = 1e-4
BATCH = 2048
MAX_EPOCHS = 40
PATIENCE = 3           # stop after this many passes without improvement
OFFSET_SD = 0.10       # champion corrections are expected within ~+-0.1 (about +-2.5 points)
MIN_TEST_GAMES = 1000
MIN_CHAMP_GAMES = 30
TEST_LAST_DEFAULT = 6
LOGIT_TO_POINTS = 25.0  # near 50%, +1 in the score ~ +25 winrate points (p(1-p) = 0.25)


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def need_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("PyTorch isn't installed, so the champion strength model was skipped.\n"
              "To install it (uses your NVIDIA graphics card), see README_WINDOWS.md, section\n"
              "'Champion strength model'.")
        sys.exit(0)


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = float(pos.sum()), float((~pos).sum())   # floats: no overflow on big groups
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def wcorr(a, b, w):
    if len(a) < 5:
        return np.nan
    a, b, w = (np.asarray(v, float) for v in (a, b, w))
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    va = np.average((a - ma) ** 2, weights=w)
    vb = np.average((b - mb) ** 2, weights=w)
    return float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, db, train_dir):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
        vec, feats = btt.champion_vectors(con)
        con.close()
        # item changes per champion and patch (how much their usual items changed, and how
        # gold-efficient their items are), from build_training_tables.py
        cpath = os.path.join(train_dir, "champion_patch.parquet")
        item_cols = ["item_stat_buff", "item_passive_buff", "item_changed_share", "item_efficiency"]
        if os.path.exists(cpath):
            import pyarrow.parquet as pq
            have = [c for c in item_cols if c in pq.read_schema(cpath).names]
            if have:
                ic = pd.read_parquet(cpath, columns=["patch", "champ_key"] + have)
                vec = vec.merge(ic, on=["patch", "champ_key"], how="left")
                for c in have:
                    vec[c] = vec[c].fillna(vec[c].mean() if c == "item_efficiency" else 0.0).fillna(0.0)
                feats = list(feats) + have
        self.feats = feats
        self.vec = vec.reset_index(drop=True)
        X = self.vec[feats].to_numpy(np.float64)
        # squash long tails (damage numbers run into the thousands), then standardize
        X = np.sign(X) * np.log1p(np.abs(X))
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        Z = np.clip((X - self.mu) / self.sd, -5, 5).astype(np.float32)
        # CHANGE inputs: each champion's numbers relative to their first patch in the data,
        # rescaled so that typical balance changes are about +-1. Without these, a 10% nerf
        # is a tiny wiggle next to the huge differences BETWEEN champions, and the network
        # barely notices it.
        order = self.vec["patch"].map(pkey)
        first = order.groupby(self.vec["champ_key"]).transform("min") == order
        ref = pd.DataFrame(Z[first.to_numpy()], index=self.vec.loc[first, "champ_key"].astype(int))
        ref = ref[~ref.index.duplicated()]
        self.zfirst = {int(k): ref.loc[k].to_numpy(np.float32) for k in ref.index}
        Dl = Z - np.stack([self.zfirst[int(k)] for k in self.vec["champ_key"]])
        nz = np.abs(Dl) > 1e-6
        self.dsd = np.array([Dl[nz[:, j], j].std() if nz[:, j].sum() > 5 else 1.0
                             for j in range(Z.shape[1])], dtype=np.float32)
        self.dsd[~np.isfinite(self.dsd) | (self.dsd < 1e-6)] = 1.0
        Dl = np.clip(Dl / self.dsd, -5, 5).astype(np.float32)
        full = np.hstack([Z, Dl])
        self.V = np.vstack([np.zeros((1, full.shape[1]), np.float32), full])   # row 0 = unknown
        self.row = {(p, int(k)): i + 1 for i, (p, k) in
                    enumerate(zip(self.vec["patch"], self.vec["champ_key"]))}
        self.names = dict(zip(self.vec["champ_key"].astype(int), self.vec["champion_id"]))
        self.code = {int(k): i + 1 for i, k in enumerate(sorted(self.vec["champ_key"].unique()))}

        path = os.path.join(train_dir, "games.parquet")
        if not os.path.exists(path):
            sys.exit("training/games.parquet not found - run build_training_tables.py first.")
        cols = ["match_id", "patch", "platform", "rank_bucket", "blue_win"] + SLOTS
        g = pd.read_parquet(path, columns=cols)
        g = g.dropna(subset=SLOTS + ["blue_win"])
        for c in SLOTS:
            g[c] = g[c].astype(np.int64)
        self.games = g.reset_index(drop=True)
        self.platforms = sorted(self.games["platform"].dropna().unique())

    def standardize(self, raw_row, key):
        """Full input row (numbers + change-since-first-patch) for a champion's raw numbers."""
        x = np.asarray(raw_row, np.float64)
        x = np.sign(x) * np.log1p(np.abs(x))
        z = np.clip((x - self.mu) / self.sd, -5, 5).astype(np.float32)
        d = np.clip((z - self.zfirst.get(int(key), z)) / self.dsd, -5, 5).astype(np.float32)
        return np.concatenate([z, d])

    def arrays(self, g, patch_override=None):
        """Number-table rows (n,10), champion codes (n,10), context (n,c), outcome (n,)."""
        n = len(g)
        vi = np.zeros((n, 10), np.int64)
        ci = np.zeros((n, 10), np.int64)
        pat = g["patch"].to_numpy() if patch_override is None else np.full(n, patch_override)
        for j, col in enumerate(SLOTS):
            keys = g[col].to_numpy()
            vi[:, j] = [self.row.get((p, k), 0) for p, k in zip(pat, keys)]
            ci[:, j] = [self.code.get(k, 0) for k in keys]
        ctx = np.column_stack([(g["rank_bucket"] == "MASTER+").to_numpy(np.float32)]
                              + [(g["platform"] == p).to_numpy(np.float32) for p in self.platforms])
        y = g["blue_win"].to_numpy(np.float32)
        return vi, ci, ctx.astype(np.float32), y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(torch, n_feats, n_champs, n_ctx):
    nn = torch.nn

    class StrengthNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.strength = nn.Sequential(
                nn.Linear(n_feats + 5, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(), nn.Linear(HIDDEN // 2, 1))
            self.offset = nn.Embedding(n_champs + 1, 1)      # per-champion correction
            nn.init.zeros_(self.offset.weight)
            self.lane = nn.Sequential(nn.Linear(2 * n_feats + 5, LANE_HIDDEN), nn.GELU(),
                                      nn.Dropout(DROPOUT), nn.Linear(LANE_HIDDEN, 1))
            self.comp_embed = nn.Sequential(nn.Linear(n_feats, 32), nn.GELU(),
                                            nn.Linear(32, COMP_DIM))
            self.comp_score = nn.Sequential(nn.Linear(COMP_DIM, 16), nn.GELU(), nn.Linear(16, 1))
            self.ctx = nn.Linear(n_ctx, 1)                   # bias = blue-side effect
            self.register_buffer("roles", torch.eye(5))

        def champ_strength(self, x, role_idx, code=None):
            s = self.strength(torch.cat([x, self.roles[role_idx]], -1)).squeeze(-1)
            return s if code is None else s + self.offset(code).squeeze(-1)

        def forward(self, x, code, ctx):                     # x (B,10,F), code (B,10)
            B = x.shape[0]
            role = torch.arange(5, device=x.device).repeat(2).expand(B, 10)
            s = self.champ_strength(x, role, code)
            team = s[:, :5].sum(1) - s[:, 5:].sum(1)
            xb, xr = x[:, :5], x[:, 5:]
            r5 = self.roles.expand(B, 5, 5)
            lane = (self.lane(torch.cat([xb, xr, r5], -1))
                    - self.lane(torch.cat([xr, xb, r5], -1))).squeeze(-1).sum(1)
            e = self.comp_embed(x)
            comp = (self.comp_score(e[:, :5].sum(1)) - self.comp_score(e[:, 5:].sum(1))).squeeze(-1)
            return team + lane + comp + self.ctx(ctx).squeeze(-1)

    return StrengthNet()


def to_dev(torch, arrs, device):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrs]


def predict(torch, model, V, arrs, device, batch=8192):
    vi, ci, ctx = to_dev(torch, arrs[:3], device)
    out = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(vi), batch):
            out.append(torch.sigmoid(model(V[vi[b:b + batch]], ci[b:b + batch],
                                           ctx[b:b + batch])).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_model(torch, D, arrs, device, seed=0, quiet=False):
    F = torch.nn.functional
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vi, ci, ctx, y = arrs
    n = len(y)
    val = rng.random(n) < 0.1                      # validation games, for early stopping
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    model = make_model(torch, D.V.shape[1], len(D.code), ctx.shape[1]).to(device)
    V = torch.from_numpy(D.V).to(device)
    vi_t, ci_t, ctx_t, y_t = to_dev(torch, [vi, ci, ctx, y], device)
    others = [p for name, p in model.named_parameters() if not name.startswith("offset")]
    opt = torch.optim.AdamW([{"params": others, "weight_decay": WEIGHT_DECAY},
                             {"params": [model.offset.weight], "weight_decay": 0.0}], lr=LR)
    va_t = torch.from_numpy(va_idx).to(device)
    best_loss, best_state, best_ep, bad = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.from_numpy(rng.permutation(tr_idx)).to(device)
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            logits = model(V[vi_t[idx]], ci_t[idx], ctx_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_t[idx])
            # champion corrections must earn their size (prior: usually within +-OFFSET_SD)
            loss = loss + (model.offset.weight ** 2).sum() / (2 * OFFSET_SD ** 2 * len(tr_idx))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for b in range(0, len(va_t), 8192):
                idx = va_t[b:b + 8192]
                vl += float(F.binary_cross_entropy_with_logits(
                    model(V[vi_t[idx]], ci_t[idx], ctx_t[idx]), y_t[idx], reduction="sum"))
            vl /= max(len(va_idx), 1)
        if vl < best_loss - 1e-5:
            best_loss, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    if not quiet:
        print(f"    trained {best_ep} passes over {len(tr_idx):,} games in {time.time() - t0:.0f}s "
              f"(validation log loss {best_loss:.4f})", flush=True)
    return model


# --------------------------------------------------------------------------- #
# Patch effect per champion
# --------------------------------------------------------------------------- #
def main_roles(g):
    rows = []
    for side in ("blue", "red"):
        for pos in POSITIONS:
            rows.append(pd.DataFrame({"champ_key": g[f"{side}_{pos}_champ"].to_numpy(), "pos": pos}))
    r = pd.concat(rows)
    return r.groupby("champ_key")["pos"].agg(lambda s: s.value_counts().index[0]).to_dict()


def why_change(torch, model, D, key, old_patch, new_patch, role, device, top=3):
    """Which changed numbers moved this champion's strength, in winrate points."""
    io, inew = D.row.get((old_patch, key)), D.row.get((new_patch, key))
    if not io or not inew:
        return 0.0, ""
    raw_o = D.vec.loc[io - 1, D.feats].to_numpy(np.float64)
    raw_n = D.vec.loc[inew - 1, D.feats].to_numpy(np.float64)
    changed = np.flatnonzero(np.abs(raw_n - raw_o) > 1e-9 * np.maximum(1, np.abs(raw_o)))
    if not len(changed):
        return 0.0, ""
    rows = [D.standardize(raw_o, key), D.standardize(raw_n, key)]
    for j in changed:
        mixed = raw_o.copy()
        mixed[j] = raw_n[j]                        # only this one number changed
        rows.append(D.standardize(mixed, key))
    x = torch.from_numpy(np.stack(rows)).to(device)
    r = torch.full((len(rows),), POSITIONS.index(role), dtype=torch.long, device=device)
    with torch.no_grad():
        s = model.champ_strength(x, r).cpu().numpy()
    total = (s[1] - s[0]) * LOGIT_TO_POINTS
    parts = sorted(((D.feats[j], (s[2 + i] - s[0]) * LOGIT_TO_POINTS) for i, j in enumerate(changed)),
                   key=lambda kv: -abs(kv[1]))
    return float(total), "; ".join(f"{k} {v:+.1f}" for k, v in parts[:top] if abs(v) >= 0.05)


def champion_predictions(torch, model, D, prev_games, target, device):
    """Previous patch's drafts, scored with old numbers and with the target patch's."""
    V = torch.from_numpy(D.V).to(device)
    arr_old = D.arrays(prev_games)
    arr_new = list(D.arrays(prev_games, patch_override=target))
    arr_new[0] = np.where(arr_new[0] == 0, arr_old[0], arr_new[0])   # no new numbers: keep old
    p_old = predict(torch, model, V, arr_old, device)
    p_new = predict(torch, model, V, arr_new, device)
    rows = []
    for j, col in enumerate(SLOTS):
        blue = j < 5
        rows.append(pd.DataFrame({"champ_key": prev_games[col].to_numpy(),
                                  "p_old": p_old if blue else 1 - p_old,
                                  "p_new": p_new if blue else 1 - p_new}))
    r = pd.concat(rows).groupby("champ_key").agg(pred_old=("p_old", "mean"),
                                                 pred_new=("p_new", "mean"),
                                                 sample=("p_old", "size"))
    r["effect"] = r["pred_new"] - r["pred_old"]
    r["champion_id"] = r.index.map(D.names)
    return r.reset_index()


def add_reasons(torch, model, D, pred, prev_games, prev, target, changed_keys, device):
    roles = main_roles(prev_games)
    strength_chg, why = [], []
    for k in pred["champ_key"]:
        if k in changed_keys and k in roles:
            t, w = why_change(torch, model, D, int(k), prev, target, roles[k], device)
        else:
            t, w = 0.0, ""
        strength_chg.append(t)
        why.append(w)
    pred["strength_change_pts"] = strength_chg
    pred["why"] = why
    return pred


# --------------------------------------------------------------------------- #
def evaluate(torch, D, cp, device, test_last):
    g = D.games
    patches = sorted(g["patch"].unique(), key=pkey)
    counts = g.groupby("patch").size()
    testable = [p for i, p in enumerate(patches) if i > 0 and counts[p] >= MIN_TEST_GAMES]
    if test_last > 0:
        testable = testable[-test_last:]
    print(f"Testing on {len(testable)} patch(es): {', '.join(testable)}")
    arr_all = D.arrays(g)
    results, detail = [], []
    for p in testable:
        i = patches.index(p)
        earlier, prev = patches[:i], patches[i - 1]
        tr = g["patch"].isin(earlier).to_numpy()
        te = (g["patch"] == p).to_numpy()
        print(f"  {p}: training on {tr.sum():,} games from {len(earlier)} earlier patch(es)",
              flush=True)
        model = train_model(torch, D, [a[tr] for a in arr_all], device)
        V = torch.from_numpy(D.V).to(device)
        y = arr_all[3][te]
        pt = predict(torch, model, V, [a[te] for a in arr_all], device)
        row = {"patch": p, "train games": int(tr.sum()), "test games": int(te.sum()),
               "coin logloss": logloss(y, np.full(len(y), arr_all[3][tr].mean())),
               "logloss": logloss(y, pt), "AUC": auc(y, pt),
               "acc": float(np.mean((pt > 0.5) == (y == 1)))}
        # champion level
        prev_games = g[g["patch"] == prev]
        pred = champion_predictions(torch, model, D, prev_games, p, device)
        truth = cp[(cp["patch"] == p) & (cp["games"] >= MIN_CHAMP_GAMES)][
            ["champ_key", "games", "winrate", "winrate_smoothed", "changed"]].copy()
        truth["champ_key"] = truth["champ_key"].astype(np.int64)
        t = truth.merge(pred, on="champ_key", how="inner")
        if len(t):
            changed_keys = set(t.loc[t["changed"] == 1, "champ_key"].astype(int))
            t = add_reasons(torch, model, D, t, prev_games, prev, p, changed_keys, device)
            w = t["games"].to_numpy(float)
            pbar = t["winrate_smoothed"]
            noise_sd = np.sqrt(pbar * (1 - pbar) / t["games"])
            err = t["pred_new"] - t["winrate"]
            ch = (t["changed"] == 1).to_numpy()
            actual = t["winrate"] - t["pred_old"]
            big = ch & (w >= 100) & (t["effect"].abs().to_numpy() > 0.002)
            row.update({
                "floor": float(np.average(noise_sd * np.sqrt(2 / np.pi), weights=w)),
                "champ MAE": float(np.average(err.abs(), weights=w)),
                "changed": int(ch.sum()),
                "floor ch": float(np.average((noise_sd * np.sqrt(2 / np.pi))[ch], weights=w[ch])) if ch.any() else np.nan,
                "champ MAE ch": float(np.average(err.abs()[ch], weights=w[ch])) if ch.any() else np.nan,
                "corr": wcorr(t["effect"][ch], actual[ch], w[ch]),
                "direction": float(np.mean(np.sign(t["effect"][big]) == np.sign(actual[big]))) if big.any() else np.nan,
                "max effect": float(t.loc[ch, "effect"].abs().max() * 100) if ch.any() else 0.0,
            })
            d = pd.DataFrame({"patch": p, "champion_id": t["champion_id"], "games": t["games"],
                              "changed": t["changed"].fillna(0).astype(int),
                              "pred_old": t["pred_old"], "predicted_winrate": t["pred_new"],
                              "effect": t["effect"], "strength_change_pts": t["strength_change_pts"],
                              "winrate": t["winrate"], "actual_change": actual, "error": err,
                              "noise_sd": noise_sd, "z": err / noise_sd, "why": t["why"]})
            detail.append(d)
        results.append(row)
        f = lambda k: row.get(k, np.nan)
        print(f"    game AUC {f('AUC'):.3f} (log loss {f('logloss'):.4f} vs coin {f('coin logloss'):.4f})"
              f" | champion miss {f('champ MAE') * 100:.2f} pts (floor {f('floor') * 100:.2f}),"
              f" changed {f('champ MAE ch') * 100:.2f} | biggest predicted effect {f('max effect'):.1f} pts",
              flush=True)
    return pd.DataFrame(results), (pd.concat(detail, ignore_index=True) if detail else pd.DataFrame())


def print_eval(res, det):
    if res.empty:
        print("Nothing to test yet (need 2+ patches with 1,000+ games).")
        return
    w = res["test games"]
    print(f"\nGame level: AUC {np.average(res['AUC'], weights=w):.3f}, log loss "
          f"{np.average(res['logloss'], weights=w):.4f} vs coin flip "
          f"{np.average(res['coin logloss'], weights=w):.4f}")
    if "champ MAE" in res:
        wc = res["changed"].clip(lower=1)
        print(f"Champion level: average miss {res['champ MAE'].mean() * 100:.2f} pts "
              f"(noise floor {res['floor'].mean() * 100:.2f}); changed champions "
              f"{np.average(res['champ MAE ch'].fillna(0), weights=wc) * 100:.2f} pts "
              f"(floor {np.average(res['floor ch'].fillna(0), weights=wc) * 100:.2f})")
        print(f"Direction right for changed champions (100+ games): {np.nanmean(res['direction']):.0%}"
              f"  (50% = coin flip)")
    if len(det):
        ch = det[det["changed"] == 1]
        top = ch.reindex(ch["effect"].abs().sort_values(ascending=False).index).head(15)
        if len(top):
            print("\nBiggest predicted patch effects vs what happened:")
            for _, r in top.iterrows():
                hit = "right" if np.sign(r["effect"]) == np.sign(r["actual_change"]) else "wrong"
                print(f"  {r['patch']:>6} {r['champion_id']:<14} predicted {r['effect'] * 100:+5.1f}  "
                      f"actual {r['actual_change'] * 100:+5.1f}  ({int(r['games'])} games, {hit})"
                      f"  {r['why']}")


def predict_patch(torch, D, cp, device, target):
    g = D.games
    earlier = [p for p in sorted(g["patch"].unique(), key=pkey) if pkey(p) < pkey(target)]
    if not earlier:
        sys.exit("Need games from a patch before the target patch.")
    if not any(pt == target for pt, _ in D.row):
        sys.exit(f"No champion numbers for patch {target} - run build_patch_state.py first.")
    prev = earlier[-1]
    tr = g["patch"].isin(earlier).to_numpy()
    print(f"Training on {tr.sum():,} games from {len(earlier)} patches before {target} ...")
    model = train_model(torch, D, [a[tr] for a in D.arrays(g)], device)
    prev_games = g[g["patch"] == prev]
    pred = champion_predictions(torch, model, D, prev_games, target, device)
    ch_keys = set(cp.loc[(cp["patch"] == target) & (cp["changed"] == 1), "champ_key"].astype(int))
    pred["changed"] = pred["champ_key"].isin(ch_keys).astype(int)
    pred = add_reasons(torch, model, D, pred, prev_games, prev, target, ch_keys, device)
    pred = pred.sort_values("pred_new", ascending=False)
    print(f"\nPatch {target} (champion drafts from {prev}, numbers from {target}):\n")
    chg = pred[pred["changed"] == 1].sort_values("effect")
    if len(chg):
        print("CHANGED champions:")
        for _, r in chg.iterrows():
            print(f"  {r['champion_id']:<14} {r['pred_old'] * 100:5.1f}% -> {r['pred_new'] * 100:5.1f}%"
                  f"  ({r['effect'] * 100:+.1f})   {r['why']}")
    for title, part in (("STRONGEST predicted", pred.head(10)),
                        ("WEAKEST predicted", pred.tail(10).iloc[::-1])):
        print(f"\n{title}:")
        for _, r in part.iterrows():
            tag = "  (changed)" if r["changed"] == 1 else ""
            print(f"  {r['champion_id']:<14} {r['pred_new'] * 100:5.1f}%{tag}")
    out = os.path.join("training", f"strength_predicted_{target}.csv")
    pred.rename(columns={"pred_old": "winrate_with_old_numbers",
                         "pred_new": "predicted_winrate"}).to_csv(out, index=False)
    print(f"\nSaved {out}")


def save_final(torch, D, device, models_dir):
    print("\nTraining the final model on all games ...", flush=True)
    model = train_model(torch, D, D.arrays(D.games), device)
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "strength_model.pt"))
    with open(os.path.join(models_dir, "strength_model.json"), "w") as f:
        json.dump({"feats": D.feats, "mu": D.mu.tolist(), "sd": D.sd.tolist(),
                   "code": {str(k): v for k, v in D.code.items()}, "platforms": D.platforms,
                   "patches": sorted(D.games["patch"].unique(), key=pkey)}, f)
    # champion corrections: what the numbers alone don't explain
    off = model.offset.weight.detach().cpu().numpy().ravel()
    inv = {v: k for k, v in D.code.items()}
    corr = pd.DataFrame({"champion_id": [D.names.get(inv[i], inv[i]) for i in range(1, len(off))],
                         "correction_pts": off[1:] * LOGIT_TO_POINTS})
    corr = corr.reindex(corr["correction_pts"].abs().sort_values(ascending=False).index)
    corr.to_csv(os.path.join(models_dir, "strength_corrections.csv"), index=False)
    print("  saved models/strength_model.pt")
    print("  champions the numbers explain least (their own correction, points):")
    for _, r in corr.head(8).iterrows():
        print(f"     {r['champion_id']:<14} {r['correction_pts']:+.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Champion strength model (neural network).")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--models", default="models")
    ap.add_argument("--test-last", type=int, default=TEST_LAST_DEFAULT,
                    help=f"test on the latest N patches (default {TEST_LAST_DEFAULT}; 0 = all)")
    ap.add_argument("--predict", default=None, help="predictions for this patch")
    ap.add_argument("--skip-eval", action="store_true", help="only train the final model")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU")
    args = ap.parse_args(argv)
    import memlimit
    memlimit.apply()
    torch = need_torch()
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        torch.set_num_threads(os.cpu_count() or 1)
        why = "--cpu" if args.cpu else "no CUDA GPU found by PyTorch"
        print(f"Device: CPU ({why})")
    else:
        device = "cuda"
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found - run this from the project folder.")
    D = Data(args.db, args.dir)
    cp = pd.read_parquet(os.path.join(args.dir, "champion_patch.parquet"))
    print(f"{len(D.games):,} games, {D.games['patch'].nunique()} patches, "
          f"{len(D.feats)} numbers per champion")
    if args.predict:
        predict_patch(torch, D, cp, device, args.predict)
        return
    if not args.skip_eval:
        res, det = evaluate(torch, D, cp, device, args.test_last)
        print_eval(res, det)
        if len(res):
            res.to_csv(os.path.join(args.dir, "strength_results.csv"), index=False)
        if len(det):
            det.to_csv(os.path.join(args.dir, "strength_predictions.csv"), index=False)
    save_final(torch, D, device, args.models)


if __name__ == "__main__":
    main()
