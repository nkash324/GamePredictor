#!/usr/bin/env python3
"""
train_change_model.py  -  predicts each champion's winrate on a patch, including how much
the patch's balance changes move it.

Two parts:

1. STRENGTH MODEL - how strong is the champion before looking at balance changes?
   Uses the champion's recent winrate (last few patches) AND long-term winrate (many
   patches), plus how popular they are (very popular champions tend to have lower
   winrates). Each champion is trusted by how many games we have on them: a champion
   with 10,000 games keeps its real winrate (e.g. 47%), one with 300 games is pulled
   toward 50%, because its winrate is mostly luck.

2. CHANGE MODEL - how much do this patch's changes move the champion?
   Learns directly from past patches: "-8% base damage on a mage who was already strong
   -> winrate moved X". Buffs and nerfs are scored separately, so nerfs can hurt more than
   buffs help. Also sees effect flags gained/lost (dash, shield, ...) and item changes:
   how much the items the champion usually buys got stronger or weaker - so a champion
   can be affected by a patch even if they weren't changed themselves.
   Kept small (~35 inputs) so it doesn't learn from a handful of champions.

Every changed champion gets TWO predictions:
  best guess  the most accurate on average. Held back when the data is noisy, so it
              often predicts small changes. Switched off entirely for a patch if it didn't
              beat "no effect" on the latest patches.
  bold        the same direction, scaled up so predicted changes are as big as real
              patch swings. Less accurate on average, but shows which champions are
              likely to move a lot. Its hit rate (right direction) is reported.
Plus a likely range for the champion's true winrate.

Testing is forward in time: each patch is predicted from earlier patches only.

Usage:
  python train_change_model.py                 evaluate on past patches
  python train_change_model.py --predict 16.19 predictions for a patch (works for a new
                                               patch before any games are played)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

MIN_GAMES = 30
MIN_TRAIN_PATCHES = 2
MAX_EFFECT = 0.10             # best-guess effect capped at +-10 winrate points
MAX_BOLD = 0.15               # bold effect capped at +-15 points
BOLD_SCALE_RANGE = (1.0, 6.0) # bold = best guess x this (learned), within these limits
ALPHAS = [30, 100, 300, 1000, 3000, 10000]
REWORK_CHANGES = 25           # this many changed numbers = a rework: % changes are meaningless
MAX_REL = 0.5                 # a single relative change counts at most +-50%
ITEM_AFFECTED_SHARE = 0.10    # champion counts as "affected" if 10%+ of their builds had a changed item
SWITCH_ON_GAIN = 0.01         # best guess must beat "no effect" by 1% on validation to be used
HIST_DECAY = 0.7              # recent strength: each older patch counts 0.7x
LONG_DECAY = 0.95             # long-term strength: each older patch counts 0.95x
TRUST_GAMES = [100, 300, 1000, 3000]   # candidate "games needed to half-trust a winrate"

# direction of each change: +1 means "higher is a buff", -1 means "lower is a buff"
CHANGE_FEATURES = {
    "hp_l1": 1, "hp_l9": 1, "hp_l16": 1, "armor_l9": 1, "armor_l16": 1, "mr_l9": 1, "mr_l16": 1,
    "ad_l1": 1, "ad_l9": 1, "ad_l16": 1, "as_l9": 1, "as_l16": 1, "stat_movespeed": 1,
    "stat_attackrange": 1, "stat_hpregen": 1, "stat_mp": 1,
    "total_dmg_base_rmax": 1, "total_dmg_ratio_ap": 1, "total_dmg_ratio_ad_all": 1,
    "total_heal_base_rmax": 1, "total_shield_base_rmax": 1,
    "basic_cd_rmax_mean": -1, "basic_cd_r1_mean": -1, "ult_cd_rmax": -1,
    "basic_cost_rmax_mean": -1, "cc_hard_total": 1, "cc_slow_max": 1, "range_max": 1,
}
# individual changes the model sees directly (all changes still count in the buff/nerf scores)
LEAN_CHANGES = ["total_dmg_base_rmax", "total_dmg_ratio_ap", "total_dmg_ratio_ad_all",
                "hp_l9", "armor_l9", "mr_l9", "ad_l9", "as_l9", "basic_cd_rmax_mean",
                "ult_cd_rmax", "total_heal_base_rmax", "total_shield_base_rmax"]
CLASSES = ["assassin", "fighter", "mage", "marksman", "support", "tank"]


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def load(path_dir):
    for ext in ("parquet", "csv.gz"):
        p = os.path.join(path_dir, f"champion_patch.{ext}")
        if os.path.exists(p):
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p, dtype={"patch": str})
    sys.exit("training/champion_patch not found - run build_training_tables.py first.")


def wavg(v, w):
    return float(np.average(np.asarray(v, float), weights=np.asarray(w, float)))


def wvar(v, w):
    v = np.asarray(v, float)
    return wavg((v - wavg(v, w)) ** 2, w)


def wcorr(a, b, w):
    if len(a) < 5:
        return np.nan
    a, b, w = np.asarray(a, float), np.asarray(b, float), np.asarray(w, float)
    va, vb = wvar(a, w), wvar(b, w)
    if va <= 0 or vb <= 0:
        return np.nan
    return float(wavg((a - wavg(a, w)) * (b - wavg(b, w)), w) / np.sqrt(va * vb))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def recent_strength(cp, decay=HIST_DECAY, with_games=False):
    """Winrate over EARLIER patches only, older patches down-weighted, 25-25 prior.
    with_games=True also returns the (down-weighted) number of games behind it."""
    order = {p: i for i, p in enumerate(sorted(cp["patch"].unique(), key=pkey))}
    pi = cp["patch"].map(order)
    wins = (cp["winrate"].fillna(0) * cp["games"].fillna(0))
    W = pd.crosstab(cp["champ_key"], pi, values=wins, aggfunc="sum").fillna(0)
    G = pd.crosstab(cp["champ_key"], pi, values=cp["games"].fillna(0), aggfunc="sum").fillna(0)
    W = W.reindex(columns=range(len(order)), fill_value=0)
    G = G.reindex(columns=range(len(order)), fill_value=0)
    sw = np.zeros(len(W))
    sg = np.zeros(len(W))
    hist, hist_games = {}, {}
    for j in range(len(order)):
        hist[j] = pd.Series((sw + 25) / (sg + 50), index=W.index)
        hist_games[j] = pd.Series(sg, index=W.index)
        sw = sw * decay + W[j].values
        sg = sg * decay + G[j].values
    h = np.array([hist[i][c] for i, c in zip(pi, cp["champ_key"])])
    hg = np.array([hist_games[i][c] for i, c in zip(pi, cp["champ_key"])])
    r = pd.Series(np.where(hg > 0, h, np.nan), index=cp.index)
    return (r, pd.Series(hg, index=cp.index)) if with_games else r


def build_features(cp):
    cp = cp.copy()
    cp = cp.sort_values("patch", key=lambda s: s.map(pkey))
    cp["recent_winrate"], cp["recent_games"] = recent_strength(cp, HIST_DECAY, True)
    cp["long_winrate"], cp["long_games"] = recent_strength(cp, LONG_DECAY, True)
    # main role: from this patch's games, else the latest earlier patch's
    cp["role"] = cp.groupby("champ_key")["main_role"].transform(lambda s: s.ffill())
    rework = cp["n_changes"].fillna(0) >= REWORK_CHANGES
    not_changed = cp["changed"].fillna(0) != 1
    zero = pd.Series(0.0, index=cp.index)

    rel = {}
    for f, direction in CHANGE_FEATURES.items():
        if f not in cp or f"delta_{f}" not in cp:
            continue
        d = cp[f"delta_{f}"].fillna(0)
        old = (cp[f] - d).abs()
        # a brand-new component (old value 0) has no meaningful % change: left at 0
        r = pd.Series(d / old.where(old > 1e-9), index=cp.index).fillna(0).clip(-MAX_REL, MAX_REL)
        r[rework] = 0.0
        # numbers that moved without counting as a balance change (renamed labels, rounding)
        r[not_changed] = 0.0
        rel[f] = (r, direction)
        cp[f"chg_{f}"] = r                      # kept for the diagnostics data checks

    directed = [r * direction for r, direction in rel.values()]
    feats = {
        "buff_part": sum((x.clip(lower=0) for x in directed), zero),   # all buffs added up
        "nerf_part": sum((x.clip(upper=0) for x in directed), zero),   # all nerfs added up
    }
    buff_score = feats["buff_part"] + feats["nerf_part"]
    cp["buff_score"] = buff_score
    for f in LEAN_CHANGES:
        if f in rel:
            feats[f"rel_{f}"] = rel[f][0]
    feats["n_changes"] = np.log1p(cp["n_changes"].fillna(0))
    feats["rework"] = rework.astype(float)
    feats["prev_strength"] = (cp["recent_winrate"] - 0.5).fillna(0)
    feats["prev_presence"] = cp["prev_pickrate"].fillna(0) + cp["prev_banrate"].fillna(0)
    feats["buff_x_strength"] = buff_score * feats["prev_strength"]   # buffing a strong champ
    for c in CLASSES:
        col = f"class_{c}"
        if col in cp:
            feats[f"{c}_x_buff"] = cp[col].fillna(0) * buff_score
    # effect flags gained / lost this patch (dash, shield, heal, ...)
    flag_deltas = [c for c in cp.columns if c.startswith("delta_flag_")]
    if flag_deltas:
        fd = cp[flag_deltas].fillna(0)
        fd[not_changed] = 0.0
        feats["flags_gained"] = fd.clip(lower=0).sum(axis=1)
        feats["flags_lost"] = -fd.clip(upper=0).sum(axis=1)
    # item changes: how much the champion's usual items got stronger (share of item cost,
    # weighted by how often they buy each item)
    cp["affected"] = (cp["changed"].fillna(0) == 1).astype(int)
    if "item_stat_buff" in cp:
        item_buff = cp["item_stat_buff"].fillna(0) + cp["item_passive_buff"].fillna(0)
        feats["item_stat_buff"] = cp["item_stat_buff"].fillna(0)
        feats["item_passive_buff"] = cp["item_passive_buff"].fillna(0)
        feats["item_changed_share"] = cp["item_changed_share"].fillna(0)
        for c in CLASSES:
            col = f"class_{c}"
            if col in cp:
                feats[f"{c}_x_item"] = cp[col].fillna(0) * item_buff
        cp["affected"] = ((cp["affected"] == 1)
                          | (cp["item_changed_share"].fillna(0) >= ITEM_AFFECTED_SHARE)).astype(int)
    X = pd.DataFrame(feats, index=cp.index)
    X = X.loc[:, X.abs().sum() > 0]
    return cp, X


# --------------------------------------------------------------------------- #
# Part 1: strength model
# --------------------------------------------------------------------------- #
class StrengthModel:
    """Expected winrate before balance changes: recent + long-term winrate (each trusted
    by how many games are behind it) + popularity. Fitted by weighted least squares."""

    def _design(self, rows, n0):
        trust = lambda n: n / (n + n0)
        rs = rows["recent_winrate"].fillna(0.5) - 0.5
        rl = rows["long_winrate"].fillna(0.5) - 0.5
        ns = rows["recent_games"].fillna(0)
        nl = rows["long_games"].fillna(0)
        pop = rows["prev_pickrate"].fillna(self.pop_mean) - self.pop_mean
        return np.column_stack([trust(ns) * rs, trust(nl) * rl, pop])

    def fit(self, rows):
        w = rows["games"].values.astype(float)
        y = rows["winrate"].values - 0.5
        pr = rows["prev_pickrate"]
        self.pop_mean = float(np.average(pr.fillna(pr.mean() if pr.notna().any() else 0),
                                         weights=w))
        sw = np.sqrt(w)
        best = None
        for n0 in TRUST_GAMES:
            A = self._design(rows, n0)
            Aw = A * sw[:, None]
            coef = np.linalg.solve(Aw.T @ Aw + 1e-6 * w.sum() * np.eye(A.shape[1]), Aw.T @ (y * sw))
            err = wavg((A @ coef - y) ** 2, w)
            if best is None or err < best[0]:
                best = (err, n0, coef)
        _, self.n0, self.coef = best
        return self

    def predict(self, rows):
        return pd.Series(0.5 + self._design(rows, self.n0) @ self.coef,
                         index=rows.index).clip(0.3, 0.7)

    def describe(self):
        return {"trust_games": self.n0, "recent_weight": float(self.coef[0]),
                "long_term_weight": float(self.coef[1]), "popularity_weight": float(self.coef[2])}


# --------------------------------------------------------------------------- #
# Part 2: change model (weighted ridge regression on standardized features)
# --------------------------------------------------------------------------- #
class ChangeModel:
    def fit(self, X, y, w, alpha):
        self.cols = list(X.columns)
        self.off = len(y) < 10
        if self.off:
            self.coef, self.b0 = np.zeros(len(self.cols)), 0.0
            return self
        A = X.values.astype(float)
        self.mu, self.sd = A.mean(0), A.std(0)
        self.sd[self.sd == 0] = 1
        Z = (A - self.mu) / self.sd
        sw = np.sqrt(w / w.mean())[:, None]
        Zw, yw = Z * sw, y * sw[:, 0]
        self.b0 = float(np.average(y, weights=w))
        self.coef = np.linalg.solve(Zw.T @ Zw + alpha * np.eye(Z.shape[1]),
                                    Zw.T @ (yw - self.b0 * sw[:, 0]))
        return self

    def _z(self, X):
        return np.clip((X[self.cols].values.astype(float) - self.mu) / self.sd, -4, 4)

    def predict(self, X):
        if self.off:
            return np.zeros(len(X))
        return np.clip(self.b0 + self._z(X) @ self.coef, -MAX_EFFECT, MAX_EFFECT)

    def contributions(self, X):
        """Per row: how much each input pushed the prediction (points of winrate)."""
        if self.off:
            return pd.DataFrame(0.0, index=X.index, columns=self.cols)
        return pd.DataFrame(self._z(X) * self.coef, index=X.index, columns=self.cols)

    def top(self, n=10):
        s = pd.Series(self.coef, index=self.cols)
        return s.reindex(s.abs().sort_values(ascending=False).index).head(n)


def fit_change(rows, X, base, alpha):
    ch = rows[rows["affected"] == 1]
    y = (ch["winrate"] - base.predict(ch)).values
    return ChangeModel().fit(X.loc[ch.index], y, ch["games"].values.astype(float), alpha)


def noise_var(rows):
    p = rows["winrate_smoothed"]
    return (p * (1 - p) / rows["games"]).values


def choose_settings(rows, X, patches):
    """From the latest training patches (each predicted from the ones before it):
    ridge strength, whether the best guess beats 'no effect', the bold scale and how
    uncertain predictions are."""
    default = {"alpha": 300.0, "on": False, "scale": 3.0, "sd_true": 0.02}
    if len(patches) < 3:
        return default
    val_patches = patches[-3:] if len(patches) >= 4 else patches[-1:]
    folds = []
    for vp in val_patches:
        tr = rows[rows["patch"].map(pkey) < pkey(vp)]
        va = rows[(rows["patch"] == vp) & (rows["affected"] == 1)]
        if va.empty or (tr["affected"] == 1).sum() < 10:
            continue
        base = StrengthModel().fit(tr)
        folds.append((tr, va, base, (va["winrate"] - base.predict(va)).values))
    if not folds:
        return default
    va_all = pd.concat([f[1] for f in folds])
    w = va_all["games"].values.astype(float)
    actual = np.concatenate([f[3] for f in folds])
    nvar = noise_var(va_all)
    err0 = wavg(np.abs(actual), w)
    best = None
    for a in ALPHAS:
        eff = np.concatenate([fit_change(tr, X, base, a).predict(X.loc[va.index])
                              for tr, va, base, _ in folds])
        err = wavg(np.abs(actual - eff), w)
        if best is None or err < best[0]:
            best = (err, a, eff)
    err, alpha, eff = best
    on = err < err0 * (1 - SWITCH_ON_GAIN)
    # bold: scale predictions so their spread matches the spread of REAL changes (after
    # removing the part of the spread that is just luck)
    sd_pred = np.sqrt(wvar(eff, w))
    sd_real = np.sqrt(max(wvar(actual, w) - wavg(nvar, w), 1e-6))
    scale = float(np.clip(sd_real / sd_pred, *BOLD_SCALE_RANGE)) if sd_pred > 1e-5 else 1.0
    used = eff if on else np.zeros_like(eff)
    sd_true = np.sqrt(max(wavg((actual - used) ** 2, w) - wavg(nvar, w), 0.0))
    return {"alpha": float(alpha), "on": bool(on), "scale": scale, "sd_true": max(sd_true, 0.005)}


class Predictor:
    """Strength model + change model + settings, trained on `rows` (earlier patches)."""

    def __init__(self, rows, X, patches):
        self.base = StrengthModel().fit(rows)
        self.s = choose_settings(rows, X, patches)
        self.change = fit_change(rows, X, self.base, self.s["alpha"])

    def parts(self, rows, X):
        base = self.base.predict(rows)
        raw = pd.Series(self.change.predict(X.loc[rows.index]), index=rows.index)
        raw[rows["affected"] != 1] = 0.0         # only champions touched by the patch get an effect
        best = raw if self.s["on"] else raw * 0.0
        bold = (raw * self.s["scale"]).clip(-MAX_BOLD, MAX_BOLD)
        return base, best, bold

    @property
    def alpha_shown(self):
        """alpha if the best guess is switched on, else inf (shown as 'none')."""
        return self.s["alpha"] if self.s["on"] else np.inf


# --------------------------------------------------------------------------- #
def champ_class(rows):
    cols = [f"class_{c}" for c in CLASSES if f"class_{c}" in rows]
    if not cols:
        return pd.Series("?", index=rows.index)
    m = rows[cols].fillna(0)
    return m.idxmax(axis=1).str.replace("class_", "").where(m.max(axis=1) > 0, "?")


def top_reasons(c, changed, n=3):
    if not changed or not c.abs().sum():
        return ""
    c = c[c.abs() > 1e-5]
    c = c.reindex(c.abs().sort_values(ascending=False).index).head(n)
    return "; ".join(f"{k} {v * 100:+.1f}" for k, v in c.items())


def eval_rows(cp):
    return cp[(cp["games"] >= MIN_GAMES) & cp["prev_winrate_smoothed"].notna()
              & (cp["prev_games"].fillna(0) > 0) & cp["recent_winrate"].notna()]


def direction_rate(pred, actual, mask):
    return float(np.mean(np.sign(pred[mask]) == np.sign(actual[mask]))) if mask.any() else np.nan


def evaluate(cp, X):
    ev = eval_rows(cp)
    patches = sorted(ev["patch"].unique(), key=pkey)
    rows, detail = [], []
    for i, p in enumerate(patches):
        past = patches[:i]
        if len(past) < MIN_TRAIN_PATCHES:
            continue
        tr = ev[ev["patch"].isin(past)]
        te = ev[ev["patch"] == p]
        pr = Predictor(tr, X, past)
        # the simple baseline: last patch's winrate pulled toward 50% by one learned amount
        x = tr["prev_winrate_smoothed"] - 0.5
        k_last = float(np.sum(tr["games"] * x * (tr["winrate"] - 0.5)) / np.sum(tr["games"] * x * x))
        last = 0.5 + k_last * (te["prev_winrate_smoothed"] - 0.5)
        base, best, bold = pr.parts(te, X)
        pred, pred_bold = base + best, base + bold
        w, truth = te["games"], te["winrate"]
        ch = te["changed"] == 1
        pbar = te["winrate_smoothed"]
        floor = np.sqrt(pbar * (1 - pbar) / w) * np.sqrt(2 / np.pi)
        actual = truth - base
        mae = lambda v, m=None: (np.average((v - truth).abs(), weights=w) if m is None else
                                 (np.average((v - truth)[m].abs(), weights=w[m]) if m.any() else np.nan))
        big = ch & (w >= 100) & (bold.abs() > 0.005)
        # the 3 changed champions the model expects to move most this patch
        cand = te[ch & (w >= 100)].index
        top3 = bold.loc[cand].abs().sort_values(ascending=False).index[:3]
        top_mask = pd.Series(False, index=te.index)
        top_mask[top3] = True
        top_mask &= bold.abs() > 0.005
        rows.append({
            "patch": p, "train patches": len(past), "alpha": pr.alpha_shown,
            "alpha_fit": pr.s["alpha"], "bold scale": pr.s["scale"],
            "floor": np.average(floor, weights=w),
            "last": mae(last), "shrunk": mae(base), "model": mae(pred), "bold": mae(pred_bold),
            "changed": int(ch.sum()),
            "floor ch": np.average(floor[ch], weights=w[ch]) if ch.any() else np.nan,
            "last ch": mae(last, ch), "shrunk ch": mae(base, ch), "model ch": mae(pred, ch),
            "bold ch": mae(pred_bold, ch),
            "corr": wcorr(bold[ch].values, actual[ch].values, w[ch].values),
            "direction": direction_rate(bold, actual, big),
            "top3 direction": direction_rate(bold, actual, top_mask),
            "top3 n": int(top_mask.sum()),
        })
        # every champion, with everything needed to see where and why it missed
        d = te[["patch", "champion_id", "games", "role"]].copy()
        d["class"] = champ_class(te)
        d["changed"] = ch.astype(int).values
        d["affected"] = te["affected"].values
        for col in ("item_stat_buff", "item_passive_buff", "item_changed_share"):
            if col in te:
                d[col] = te[col].fillna(0).values
        d["rework"] = X.loc[te.index, "rework"].values if "rework" in X else 0
        d["n_changes"] = te["n_changes"].fillna(0).values
        d["buff_score"] = te["buff_score"].values
        d["prev_winrate"] = te["prev_winrate_smoothed"].values
        d["recent_winrate"] = te["recent_winrate"].values
        d["long_winrate"] = te["long_winrate"].values
        d["last_patch_pred"] = last.values
        d["base_pred"] = base.values
        d["predicted"] = best.values               # best-guess balance effect
        d["pred_winrate"] = pred.values
        d["bold"] = bold.values                    # bold balance effect
        d["bold_winrate"] = pred_bold.values
        d["range_lo"] = (pred - 1.96 * pr.s["sd_true"]).values   # likely true winrate
        d["range_hi"] = (pred + 1.96 * pr.s["sd_true"]).values
        d["winrate"] = truth.values
        d["actual"] = actual.values                # winrate change beyond expected strength
        d["error"] = (pred - truth).values
        d["noise_sd"] = np.sqrt(pbar * (1 - pbar) / w).values
        d["z"] = d["error"] / d["noise_sd"]        # |z| > 2: bigger miss than luck explains
        d["model_on"] = int(pr.s["on"])
        contrib = pr.change.contributions(X.loc[te.index])
        d["why"] = [top_reasons(contrib.loc[i], ch.loc[i]) for i in te.index]
        detail.append(d)
    return pd.DataFrame(rows), (pd.concat(detail) if detail else pd.DataFrame())


def print_eval(res, detail):
    if res.empty:
        print(f"Not enough history yet: need {MIN_TRAIN_PATCHES + 1}+ consecutive patches with games.")
        return
    f = lambda v: "    -" if pd.isna(v) else f"{v * 100:5.2f}"
    g = lambda v: "    -" if pd.isna(v) else f"{v:5.2f}"
    a = lambda v: "  none" if not np.isfinite(v) else f"{v:6.0f}"
    print("Average miss in winrate points (lower is better) - each patch predicted from earlier ones")
    print("  last   = last patch's winrate pulled toward 50% (simple baseline)")
    print("  base   = strength model (recent + long-term winrate, popularity)")
    print("  best   = base + best-guess effect of the balance changes")
    print("  bold   = base + bold effect (sized like real patch swings)")
    print("  alpha  = how strongly the best guess is held back ('none' = switched off)\n")
    print(f"{'patch':>7} {'alpha':>6} | {'ALL champions':^27} | {'CHANGED champions only':^44}")
    print(f"{'':>7} {'':>6} | {'floor':>5} {'last':>6} {'base':>6} {'best':>6} | {'n':>3} {'floor':>5} "
          f"{'last':>6} {'base':>6} {'best':>6} {'bold':>6} {'dir':>5}")
    for _, r in res.iterrows():
        print(f"{r['patch']:>7} {a(r['alpha'])} | {f(r['floor'])} {f(r['last']):>6} {f(r['shrunk']):>6} "
              f"{f(r['model']):>6} | {int(r['changed']):>3} {f(r['floor ch'])} {f(r['last ch']):>6} "
              f"{f(r['shrunk ch']):>6} {f(r['model ch']):>6} {f(r['bold ch']):>6} {g(r['direction'])}")
    w = res["changed"].clip(lower=1)
    avg = lambda c, ww=None: np.average(res[c].fillna(0), weights=ww) * 100
    print(f"\nAll champions:     last {avg('last'):.2f} | base {avg('shrunk'):.2f} | best "
          f"{avg('model'):.2f}   (noise floor {avg('floor'):.2f})")
    print(f"Changed champions: last {avg('last ch', w):.2f} | base {avg('shrunk ch', w):.2f} | best "
          f"{avg('model ch', w):.2f} | bold {avg('bold ch', w):.2f}   (noise floor "
          f"{avg('floor ch', w):.2f})")
    on = int(np.isfinite(res["alpha"]).sum())
    print(f"Best guess switched on for {on} of {len(res)} patches. Bold scale: "
          f"{res['bold scale'].iloc[-1]:.1f}x on the latest patch.")
    print(f"Bold direction right (changed champions, 100+ games): {np.nanmean(res['direction']):.0%}"
          f"   |   for the 3 biggest predicted movers per patch: "
          f"{np.nansum(res['top3 direction'] * res['top3 n']) / max(res['top3 n'].sum(), 1):.0%} "
          f"of {int(res['top3 n'].sum())}")
    print("  (50% = coin flip)")
    if len(detail):
        dd = detail[detail["changed"] == 1]
        top = dd.reindex(dd["bold"].abs().sort_values(ascending=False).index).head(15)
        top = top[top["bold"] != 0]
        if len(top):
            print("\nBiggest bold predictions vs what happened (change beyond expected strength):")
            for _, r in top.iterrows():
                hit = "right" if np.sign(r["bold"]) == np.sign(r["actual"]) else "wrong"
                print(f"  {r['patch']:>6} {r['champion_id']:<14} bold {r['bold'] * 100:+5.1f}  best "
                      f"{r['predicted'] * 100:+5.1f}  actual {r['actual'] * 100:+5.1f}  "
                      f"({int(r['games'])} games, {hit} direction)")


def predict_patch(cp, X, target):
    ev = eval_rows(cp)
    ev = ev[ev["patch"].map(pkey) < pkey(target)]
    past = sorted(ev["patch"].unique(), key=pkey)
    if len(past) < MIN_TRAIN_PATCHES:
        sys.exit("Not enough earlier patches with games to learn from.")
    pr = Predictor(ev, X, past)
    rows = cp[(cp["patch"] == target) & (cp["recent_games"].fillna(0) > 0)]
    if rows.empty:
        sys.exit(f"No champions found for patch {target} (is its balance data loaded?).")
    base, best, bold = pr.parts(rows, X)
    cols = ["champion_id", "changed", "affected", "n_changes", "recent_winrate", "long_winrate"]
    cols += [c for c in ("item_stat_buff", "item_passive_buff", "item_changed_share") if c in rows]
    out = rows[cols].copy()
    out["changed"] = out["changed"].fillna(0).astype(int)
    out["expected_before_changes"] = base
    out["best_change"] = best
    out["bold_change"] = bold
    out["predicted_winrate"] = base + best
    out["bold_winrate"] = base + bold
    out["likely_low"] = out["predicted_winrate"] - 1.96 * pr.s["sd_true"]
    out["likely_high"] = out["predicted_winrate"] + 1.96 * pr.s["sd_true"]
    out = out.sort_values("bold_winrate", ascending=False)

    pct = lambda v: "     -" if pd.isna(v) else f"{v * 100:5.1f}%"
    print(f"Patch {target}: predictions learned from {len(past)} earlier patches\n")
    d = pr.base.describe()
    print(f"Strength model: a champion is half-trusted after {d['trust_games']:,} games; weights "
          f"recent {d['recent_weight']:.2f}, long-term {d['long_term_weight']:.2f}, "
          f"popularity {d['popularity_weight']:+.2f}")
    print(f"Best guess {'ON' if pr.s['on'] else 'OFF (did not beat no-effect on recent patches)'}; "
          f"bold = {pr.s['scale']:.1f}x the model's raw effect. Likely range: +-"
          f"{1.96 * pr.s['sd_true'] * 100:.1f} pts around the best guess.\n")
    chg = out[out["affected"] == 1].sort_values("bold_change")
    if len(chg):
        print("CHANGED champions, and champions whose usual items changed (sorted by bold change):")
        print(f"  {'champion':<14} {'before':>7} {'best':>6} {'bold':>6}   {'predicted':>9} "
              f"{'bold':>7}   likely range")
        for _, r in chg.iterrows():
            print(f"  {r['champion_id']:<14} {pct(r['expected_before_changes'])} "
                  f"{r['best_change'] * 100:+6.1f} {r['bold_change'] * 100:+6.1f}   "
                  f"{pct(r['predicted_winrate']):>9} {pct(r['bold_winrate']):>7}   "
                  f"{pct(r['likely_low']).strip()} - {pct(r['likely_high']).strip()}")
    for title, part in (("STRONGEST predicted (bold)", out.head(10)),
                        ("WEAKEST predicted (bold)", out.tail(10).iloc[::-1])):
        print(f"\n{title}:")
        for _, r in part.iterrows():
            tag = ("  (changed)" if r["changed"] == 1 else
                   "  (items changed)" if r["affected"] == 1 else "")
            print(f"  {r['champion_id']:<14} {pct(r['bold_winrate'])}   best guess "
                  f"{pct(r['predicted_winrate'])}{tag}")
    os.makedirs("training", exist_ok=True)
    path = os.path.join("training", f"predicted_changes_{target}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path} (all champions)")
    print("\nWhat the change model has learned matters most (standardized weights, points):")
    for name, v in ([] if pr.change.off else pr.change.top(10).items()):
        print(f"  {name:<30} {v * 100:+.2f}")


def save_weights(cp, X, out_dir):
    """What the models learned from ALL patches so far (for the diagnostics report)."""
    ev = eval_rows(cp)
    past = sorted(ev["patch"].unique(), key=pkey)
    pr = Predictor(ev, X, past)
    ch = ev["changed"] == 1
    m = pr.change
    w = pd.DataFrame({"feature": m.cols, "weight": m.coef,
                      "used_by_changed": (X.loc[ev.index[ch], m.cols] != 0).sum().values})
    w = w.reindex(w["weight"].abs().sort_values(ascending=False).index)
    d = pr.base.describe()
    w["alpha"] = pr.s["alpha"]
    w["switched_on"] = int(pr.s["on"])
    w["bold_scale"] = pr.s["scale"]
    w["sd_true"] = pr.s["sd_true"]
    w["k_recent"] = d["recent_weight"]
    w["k_long"] = d["long_term_weight"]
    w["k_popularity"] = d["popularity_weight"]
    w["trust_games"] = d["trust_games"]
    w.to_csv(os.path.join(out_dir, "change_model_weights.csv"), index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model of winrate changes caused by balance changes.")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--predict", default=None, help="predictions for this patch")
    args = ap.parse_args(argv)
    import memlimit
    memlimit.apply()          # stay under the RAM limit (default 7 GB)
    cp, X = build_features(load(args.dir))
    if args.predict:
        predict_patch(cp, X, args.predict)
        return
    res, detail = evaluate(cp, X)
    print_eval(res, detail)
    if not res.empty:
        res.to_csv(os.path.join(args.dir, "change_model_results.csv"), index=False)
        detail.to_csv(os.path.join(args.dir, "change_model_predictions.csv"), index=False)
        save_weights(cp, X, args.dir)


if __name__ == "__main__":
    main()
