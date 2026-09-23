#!/usr/bin/env python3
"""
train_change_model.py  -  predicts how much each champion's winrate moves when a patch
changes their numbers.

Unlike the game model (which predicts single games and only "notices" balance changes
indirectly), this model learns DIRECTLY from past patches:
    "this champion got +10% AP ratio, is a mage, played mid, was already strong ...
     and their winrate moved by X"

Inputs per champion per patch (all known before the patch goes live):
  - what changed, as a share of the old value (e.g. -8% base damage, +5% armor)
  - a single buff/nerf score (all changes combined, cooldown/cost cuts counted as buffs)
  - how many things changed
  - the champion's class and main role, combined with the changes
    ("AP ratio buff on a mage", "armor nerf on a tank", ...)
  - how strong and how popular the champion was last patch
Target: the winrate change beyond what the champion's recent strength predicts.
"Recent strength" is the last few patches' winrate (older patches count less), pulled
toward 50% by an amount learned from the past. It is steadier than last patch alone,
which with ~500 games per champion is mostly luck.

The strength of the change model (ridge alpha) is picked by testing on the latest
training patches. If no setting beats "no effect", the model predicts no effect: that
is the data saying the games are still too noisy to tell (more games fix this).

Testing is forward in time: each patch is predicted from earlier patches only.
Compared against "shrunk last patch" (the best simple baseline), overall and on the
champions that were actually changed (where the model matters).

Usage:
  python train_change_model.py                 evaluate on past patches
  python train_change_model.py --predict 16.18 predicted winrate changes for a patch
                                               (works for a new patch before any games)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

MIN_GAMES = 30
MAX_EFFECT = 0.10             # predictions are capped at +-10 winrate points
MIN_TRAIN_PATCHES = 2
ALPHAS = [30, 100, 300, 1000, 3000, 10000, np.inf]   # inf = "no effect" (data too noisy)
REWORK_CHANGES = 25           # this many changed numbers = a rework: relative changes are meaningless
MAX_REL = 0.5                 # a single relative change counts at most +-50%
SWITCH_ON_GAIN = 0.01         # change model must beat "no effect" by 1% on validation to be used
HIST_DECAY = 0.7              # weight of each older patch when judging a champion's strength

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
# changes that get combined with class / role
KEY_CHANGES = ["total_dmg_base_rmax", "total_dmg_ratio_ap", "total_dmg_ratio_ad_all",
               "hp_l9", "armor_l9", "mr_l9", "ad_l9", "as_l9", "basic_cd_rmax_mean",
               "ult_cd_rmax", "total_heal_base_rmax", "total_shield_base_rmax"]
CLASSES = ["assassin", "fighter", "mage", "marksman", "support", "tank"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def load(path_dir):
    for ext in ("parquet", "csv.gz"):
        p = os.path.join(path_dir, f"champion_patch.{ext}")
        if os.path.exists(p):
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    sys.exit("training/champion_patch not found - run build_training_tables.py first.")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def recent_strength(cp, decay=HIST_DECAY):
    """Winrate over EARLIER patches only, older patches down-weighted, 25-25 prior."""
    order = {p: i for i, p in enumerate(sorted(cp["patch"].unique(), key=pkey))}
    pi = cp["patch"].map(order)
    wins = (cp["winrate"].fillna(0) * cp["games"].fillna(0))
    W = pd.crosstab(cp["champ_key"], pi, values=wins, aggfunc="sum").fillna(0)
    G = pd.crosstab(cp["champ_key"], pi, values=cp["games"].fillna(0), aggfunc="sum").fillna(0)
    W = W.reindex(columns=range(len(order)), fill_value=0)
    G = G.reindex(columns=range(len(order)), fill_value=0)
    sw = np.zeros(len(W)); sg = np.zeros(len(W))
    hist, hist_games = {}, {}
    for j in range(len(order)):
        hist[j] = pd.Series((sw + 25) / (sg + 50), index=W.index)
        hist_games[j] = pd.Series(sg, index=W.index)
        sw = sw * decay + W[j].values
        sg = sg * decay + G[j].values
    h = np.array([hist[i][c] for i, c in zip(pi, cp["champ_key"])])
    hg = np.array([hist_games[i][c] for i, c in zip(pi, cp["champ_key"])])
    return pd.Series(np.where(hg > 0, h, np.nan), index=cp.index)


def build_features(cp):
    cp = cp.copy()
    # main role: from this patch's games, else the latest earlier patch's
    cp = cp.sort_values("patch", key=lambda s: s.map(pkey))
    cp["recent_winrate"] = recent_strength(cp)
    cp["role"] = cp.groupby("champ_key")["main_role"].transform(lambda s: s.ffill())
    feats = {}
    rel_cols = []
    rework = cp["n_changes"].fillna(0) >= REWORK_CHANGES
    for f, direction in CHANGE_FEATURES.items():
        if f not in cp or f"delta_{f}" not in cp:
            continue
        d = cp[f"delta_{f}"].fillna(0)
        old = (cp[f] - d).abs()
        # a brand-new component (old value 0) has no meaningful % change: left at 0
        rel = pd.Series(d / old.where(old > 1e-9), index=cp.index).fillna(0).clip(-MAX_REL, MAX_REL)
        rel[rework] = 0.0
        # numbers that moved without counting as a balance change (renamed labels, rounding)
        rel[cp["changed"].fillna(0) != 1] = 0.0
        feats[f"rel_{f}"] = rel
        rel_cols.append((f, direction))
    feats["buff_score"] = sum(feats[f"rel_{f}"] * direction for f, direction in rel_cols)
    feats["n_changes"] = np.log1p(cp["n_changes"].fillna(0))
    feats["changed"] = cp["changed"].fillna(0)
    feats["rework"] = rework.astype(float)
    feats["prev_strength"] = (cp["recent_winrate"] - 0.5).fillna(0)
    feats["prev_presence"] = cp["prev_pickrate"].fillna(0) + cp["prev_banrate"].fillna(0)
    for c in CLASSES:
        col = f"class_{c}"
        if col in cp:
            for f in KEY_CHANGES:
                if f"rel_{f}" in feats:
                    feats[f"{c}_x_{f}"] = cp[col].fillna(0) * feats[f"rel_{f}"]
            feats[f"{c}_x_buff"] = cp[col].fillna(0) * feats["buff_score"]
    for r in ROLES:
        is_r = (cp["role"] == r).astype(float)
        feats[f"{r}_x_buff"] = is_r * feats["buff_score"]
        for f in ("total_dmg_base_rmax", "hp_l9", "ad_l9"):
            if f"rel_{f}" in feats:
                feats[f"{r}_x_{f}"] = is_r * feats[f"rel_{f}"]
    feats["buff_x_strength"] = feats["buff_score"] * feats["prev_strength"]
    X = pd.DataFrame(feats, index=cp.index)
    X = X.loc[:, X.abs().sum() > 0]
    return cp, X


# --------------------------------------------------------------------------- #
# Model: weighted ridge regression on standardized features
# --------------------------------------------------------------------------- #
def fit_shrink(rows, col="recent_winrate"):
    x = rows[col].values - 0.5
    y = rows["winrate"].values - 0.5
    w = rows["games"].values
    den = np.sum(w * x * x)
    return float(np.sum(w * x * y) / den) if den > 0 else 0.5


class ChangeModel:
    def fit(self, X, y, w, alpha):
        self.cols = list(X.columns)
        self.off = not np.isfinite(alpha) or len(y) < 10
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

    def predict(self, X):
        if self.off:
            return np.zeros(len(X))
        Z = np.clip((X[self.cols].values.astype(float) - self.mu) / self.sd, -4, 4)
        return np.clip(self.b0 + Z @ self.coef, -MAX_EFFECT, MAX_EFFECT)

    def contributions(self, X):
        """Per row: how much each input pushed the prediction (points of winrate)."""
        if self.off:
            return pd.DataFrame(0.0, index=X.index, columns=self.cols)
        Z = np.clip((X[self.cols].values.astype(float) - self.mu) / self.sd, -4, 4)
        return pd.DataFrame(Z * self.coef, index=X.index, columns=self.cols)

    def base(self, rows, k):
        return 0.5 + k * (rows["recent_winrate"] - 0.5)

    def top(self, n=10):
        s = pd.Series(self.coef, index=self.cols)
        return s.reindex(s.abs().sort_values(ascending=False).index).head(n)


def choose_alpha(train_rows, X, patches):
    """Pick regularisation by predicting the last few training patches from earlier ones."""
    val_patches = patches[-3:] if len(patches) >= 4 else patches[-1:]
    best, best_err, base_err = np.inf, np.inf, np.inf
    for a in [np.inf] + [a for a in ALPHAS if np.isfinite(a)]:   # "no effect" wins ties
        err = n = 0.0
        for vp in val_patches:
            tr = train_rows[train_rows["patch"].map(pkey) < pkey(vp)]
            va = train_rows[(train_rows["patch"] == vp) & (train_rows["changed"] == 1)]
            if va.empty or (tr["changed"] == 1).sum() < 10:
                continue
            m, k = fit_model(tr, X, a)
            pred = m.base(va, k) + m.predict(X.loc[va.index])
            err += float(np.sum(va["games"] * np.abs(pred - va["winrate"])))
            n += float(va["games"].sum())
        if not n:
            continue
        if not np.isfinite(a):
            base_err = best_err = err / n
        elif err / n < best_err and err / n < base_err * (1 - SWITCH_ON_GAIN):
            best, best_err = a, err / n
    return best


def fit_model(rows, X, alpha):
    """Drift factor from all champions; change model from changed champions only."""
    k = fit_shrink(rows)
    ch = rows[rows["changed"] == 1]
    y = ch["winrate"] - (0.5 + k * (ch["recent_winrate"] - 0.5))
    return ChangeModel().fit(X.loc[ch.index], y.values, ch["games"].values, alpha), k


def train(rows, X, patches):
    alpha = choose_alpha(rows, X, patches) if len(patches) >= 3 else np.inf
    model, k = fit_model(rows, X, alpha)
    return model, k, alpha


# --------------------------------------------------------------------------- #
def wcorr(a, b, w):
    if len(a) < 5:
        return np.nan
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    va, vb = np.average((a - ma) ** 2, weights=w), np.average((b - mb) ** 2, weights=w)
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


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


def evaluate(cp, X):
    ev = cp[(cp["games"] >= MIN_GAMES) & cp["prev_winrate_smoothed"].notna()
            & (cp["prev_games"].fillna(0) > 0) & cp["recent_winrate"].notna()]
    patches = sorted(ev["patch"].unique(), key=pkey)
    rows, detail = [], []
    for i, p in enumerate(patches):
        past = patches[:i]
        if len(past) < MIN_TRAIN_PATCHES:
            continue
        tr = ev[ev["patch"].isin(past)]
        te = ev[ev["patch"] == p]
        model, k, alpha = train(tr, X, past)
        k_last = fit_shrink(tr, "prev_winrate_smoothed")
        last = 0.5 + k_last * (te["prev_winrate_smoothed"] - 0.5)   # the simple baseline
        shrunk = model.base(te, k)                                     # recent patches
        effect = pd.Series(model.predict(X.loc[te.index]), index=te.index)
        effect[te["changed"] != 1] = 0.0           # only changed champions get an effect
        pred = shrunk + effect
        w, truth = te["games"], te["winrate"]
        ch = te["changed"] == 1
        pbar = te["winrate_smoothed"]
        floor = np.sqrt(pbar * (1 - pbar) / w) * np.sqrt(2 / np.pi)
        actual = truth - shrunk
        big = ch & (w >= 100) & (effect.abs() > 0.002)
        rows.append({
            "patch": p, "train patches": len(past), "alpha": alpha,
            "floor": np.average(floor, weights=w),
            "last": np.average((last - truth).abs(), weights=w),
            "shrunk": np.average((shrunk - truth).abs(), weights=w),
            "model": np.average((pred - truth).abs(), weights=w),
            "changed": int(ch.sum()),
            "floor ch": np.average(floor[ch], weights=w[ch]) if ch.any() else np.nan,
            "last ch": np.average((last - truth)[ch].abs(), weights=w[ch]) if ch.any() else np.nan,
            "shrunk ch": np.average((shrunk - truth)[ch].abs(), weights=w[ch]) if ch.any() else np.nan,
            "model ch": np.average((pred - truth)[ch].abs(), weights=w[ch]) if ch.any() else np.nan,
            "corr": wcorr(effect[ch].values, actual[ch].values, w[ch].values),
            "direction": float(np.mean(np.sign(effect[big]) == np.sign(actual[big]))) if big.any() else np.nan,
        })
        # every champion, with everything needed to see where and why it missed
        d = te[["patch", "champion_id", "games", "role"]].copy()
        d["class"] = champ_class(te)
        d["changed"] = ch.astype(int).values
        d["rework"] = X.loc[te.index, "rework"].values if "rework" in X else 0
        d["n_changes"] = te["n_changes"].fillna(0).values
        d["buff_score"] = X.loc[te.index, "buff_score"].values
        d["prev_winrate"] = te["prev_winrate_smoothed"].values
        d["recent_winrate"] = te["recent_winrate"].values
        d["last_patch_pred"] = last.values
        d["base_pred"] = shrunk.values
        d["predicted"] = effect.values            # balance-change effect only
        d["pred_winrate"] = pred.values
        d["winrate"] = truth.values
        d["actual"] = actual.values                # winrate change beyond recent strength
        d["error"] = (pred - truth).values
        d["noise_sd"] = np.sqrt(pbar * (1 - pbar) / w).values
        d["z"] = d["error"] / d["noise_sd"]        # |z| > 2: bigger miss than luck explains
        d["model_on"] = int(np.isfinite(alpha))
        contrib = model.contributions(X.loc[te.index])
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
    print("  recent = last few patches' winrate pulled toward 50%")
    print("  model  = recent + predicted effect of the balance changes")
    print("  alpha  = how strongly the change model is held back ('none' = data too noisy, no effect)\n")
    print(f"{'patch':>7} {'alpha':>6} | {'ALL champions':^26} | {'CHANGED champions only':^38}")
    print(f"{'':>7} {'':>6} | {'floor':>5} {'last':>6} {'recent':>6} {'model':>6} | {'n':>3} {'floor':>5} "
          f"{'last':>6} {'recent':>6} {'model':>6} {'corr':>5}")
    for _, r in res.iterrows():
        print(f"{r['patch']:>7} {a(r['alpha'])} | {f(r['floor'])} {f(r['last']):>6} {f(r['shrunk']):>6} "
              f"{f(r['model']):>6} | {int(r['changed']):>3} {f(r['floor ch'])} {f(r['last ch']):>6} "
              f"{f(r['shrunk ch']):>6} {f(r['model ch']):>6} {g(r['corr'])}")
    w = res["changed"].clip(lower=1)
    avg = lambda c, ww=None: np.average(res[c].fillna(0), weights=ww) * 100
    print(f"\nAll champions:     last {avg('last'):.2f} | recent {avg('shrunk'):.2f} | model "
          f"{avg('model'):.2f}   (noise floor {avg('floor'):.2f})")
    print(f"Changed champions: last {avg('last ch', w):.2f} | recent {avg('shrunk ch', w):.2f} | model "
          f"{avg('model ch', w):.2f}   (noise floor {avg('floor ch', w):.2f})")
    on = res[np.isfinite(res["alpha"])]
    print(f"Change model switched on for {len(on)} of {len(res)} patches.")
    if len(on):
        print(f"Direction right when on (changed champions, 100+ games): {np.nanmean(on['direction']):.0%}")
    if len(detail):
        detail = detail[detail["changed"] == 1]
    if len(detail) and detail["predicted"].abs().max() > 0:
        top = detail.reindex(detail["predicted"].abs().sort_values(ascending=False).index).head(15)
        top = top[top["predicted"] != 0]
        print("\nBiggest predictions vs what happened (winrate change beyond recent strength):")
        for _, r in top.iterrows():
            print(f"  {r['patch']:>6} {r['champion_id']:<14} predicted {r['predicted'] * 100:+5.1f}  "
                  f"actual {r['actual'] * 100:+5.1f}  ({int(r['games'])} games)")


def predict_patch(cp, X, target):
    ev = cp[(cp["games"] >= MIN_GAMES) & cp["prev_winrate_smoothed"].notna()
            & (cp["prev_games"].fillna(0) > 0) & (cp["patch"].map(pkey) < pkey(target))]
    past = sorted(ev["patch"].unique(), key=pkey)
    if len(past) < MIN_TRAIN_PATCHES:
        sys.exit("Not enough earlier patches with games to learn from.")
    model, k, alpha = train(ev, X, past)
    rows = cp[(cp["patch"] == target) & (cp["changed"] == 1)]
    if rows.empty:
        sys.exit(f"No changed champions found for patch {target} (is its balance data loaded?).")
    eff = model.predict(X.loc[rows.index])
    out = rows[["champion_id", "prev_winrate_smoothed", "recent_winrate", "prev_games",
                "n_changes"]].copy()
    out["predicted_change"] = eff
    base = 0.5 + k * (out["recent_winrate"].fillna(0.5) - 0.5)
    out["predicted_winrate"] = base + eff
    out = out.sort_values("predicted_change")
    print(f"Predicted effect of patch {target}'s balance changes (learned from {len(past)} earlier "
          f"patches):\n")
    if model.off:
        print("  (change model is switched off: on recent patches it didn't beat 'no effect', so\n"
              "   these are recent strength only - more games per patch will switch it on)\n")
    print(f"  {'champion':<14} {'change':>7}   {'recent':>10}  ->  {'predicted':>9}")
    for _, r in out.iterrows():
        last = "         -" if pd.isna(r["recent_winrate"]) else f"{r['recent_winrate']:10.1%}"
        print(f"  {r['champion_id']:<14} {r['predicted_change'] * 100:+6.1f}   {last}  ->  "
              f"{r['predicted_winrate']:9.1%}")
    os.makedirs("training", exist_ok=True)
    path = os.path.join("training", f"predicted_changes_{target}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")
    print("\nWhat the model has learned matters most (standardized weights):")
    for name, v in ([] if model.off else model.top(10).items()):
        print(f"  {name:<36} {v * 100:+.2f}")


def save_weights(cp, X, out_dir):
    """What the model learned from ALL patches so far (for the diagnostics report)."""
    ev = cp[(cp["games"] >= MIN_GAMES) & cp["prev_winrate_smoothed"].notna()
            & (cp["prev_games"].fillna(0) > 0) & cp["recent_winrate"].notna()]
    past = sorted(ev["patch"].unique(), key=pkey)
    model, k, alpha = train(ev, X, past)
    ch = ev["changed"] == 1
    w = pd.DataFrame({"feature": model.cols, "weight": model.coef,
                      "used_by_changed": (X.loc[ev.index[ch], model.cols] != 0).sum().values})
    w = w.reindex(w["weight"].abs().sort_values(ascending=False).index)
    w["alpha"], w["k_recent"] = alpha, k
    w.to_csv(os.path.join(out_dir, "change_model_weights.csv"), index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model of winrate changes caused by balance changes.")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--predict", default=None, help="predict changes for this patch")
    args = ap.parse_args(argv)
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
