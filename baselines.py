#!/usr/bin/env python3
"""
baselines.py  -  the simple predictions every real model has to beat.

For every patch that has a previous patch with games, each baseline predicts each
champion's winrate on the new patch WITHOUT looking at that patch's games, and is
scored against what actually happened. Testing is FORWARD IN TIME: anything a baseline
learns comes only from patches BEFORE the one being tested - exactly the situation
when predicting a real new patch.

  coin flip          every champion 50%
  nothing changes    last patch's winrate
  shrunk last patch  last patch's winrate pulled toward 50% (how much is learned
                     from earlier patches - accounts for winrates being noisy)
  recent patches     the last few patches' winrate (older ones count less), pulled
                     toward 50% - steadier than one patch of noisy games
  balance changes    shrunk last patch + a ridge regression on what changed in the
                     champion's numbers this patch (trained on earlier patches)

Scores (games-weighted, champions with >= 30 games):
  MAE          average miss in winrate points
  noise floor  how close a PERFECT prediction could get, given sampling noise with
               this many games - nothing can beat it, so it tells you what "good" means
  change corr  for champions whose numbers changed: correlation between the predicted
               and actual change in winrate
  direction    for changed champions with >= 100 games: share where the predicted
               direction (up / down) matched the actual one

Usage:  python baselines.py          (after build_training_tables.py)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

MIN_GAMES = 30
MAX_CHANGE = 0.05   # a baseline shouldn't predict a winrate swing bigger than 5 points
# a small set of meaningful "what changed" signals (keeps the baseline from overfitting)
CHANGE_FEATURES = ["hp_l9", "hp_l16", "armor_l9", "mr_l9", "ad_l9", "ad_l16", "as_l9",
                   "stat_movespeed", "total_dmg_base_rmax", "total_dmg_ratio_ap",
                   "total_dmg_ratio_ad_all", "total_heal_base_rmax", "total_shield_base_rmax",
                   "basic_cd_rmax_mean", "ult_cd_rmax", "basic_cost_rmax_mean", "cc_hard_total"]


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def load(path_dir):
    for ext in ("parquet", "csv.gz"):
        p = os.path.join(path_dir, f"champion_patch.{ext}")
        if os.path.exists(p):
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    sys.exit("training/champion_patch not found - run build_training_tables.py first.")


def wmae(pred, truth, w):
    return float(np.sum(w * np.abs(pred - truth)) / np.sum(w))


def wcorr(a, b, w):
    if len(a) < 5:
        return np.nan
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    va, vb = np.average((a - ma) ** 2, weights=w), np.average((b - mb) ** 2, weights=w)
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


def fit_shrink(train):
    """Best k in  pred = 0.5 + k * (prev - 0.5), fitted on other patches."""
    x = train["prev_winrate_smoothed"].values - 0.5
    y = train["winrate"].values - 0.5
    w = train["games"].values
    return float(np.sum(w * x * y) / np.sum(w * x * x)) if np.sum(w * x * x) > 0 else 0.0


def fit_ridge(train, cols, k, alpha):
    """Ridge on standardized deltas, predicting the part the shrunk baseline misses."""
    X = train[cols].values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    base = 0.5 + k * (train["prev_winrate_smoothed"].values - 0.5)
    y = train["winrate"].values - base
    w = train["games"].values
    W = np.sqrt(w)[:, None]
    A = Xs * W
    coef = np.linalg.solve(A.T @ A + alpha * np.eye(len(cols)), A.T @ (y * W[:, 0]))
    return lambda df: 0.5 + k * (df["prev_winrate_smoothed"].values - 0.5) + \
        ((df[cols].values - mu) / sd) @ coef


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate simple baselines per patch.")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--alpha", type=float, default=300.0, help="ridge strength")
    args = ap.parse_args(argv)

    cp = load(args.dir)
    from train_change_model import recent_strength
    cp["recent_winrate"] = recent_strength(cp)
    ev = cp[(cp["games"] >= MIN_GAMES) & cp["prev_winrate_smoothed"].notna()
            & (cp["prev_games"].fillna(0) > 0)].copy()
    patches = sorted(ev["patch"].unique(), key=pkey)
    if not patches:
        sys.exit("No patch with games on both it and the previous patch yet - collect more "
                 "patches first (the collector fills them automatically).")
    # relative changes (e.g. +5% damage), so champions of different scales compare
    delta_cols = []
    for f in CHANGE_FEATURES:
        if f"delta_{f}" in ev and f in ev:
            base = (ev[f] - ev[f"delta_{f}"]).abs().replace(0, np.nan)
            ev[f"rel_{f}"] = (ev[f"delta_{f}"] / base).fillna(0).clip(-1, 1)
            if ev[f"rel_{f}"].abs().sum() > 0:
                delta_cols.append(f"rel_{f}")

    rows, detail = [], []
    for p in patches:
        test = ev[ev["patch"] == p]
        train = ev[ev["patch"].map(pkey) < pkey(p)]   # only the past
        truth, w = test["winrate"].values, test["games"].values
        prev = test["prev_winrate_smoothed"].values
        # sampling noise: expected miss of a perfect prediction
        pbar = test["winrate_smoothed"].values
        floor = float(np.sum(w * np.sqrt(pbar * (1 - pbar) / w) * np.sqrt(2 / np.pi)) / np.sum(w))
        preds = {"coin flip": np.full(len(test), 0.5), "nothing changes": prev}
        k = fit_shrink(train) if len(train) else 0.5   # first patch: no history yet
        preds["shrunk last patch"] = 0.5 + k * (prev - 0.5)
        if len(train):
            tr = train[train["recent_winrate"].notna()]
            x = tr["recent_winrate"].values - 0.5
            kr = float(np.sum(tr["games"] * x * (tr["winrate"] - 0.5)) / np.sum(tr["games"] * x * x))
            rec = test["recent_winrate"].fillna(test["prev_winrate_smoothed"]).values
            preds["recent patches"] = 0.5 + kr * (rec - 0.5)
        if len(train) >= 50 and delta_cols:
            raw = fit_ridge(train, delta_cols, k, args.alpha)(test)
            base = preds["shrunk last patch"]
            preds["balance changes"] = base + np.clip(raw - base, -MAX_CHANGE, MAX_CHANGE)
        changed = (test["changed"] == 1).values
        big = changed & (w >= 100)
        actual_chg = truth - prev
        for name, pred in preds.items():
            pred_chg = pred - prev
            r = {"patch": p, "baseline": name, "champions": len(test),
                 "MAE": wmae(pred, truth, w), "noise floor": floor,
                 "change corr": wcorr(pred_chg[changed], actual_chg[changed], w[changed])
                 if name == "balance changes" else np.nan,
                 "direction": float(np.mean(np.sign(pred_chg[big]) == np.sign(actual_chg[big])))
                 if name == "balance changes" and big.sum() else np.nan,
                 "changed champs": int(changed.sum())}
            rows.append(r)
        if "balance changes" in preds:
            t = test[["patch", "champion_id", "games", "changed", "prev_winrate_smoothed",
                      "winrate"]].copy()
            t["predicted"] = preds["balance changes"]
            detail.append(t)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    fmt = res.copy()
    for c in ("MAE", "noise floor"):
        fmt[c] = (fmt[c] * 100).map(lambda v: f"{v:.2f} pts")
    for c in ("change corr", "direction"):
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    print(fmt.to_string(index=False))

    overall = {}
    for name in dict.fromkeys(res["baseline"]):  # keeps first-seen order
        d = res[res["baseline"] == name]
        overall[name] = {"MAE": np.average(d["MAE"], weights=d["champions"]),
                         "noise floor": np.average(d["noise floor"], weights=d["champions"])}
    print("\nOverall (all evaluated patches):")
    for name, r in overall.items():
        gap = (r["MAE"] - r["noise floor"]) * 100
        print(f"  {name:<18} MAE {r['MAE'] * 100:.2f} pts   (noise floor {r['noise floor'] * 100:.2f}"
              f" pts, {gap:+.2f} above it)")
    print(f"\nEvaluated {len(patches)} patch(es): {', '.join(patches)}. Each was predicted "
          f"using only earlier patches, so early patches have little history to learn from.")
    os.makedirs(args.dir, exist_ok=True)
    res.to_csv(os.path.join(args.dir, "baseline_results.csv"), index=False)
    if detail:
        d = pd.concat(detail)
        d.to_csv(os.path.join(args.dir, "baseline_predictions.csv"), index=False)
        ch = d[d["changed"] == 1].copy()
        if len(ch):
            ch["pred_change"] = ch["predicted"] - ch["prev_winrate_smoothed"]
            ch["actual_change"] = ch["winrate"] - ch["prev_winrate_smoothed"]
            print("\nChanged champions, predicted vs actual winrate change (biggest predictions):")
            top = ch.reindex(ch["pred_change"].abs().sort_values(ascending=False).index).head(12)
            for _, r in top.iterrows():
                print(f"  {r['patch']:>6} {r['champion_id']:<14} predicted {r['pred_change'] * 100:+5.1f}"
                      f"  actual {r['actual_change'] * 100:+5.1f}  ({int(r['games'])} games)")
    print(f"\nSaved {os.path.join(args.dir, 'baseline_results.csv')}")


if __name__ == "__main__":
    main()
