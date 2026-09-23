#!/usr/bin/env python3
"""
train_game_model.py  -  the main model: "given these two teams on this patch, who wins?"

A LightGBM model (thousands of small decision trees) learns from every collected game.
Each team is described by its champions' NUMBERS on that patch (stats, ability damage,
crowd control, class mix, lane matchups ...) - see build_training_tables.py.

What this script does:
  1. Honest test, forward in time: for each patch, train only on EARLIER patches and
     predict that patch's games. Scores: log loss / AUC / accuracy vs a coin flip.
  2. Patch effect: take the previous patch's real games, swap in the new patch's
     champion numbers, and measure how each champion's predicted win chance moves.
     That is the model's estimate of what the balance changes did. Added to the
     "shrunk last patch" baseline, it gives a winrate prediction per champion, scored
     against what really happened (same scoring as baselines.py).
  3. Two versions:  numbers   - champions only as numbers (can generalise to new metas)
                    identity  - numbers + which champion it is (can memorise champions)
  4. Trains the final models on all games and saves them to models/.

Usage:
  pip install lightgbm pandas numpy pyarrow
  python train_game_model.py                   evaluate + train final models
  python train_game_model.py --predict 16.18   predicted winrate changes for a patch,
                                               from the previous patch's games
"""

import argparse
import json
import os
import sys
import warnings

# Windows 11 lacks the tool a helper library uses to count CPU cores; tell it directly
# (silences a long, harmless warning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*categorical_feature.*")
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("LightGBM is not installed. Run:  pip install lightgbm")

import build_training_tables as btt  # same folder: reuses the feature code

POSITIONS = btt.POSITIONS
MIN_TEST_GAMES = 1000
MIN_CHAMP_GAMES = 30
MAX_CHANGE = 0.05
# Cautious settings: small trees that need lots of games to form a rule. With tens of
# thousands of games (and games being close to coin flips in high elo), bigger trees just
# memorise luck.
PARAMS = dict(n_estimators=4000, learning_rate=0.01, num_leaves=7, min_child_samples=800,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=20.0,
              verbose=-1)


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def load_table(path_dir, name):
    for ext in ("parquet", "csv.gz"):
        p = os.path.join(path_dir, f"{name}.{ext}")
        if os.path.exists(p):
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    sys.exit(f"{path_dir}/{name} not found - run build_training_tables.py first.")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
ID_COLS = [f"{s}_{p}_champ" for s in ("blue", "red") for p in POSITIONS]


def number_cols(df):
    return [c for c in df.columns if c.startswith(("diff_", "lane_"))]


def add_context(df, platforms):
    out = pd.DataFrame(index=df.index)
    out["ctx_master"] = (df["rank_bucket"] == "MASTER+").astype(float)
    for p in platforms:
        out[f"ctx_{p}"] = (df["platform"] == p).astype(float)
    return out


def mirror(X, variant):
    """Same game seen from the red side: swap teams, flip differences."""
    M = X.copy()
    for c in X.columns:
        if c.startswith(("diff_", "lane_")):
            M[c] = -X[c]
    if variant == "identity":
        for p in POSITIONS:
            M[f"blue_{p}_champ"], M[f"red_{p}_champ"] = X[f"red_{p}_champ"], X[f"blue_{p}_champ"]
    M["ctx_blue_side"] = 1.0 - X["ctx_blue_side"]
    return M


def design(df, variant, platforms, num_cols):
    X = df[num_cols].astype(float).copy()
    X = pd.concat([X, add_context(df, platforms)], axis=1)
    X["ctx_blue_side"] = 1.0
    if variant == "identity":
        for c in ID_COLS:
            X[c] = df[c].astype("float").fillna(-1).astype(int)
    return X


class GameModel:
    """
    LightGBM "does blue win?" model. Data is handed to LightGBM as plain, contiguous
    number arrays (the most compatible format - avoids crashes some Windows installs
    have with pandas input).
    """

    def __init__(self, variant, platforms, num_cols):
        self.variant, self.platforms, self.num_cols = variant, platforms, num_cols
        self.booster = None
        self.cat_levels = None   # champion keys, identity model only
        self.names = None

    def _matrix(self, X):
        X = X.copy()
        if self.variant == "identity":
            code = {k: i for i, k in enumerate(self.cat_levels)}
            for c in ID_COLS:
                X[c] = X[c].map(code).astype(float)          # unknown champion -> NaN
        return np.ascontiguousarray(X.to_numpy(dtype=np.float32))

    @staticmethod
    def _params(seed):
        p = PARAMS
        return {"objective": "binary", "learning_rate": p["learning_rate"],
                "num_leaves": p["num_leaves"], "min_data_in_leaf": p["min_child_samples"],
                "bagging_fraction": p["subsample"], "bagging_freq": p["subsample_freq"],
                "feature_fraction": p["colsample_bytree"], "lambda_l2": p["reg_lambda"],
                "verbose": -1, "seed": seed, "num_threads": os.cpu_count() or 1}

    def fit(self, df, seed=0):
        X = design(df, self.variant, self.platforms, self.num_cols)
        y = df["blue_win"].to_numpy(dtype=np.float64)
        if self.variant == "identity":
            self.cat_levels = sorted(int(v) for v in np.unique(X[ID_COLS].values) if v >= 0)
        rng = np.random.default_rng(seed)
        val = rng.random(len(X)) < 0.15             # validation split, by game
        Xa = pd.concat([X, mirror(X, self.variant)], ignore_index=True)
        self.names = list(Xa.columns)
        M = self._matrix(Xa)
        ya = np.ascontiguousarray(np.concatenate([y, 1.0 - y]))
        va = np.concatenate([val, val])
        cats = [self.names.index(c) for c in ID_COLS] if self.variant == "identity" else "auto"
        dtrain = lgb.Dataset(np.ascontiguousarray(M[~va]), label=np.ascontiguousarray(ya[~va]),
                             feature_name=self.names, categorical_feature=cats)
        dval = lgb.Dataset(np.ascontiguousarray(M[va]), label=np.ascontiguousarray(ya[va]),
                           reference=dtrain)
        self.booster = lgb.train(self._params(seed), dtrain,
                                 num_boost_round=PARAMS["n_estimators"], valid_sets=[dval],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
        self.trees = self.booster.best_iteration or PARAMS["n_estimators"]
        return self

    def predict(self, df):
        """P(blue wins), averaged over both side views so it's symmetric."""
        X = design(df, self.variant, self.platforms, self.num_cols)
        p1 = self.booster.predict(self._matrix(X), num_iteration=self.trees)
        p2 = self.booster.predict(self._matrix(mirror(X, self.variant)), num_iteration=self.trees)
        return (p1 + (1 - p2)) / 2

    def importance(self, top=15):
        imp = pd.Series(self.booster.feature_importance("gain"),
                        index=self.booster.feature_name())
        return (imp / imp.sum()).sort_values(ascending=False).head(top)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def wcorr(a, b, w):
    if len(a) < 5:
        return np.nan
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    va, vb = np.average((a - ma) ** 2, weights=w), np.average((b - mb) ** 2, weights=w)
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


# --------------------------------------------------------------------------- #
# Patch effect: re-score last patch's games with the new patch's numbers
# --------------------------------------------------------------------------- #
class Rescorer:
    def __init__(self, db):
        import sqlite3
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
        self.vec, self.feats = btt.champion_vectors(con)
        self.m, self.p, _ = btt.load_games(con)
        self.names = dict(zip(self.vec["champ_key"], self.vec["champion_id"]))

    def features_as(self, games_prev, target_patch):
        """games_prev's drafts, but with every champion's numbers from target_patch."""
        m = self.m[self.m["match_id"].isin(set(games_prev["match_id"]))].copy()
        m["patch"] = target_patch
        p = self.p[self.p["match_id"].isin(set(m["match_id"]))]
        tf, _ = btt.team_features(m, p, self.vec, self.feats)
        ctx = games_prev.set_index("match_id")[["platform", "rank_bucket", "patch", "blue_win"]]
        return tf.join(ctx, how="inner").reset_index()

    def champion_effect(self, model, games_prev, target_patch):
        """Per champion: mean predicted win chance with new numbers minus old numbers."""
        new = self.features_as(games_prev, target_patch)
        old = games_prev[games_prev["match_id"].isin(set(new["match_id"]))]
        new = new.set_index("match_id").loc[old["match_id"]].reset_index()
        p_old, p_new = model.predict(old), model.predict(new)
        rows = []
        for side, team in (("blue", 100), ("red", 200)):
            sign_old = p_old if team == 100 else 1 - p_old
            sign_new = p_new if team == 100 else 1 - p_new
            for pos in POSITIONS:
                rows.append(pd.DataFrame({"champ_key": old[f"{side}_{pos}_champ"].values,
                                          "p_old": sign_old, "p_new": sign_new}))
        r = pd.concat(rows)
        eff = r.groupby("champ_key").agg(p_old=("p_old", "mean"), p_new=("p_new", "mean"),
                                         sample=("p_old", "size"))
        eff["effect"] = eff["p_new"] - eff["p_old"]
        eff["champion_id"] = eff.index.map(self.names)
        return eff.reset_index()


def shrink_k(cp, exclude):
    t = cp[(cp["patch"] != exclude) & (cp["games"] >= MIN_CHAMP_GAMES)
           & cp["prev_winrate_smoothed"].notna() & (cp["prev_games"].fillna(0) > 0)]
    if t.empty:
        return 0.5
    x, y, w = t["prev_winrate_smoothed"] - 0.5, t["winrate"] - 0.5, t["games"]
    den = float(np.sum(w * x * x))
    return float(np.sum(w * x * y) / den) if den > 0 else 0.5


# --------------------------------------------------------------------------- #
def evaluate(games, cp, rescorer, platforms, num_cols):
    patches = sorted(games["patch"].unique(), key=pkey)
    counts = games.groupby("patch").size()
    results, champ_rows = [], []
    for i, test_patch in enumerate(patches):
        earlier = patches[:i]
        if counts[test_patch] < MIN_TEST_GAMES or not earlier:
            continue
        train = games[games["patch"].isin(earlier)]
        test = games[games["patch"] == test_patch]
        y = test["blue_win"].values
        base_p = np.full(len(test), train["blue_win"].mean())
        row = {"patch": test_patch, "train games": len(train), "test games": len(test),
               "coin logloss": logloss(y, base_p)}
        # patch-level truth for this patch
        truth = cp[(cp["patch"] == test_patch) & (cp["games"] >= MIN_CHAMP_GAMES)
                   & cp["prev_winrate_smoothed"].notna() & (cp["prev_games"].fillna(0) > 0)]
        k = shrink_k(cp, test_patch)
        prev_patch = earlier[-1]
        prev_games = games[games["patch"] == prev_patch]
        for variant in ("numbers", "identity"):
            gm = GameModel(variant, platforms, num_cols).fit(train)
            p = gm.predict(test)
            row[f"{variant} logloss"] = logloss(y, p)
            row[f"{variant} AUC"] = auc(y, p)
            row[f"{variant} acc"] = float(np.mean((p > 0.5) == (y == 1)))
            if len(truth):
                eff = rescorer.champion_effect(gm, prev_games, test_patch)
                t = truth.merge(eff[["champ_key", "effect"]], on="champ_key", how="left")
                t["effect"] = t["effect"].fillna(0).clip(-MAX_CHANGE, MAX_CHANGE)
                shrunk = 0.5 + k * (t["prev_winrate_smoothed"] - 0.5)
                pred = shrunk + t["effect"]
                w = t["games"].values
                row[f"{variant} champ MAE"] = float(np.average(np.abs(pred - t["winrate"]), weights=w))
                row["shrunk MAE"] = float(np.average(np.abs(shrunk - t["winrate"]), weights=w))
                pbar = t["winrate_smoothed"].values
                row["noise floor"] = float(np.average(np.sqrt(pbar * (1 - pbar) / w)
                                                      * np.sqrt(2 / np.pi), weights=w))
                ch = (t["changed"] == 1).values
                actual = (t["winrate"] - t["prev_winrate_smoothed"]).values
                predc = (pred - t["prev_winrate_smoothed"]).values
                row[f"{variant} change corr"] = wcorr(t["effect"].values[ch], actual[ch], w[ch])
                big = ch & (w >= 100) & (np.abs(t["effect"].values) > 0.001)
                row[f"{variant} direction"] = float(np.mean(np.sign(t["effect"].values[big])
                                                            == np.sign(actual[big]))) if big.sum() else np.nan
                if variant == "numbers":
                    tt = t[ch][["patch", "champion_id", "games", "effect"]].copy()
                    tt["actual_change"] = actual[ch]
                    champ_rows.append(tt)
        results.append(row)
        print(f"  tested {test_patch}: trained on {len(train):,} games from {len(earlier)} "
              f"earlier patch(es)")
    return pd.DataFrame(results), (pd.concat(champ_rows, ignore_index=True) if champ_rows
                                   else pd.DataFrame())


def print_results(res, champs):
    if res.empty:
        print("Not enough patches yet: need a patch with 1,000+ games and an earlier patch.")
        return
    print("\nGAME LEVEL - predicting who wins each game of a patch the model never saw")
    print(f"{'patch':>7} {'train':>8} {'test':>6} | {'coin':>6} | {'numbers':>8} {'AUC':>5} {'acc':>5}"
          f" | {'identity':>8} {'AUC':>5} {'acc':>5}")
    for _, r in res.iterrows():
        print(f"{r['patch']:>7} {r['train games']:>8,} {r['test games']:>6,} | {r['coin logloss']:.4f} |"
              f" {r['numbers logloss']:.4f}  {r['numbers AUC']:.3f} {r['numbers acc']:.3f} |"
              f" {r['identity logloss']:.4f}  {r['identity AUC']:.3f} {r['identity acc']:.3f}")
    print("  (log loss: lower is better, coin flip = 0.693. AUC: 0.5 = no skill.)")
    if "shrunk MAE" in res:
        print("\nCHAMPION LEVEL - predicted winrate per champion on the new patch (avg miss, points)")
        print(f"{'patch':>7} | {'floor':>5} {'shrunk':>6} | {'numbers':>7} {'corr':>5} {'dir':>4} |"
              f" {'identity':>8} {'corr':>5} {'dir':>4}")
        for _, r in res.iterrows():
            f = lambda v: "  -" if pd.isna(v) else f"{v:.2f}"
            print(f"{r['patch']:>7} | {r['noise floor'] * 100:5.2f} {r['shrunk MAE'] * 100:6.2f} |"
                  f" {r['numbers champ MAE'] * 100:7.2f} {f(r['numbers change corr']):>5}"
                  f" {f(r['numbers direction']):>4} | {r['identity champ MAE'] * 100:8.2f}"
                  f" {f(r['identity change corr']):>5} {f(r['identity direction']):>4}")
        print("  (floor = best possible; shrunk = best baseline; corr / dir = for changed champions)")
    if len(champs):
        top = champs.reindex(champs["effect"].abs().sort_values(ascending=False).index).head(12)
        print("\nBiggest predicted balance effects (numbers model) vs what happened:")
        for _, r in top.iterrows():
            print(f"  {r['patch']:>6} {r['champion_id']:<14} predicted {r['effect'] * 100:+5.1f}"
                  f"  actual {r['actual_change'] * 100:+5.1f}  ({int(r['games'])} games)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train and evaluate the main win model.")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--models", default="models")
    ap.add_argument("--predict", default=None, help="predict winrate changes for this patch")
    ap.add_argument("--skip-eval", action="store_true", help="only train the final models")
    args = ap.parse_args(argv)

    games = load_table(args.dir, "games")
    cp = load_table(args.dir, "champion_patch")
    num_cols = number_cols(games)
    platforms = sorted(games["platform"].unique())
    print(f"{len(games):,} games, {games['patch'].nunique()} patches, {len(num_cols)} number "
          f"features per game")
    rescorer = Rescorer(args.db)

    if args.predict:
        target = args.predict
        earlier = [p for p in sorted(games["patch"].unique(), key=pkey) if pkey(p) < pkey(target)]
        if not earlier:
            sys.exit("Need games from a patch before the target patch.")
        prev = earlier[-1]
        train = games[games["patch"].isin(earlier)]
        gm = GameModel("numbers", platforms, num_cols).fit(train)
        eff = rescorer.champion_effect(gm, games[games["patch"] == prev], target)
        changed = set(cp[(cp["patch"] == target) & (cp["changed"] == 1)]["champion_id"])
        eff = eff[eff["champion_id"].isin(changed)].sort_values("effect")
        print(f"\nPredicted effect of {prev} -> {target} balance changes on winrate "
              f"(model trained on {len(train):,} games before {target}):")
        for _, r in eff.iterrows():
            print(f"  {r['champion_id']:<14} {r['effect'] * 100:+5.2f} pts")
        return

    if not args.skip_eval:
        print("\nEvaluating (train on earlier patches, test on the next one) ...")
        res, champs = evaluate(games, cp, rescorer, platforms, num_cols)
        print_results(res, champs)
        os.makedirs(args.dir, exist_ok=True)
        res.to_csv(os.path.join(args.dir, "model_results.csv"), index=False)

    print("\nTraining final models on all games ...")
    os.makedirs(args.models, exist_ok=True)
    for variant in ("numbers", "identity"):
        gm = GameModel(variant, platforms, num_cols).fit(games)
        path = os.path.join(args.models, f"game_model_{variant}.txt")
        gm.booster.save_model(path, num_iteration=gm.trees)
        with open(os.path.join(args.models, f"game_model_{variant}.json"), "w") as f:
            json.dump({"variant": variant, "platforms": platforms, "num_cols": num_cols,
                       "cat_levels": [int(v) for v in gm.cat_levels] if gm.cat_levels else None,
                       "trees": int(gm.trees),
                       "patches": sorted(games["patch"].unique(), key=pkey)}, f)
        print(f"  saved {path} ({gm.trees} trees)")
        if variant == "numbers":
            print("  what the numbers model relies on most:")
            for name, share in gm.importance(12).items():
                print(f"     {name:<40} {share:5.1%}")


if __name__ == "__main__":
    main()
