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
    lgb = None

# If LightGBM is missing or crashes (some Windows installs of LightGBM 4.7 do), the
# model automatically switches to scikit-learn's HistGradientBoosting - the same idea
# (many small decision trees), just a different library.
ENGINE = {"use_sklearn": lgb is None, "warned": False,
          "choice": "lightgbm",   # or "xgboost"  (--engine)
          "gpu": False}           # --gpu: train on the graphics card, falls back to CPU

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
# Linear trees: each leaf fits a small straight line instead of one number, so the model
# reacts smoothly to small stat changes. Tested on 82k games (Sept 2026): same accuracy as
# normal trees, so it's off by default. Try again with more games: --linear-trees
LINEAR_TREES = {"on": False, "linear_lambda": 1.0}


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


GAME_KEEP = ["match_id", "patch", "platform", "rank_bucket", "blue_win"]


def load_games(path_dir):
    """Only the columns the model needs, numbers as float32 (half the memory)."""
    p = os.path.join(path_dir, "games.parquet")
    if os.path.exists(p):
        try:
            import pyarrow.parquet as pq
            names = pq.read_schema(p).names
            keep = [c for c in names if c.startswith(("diff_", "lane_")) or c in ID_COLS
                    or c in GAME_KEEP]
            g = pd.read_parquet(p, columns=keep)
        except ImportError:
            g = pd.read_parquet(p)
    else:
        g = load_table(path_dir, "games")
    drop = [c for c in g.columns if c.startswith(("blue_", "red_")) and c not in ID_COLS
            and c != "blue_win"]
    g = g.drop(columns=drop)
    for c in number_cols(g):                    # column by column: no big temporary copy
        g[c] = g[c].astype(np.float32)
    return g


class GameModel:
    """
    LightGBM "does blue win?" model. Each game is used twice: as it happened, and seen
    from the red side (teams swapped, differences flipped), so the model has no side bias
    beyond the real blue-side effect.

    Data is written straight into one compact float32 number array (no pandas copies),
    which keeps RAM use low and is also the most compatible format for LightGBM on Windows.
    """

    def __init__(self, variant, platforms, num_cols):
        self.variant, self.platforms, self.num_cols = variant, platforms, num_cols
        self.booster = None
        self.cat_levels = None   # champion keys, identity model only
        self.names = (list(num_cols) + ["ctx_master"] + [f"ctx_{p}" for p in platforms]
                      + ["ctx_blue_side"] + (ID_COLS if variant == "identity" else []))

    # -- building the number array ------------------------------------------------ #
    def _lut(self):
        top = max(self.cat_levels) if self.cat_levels else 0
        lut = np.full(top + 2, np.nan, dtype=np.float32)
        lut[self.cat_levels] = np.arange(len(self.cat_levels), dtype=np.float32)
        return lut

    def _ids(self, df, col, rows):
        v = df[col].to_numpy(dtype=np.float64, na_value=np.nan)[rows]
        out = np.full(len(v), np.nan, dtype=np.float32)
        ok = ~np.isnan(v) & (v >= 0) & (v < len(self._lut_cache) - 1)
        out[ok] = self._lut_cache[v[ok].astype(np.int64)]
        return out                                   # unknown champion -> NaN

    def _fill(self, M, start, df, rows, mirrored):
        """Write games `rows` of df into M[start:start+len(rows)] (as blue, or mirrored)."""
        end = start + len(rows)
        k = len(self.num_cols)
        sign = np.float32(-1.0 if mirrored else 1.0)
        for j, c in enumerate(self.num_cols):
            M[start:end, j] = df[c].to_numpy()[rows] * sign
        M[start:end, k] = (df["rank_bucket"].to_numpy()[rows] == "MASTER+")
        plat = df["platform"].to_numpy()[rows]
        for i, p in enumerate(self.platforms):
            M[start:end, k + 1 + i] = (plat == p)
        j = k + 1 + len(self.platforms)
        M[start:end, j] = 0.0 if mirrored else 1.0           # ctx_blue_side
        if self.variant == "identity":
            for i, pos in enumerate(POSITIONS):
                b, r = f"blue_{pos}_champ", f"red_{pos}_champ"
                src_b, src_r = (r, b) if mirrored else (b, r)
                M[start:end, j + 1 + i] = self._ids(df, src_b, rows)
                M[start:end, j + 1 + len(POSITIONS) + i] = self._ids(df, src_r, rows)

    def _matrix(self, df, rows, both_sides=True):
        n = len(rows)
        M = np.empty((2 * n if both_sides else n, len(self.names)), dtype=np.float32)
        self._fill(M, 0, df, rows, False)
        if both_sides:
            self._fill(M, n, df, rows, True)
        return M

    # -- training ------------------------------------------------------------------ #
    @staticmethod
    def _params(seed):
        p = PARAMS
        return {"objective": "binary", "learning_rate": p["learning_rate"],
                "num_leaves": p["num_leaves"], "min_data_in_leaf": p["min_child_samples"],
                "bagging_fraction": p["subsample"], "bagging_freq": p["subsample_freq"],
                "feature_fraction": p["colsample_bytree"], "lambda_l2": p["reg_lambda"],
                "verbose": -1, "seed": seed, "num_threads": os.cpu_count() or 1,
                **({"device_type": "gpu"} if ENGINE["gpu"] else {}),
                **({"linear_tree": True, "linear_lambda": LINEAR_TREES["linear_lambda"]}
                   if LINEAR_TREES["on"] else {})}

    def fit(self, df, seed=0):
        y = df["blue_win"].to_numpy(dtype=np.float64)
        if self.variant == "identity":
            vals = np.concatenate([df[c].to_numpy(dtype=np.float64, na_value=-1) for c in ID_COLS])
            self.cat_levels = sorted(int(v) for v in np.unique(vals) if v >= 0)
            self._lut_cache = self._lut()
        rng = np.random.default_rng(seed)
        val = rng.random(len(df)) < 0.15             # validation split, by game
        tr_rows, va_rows = np.flatnonzero(~val), np.flatnonzero(val)
        cats = [self.names.index(c) for c in ID_COLS] if self.variant == "identity" else []
        if ENGINE["choice"] == "xgboost":
            self._fit_xgb(df, y, tr_rows, va_rows, cats, seed)
            return self
        if not ENGINE["use_sklearn"]:
            try:
                ytr = np.concatenate([y[tr_rows], 1.0 - y[tr_rows]])
                yva = np.concatenate([y[va_rows], 1.0 - y[va_rows]])
                dtrain = lgb.Dataset(self._matrix(df, tr_rows), label=ytr,
                                     feature_name=self.names, categorical_feature=cats or "auto",
                                     free_raw_data=True)
                dval = lgb.Dataset(self._matrix(df, va_rows), label=yva, reference=dtrain,
                                   free_raw_data=True)
                self.booster = lgb.train(self._params(seed), dtrain,
                                         num_boost_round=PARAMS["n_estimators"],
                                         valid_sets=[dval],
                                         callbacks=[lgb.early_stopping(100, verbose=False)])
                del dtrain, dval
                self.engine = "lightgbm"
                self.trees = self.booster.best_iteration or PARAMS["n_estimators"]
                return self
            except Exception as e:
                if ENGINE["gpu"]:
                    ENGINE["gpu"] = False
                    print(f"  NOTE: LightGBM couldn't use the GPU ({str(e)[:120]}). Using the "
                          f"CPU instead.", flush=True)
                    return self.fit(df, seed)
                if not isinstance(e, OSError):
                    raise
                ENGINE["use_sklearn"] = True
                print(f"  NOTE: LightGBM crashed on this computer ({e}). Switching to "
                      f"scikit-learn's HistGradientBoosting (same technique). To try LightGBM "
                      f"again later:  pip install \"lightgbm==4.6.0\"", flush=True)
        all_rows = np.arange(len(df))
        self._fit_sklearn(self._matrix(df, all_rows), np.concatenate([y, 1.0 - y]), cats, seed)
        return self

    def _fit_xgb(self, df, y, tr_rows, va_rows, cats, seed):
        import xgboost as xgb
        ftypes = ["c" if i in cats else "q" for i in range(len(self.names))]
        ytr = np.concatenate([y[tr_rows], 1.0 - y[tr_rows]])
        yva = np.concatenate([y[va_rows], 1.0 - y[va_rows]])
        dtrain = xgb.DMatrix(self._matrix(df, tr_rows), label=ytr, feature_names=self.names,
                             feature_types=ftypes, enable_categorical=bool(cats))
        dval = xgb.DMatrix(self._matrix(df, va_rows), label=yva, feature_names=self.names,
                           feature_types=ftypes, enable_categorical=bool(cats))
        p = PARAMS
        params = {"objective": "binary:logistic", "eval_metric": "logloss",
                  "eta": p["learning_rate"], "max_depth": 3,          # ~ LightGBM's 7 leaves
                  # LightGBM's 800 games per leaf, in XGBoost's units (each game counts ~0.25)
                  "min_child_weight": p["min_child_samples"] * 0.25,
                  "subsample": p["subsample"], "colsample_bytree": p["colsample_bytree"],
                  "lambda": p["reg_lambda"], "tree_method": "hist", "max_cat_to_onehot": 1,
                  "seed": seed, "nthread": os.cpu_count() or 1,
                  "device": "cuda" if ENGINE["gpu"] else "cpu"}
        try:
            self.booster = xgb.train(params, dtrain, num_boost_round=p["n_estimators"],
                                     evals=[(dval, "val")], early_stopping_rounds=100,
                                     verbose_eval=False)
        except Exception as e:
            if not ENGINE["gpu"]:
                raise
            ENGINE["gpu"] = False
            print(f"  NOTE: XGBoost couldn't use the GPU ({str(e)[:120]}). Using the CPU instead.",
                  flush=True)
            params["device"] = "cpu"
            self.booster = xgb.train(params, dtrain, num_boost_round=p["n_estimators"],
                                     evals=[(dval, "val")], early_stopping_rounds=100,
                                     verbose_eval=False)
        self.engine = "xgboost"
        self.trees = int(getattr(self.booster, "best_iteration", p["n_estimators"] - 1)) + 1
        self._xgb_types = (ftypes, bool(cats))

    def _fit_sklearn(self, M, y, cats, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        if not ENGINE["warned"] and lgb is None:
            print("  NOTE: LightGBM isn't installed - using scikit-learn's HistGradientBoosting.",
                  flush=True)
        ENGINE["warned"] = True
        mask = np.zeros(M.shape[1], dtype=bool)
        mask[cats] = True
        kw = dict(learning_rate=PARAMS["learning_rate"], max_iter=PARAMS["n_estimators"],
                  max_leaf_nodes=max(PARAMS["num_leaves"], 2),
                  min_samples_leaf=PARAMS["min_child_samples"],
                  l2_regularization=PARAMS["reg_lambda"], early_stopping=True,
                  validation_fraction=0.15, n_iter_no_change=100, random_state=seed,
                  categorical_features=mask if mask.any() else None)
        try:
            clf = HistGradientBoostingClassifier(max_features=PARAMS["colsample_bytree"], **kw)
        except TypeError:  # older scikit-learn without max_features
            clf = HistGradientBoostingClassifier(**kw)
        clf.fit(M, y.astype(int))
        self.booster, self.engine, self.trees = clf, "sklearn", int(clf.n_iter_)

    def predict(self, df, batch=50000):
        """P(blue wins), averaged over both side views so it's symmetric. Done in batches."""
        out = np.empty(len(df))
        for s0 in range(0, len(df), batch):
            rows = np.arange(s0, min(s0 + batch, len(df)))
            n = len(rows)
            M = self._matrix(df, rows)
            if self.engine == "sklearn":
                pr = self.booster.predict_proba(M)[:, 1]
            elif self.engine == "xgboost":
                import xgboost as xgb
                ftypes, cat = self._xgb_types
                pr = self.booster.predict(xgb.DMatrix(M, feature_names=self.names,
                                                      feature_types=ftypes, enable_categorical=cat),
                                          iteration_range=(0, self.trees))
            else:
                pr = self.booster.predict(M, num_iteration=self.trees)
            out[s0:s0 + n] = (pr[:n] + (1 - pr[n:])) / 2
        return out

    def importance(self, top=15):
        """Share of the model's total gain per input (all inputs if top is None)."""
        if self.engine == "sklearn":
            return pd.Series(dtype=float)  # not available for this engine
        if self.engine == "xgboost":
            g = self.booster.get_score(importance_type="total_gain")
            imp = pd.Series({n: g.get(n, 0.0) for n in self.names}, dtype=float)
        else:
            imp = pd.Series(self.booster.feature_importance("gain"),
                            index=self.booster.feature_name())
        imp = (imp / imp.sum()).sort_values(ascending=False)
        return imp if top is None else imp.head(top)

    def save(self, models_dir, variant):
        if self.engine == "sklearn":
            import joblib
            path = os.path.join(models_dir, f"game_model_{variant}.joblib")
            joblib.dump(self.booster, path)
        elif self.engine == "xgboost":
            path = os.path.join(models_dir, f"game_model_{variant}_xgb.json")
            self.booster.save_model(path)
        else:
            path = os.path.join(models_dir, f"game_model_{variant}.txt")
            self.booster.save_model(path, num_iteration=self.trees)
        return path


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


def shrink_k(cp, before):
    """How much of last patch's winrate carries over, learned from patches BEFORE `before`."""
    t = cp[(cp["patch"].map(pkey) < pkey(before)) & (cp["games"] >= MIN_CHAMP_GAMES)
           & cp["prev_winrate_smoothed"].notna() & (cp["prev_games"].fillna(0) > 0)]
    if t.empty:
        return 0.5
    x, y, w = t["prev_winrate_smoothed"] - 0.5, t["winrate"] - 0.5, t["games"]
    den = float(np.sum(w * x * x))
    return float(np.sum(w * x * y) / den) if den > 0 else 0.5


# --------------------------------------------------------------------------- #
def evaluate(games, cp, rescorer, platforms, num_cols, test_last=0):
    patches = sorted(games["patch"].unique(), key=pkey)
    counts = games.groupby("patch").size()
    results, champ_rows, game_rows, eff_rows = [], [], [], []
    testable = [p for i, p in enumerate(patches) if i > 0 and counts[p] >= MIN_TEST_GAMES]
    if test_last > 0:
        testable = testable[-test_last:]
    for i, test_patch in enumerate(patches):
        earlier = patches[:i]
        if test_patch not in testable:
            continue
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
        gp = test[["match_id", "patch", "platform", "rank_bucket", "blue_win"] + ID_COLS].copy()
        for variant in ("numbers", "identity"):
            gm = GameModel(variant, platforms, num_cols).fit(train)
            p = gm.predict(test)
            gp[f"p_{variant}"] = p
            row[f"{variant} logloss"] = logloss(y, p)
            row[f"{variant} AUC"] = auc(y, p)
            row[f"{variant} acc"] = float(np.mean((p > 0.5) == (y == 1)))
            if len(truth) and rescorer is not None:
                eff = rescorer.champion_effect(gm, prev_games, test_patch)
                e = eff[["champ_key", "champion_id", "effect", "sample"]].copy()
                e["patch"], e["variant"] = test_patch, variant
                eff_rows.append(e)
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
        game_rows.append(gp)
        results.append(row)
        print(f"  tested {test_patch}: trained on {len(train):,} games from {len(earlier)} "
              f"earlier patch(es)")
    cat = lambda rows: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.DataFrame(results), cat(champ_rows), cat(game_rows), cat(eff_rows)


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
    ap.add_argument("--linear-trees", action="store_true",
                    help="LightGBM linear trees (smoother reaction to small stat changes)")
    ap.add_argument("--engine", choices=["lightgbm", "xgboost"], default="lightgbm",
                    help="tree library (default lightgbm)")
    ap.add_argument("--gpu", action="store_true",
                    help="train on the graphics card (falls back to the CPU if it can't)")
    ap.add_argument("--test-last", type=int, default=0,
                    help="only test the latest N patches (much faster; 0 = test every patch)")
    args = ap.parse_args(argv)
    LINEAR_TREES["on"] = args.linear_trees
    ENGINE["choice"], ENGINE["gpu"] = args.engine, args.gpu
    if args.engine == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError:
            sys.exit("XGBoost isn't installed:  pip install xgboost")
    print(f"Engine: {args.engine}{' (GPU)' if args.gpu else ''}")
    import memlimit
    memlimit.apply()          # stay under the RAM limit (default 7 GB) instead of freezing the PC

    games = load_games(args.dir)
    cp = load_table(args.dir, "champion_patch")
    num_cols = number_cols(games)
    platforms = sorted(games["platform"].unique())
    print(f"{len(games):,} games, {games['patch'].nunique()} patches, {len(num_cols)} number "
          f"features per game")
    rescorer = Rescorer(args.db) if os.path.exists(args.db) else None
    if rescorer is None:
        print(f"  NOTE: {args.db} not found - champion-level patch effects are skipped.")

    if args.predict and rescorer is None:
        sys.exit(f"--predict needs {args.db}.")
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
        res, champs, game_preds, effects = evaluate(games, cp, rescorer, platforms, num_cols,
                                                    test_last=args.test_last)
        print_results(res, champs)
        os.makedirs(args.dir, exist_ok=True)
        res.to_csv(os.path.join(args.dir, "model_results.csv"), index=False)
        # for the diagnostics report: every test game's prediction + per-champion effects
        if len(game_preds):
            try:
                game_preds.to_parquet(os.path.join(args.dir, "game_test_predictions.parquet"),
                                      index=False)
            except Exception:
                game_preds.to_csv(os.path.join(args.dir, "game_test_predictions.csv.gz"),
                                  index=False)
        if len(effects):
            effects.to_csv(os.path.join(args.dir, "game_model_effects.csv"), index=False)

    print("\nTraining final models on all games ...")
    os.makedirs(args.models, exist_ok=True)
    for variant in ("numbers", "identity"):
        gm = GameModel(variant, platforms, num_cols).fit(games)
        path = gm.save(args.models, variant)
        with open(os.path.join(args.models, f"game_model_{variant}.json"), "w") as f:
            json.dump({"variant": variant, "engine": gm.engine, "platforms": platforms, "num_cols": num_cols,
                       "cat_levels": [int(v) for v in gm.cat_levels] if gm.cat_levels else None,
                       "trees": int(gm.trees),
                       "patches": sorted(games["patch"].unique(), key=pkey)}, f)
        print(f"  saved {path} ({gm.trees} trees)")
        if variant == "numbers" and gm.engine != "sklearn":
            gm.importance(None).rename("share").rename_axis("feature").reset_index().to_csv(
                os.path.join(args.models, "importance_numbers.csv"), index=False)
        if variant == "numbers":
            print("  what the numbers model relies on most:")
            for name, share in gm.importance(12).items():
                print(f"     {name:<40} {share:5.1%}")


if __name__ == "__main__":
    main()
