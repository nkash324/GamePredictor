#!/usr/bin/env python3
"""
build_training_tables.py  -  turn collected games + patch features into model inputs.

Writes two tables to the training/ folder (Parquet, or .csv.gz if pyarrow is missing):

  games          one row per game - the main model's training data
                   label       blue_win (1/0)
                   inputs      each team's champions described by their NUMBERS on that
                               patch (stats, ability damage/ratios, CC, class, ...):
                               blue_* and red_* team totals, diff_* (blue - red),
                               lane_* (blue minus red for the same position)
                   identity    blue_TOP_champ ... red_UTILITY_champ (champion keys) - only
                               for the "numbers + identity" model variant
                   context     platform, rank_bucket, patch (patch is NOT a model input)
                   aux_*       results used only as extra training targets, never inputs
                               (game length, gold difference at 15 min)
  champion_patch one row per champion per patch - for baselines and checking predictions
                   outcomes    games, winrate (raw + smoothed), pick / ban rate
                   features    the champion's numbers on that patch
                   delta_*     change from the previous patch
                   prev_*      previous patch's outcomes

Only information known BEFORE a game starts is used as model input. Kills, gold,
damage, final items and game length are results and would let a model "cheat".

Usage:
  pip install pandas numpy pyarrow
  python build_training_tables.py
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
MIN_GAME_SECONDS = 600       # shorter games are almost always AFK / early surrender
PRIOR_GAMES = 50             # smoothing strength for champion winrates

# Core per-champion features used for the model (general, not champion-specific names).
# Anything matching these prefixes/names is included when present.
CORE_EXACT = [
    "hp_l1", "hp_l9", "hp_l16", "armor_l1", "armor_l9", "armor_l16", "mr_l1", "mr_l9", "mr_l16",
    "ad_l1", "ad_l9", "ad_l16", "as_l1", "as_l9", "as_l16", "ehp_phys_l9", "ehp_phys_l16",
    "stat_movespeed", "stat_attackrange", "stat_hpregen", "stat_mp", "is_ranged",
    "cc_hard_total", "cc_slow_max",
    "rating_attack", "rating_defense", "rating_magic", "rating_difficulty",
]
CORE_PREFIXES = ["class_", "resource_", "total_", "cc_"]
# ability-level summaries computed here
DERIVED = ["basic_cd_rmax_mean", "basic_cd_r1_mean", "ult_cd_rmax", "basic_cost_rmax_mean",
           "range_max", "n_damage_abilities"]
# per-lane head-to-head differences (blue minus red at the same position)
LANE_FEATURES = ["hp_l9", "ad_l9", "ehp_phys_l9", "armor_l9", "mr_l9", "stat_attackrange",
                 "total_dmg_base_rmax", "total_dmg_ratio_ap", "total_dmg_ratio_ad_all",
                 "cc_hard_total"]


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def save(df, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = os.path.join(out_dir, f"{name}.csv.gz")
        df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Champion feature vectors per patch
# --------------------------------------------------------------------------- #
def champion_vectors(con):
    """DataFrame indexed by (patch, champ_key) with the core features."""
    raw = pd.read_sql_query(
        "SELECT patch, champion_id, champ_key, feature, value FROM patch_champion_features", con)
    if raw.empty:
        sys.exit("patch_champion_features is empty - run build_patch_state.py first.")
    wide = raw.pivot_table(index=["patch", "champion_id", "champ_key"], columns="feature",
                           values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    cols = [c for c in wide.columns if c in CORE_EXACT or any(c.startswith(p) for p in CORE_PREFIXES)]
    out = wide[["patch", "champion_id", "champ_key"] + cols].copy()

    def mean_of(df, names):
        present = [n for n in names if n in df]
        return df[present].mean(axis=1) if present else 0.0

    out["basic_cd_rmax_mean"] = mean_of(wide, ["Q_cd_rmax", "W_cd_rmax", "E_cd_rmax"])
    out["basic_cd_r1_mean"] = mean_of(wide, ["Q_cd_r1", "W_cd_r1", "E_cd_r1"])
    out["ult_cd_rmax"] = wide.get("R_cd_rmax", 0.0)
    out["basic_cost_rmax_mean"] = mean_of(wide, ["Q_cost_rmax", "W_cost_rmax", "E_cost_rmax"])
    rng = [c for c in ["Q_range_rmax", "W_range_rmax", "E_range_rmax", "R_range_rmax"] if c in wide]
    out["range_max"] = wide[rng].clip(upper=5000).max(axis=1) if rng else 0.0
    dmg_cols = [c for c in wide if c.endswith("_base_rmax") and c[:2] in ("Q_", "W_", "E_", "R_")
                and any(t in c for t in ("_magic_", "_physical_", "_true_", "_untyped_"))]
    out["n_damage_abilities"] = sum((wide[c].fillna(0) > 0).astype(float) for c in dmg_cols) \
        if dmg_cols else 0.0

    # compact damage summaries used for lanes
    def s(names):
        present = [n for n in names if n in out]
        return out[present].fillna(0).sum(axis=1) if present else 0.0
    out["total_dmg_base_rmax"] = s(["total_magic_base_rmax", "total_physical_base_rmax",
                                    "total_true_base_rmax", "total_untyped_base_rmax"])
    out["total_dmg_ratio_ap"] = s(["total_magic_ratio_ap", "total_physical_ratio_ap",
                                   "total_true_ratio_ap"])
    out["total_dmg_ratio_ad_all"] = s([c for c in out if c.startswith("total_")
                                       and ("_ratio_ad" in c) and "heal" not in c
                                       and "shield" not in c])
    feats = [c for c in out.columns if c not in ("patch", "champion_id", "champ_key")]
    out[feats] = out[feats].astype(float).fillna(0.0)
    # drop features that never vary (no information)
    varying = [c for c in feats if out[c].nunique() > 1]
    out = out[["patch", "champion_id", "champ_key"] + varying]
    return out, varying


# --------------------------------------------------------------------------- #
# Games table
# --------------------------------------------------------------------------- #
def load_games(con):
    m = pd.read_sql_query("""
        SELECT match_id, platform, patch, game_start, duration_s, winning_team,
               rank_bucket, rank_source
        FROM matches""", con)
    p = pd.read_sql_query("""
        SELECT match_id, team_id, champion_id AS champ_key, champion_name, position, keystone
        FROM participants""", con)
    stats = {"total": len(m)}
    m = m[m["winning_team"].isin([100, 200])]
    stats["no_winner"] = stats["total"] - len(m)
    n = len(m)
    m = m[m["duration_s"] >= MIN_GAME_SECONDS]
    stats["too_short"] = n - len(m)
    # exactly one player per position per team
    ok_pos = p[p["position"].isin(POSITIONS)]
    counts = ok_pos.groupby(["match_id", "team_id"])["position"].nunique()
    good = counts[counts == 5].reset_index().groupby("match_id").size()
    good = set(good[good == 2].index)
    n = len(m)
    m = m[m["match_id"].isin(good)]
    stats["bad_positions"] = n - len(m)
    p = ok_pos[ok_pos["match_id"].isin(set(m["match_id"]))]
    return m, p, stats


def team_features(m, p, vec, feats):
    """Sum each team's champion vectors, plus blue-red and lane differences."""
    pv = p.merge(m[["match_id", "patch"]], on="match_id").merge(
        vec.drop(columns=["champion_id"]), on=["patch", "champ_key"], how="left")
    missing = pv[feats[0]].isna().groupby(pv["match_id"]).any()
    bad = set(missing[missing].index)            # games with a champion we have no numbers for
    pv = pv[~pv["match_id"].isin(bad)]
    pv[feats] = pv[feats].fillna(0.0)

    sums = pv.groupby(["match_id", "team_id"])[feats].sum()
    blue = sums.xs(100, level="team_id").add_prefix("blue_")
    red = sums.xs(200, level="team_id").add_prefix("red_")
    diff = pd.DataFrame(blue.values - red.values, index=blue.index,
                        columns=[f"diff_{c}" for c in feats])

    lane_cols = [c for c in LANE_FEATURES if c in feats]
    lanes = pv.pivot_table(index="match_id", columns=["team_id", "position"],
                           values=lane_cols, aggfunc="first")
    lane_df = pd.DataFrame({f"lane_{pos}_{f}": lanes[(f, 100, pos)] - lanes[(f, 200, pos)]
                            for f in lane_cols for pos in POSITIONS}, index=lanes.index)

    ids = p.pivot_table(index="match_id", columns=["team_id", "position"], values="champ_key",
                        aggfunc="first")
    id_df = pd.DataFrame({f"{side}_{pos}_champ": ids[(team, pos)].astype("Int64")
                          for team, side in ((100, "blue"), (200, "red")) for pos in POSITIONS},
                         index=ids.index)
    out = pd.concat([blue, red, diff, lane_df], axis=1).join(id_df, how="left")
    return out, len(bad)


def gold_diff_15(con):
    try:
        t = pd.read_sql_query("""
            SELECT f.match_id, p.team_id, SUM(f.gold) AS gold
            FROM timeline_frames f
            JOIN participants p ON p.match_id = f.match_id AND p.participant_id = f.participant_id
            WHERE f.minute = 15 GROUP BY f.match_id, p.team_id""", con)
    except Exception:
        return pd.Series(dtype=float)
    if t.empty:
        return pd.Series(dtype=float)
    w = t.pivot(index="match_id", columns="team_id", values="gold")
    return (w.get(100) - w.get(200)).rename("aux_gold_diff_15")


def build_games(con, vec, feats):
    m, p, stats = load_games(con)
    tf, n_missing = team_features(m, p, vec, feats)
    stats["no_champion_numbers"] = n_missing
    games = m.set_index("match_id").join(tf, how="inner")
    games["blue_win"] = (games["winning_team"] == 100).astype(int)
    games["aux_duration_min"] = games["duration_s"] / 60.0
    games = games.join(gold_diff_15(con), how="left")
    games = games.drop(columns=["winning_team", "duration_s"]).reset_index()
    stats["kept"] = len(games)
    return games, stats, p


# --------------------------------------------------------------------------- #
# Champion x patch table
# --------------------------------------------------------------------------- #
def build_champion_patch(con, games, parts, vec, feats):
    kept = parts[parts["match_id"].isin(set(games["match_id"]))]
    wins = games.set_index("match_id")["blue_win"]
    kept = kept.join(wins, on="match_id")
    kept["win"] = np.where(kept["team_id"] == 100, kept["blue_win"], 1 - kept["blue_win"])
    kept = kept.merge(games[["match_id", "patch"]], on="match_id")
    n_games = games.groupby("patch").size().rename("patch_games")

    agg = kept.groupby(["patch", "champ_key"]).agg(games=("win", "size"), wins=("win", "sum"))
    main_role = (kept.groupby(["patch", "champ_key"])["position"]
                 .agg(lambda s: s.value_counts().index[0]).rename("main_role"))
    bans = pd.read_sql_query("""
        SELECT m.patch, b.champion_id AS champ_key, COUNT(DISTINCT b.match_id) AS bans
        FROM bans b JOIN matches m USING (match_id) WHERE b.champion_id > 0
        GROUP BY m.patch, b.champion_id""", con)
    bans = bans[bans["patch"].isin(n_games.index)]

    cp = vec.merge(agg.reset_index(), on=["patch", "champ_key"], how="left")
    cp = cp.merge(main_role.reset_index(), on=["patch", "champ_key"], how="left")
    cp = cp.merge(bans, on=["patch", "champ_key"], how="left")
    cp = cp.join(n_games, on="patch")
    has_data = cp["patch_games"].notna()
    cp.loc[has_data, ["games", "wins", "bans"]] = cp.loc[has_data, ["games", "wins", "bans"]].fillna(0)
    cp["winrate"] = cp["wins"] / cp["games"].where(cp["games"] > 0)
    cp["winrate_smoothed"] = (cp["wins"] + 0.5 * PRIOR_GAMES) / (cp["games"] + PRIOR_GAMES)
    cp["pickrate"] = cp["games"] / cp["patch_games"]
    cp["banrate"] = cp["bans"] / cp["patch_games"]
    cp["presence"] = cp["pickrate"] + cp["banrate"]

    # changes from the previous patch (balance) and previous outcomes (lags)
    order = sorted(cp["patch"].unique(), key=pkey)
    prev_of = {p: order[i - 1] for i, p in enumerate(order) if i > 0}
    cp["prev_patch"] = cp["patch"].map(prev_of)
    prev = cp[["patch", "champ_key"] + feats + ["winrate_smoothed", "pickrate", "banrate",
                                                 "games"]].rename(columns={"patch": "prev_patch"})
    prev = prev.rename(columns={c: f"prev_{c}" for c in prev.columns
                                if c not in ("prev_patch", "champ_key")})
    cp = cp.merge(prev, on=["prev_patch", "champ_key"], how="left")
    deltas = pd.DataFrame({f"delta_{f}": cp[f] - cp[f"prev_{f}"] for f in feats}, index=cp.index)
    cp = pd.concat([cp.drop(columns=[f"prev_{f}" for f in feats]), deltas], axis=1)
    nchg = pd.read_sql_query("""SELECT patch, entity_id AS champion_id, COUNT(*) AS n_changes
                                FROM patch_changes WHERE entity = 'champion'
                                GROUP BY patch, entity_id""", con)
    cp = cp.merge(nchg, on=["patch", "champion_id"], how="left")
    cp["n_changes"] = cp["n_changes"].fillna(0)
    cp.loc[cp["prev_patch"].isna(), "n_changes"] = np.nan
    cp["changed"] = (cp["n_changes"] > 0).astype(float).where(cp["n_changes"].notna())
    cp["winrate_change"] = cp["winrate_smoothed"] - cp["prev_winrate_smoothed"]
    return cp


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build model training tables.")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--out", default="training")
    args = ap.parse_args(argv)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=60)

    vec, feats = champion_vectors(con)
    print(f"Champion features: {len(feats)} numbers per champion, "
          f"{vec['patch'].nunique()} patches")
    games, stats, parts = build_games(con, vec, feats)
    print(f"Games: {stats['total']:,} collected -> {stats['kept']:,} kept "
          f"(dropped: {stats['no_winner']} no winner, {stats['too_short']} under "
          f"{MIN_GAME_SECONDS // 60} min, {stats['bad_positions']} missing positions, "
          f"{stats['no_champion_numbers']} champion without numbers)")
    by = games.groupby("patch").size()
    print("Games per patch: " + ", ".join(f"{p} {n:,}" for p, n in
                                          sorted(by.items(), key=lambda kv: pkey(kv[0]), reverse=True)))
    model_cols = [c for c in games if c.startswith(("blue_", "red_", "diff_", "lane_"))
                  and not c.endswith("_champ")]
    print(f"Model inputs per game: {len(model_cols)} numbers "
          f"(+10 champion ids, + platform and rank)")
    print(f"Blue side win rate: {games['blue_win'].mean():.1%}  |  "
          f"timeline gold@15 available: {games['aux_gold_diff_15'].notna().mean():.0%} of games")

    cp = build_champion_patch(con, games, parts, vec, feats)
    with_data = cp[cp["games"].notna()]
    print(f"Champion-patch rows: {len(cp):,} ({len(with_data):,} with games; "
          f"median {with_data['games'].median():.0f} games per champion per patch)")

    p1 = save(games, "games", args.out)
    p2 = save(cp, "champion_patch", args.out)
    print(f"Saved {p1} and {p2}")


if __name__ == "__main__":
    main()
