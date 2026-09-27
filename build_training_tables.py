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
import re
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
CORE_PREFIXES = ["class_", "resource_", "total_", "cc_", "flag_"]
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
    # only the features used below - read in pieces, so the (large) full table is never
    # in memory at once
    def needed(f):
        return (f in CORE_EXACT or f.startswith(tuple(CORE_PREFIXES))
                or (f[:2] in ("Q_", "W_", "E_", "R_")
                    and (f.endswith(("_cd_rmax", "_cd_r1", "_cost_rmax", "_range_rmax"))
                         or f.endswith("_base_rmax"))))
    pieces = []
    for chunk in pd.read_sql_query(
            "SELECT patch, champion_id, champ_key, feature, value FROM patch_champion_features",
            con, chunksize=200_000):
        pieces.append(chunk[chunk["feature"].map(needed)])
    raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    del pieces
    if raw.empty:
        sys.exit("patch_champion_features is empty - run build_patch_state.py first.")
    raw = raw.drop_duplicates(["patch", "champion_id", "champ_key", "feature"])
    wide = raw.set_index(["patch", "champion_id", "champ_key", "feature"])["value"].unstack(
        "feature").reset_index()
    del raw
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
    import memlimit
    m = pd.read_sql_query("""
        SELECT match_id, platform, patch, game_start, duration_s, winning_team,
               rank_bucket, rank_source
        FROM matches""", con)
    pieces = []
    for chunk in pd.read_sql_query("""
            SELECT match_id, team_id, champion_id AS champ_key, position
            FROM participants""", con, chunksize=500_000):
        chunk["team_id"] = chunk["team_id"].astype("int16")
        pieces.append(chunk[chunk["position"].isin(POSITIONS)])   # drop junk rows early
    p = pd.concat(pieces, ignore_index=True)
    del pieces
    print(f"  loaded {len(m):,} games / {len(p):,} players{memlimit.ram_note()}", flush=True)
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


CHUNK_GAMES = 20000           # games processed at a time (keeps RAM use low)


def team_features(m, p, vec, feats, team_sums=False, chunk=CHUNK_GAMES):
    """Blue-red differences of the teams' champion numbers, lane-vs-lane differences and
    the 10 champion ids. Works through the games in chunks and stores numbers as float32
    (half the memory of the default), so RAM use stays low as the dataset grows.
    team_sums=True also keeps each team's own totals (blue_..., red_...; not used by the
    models)."""
    vec = vec.drop(columns=["champion_id"], errors="ignore").copy()
    vec[feats] = vec[feats].astype(np.float32)
    ids_all = m["match_id"].unique()
    outs, n_bad = [], 0
    for i in range(0, len(ids_all), chunk):
        part = set(ids_all[i:i + chunk])
        o, b = _team_features_chunk(m[m["match_id"].isin(part)], p[p["match_id"].isin(part)],
                                    vec, feats, team_sums)
        outs.append(o)
        n_bad += b
        if len(ids_all) > chunk:
            import memlimit
            print(f"  game numbers: {min(i + chunk, len(ids_all)):,} / {len(ids_all):,}"
                  f"{memlimit.ram_note()}", flush=True)
    return (pd.concat(outs) if outs else pd.DataFrame()), n_bad


def _team_features_chunk(m, p, vec, feats, team_sums):
    pv = p[["match_id", "team_id", "champ_key", "position"]].merge(
        m[["match_id", "patch"]], on="match_id").merge(vec, on=["patch", "champ_key"], how="left")
    missing = pv[feats[0]].isna().groupby(pv["match_id"]).any()
    bad = set(missing[missing].index)            # games with a champion we have no numbers for
    pv = pv[~pv["match_id"].isin(bad)]
    pv[feats] = pv[feats].fillna(0.0).astype(np.float32)

    sums = pv.groupby(["match_id", "team_id"])[feats].sum()
    blue = sums.xs(100, level="team_id")
    red = sums.xs(200, level="team_id").reindex(blue.index)
    parts = [pd.DataFrame((blue.values - red.values).astype(np.float32), index=blue.index,
                          columns=[f"diff_{c}" for c in feats])]
    if team_sums:
        parts = [blue.add_prefix("blue_"), red.add_prefix("red_")] + parts
    del sums

    lane_cols = [c for c in LANE_FEATURES if c in feats]
    lanes = pv.pivot_table(index="match_id", columns=["team_id", "position"],
                           values=lane_cols, aggfunc="first")
    parts.append(pd.DataFrame({f"lane_{pos}_{f}": (lanes[(f, 100, pos)] - lanes[(f, 200, pos)])
                               .astype(np.float32)
                               for f in lane_cols for pos in POSITIONS}, index=lanes.index))
    del pv, lanes

    ids = p.pivot_table(index="match_id", columns=["team_id", "position"], values="champ_key",
                        aggfunc="first")
    id_df = pd.DataFrame({f"{side}_{pos}_champ": ids[(team, pos)].astype("Int64")
                          for team, side in ((100, "blue"), (200, "red")) for pos in POSITIONS},
                         index=ids.index)
    out = pd.concat(parts, axis=1).join(id_df, how="left")
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


# --------------------------------------------------------------------------- #
# Item changes -> which champions they touch
# --------------------------------------------------------------------------- #
# gold value of one unit of each Data Dragon stat (standard gold-efficiency values)
GOLD_PER_STAT = {
    "FlatPhysicalDamageMod": 35.0, "FlatMagicDamageMod": 20.0, "FlatHPPoolMod": 2.67,
    "FlatArmorMod": 20.0, "FlatSpellBlockMod": 18.0, "PercentAttackSpeedMod": 2500.0,
    "FlatCritChanceMod": 4000.0, "FlatMovementSpeedMod": 12.0, "PercentMovementSpeedMod": 3950.0,
    "PercentLifeStealMod": 3750.0, "FlatMPPoolMod": 1.4, "FlatHPRegenMod": 3.0,
}
# passive numbers where LOWER is better for the owner (cooldowns, delays, costs)
LOWER_IS_BUFF = re.compile(r"cooldown|\bcd\b|_cd|cd_|delay|cost|recharge", re.I)
MAX_ITEM_REL = 0.5
# a passive is only part of an item's value: a 20% stronger passive counts as the item being
# about 7% stronger (on top of any stat / price change)
PASSIVE_SHARE = 0.35


def item_table(con):
    """{patch: {item_id: {feature: value}}} for finished items and boots."""
    rows = pd.read_sql_query("SELECT patch, item_id, feature, value FROM patch_item_features", con)
    out = {}
    for (p, iid), g in rows.groupby(["patch", "item_id"]):
        f = dict(zip(g["feature"], g["value"]))
        if f.get("is_completed") or f.get("tag_boots"):
            out.setdefault(p, {})[str(iid)] = f
    return out


def item_names(con):
    names = {}
    for p, v in con.execute("SELECT patch, version FROM patches"):
        for iid, n in con.execute("SELECT item_id, name FROM items WHERE version = ?", (v,)):
            names.setdefault(p, {})[str(iid)] = n
    return names


def stat_gold(f):
    return sum(f.get(f"stat_{s}", 0.0) * g for s, g in GOLD_PER_STAT.items())


def item_changes(items):
    """Per patch and changed item: size of the change, as a share of the item's cost.
      stat_pct     (stat gold value change - price change) / old price
      passive_pct  average change of the passive numbers that changed (+ = stronger),
                   times PASSIVE_SHARE (the passive's share of the item's value)
    """
    order = sorted(items, key=pkey)
    rows = []
    for prev, cur in zip(order, order[1:]):
        for iid, fc in items[cur].items():
            fp = items[prev].get(iid)
            if not fp or not fp.get("gold_total"):
                continue
            d_cost = fc.get("gold_total", 0) - fp.get("gold_total", 0)
            stat_pct = ((stat_gold(fc) - stat_gold(fp)) - d_cost) / fp["gold_total"]
            rels, desc = [], []
            for k in set(fc) & set(fp):
                if not k.startswith("pv_"):
                    continue
                o, n = fp[k], fc[k]
                if o is None or n is None or abs(n - o) <= 1e-6 * max(1, abs(o)):
                    continue
                r = np.clip((n - o) / abs(o), -MAX_ITEM_REL, MAX_ITEM_REL) if o else np.sign(n - o) * 0.2
                if LOWER_IS_BUFF.search(k):
                    r = -r
                rels.append(r)
                desc.append(f"{k[3:]} {o:g}->{n:g}")
            flags_gained = [k[5:] for k in fc if k.startswith("flag_") and fc[k] and not fp.get(k)]
            flags_lost = [k[5:] for k in fp if k.startswith("flag_") and fp[k] and not fc.get(k)]
            stat_desc = [f"{k[5:]} {fp.get(k, 0):g}->{fc.get(k, 0):g}" for k in set(fc) | set(fp)
                         if k.startswith("stat_") and abs(fc.get(k, 0) - fp.get(k, 0)) > 1e-9]
            if abs(stat_pct) < 1e-6 and not rels and not flags_gained and not flags_lost:
                continue
            rows.append({"patch": cur, "item_id": iid, "cost_change": d_cost,
                         "stat_pct": stat_pct,
                         "passive_pct": PASSIVE_SHARE * float(np.mean(rels)) if rels else 0.0,
                         "n_passive_changed": len(rels),
                         "flags_gained": ",".join(flags_gained), "flags_lost": ",".join(flags_lost),
                         "changes": "; ".join(([f"cost {fp.get('gold_total', 0):g}->"
                                                f"{fc.get('gold_total', 0):g}"] if d_cost else [])
                                              + stat_desc + desc)[:300]})
    return pd.DataFrame(rows)


def item_exposure(con, games):
    """Share of each champion's games (per patch) that ended with each item."""
    patch_of = games.set_index("match_id")["patch"]
    counts = []
    for chunk in pd.read_sql_query("SELECT match_id, champion_id AS champ_key, items "
                                   "FROM participants", con, chunksize=400_000):
        chunk = chunk[chunk["match_id"].isin(patch_of.index)]
        if chunk.empty:
            continue
        chunk["patch"] = chunk["match_id"].map(patch_of)
        chunk["player"] = np.arange(len(chunk))
        ex = chunk[["patch", "champ_key", "player", "items"]].assign(
            item=chunk["items"].fillna("").str.split(",")).explode("item")
        ex = ex[ex["item"].str.len() > 0].drop_duplicates(["player", "item"])
        counts.append(ex.groupby(["patch", "champ_key", "item"]).size())
    if not counts:
        return pd.DataFrame(columns=["patch", "champ_key", "item", "share"])
    c = pd.concat(counts).groupby(level=[0, 1, 2]).sum().rename("n").reset_index()
    tot = games.melt(value_vars=[f"{s}_{p}_champ" for s in ("blue", "red") for p in POSITIONS],
                     id_vars=["patch"], value_name="champ_key").groupby(["patch", "champ_key"]).size()
    c = c.join(tot.rename("games"), on=["patch", "champ_key"])
    c["share"] = c["n"] / c["games"]
    c["champ_key"] = c["champ_key"].astype(int)
    return c[["patch", "champ_key", "item", "share", "games"]]


def add_item_scores(con, cp, games, out_dir):
    """Item buff scores per champion and patch, plus training/item_changes.csv."""
    if "patch_item_features" not in {r[0] for r in con.execute("SELECT name FROM sqlite_master")}:
        return cp
    items = item_table(con)
    if not items:
        return cp
    chg = item_changes(items)
    expo = item_exposure(con, games)
    if expo.empty:
        return cp
    game_patches = sorted(expo["patch"].unique(), key=pkey)
    all_patches = sorted(cp["patch"].unique(), key=pkey)
    # builds known BEFORE each patch: the latest earlier patch with games
    src = {}
    for p in all_patches:
        earlier = [q for q in game_patches if pkey(q) < pkey(p)]
        if earlier:
            src[p] = earlier[-1]
    eff = {p: {iid: stat_gold(f) / f["gold_total"] for iid, f in its.items() if f.get("gold_total")}
           for p, its in items.items()}
    by_patch = {p: g.set_index("item_id") for p, g in chg.groupby("patch")} if len(chg) else {}
    expo_by = {p: g for p, g in expo.groupby("patch")}
    rows = []
    for p in all_patches:
        s = src.get(p)
        if s is None:
            continue
        e = expo_by[s]
        c = by_patch.get(p)
        stat = e["item"].map(c["stat_pct"]).fillna(0.0) if c is not None else 0.0
        pas = e["item"].map(c["passive_pct"]).fillna(0.0) if c is not None else 0.0
        touched = e["item"].isin(c.index).astype(float) if c is not None else 0.0
        ef = e["item"].map(eff.get(p, {}))
        t = pd.DataFrame({"champ_key": e["champ_key"], "stat": e["share"] * stat,
                          "pas": e["share"] * pas, "touch": e["share"] * touched,
                          "eff_w": e["share"] * ef.fillna(0), "w": e["share"] * ef.notna()})
        g = t.groupby("champ_key").sum()
        rows.append(pd.DataFrame({"patch": p, "champ_key": g.index,
                                  "item_stat_buff": g["stat"].values,
                                  "item_passive_buff": g["pas"].values,
                                  "item_changed_share": g["touch"].values,
                                  "item_efficiency": (g["eff_w"] / g["w"].where(g["w"] > 0)).values}))
    if rows:
        sc = pd.concat(rows, ignore_index=True)
        cp = cp.merge(sc, on=["patch", "champ_key"], how="left")
        for col in ("item_stat_buff", "item_passive_buff", "item_changed_share"):
            cp[col] = cp[col].fillna(0.0)
    # item change list with each item's main users (for the report)
    if len(chg):
        names = item_names(con)
        champ_names = dict(zip(cp["champ_key"], cp["champion_id"]))
        users, n_users = [], []
        for _, r in chg.iterrows():
            s = src.get(r["patch"])
            if s not in expo_by:
                users.append("")
                n_users.append(0)
                continue
            u = expo_by[s][expo_by[s]["item"] == r["item_id"]]
            u = u[u["games"] >= 100].sort_values("share", ascending=False)
            users.append("; ".join(f"{champ_names.get(k, k)} {sh:.0%}"
                                   for k, sh in zip(u["champ_key"][:6], u["share"][:6])))
            n_users.append(int((u["share"] >= 0.2).sum()))
        chg["name"] = [names.get(p, {}).get(i, i) for p, i in zip(chg["patch"], chg["item_id"])]
        chg["main_users"] = users
        chg["n_champs_20pct"] = n_users
        os.makedirs(out_dir, exist_ok=True)
        chg.to_csv(os.path.join(out_dir, "item_changes.csv"), index=False)
        # also the exposure of changed items per champion (for the report's per-item check)
        ex = []
        for _, r in chg.iterrows():
            s = src.get(r["patch"])
            if s in expo_by:
                u = expo_by[s][expo_by[s]["item"] == r["item_id"]]
                ex.append(pd.DataFrame({"patch": r["patch"], "item_id": r["item_id"],
                                        "champion_id": u["champ_key"].map(champ_names),
                                        "share": u["share"]}))
        if ex:
            pd.concat(ex, ignore_index=True).to_csv(os.path.join(out_dir, "item_users.csv"),
                                                    index=False)
    return cp


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build model training tables.")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--out", default="training")
    args = ap.parse_args(argv)
    import memlimit
    memlimit.apply()          # stay under the RAM limit (default 7 GB)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=60)

    vec, feats = champion_vectors(con)
    print(f"Champion features: {len(feats)} numbers per champion, "
          f"{vec['patch'].nunique()} patches{memlimit.ram_note()}", flush=True)
    games, stats, parts = build_games(con, vec, feats)
    print(f"Games: {stats['total']:,} collected -> {stats['kept']:,} kept "
          f"(dropped: {stats['no_winner']} no winner, {stats['too_short']} under "
          f"{MIN_GAME_SECONDS // 60} min, {stats['bad_positions']} missing positions, "
          f"{stats['no_champion_numbers']} champion without numbers)")
    by = games.groupby("patch").size()
    print("Games per patch: " + ", ".join(f"{p} {n:,}" for p, n in
                                          sorted(by.items(), key=lambda kv: pkey(kv[0]), reverse=True)))
    model_cols = [c for c in games if c.startswith(("diff_", "lane_"))]
    print(f"Model inputs per game: {len(model_cols)} numbers "
          f"(+10 champion ids, + platform and rank)")
    print(f"Blue side win rate: {games['blue_win'].mean():.1%}  |  "
          f"timeline gold@15 available: {games['aux_gold_diff_15'].notna().mean():.0%} of games")

    cp = build_champion_patch(con, games, parts, vec, feats)
    del parts
    cp = add_item_scores(con, cp, games, args.out)
    if "item_stat_buff" in cp:
        aff = (cp["item_changed_share"] > 0.1).sum()
        print(f"Item changes: {cp.loc[cp['item_changed_share'] > 0, 'patch'].nunique()} patches with "
              f"changed items; {aff:,} champion-patches where 10%+ of builds had a changed item"
              f"{memlimit.ram_note()}", flush=True)
    with_data = cp[cp["games"].notna()]
    print(f"Champion-patch rows: {len(cp):,} ({len(with_data):,} with games; "
          f"median {with_data['games'].median():.0f} games per champion per patch)")

    print(f"  tables built{memlimit.ram_note()}", flush=True)
    p1 = save(games, "games", args.out)
    p2 = save(cp, "champion_patch", args.out)
    print(f"Saved {p1} and {p2}")


if __name__ == "__main__":
    main()
