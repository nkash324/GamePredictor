#!/usr/bin/env python3
"""
build_patch_state.py  -  turn raw balance data into model-ready numbers per patch.

Reads what fetch_balance_data.py (Data Dragon) and fetch_ability_data.py
(CommunityDragon) stored in lol_data.db and writes:

  patch_champion_features  (patch, champion_id, champ_key, feature, value)
      every champion as a list of numbers for that patch:
        - base stats and growth, plus stats at levels 1 / 9 / 16
        - class (Mage, Tank, ...), resource type, Riot's difficulty ratings
        - per ability (Q/W/E/R): cooldown, cost, range (Data Dragon)
        - per ability: main magic / physical / true damage, heal and shield
          formulas - base at rank 1 and max rank, and AP / AD / HP ratios
        - crowd control: longest stun / root / knock-up / charm / ... and slows
        - totals across abilities (total damage, total AP ratio, ...)
  patch_item_features      (patch, item_id, feature, value)
      Summoner's Rift items: price, stats, type tags, completed or component
  patch_changes            (patch, prev_patch, entity, entity_id, feature, old, new, delta, pct)
      everything that changed from the previous patch - this is what the model
      learns from ("what did the balance team change?")

Usage:
  python build_patch_state.py               rebuild all tables (a few seconds to a minute)
  python build_patch_state.py --report 16.18   list the biggest changes in a patch
  python build_patch_state.py --export-csv     also write csv/champion_features_wide.csv

Safe to run while the collector is running; re-run it after new patches or after
fetch_ability_data.py.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict

SCHEMA = """
DROP TABLE IF EXISTS patch_champion_features;
CREATE TABLE patch_champion_features (
    patch TEXT NOT NULL, champion_id TEXT NOT NULL, champ_key INTEGER,
    feature TEXT NOT NULL, value REAL,
    PRIMARY KEY (patch, champion_id, feature));
DROP TABLE IF EXISTS patch_item_features;
CREATE TABLE patch_item_features (
    patch TEXT NOT NULL, item_id TEXT NOT NULL, feature TEXT NOT NULL, value REAL,
    PRIMARY KEY (patch, item_id, feature));
DROP TABLE IF EXISTS patch_changes;
CREATE TABLE patch_changes (
    patch TEXT NOT NULL, prev_patch TEXT, entity TEXT NOT NULL, entity_id TEXT NOT NULL,
    feature TEXT NOT NULL, old REAL, new REAL, delta REAL, pct REAL,
    PRIMARY KEY (patch, entity, entity_id, feature));
CREATE INDEX IF NOT EXISTS ix_changes_entity ON patch_changes(entity, entity_id);
"""

SLOTS = ["Q", "W", "E", "R"]
DAMAGE_CATS = ["magic", "physical", "true", "heal", "shield"]
CLASSES = ["Assassin", "Fighter", "Mage", "Marksman", "Support", "Tank"]
# crowd control words found in ability data-value names -> cc type
CC_WORDS = [("stun", "stun"), ("root", "root"), ("snare", "root"), ("knockup", "knockup"),
            ("airborne", "knockup"), ("knockback", "knockup"), ("charm", "charm"),
            ("fear", "fear"), ("flee", "fear"), ("taunt", "taunt"), ("silence", "silence"),
            ("sleep", "sleep"), ("suppress", "suppress"), ("polymorph", "polymorph"),
            ("blind", "blind"), ("ground", "ground")]
HARD_CC = {"stun", "root", "knockup", "charm", "fear", "taunt", "sleep", "suppress", "polymorph"}


FIXES = []  # (patch, stat) where Data Dragon was replaced by game-file values


def pkey(p):
    return tuple(int(x) for x in p.split(".")[:2])


def stat_at(base, growth, level):
    """League's stat growth formula."""
    return base + growth * (level - 1) * (0.7025 + 0.0175 * (level - 1))


# --------------------------------------------------------------------------- #
# Champion features
# --------------------------------------------------------------------------- #
def champion_features(con, patch, version):
    feats = defaultdict(dict)   # champion_id -> {feature: value}
    keys = {}

    for cid, key, tags, partype, a, d, m, diff in con.execute(
            "SELECT champion_id, champ_key, tags, partype, info_attack, info_defense, "
            "info_magic, info_difficulty FROM champions WHERE version = ?", (version,)):
        if "_" in cid:
            continue  # special-mode variants (e.g. Jade_Ahri) - never in ranked games
        keys[cid] = key
        f = feats[cid]
        tagset = set((tags or "").split(","))
        for c in CLASSES:
            f[f"class_{c.lower()}"] = float(c in tagset)
        res = (partype or "").lower()
        f["resource_mana"] = float("mana" in res)
        f["resource_energy"] = float("energy" in res)
        f["resource_none"] = float(res in ("", "none", "no cost"))
        f["resource_other"] = float(not (f["resource_mana"] or f["resource_energy"]
                                         or f["resource_none"]))  # fury, rage, heat, ...
        for name, v in (("rating_attack", a), ("rating_defense", d), ("rating_magic", m),
                        ("rating_difficulty", diff)):
            if v is not None:
                f[name] = float(v)

    stats = defaultdict(dict)
    for cid, stat, value in con.execute(
            "SELECT champion_id, stat, value FROM champion_stats WHERE version = ?", (version,)):
        stats[cid][stat] = value or 0.0
    # Data Dragon sometimes breaks a stat for every champion (e.g. AD growth = 0 from
    # 16.5 on). When a stat is zero for all champions but the game files have real
    # values, use the game files' values instead.
    cd_stats = defaultdict(dict)
    if con.execute("SELECT name FROM sqlite_master WHERE name='cd_champion_stats'").fetchone():
        for cid, stat, value in con.execute(
                "SELECT champion_id, stat, value FROM cd_champion_stats WHERE patch = ?", (patch,)):
            cd_stats[cid][stat] = value
    for stat in {st for d in cd_stats.values() for st in d}:
        dd_vals = [s.get(stat, 0) for cid, s in stats.items() if "_" not in cid]
        cd_vals = [d[stat] for d in cd_stats.values() if d.get(stat)]
        if dd_vals and not any(dd_vals) and len(cd_vals) > len(dd_vals) / 2:
            for cid, d in cd_stats.items():
                if stat in d and cid in stats:
                    stats[cid][stat] = d[stat]
            FIXES.append((patch, stat))
    for cid, s in stats.items():
        f = feats[cid]
        for stat, v in s.items():
            f[f"stat_{stat}"] = v
        for lvl in (1, 9, 16):
            f[f"hp_l{lvl}"] = stat_at(s.get("hp", 0), s.get("hpperlevel", 0), lvl)
            f[f"armor_l{lvl}"] = stat_at(s.get("armor", 0), s.get("armorperlevel", 0), lvl)
            f[f"mr_l{lvl}"] = stat_at(s.get("spellblock", 0), s.get("spellblockperlevel", 0), lvl)
            f[f"ad_l{lvl}"] = stat_at(s.get("attackdamage", 0), s.get("attackdamageperlevel", 0), lvl)
            f[f"as_l{lvl}"] = s.get("attackspeed", 0) * (1 + stat_at(0, s.get("attackspeedperlevel", 0), lvl) / 100)
            # effective HP against physical damage: hp * (1 + armor/100)
            f[f"ehp_phys_l{lvl}"] = f[f"hp_l{lvl}"] * (1 + f[f"armor_l{lvl}"] / 100)
        f["is_ranged"] = float(s.get("attackrange", 0) >= 300)

    # Data Dragon ability basics
    for cid, slot, mr, cd, cost, rng in con.execute(
            "SELECT champion_id, slot, max_rank, cooldown, cost, range FROM abilities "
            "WHERE version = ? AND slot != 'P'", (version,)):
        f = feats[cid]
        for name, raw in (("cd", cd), ("cost", cost), ("range", rng)):
            try:
                vals = [float(x) for x in json.loads(raw)] if raw else []
            except (ValueError, TypeError):
                vals = []
            if vals:
                f[f"{slot}_{name}_r1"] = vals[0]
                f[f"{slot}_{name}_rmax"] = vals[-1]
        if mr:
            f[f"{slot}_max_rank"] = float(mr)

    # CommunityDragon formulas (if fetched)
    calcs = defaultdict(list)   # (cid, slot) -> [(calc, category, complex)]
    for cid, slot, calc, cat, cx in con.execute(
            "SELECT champion_id, slot, calc, category, complex FROM ability_calcs "
            "WHERE patch = ?", (patch,)):
        calcs[(cid, slot)].append((calc, cat, cx))
    base = defaultdict(dict)    # (cid, slot, calc) -> {rank: value}
    ratio = defaultdict(dict)   # (cid, slot, calc) -> {stat: value at max rank}
    data = defaultdict(dict)    # (cid, slot) -> {name: [values by rank]}
    for cid, slot, kind, name, rank, value in con.execute(
            "SELECT champion_id, slot, kind, name, rank, value FROM ability_values "
            "WHERE patch = ?", (patch,)):
        if kind == "calc_base":
            base[(cid, slot, name)][rank] = value
        elif kind == "calc_ratio":
            calc, stat = name.split("|", 1)
            prev = ratio[(cid, slot, calc)].get(stat)
            # keep the value at the highest rank seen
            if prev is None or rank >= prev[0]:
                ratio[(cid, slot, calc)][stat] = (rank, value)
        elif kind == "data":
            data[(cid, slot)].setdefault(name, {})[rank] = value

    have_cd = set()
    for (cid, slot), lst in calcs.items():
        f = feats[cid]
        have_cd.add(cid)
        for cat in DAMAGE_CATS:
            options = []
            for calc, c, cx in lst:
                if c != cat:
                    continue
                b = base.get((cid, slot, calc), {})
                if b:
                    options.append((b[max(b)], calc, cx))
            if not options:
                continue
            _, calc, cx = max(options)            # main formula = biggest at max rank
            b = base[(cid, slot, calc)]
            f[f"{slot}_{cat}_base_r1"] = b[min(b)]
            f[f"{slot}_{cat}_base_rmax"] = b[max(b)]
            for stat, (_, v) in ratio.get((cid, slot, calc), {}).items():
                f[f"{slot}_{cat}_ratio_{stat}"] = v
            if cx:
                f[f"{slot}_{cat}_partial"] = 1.0
            f[f"total_{cat}_base_rmax"] = f.get(f"total_{cat}_base_rmax", 0) + b[max(b)]
            for stat, (_, v) in ratio.get((cid, slot, calc), {}).items():
                f[f"total_{cat}_ratio_{stat}"] = f.get(f"total_{cat}_ratio_{stat}", 0) + v

    for (cid, slot), dvs in data.items():
        f = feats[cid]
        for name, byrank in dvs.items():
            v = byrank[max(byrank)]
            if not v:
                continue
            for word, cc in CC_WORDS:
                if word in name and ("duration" in name or "time" in name or name == word):
                    key = f"cc_{cc}_max"
                    if 0 < v <= 10:  # seconds; ignore obviously non-duration values
                        f[key] = max(f.get(key, 0), v)
                    break
            if "slow" in name and any(w in name for w in ("percent", "amount", "mod", "pct")):
                amt = v / 100 if v > 1 else v
                if 0 < amt <= 1:
                    f["cc_slow_max"] = max(f.get("cc_slow_max", 0), amt)
    for cid in feats:
        f = feats[cid]
        f["cc_hard_total"] = sum(v for k, v in f.items()
                                 if k.startswith("cc_") and k[3:-4] in HARD_CC)
        f["has_ability_numbers"] = float(cid in have_cd)
    return {cid: f for cid, f in feats.items() if cid in keys}, keys


# --------------------------------------------------------------------------- #
# Item features
# --------------------------------------------------------------------------- #
def item_features(con, version):
    feats = defaultdict(dict)
    for iid, name, total, gbase, purch, on_sr, tags, bfrom, binto in con.execute(
            "SELECT item_id, name, gold_total, gold_base, purchasable, on_sr, tags, "
            "builds_from, builds_into FROM items WHERE version = ?", (version,)):
        # model-relevant items: buyable on Summoner's Rift, not free, not a mode variant
        if not on_sr or not purch or not total or len(str(iid)) > 4:
            continue
        f = feats[iid]
        f["gold_total"] = float(total)
        f["gold_base"] = float(gbase or 0)
        f["is_completed"] = float(not binto)
        f["n_components"] = float(len([x for x in (bfrom or "").split(",") if x]))
        for t in (tags or "").split(","):
            if t:
                f[f"tag_{t.lower()}"] = 1.0
    for iid, stat, value in con.execute(
            "SELECT item_id, stat, value FROM item_stats WHERE version = ?", (version,)):
        if iid in feats and value:
            feats[iid][f"stat_{stat}"] = value
    return feats


# --------------------------------------------------------------------------- #
def changes(prev, cur):
    """Yield (entity_id, feature, old, new) for everything that differs."""
    for eid in sorted(set(prev) | set(cur)):
        a, b = prev.get(eid, {}), cur.get(eid, {})
        if not a and b:
            yield eid, "_new", None, 1.0
            continue
        if a and not b:
            yield eid, "_removed", 1.0, None
            continue
        for feat in sorted(set(a) | set(b)):
            o, n = a.get(feat, 0.0), b.get(feat, 0.0)
            if abs((o or 0) - (n or 0)) > 1e-9:
                yield eid, feat, o, n


def build(con):
    con.executescript(SCHEMA)
    patches = sorted(con.execute("SELECT patch, version FROM patches").fetchall(),
                     key=lambda r: pkey(r[0]))
    have_cd = {r[0] for r in con.execute("SELECT DISTINCT patch FROM ability_calcs")} \
        if con.execute("SELECT name FROM sqlite_master WHERE name='ability_calcs'").fetchone() \
        else set()
    if not have_cd:
        con.executescript("CREATE TABLE IF NOT EXISTS ability_calcs (patch, champion_id, slot, "
                          "calc, category, complex, level_scaling);"
                          "CREATE TABLE IF NOT EXISTS ability_values (patch, champion_id, slot, "
                          "kind, name, rank, value);")
        print("Note: no CommunityDragon ability numbers yet (run fetch_ability_data.py) - "
              "building from Data Dragon only.")
    prev_c = prev_i = prev_patch = None
    summary = []
    for patch, version in patches:
        cf, keys = champion_features(con, patch, version)
        itf = item_features(con, version)
        con.executemany("INSERT INTO patch_champion_features VALUES (?,?,?,?,?)",
                        [(patch, cid, keys.get(cid), k, v)
                         for cid, f in cf.items() for k, v in f.items()])
        con.executemany("INSERT INTO patch_item_features VALUES (?,?,?,?)",
                        [(patch, iid, k, v) for iid, f in itf.items() for k, v in f.items()])
        n_c = n_i = 0
        if prev_c is not None:
            rows = []
            skip_cd = (patch in have_cd) != (prev_patch in have_cd)  # avoid fake "changes"
            for entity, prev, cur in (("champion", prev_c, cf), ("item", prev_i, itf)):
                for eid, feat, o, n in changes(prev, cur):
                    if skip_cd and entity == "champion" and _is_cd_feature(feat):
                        continue
                    d = (n or 0) - (o or 0) if o is not None and n is not None else None
                    pct = d / o if d is not None and o else None
                    rows.append((patch, prev_patch, entity, eid, feat, o, n, d, pct))
            con.executemany("INSERT INTO patch_changes VALUES (?,?,?,?,?,?,?,?,?)", rows)
            n_c = len({r[3] for r in rows if r[2] == "champion"})
            n_i = len({r[3] for r in rows if r[2] == "item"})
        summary.append((patch, len(cf), len(itf), n_c, n_i, patch in have_cd))
        prev_c, prev_i, prev_patch = cf, itf, patch
    con.commit()
    if FIXES:
        by_stat = defaultdict(list)
        for p, st in FIXES:
            by_stat[st].append(p)
        for st, ps in by_stat.items():
            print(f"Note: Data Dragon has {st} = 0 for every champion in {len(ps)} patches "
                  f"({ps[0]} .. {ps[-1]}); used the game files' values instead.")
    missing = [p for p, *_ in summary if not any(fp == p for fp, _ in FIXES)]
    zero_ad = [p for p in missing if not con.execute(
        "SELECT SUM(value > 0) FROM champion_stats s JOIN patches USING (version) "
        "WHERE patch = ? AND stat = 'attackdamageperlevel'", (p,)).fetchone()[0]]
    if zero_ad:
        print(f"!!! AD growth is 0 for every champion in {len(zero_ad)} patches ({zero_ad[0]} .. "
              f"{zero_ad[-1]}) and no game-file values are available yet - run "
              f"fetch_ability_data.py, then this script again.")
    print("\npatch   champions  items  champions changed  items changed  ability numbers")
    for p, nc, ni, cc, ic, cd in summary:
        print(f"{p:>6} {nc:>10} {ni:>6} {cc:>18} {ic:>14}  {'yes' if cd else 'no'}")
    return summary


def _is_cd_feature(feat):
    return bool(re.match(r"^([QWER]_(magic|physical|true|heal|shield)_|total_|cc_|has_ability)", feat))


def report(con, patch):
    rows = con.execute("SELECT entity, entity_id, feature, old, new, pct FROM patch_changes "
                       "WHERE patch = ? ORDER BY entity, entity_id, feature", (patch,)).fetchall()
    if not rows:
        sys.exit(f"No changes stored for patch {patch} (first patch, or not built yet).")
    prev = rows and con.execute("SELECT prev_patch FROM patch_changes WHERE patch = ? LIMIT 1",
                                (patch,)).fetchone()[0]
    print(f"Changes {prev} -> {patch}:")
    cur = None
    for entity, eid, feat, o, n, pct in rows:
        if (entity, eid) != cur:
            cur = (entity, eid)
            print(f"\n  {entity} {eid}")
        fmt = lambda x: "-" if x is None else f"{x:g}"
        p = f"  ({pct:+.0%})" if pct is not None else ""
        print(f"     {feat:<34} {fmt(o):>10} -> {fmt(n):<10}{p}")


def export_csv(con):
    os.makedirs("csv", exist_ok=True)
    rows = con.execute("SELECT patch, champion_id, feature, value FROM patch_champion_features")
    wide, feats = defaultdict(dict), set()
    for p, cid, f, v in rows:
        wide[(p, cid)][f] = v
        feats.add(f)
    cols = sorted(feats)
    with open("csv/champion_features_wide.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patch", "champion_id"] + cols)
        for (p, cid), f in sorted(wide.items(), key=lambda kv: (pkey(kv[0][0]), kv[0][1])):
            w.writerow([p, cid] + [f.get(c, "") for c in cols])
    print(f"Wrote csv/champion_features_wide.csv ({len(wide):,} rows, {len(cols)} features)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build per-patch champion/item features and changes.")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--report", default=None, help="print the changes for one patch, e.g. 16.18")
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args(argv)
    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    if args.report:
        report(con, args.report)
        return
    build(con)
    if args.export_csv:
        export_csv(con)
    con.close()


if __name__ == "__main__":
    main()
