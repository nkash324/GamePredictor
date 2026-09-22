#!/usr/bin/env python3
"""
fetch_balance_data.py  -  Part 1 of the LoL prediction project.

Downloads per-patch game balance data from Riot's Data Dragon (free, no API key)
and stores it in a single SQLite database (lol_data.db) that later parts of the
project (outcome data, feature building, modelling) will read from.

What it collects for every patch:
  - champions        : name, roles/tags, resource type, Riot's 1-10 ratings
  - champion_stats   : base stats + per-level growth (hp, armor, ad, attack speed ...)
  - abilities        : passive/Q/W/E/R cooldowns, costs, ranges per rank + text
  - items            : price, build path, Summoner's Rift availability
  - item_stats       : the stats each item gives

Raw JSON is also cached in ./raw/<version>/ so nothing is downloaded twice and
no information is lost if we want to parse more fields later.

Usage:
  pip install requests
  python fetch_balance_data.py                  # last 10 patches
  python fetch_balance_data.py --patches 40     # last 40 patches
  python fetch_balance_data.py --since 13.1     # every patch from 13.1 onwards
  python fetch_balance_data.py --export-csv     # also write wide CSVs to ./csv/
"""

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DDRAGON = "https://ddragon.leagueoflegends.com"
SUMMONERS_RIFT = "11"
SLOTS = ["Q", "W", "E", "R"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS patches (
    version     TEXT PRIMARY KEY,   -- Data Dragon version, e.g. 14.23.1
    patch       TEXT NOT NULL,      -- short patch, e.g. 14.23
    major       INTEGER NOT NULL,
    minor       INTEGER NOT NULL,
    fetched_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS champions (
    version      TEXT NOT NULL,
    champion_id  TEXT NOT NULL,     -- e.g. MonkeyKing
    champ_key    INTEGER,           -- numeric id used by the match API
    name         TEXT,              -- e.g. Wukong
    title        TEXT,
    tags         TEXT,              -- comma separated, e.g. Fighter,Tank
    partype      TEXT,              -- resource: Mana, Energy, None ...
    info_attack     INTEGER,
    info_defense    INTEGER,
    info_magic      INTEGER,
    info_difficulty INTEGER,
    PRIMARY KEY (version, champion_id)
);
CREATE TABLE IF NOT EXISTS champion_stats (
    version      TEXT NOT NULL,
    champion_id  TEXT NOT NULL,
    stat         TEXT NOT NULL,     -- e.g. hp, hpperlevel, armor, attackspeed
    value        REAL,
    PRIMARY KEY (version, champion_id, stat)
);
CREATE TABLE IF NOT EXISTS abilities (
    version      TEXT NOT NULL,
    champion_id  TEXT NOT NULL,
    slot         TEXT NOT NULL,     -- P, Q, W, E, R
    spell_id     TEXT,
    name         TEXT,
    max_rank     INTEGER,
    cooldown     TEXT,              -- JSON list, one value per rank
    cost         TEXT,              -- JSON list
    cost_type    TEXT,
    range        TEXT,              -- JSON list
    description  TEXT,
    tooltip      TEXT,
    PRIMARY KEY (version, champion_id, slot)
);
CREATE TABLE IF NOT EXISTS items (
    version        TEXT NOT NULL,
    item_id        TEXT NOT NULL,
    name           TEXT,
    gold_total     INTEGER,
    gold_base      INTEGER,
    gold_sell      INTEGER,
    purchasable    INTEGER,
    on_sr          INTEGER,         -- 1 if available on Summoner's Rift
    tags           TEXT,
    builds_from    TEXT,
    builds_into    TEXT,
    description    TEXT,
    PRIMARY KEY (version, item_id)
);
CREATE TABLE IF NOT EXISTS item_stats (
    version   TEXT NOT NULL,
    item_id   TEXT NOT NULL,
    stat      TEXT NOT NULL,        -- e.g. FlatPhysicalDamageMod
    value     REAL,
    PRIMARY KEY (version, item_id, stat)
);
"""


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
class DataDragon:
    def __init__(self, base_url, raw_dir, lang, session=None):
        self.base = base_url.rstrip("/")
        self.raw_dir = Path(raw_dir)
        self.lang = lang
        self.s = session or requests.Session()

    def _get_json(self, url, cache_path=None, retries=3):
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        for attempt in range(1, retries + 1):
            try:
                r = self.s.get(url, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt == retries:
                    raise
                print(f"    retry {attempt} for {url} ({e})")
                time.sleep(2 * attempt)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.2)  # be polite to the CDN
        return data

    def versions(self):
        return self._get_json(f"{self.base}/api/versions.json")

    def file(self, version, name):
        url = f"{self.base}/cdn/{version}/data/{self.lang}/{name}"
        return self._get_json(url, self.raw_dir / version / name)


def pick_patch_versions(all_versions, n_patches=None, since=None):
    """One Data Dragon version per patch (the latest hotfix), newest first."""
    best = {}
    for v in all_versions:
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
        if not m:
            continue  # skips things like 'lolpatch_7.20'
        major, minor, sub = map(int, m.groups())
        key = (major, minor)
        if key not in best or sub > best[key][0]:
            best[key] = (sub, v)
    keys = sorted(best, reverse=True)
    if since:
        smaj, smin = map(int, since.split(".")[:2])
        keys = [k for k in keys if k >= (smaj, smin)]
    if n_patches:
        keys = keys[:n_patches]
    return [best[k][1] for k in keys]


# --------------------------------------------------------------------------- #
# Parsing + storing
# --------------------------------------------------------------------------- #
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def store_champions(con, version, champ_full):
    champs, stats, abil = [], [], []
    for cid, c in champ_full["data"].items():
        info = c.get("info", {})
        champs.append((
            version, cid, int(c["key"]) if str(c.get("key", "")).isdigit() else None,
            c.get("name"), c.get("title"), ",".join(c.get("tags", [])),
            c.get("partype"), info.get("attack"), info.get("defense"),
            info.get("magic"), info.get("difficulty"),
        ))
        for stat, val in c.get("stats", {}).items():
            stats.append((version, cid, stat, _num(val)))

        p = c.get("passive") or {}
        abil.append((version, cid, "P", None, p.get("name"), None, None, None,
                     None, None, p.get("description"), None))
        for slot, sp in zip(SLOTS, c.get("spells", [])):
            abil.append((
                version, cid, slot, sp.get("id"), sp.get("name"), sp.get("maxrank"),
                json.dumps(sp.get("cooldown")), json.dumps(sp.get("cost")),
                sp.get("costType"), json.dumps(sp.get("range")),
                sp.get("description"), sp.get("tooltip"),
            ))
    con.executemany("INSERT OR REPLACE INTO champions VALUES (?,?,?,?,?,?,?,?,?,?,?)", champs)
    con.executemany("INSERT OR REPLACE INTO champion_stats VALUES (?,?,?,?)", stats)
    con.executemany("INSERT OR REPLACE INTO abilities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", abil)
    return len(champs)


def store_items(con, version, item_json):
    items, stats = [], []
    for iid, it in item_json["data"].items():
        gold = it.get("gold", {})
        on_sr = int(bool(it.get("maps", {}).get(SUMMONERS_RIFT, False)))
        items.append((
            version, iid, it.get("name"), gold.get("total"), gold.get("base"),
            gold.get("sell"), int(bool(gold.get("purchasable", False))), on_sr,
            ",".join(it.get("tags", [])), ",".join(it.get("from", [])),
            ",".join(it.get("into", [])), it.get("plaintext") or it.get("description"),
        ))
        for stat, val in it.get("stats", {}).items():
            stats.append((version, iid, stat, _num(val)))
    con.executemany("INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", items)
    con.executemany("INSERT OR REPLACE INTO item_stats VALUES (?,?,?,?)", stats)
    return len(items)


# --------------------------------------------------------------------------- #
# Optional CSV export (wide format, one row per champion per patch)
# --------------------------------------------------------------------------- #
def export_csv(con, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stat_names = [r[0] for r in con.execute(
        "SELECT DISTINCT stat FROM champion_stats ORDER BY stat")]
    rows = con.execute("""
        SELECT c.version, p.patch, c.champion_id, c.name, c.tags, c.partype,
               s.stat, s.value
        FROM champions c
        JOIN patches p ON p.version = c.version
        LEFT JOIN champion_stats s
               ON s.version = c.version AND s.champion_id = c.champion_id
        ORDER BY p.major, p.minor, c.champion_id
    """)
    wide = {}
    for version, patch, cid, name, tags, partype, stat, value in rows:
        rec = wide.setdefault((version, cid), {
            "version": version, "patch": patch, "champion_id": cid,
            "name": name, "tags": tags, "partype": partype})
        if stat:
            rec[stat] = value
    header = ["version", "patch", "champion_id", "name", "tags", "partype"] + stat_names
    with open(out / "champion_stats_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(wide.values())

    for table in ["abilities", "items"]:
        cur = con.execute(f"SELECT * FROM {table}")
        with open(out / f"{table}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([d[0] for d in cur.description])
            w.writerows(cur)
    print(f"CSV files written to {out.resolve()}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Download LoL balance data per patch.")
    ap.add_argument("--patches", type=int, default=None,
                    help="number of most recent patches to fetch (default 10)")
    ap.add_argument("--since", default=None, help="fetch every patch since e.g. 13.1")
    ap.add_argument("--db", default="lol_data.db", help="SQLite output file")
    ap.add_argument("--raw-dir", default="raw", help="folder for cached raw JSON")
    ap.add_argument("--lang", default="en_US")
    ap.add_argument("--refresh", action="store_true",
                    help="re-parse patches that are already in the database")
    ap.add_argument("--export-csv", action="store_true",
                    help="also write CSV files to ./csv/")
    ap.add_argument("--base-url", default=DDRAGON, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.patches is None and args.since is None:
        args.patches = 10

    dd = DataDragon(args.base_url, args.raw_dir, args.lang)
    con = sqlite3.connect(args.db, timeout=120)  # wait if the collector is writing
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)

    print("Fetching version list from Data Dragon ...")
    versions = pick_patch_versions(dd.versions(), args.patches, args.since)
    done = {r[0] for r in con.execute("SELECT version FROM patches")}
    print(f"{len(versions)} patches selected: {versions[-1]} -> {versions[0]}")

    for v in versions:
        if v in done and not args.refresh:
            print(f"  {v}: already in database, skipping")
            continue
        try:
            n_c = store_champions(con, v, dd.file(v, "championFull.json"))
            n_i = store_items(con, v, dd.file(v, "item.json"))
        except Exception as e:  # keep going if one old patch is odd
            con.rollback()
            print(f"  {v}: FAILED ({e})", file=sys.stderr)
            continue
        major, minor = map(int, v.split(".")[:2])
        con.execute("INSERT OR REPLACE INTO patches VALUES (?,?,?,?,?)",
                    (v, f"{major}.{minor}", major, minor,
                     datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()
        print(f"  {v}: {n_c} champions, {n_i} items")

    if args.export_csv:
        export_csv(con, "csv")

    total = con.execute("SELECT COUNT(*) FROM patches").fetchone()[0]
    print(f"Done. {total} patches stored in {Path(args.db).resolve()}")
    con.close()


if __name__ == "__main__":
    main()
