#!/usr/bin/env python3
"""
fetch_item_data.py  -  item passive numbers and item texts per patch.

Data Dragon (fetch_balance_data.py) only has an item's basic stats and cost. What an
item's passive does - Liandry's burn, a shield's size, an on-hit's damage, a cooldown -
lives in the game files. This script reads them for every patch:

  1. CommunityDragon's item file for the patch (the game's own item data): every number
     attached to each finished item and boots, e.g. BurnDamage 0.01, ShieldAmount 250.
     -> table item_passives (patch, item_id, name, value)
  2. Data Dragon's full item descriptions (tooltip text), used for the effect flags
     (shield, heal, anti-heal, ...).
     -> table item_texts (patch, item_id, name, description)

Only finished items and boots are kept (what players end games with).
Downloads are cached (gzip) in cdragon_cache/items/ and ddragon_cache/, so only new
patches are downloaded. Needs no API key; safe to run while the collector runs.

Usage:
  python fetch_item_data.py                     all patches in the database
  python fetch_item_data.py --inspect Liandry   what was read for one item (name or id)
  python fetch_item_data.py --reparse           re-read cached files (no downloading)
"""

import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

CDRAGON = "https://raw.communitydragon.org"
DDRAGON = "https://ddragon.leagueoflegends.com"
ITEM_FILES = ["game/items.cdtb.bin.json", "game/global/items/items.bin.json"]
DATA_VALUE_LISTS = ("mDataValues", "DataValues", "dataValues", "mItemDataValues", "itemDataValues")

SCHEMA = """
CREATE TABLE IF NOT EXISTS item_passives (
    patch    TEXT NOT NULL,
    item_id  TEXT NOT NULL,
    name     TEXT NOT NULL,     -- the game's name for the number, e.g. burndamage
    value    REAL,
    PRIMARY KEY (patch, item_id, name)
);
CREATE TABLE IF NOT EXISTS item_texts (
    patch        TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    name         TEXT,
    description  TEXT,          -- full tooltip text (with tags like <passive>, <shield>)
    PRIMARY KEY (patch, item_id)
);
CREATE TABLE IF NOT EXISTS cd_item_files (
    patch      TEXT PRIMARY KEY,
    status     INTEGER,         -- 200 = read ok
    n_items    INTEGER,         -- finished items / boots found in the file
    n_values   INTEGER,         -- passive numbers read
    fetched_at TEXT
);
"""


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def fnv1a(name):
    h = 0x811c9dc5
    for ch in name.lower().encode():
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return f"{{{h:08x}}}"


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):                       # e.g. {"baseValue": 12}
        for k in ("baseValue", "mValue", "value"):
            if k in x:
                return _num(x[k])
    return None


# --------------------------------------------------------------------------- #
# Downloads (cached)
# --------------------------------------------------------------------------- #
SESSION = requests.Session()


def get_json(url, cache, use_cache=True):
    if use_cache and os.path.exists(cache):
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return 200, json.load(f)
    if not use_cache and not os.path.exists(cache):
        return 0, None
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=120)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            doc = r.json()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with gzip.open(cache, "wt", encoding="utf-8") as f:
                json.dump(doc, f)
            return 200, doc
        if r.status_code in (403, 404):
            return r.status_code, None
        time.sleep(3 * (attempt + 1))
    return 0, None


def cdragon_items(base, patch, cache_dir, download=True):
    cache = os.path.join(cache_dir, "items", f"{patch}.json.gz")
    if os.path.exists(cache):
        return get_json(None, cache)
    if not download:
        return 0, None
    status = 0
    for path in ITEM_FILES:
        status, doc = get_json(f"{base}/{patch}/{path}", cache)
        if doc is not None:
            return status, doc
    return status, None


def ddragon_items(version, cache_dir, download=True):
    cache = os.path.join(cache_dir, f"item_{version}.json.gz")
    if os.path.exists(cache):
        return get_json(None, cache)
    if not download:
        return 0, None
    return get_json(f"{DDRAGON}/cdn/{version}/data/en_US/item.json", cache)


# --------------------------------------------------------------------------- #
# Parsing the game's item file
# --------------------------------------------------------------------------- #
def item_id_of(entry):
    for k in ("itemID", "mItemID", "itemId", "ItemID"):
        v = entry.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(int(v))
    return None


def data_values(obj):
    """{name: value} from any data-value list inside obj (searched a few levels deep)."""
    out = {}

    def add_list(lst):
        for d in lst:
            if not isinstance(d, dict):
                continue
            name = d.get("mName", d.get("name"))
            if name is None:
                continue
            v = d.get("mValue", d.get("value"))
            if v is None:
                vals = d.get("mValues", d.get("values"))
                if isinstance(vals, list) and vals:
                    # per-rank lists: use the first non-zero entry (items have no ranks)
                    nums = [_num(x) for x in vals]
                    nums = [x for x in nums if x is not None]
                    v = next((x for x in nums if x != 0), nums[0] if nums else None)
            v = _num(v) if v is not None else 0.0     # a missing value means 0
            if v is not None:
                out.setdefault(str(name).lower(), v)

    def walk(o, depth):
        if depth > 4:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k in DATA_VALUE_LISTS and isinstance(v, list):
                    add_list(v)
                elif isinstance(v, (dict, list)):
                    walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)
    walk(obj, 0)
    return out


def referenced(entry, doc, hashed):
    """Other objects in the file that this item points to (e.g. its passive's spell)."""
    refs = []

    def walk(o, depth):
        if depth > 3:
            return
        if isinstance(o, str):
            key = o if o in doc else hashed.get(o.lower())
            # follow links to spells/buffs, but not to other items (e.g. its components)
            if key and isinstance(doc.get(key), dict) and item_id_of(doc[key]) is None:
                refs.append(key)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)
    walk(entry, 0)
    return refs


def parse_items(doc, wanted):
    """{item_id: {name: value}} for the wanted item ids."""
    hashed = {}
    for k in doc:
        if isinstance(k, str) and not k.startswith("{"):
            hashed[fnv1a(k)] = k
    out = {}
    for key, entry in doc.items():
        if not isinstance(entry, dict):
            continue
        iid = item_id_of(entry)
        if iid is None or iid not in wanted:
            continue
        vals = data_values(entry)
        for ref in referenced(entry, doc, hashed):
            if ref == key:
                continue
            for n, v in data_values(doc[ref]).items():
                vals.setdefault(n, v)
        # stat fields written directly on the item (flat numbers), e.g. flatHPPoolMod
        for k, v in entry.items():
            if k.lower().startswith(("flat", "percent", "mflat", "mpercent")):
                x = _num(v)
                if x:
                    vals.setdefault(f"stat_{k.lower().lstrip('m')}", x)
        if iid in out:        # the same item listed twice (mode variants): keep the richer one
            if len(vals) <= len(out[iid]):
                continue
        out[iid] = vals
    return out


def wanted_items(con, version):
    """Finished items and boots on Summoner's Rift for this Data Dragon version."""
    ids = set()
    for iid, total, purch, on_sr, tags, binto in con.execute(
            "SELECT item_id, gold_total, purchasable, on_sr, tags, builds_into FROM items "
            "WHERE version = ?", (version,)):
        if not on_sr or not purch or not total or len(str(iid)) > 4:
            continue
        is_boots = "Boots" in (tags or "")
        if (not binto and total >= 900) or is_boots:
            ids.add(str(iid))
    return ids


# --------------------------------------------------------------------------- #
def store(con, patch, status, parsed, texts):
    con.execute("DELETE FROM item_passives WHERE patch = ?", (patch,))
    con.execute("DELETE FROM item_texts WHERE patch = ?", (patch,))
    if parsed:
        con.executemany("INSERT OR REPLACE INTO item_passives VALUES (?,?,?,?)",
                        [(patch, iid, n, v) for iid, vals in parsed.items() for n, v in vals.items()])
    if texts:
        con.executemany("INSERT OR REPLACE INTO item_texts VALUES (?,?,?,?)",
                        [(patch, iid, n, d) for iid, (n, d) in texts.items()])
    n_vals = sum(len(v) for v in (parsed or {}).values())
    con.execute("INSERT OR REPLACE INTO cd_item_files VALUES (?,?,?,?,?)",
                (patch, status, len(parsed or {}), n_vals,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def inspect(con, query):
    row = con.execute("SELECT patch FROM cd_item_files WHERE status = 200 ORDER BY rowid DESC").fetchall()
    if not row:
        sys.exit("No item data yet - run fetch_item_data.py first.")
    patches = sorted({r[0] for r in row}, key=pkey)
    last = patches[-1]
    hits = con.execute("SELECT item_id, name FROM item_texts WHERE patch = ? AND "
                       "(item_id = ? OR lower(name) LIKE ?)", (last, query, f"%{query.lower()}%")).fetchall()
    if not hits:
        sys.exit(f"No finished item matching '{query}' on patch {last}.")
    for iid, name in hits:
        print(f"\n{name} ({iid}), patch {last}")
        vals = con.execute("SELECT name, value FROM item_passives WHERE patch = ? AND item_id = ? "
                           "ORDER BY name", (last, iid)).fetchall()
        if not vals:
            print("  no passive numbers found")
        for n, v in vals:
            print(f"  {n:<40} {v:g}")
        desc = con.execute("SELECT description FROM item_texts WHERE patch = ? AND item_id = ?",
                           (last, iid)).fetchone()[0] or ""
        print("  tooltip: " + re.sub(r"\s+", " ", desc)[:400])
        # history of each number
        hist = con.execute("SELECT patch, name, value FROM item_passives WHERE item_id = ?",
                           (iid,)).fetchall()
        by = {}
        for p, n, v in hist:
            by.setdefault(n, {})[p] = v
        changed = {n: s for n, s in by.items() if len(set(s.values())) > 1}
        if changed:
            print("  changed across patches:")
            for n, s in changed.items():
                seq = [f"{p}: {s[p]:g}" for p in sorted(s, key=pkey)]
                print(f"    {n}: " + ", ".join(seq[-6:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Item passive numbers and texts per patch.")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--base-url", default=CDRAGON)
    ap.add_argument("--cache-dir", default="cdragon_cache")
    ap.add_argument("--dd-cache-dir", default="ddragon_cache")
    ap.add_argument("--reparse", action="store_true", help="re-read cached files, no downloads")
    ap.add_argument("--inspect", default=None, help="show what was read for one item")
    args = ap.parse_args(argv)
    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found - run this from the project folder.")
    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    if args.inspect:
        inspect(con, args.inspect)
        return
    patches = sorted(con.execute("SELECT patch, version FROM patches").fetchall(),
                     key=lambda r: pkey(r[0]))
    done = {r[0] for r in con.execute("SELECT patch FROM cd_item_files WHERE status = 200")}
    todo = [(p, v) for p, v in patches if args.reparse or p not in done]
    print(f"{len(todo)} of {len(patches)} patches to read"
          + (" (from the cache)" if args.reparse else ""))
    for patch, version in todo:
        wanted = wanted_items(con, version)
        status, doc = cdragon_items(args.base_url, patch, args.cache_dir,
                                    download=not args.reparse)
        parsed = parse_items(doc, wanted) if doc else {}
        _, dd = ddragon_items(version, args.dd_cache_dir, download=not args.reparse)
        texts = {}
        for iid, it in ((dd or {}).get("data") or {}).items():
            if iid in wanted:
                texts[iid] = (it.get("name"), it.get("description") or "")
        store(con, patch, status if doc else (status or 0), parsed, texts)
        n_with = sum(1 for v in parsed.values() if any(not k.startswith("stat_") for k in v))
        print(f"  {patch:>6}: {len(wanted)} finished items/boots | game file "
              f"{'ok' if doc else f'missing (HTTP {status})'}: {n_with} items with passive numbers, "
              f"{sum(len(v) for v in parsed.values())} numbers | {len(texts)} tooltips", flush=True)
    rows = con.execute("SELECT patch, status, n_items, n_values FROM cd_item_files").fetchall()
    bad = [p for p, s, *_ in rows if s != 200]
    if bad:
        print(f"\n!!! No game item file for {len(bad)} patch(es): {', '.join(sorted(bad, key=pkey))}. "
              f"Item passives for those patches are copied from the nearest patch later.")


if __name__ == "__main__":
    main()
