#!/usr/bin/env python3
"""
fetch_ability_data.py  -  ability numbers per patch from CommunityDragon.

Data Dragon (fetch_balance_data.py) gives cooldowns, costs and ranges, but not the
numbers that matter most: base damage per rank, AP/AD/HP ratios, heal and shield
amounts, crowd-control durations. CommunityDragon publishes the game's own data
files for every patch; this script downloads each champion's file, parses the
ability formulas and stores clean numbers in lol_data.db:

  ability_values   every number per champion / patch / ability / rank:
                     kind 'data'       raw data values (e.g. BaseDamage, CharmDuration)
                     kind 'calc_base'  flat part of a formula (e.g. TotalDamage base)
                     kind 'calc_ratio' scaling part, name 'formula|stat' (e.g. totaldamage|ap)
  ability_calcs    one row per formula: what it is (magic/physical/true damage, heal,
                   shield - read from the ability's tooltip) and whether it was too
                   complex to capture fully
  cd_files         which files were fetched, and a self-check against Data Dragon

Usage:
  python fetch_ability_data.py                  all patches in the database (~1 hour first time)
  python fetch_ability_data.py --inspect Ahri   show what was parsed for one champion
  python fetch_ability_data.py --reparse        re-parse cached files (no downloading)

Downloads are cached (gzip) in cdragon_cache/, about 0.5-1 GB for 25 patches.
Needs no API key and does not touch the Riot API rate limit, so it can run while the
collector is running.
"""

import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

CDRAGON = "https://raw.communitydragon.org"
SLOTS = ["Q", "W", "E", "R"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS cd_files (
    patch        TEXT NOT NULL,
    champion_id  TEXT NOT NULL,
    status       INTEGER,        -- HTTP status (200 = ok)
    rank_offset  INTEGER,        -- where rank 1 sits in the game's arrays (self-check)
    cd_match     INTEGER,        -- abilities whose cooldowns matched Data Dragon
    cd_checked   INTEGER,        -- abilities that could be checked
    fetched_at   TEXT,
    PRIMARY KEY (patch, champion_id)
);
CREATE TABLE IF NOT EXISTS ability_values (
    patch        TEXT NOT NULL,
    champion_id  TEXT NOT NULL,
    slot         TEXT NOT NULL,  -- P Q W E R
    kind         TEXT NOT NULL,  -- data | calc_base | calc_ratio
    name         TEXT NOT NULL,  -- lower-case name; calc_ratio: 'formula|stat'
    rank         INTEGER NOT NULL,
    value        REAL,
    PRIMARY KEY (patch, champion_id, slot, kind, name, rank)
);
CREATE TABLE IF NOT EXISTS cd_champion_stats (   -- base stats straight from the game files
    patch        TEXT NOT NULL,
    champion_id  TEXT NOT NULL,
    stat         TEXT NOT NULL,  -- same names as Data Dragon's champion_stats
    value        REAL,
    PRIMARY KEY (patch, champion_id, stat)
);
CREATE TABLE IF NOT EXISTS ability_calcs (
    patch          TEXT NOT NULL,
    champion_id    TEXT NOT NULL,
    slot           TEXT NOT NULL,
    calc           TEXT NOT NULL,
    category       TEXT,         -- magic | physical | true | heal | shield | other
    complex        INTEGER,      -- 1 = contains parts we could not fully capture
    level_scaling  INTEGER,      -- 1 = scales with champion level
    PRIMARY KEY (patch, champion_id, slot, calc)
);
"""

# The game's stat ids (mStat) and formula ids (mStatFormula). Unknown ids are kept
# as 'statN' so nothing is lost even if a name is wrong.
STAT_NAMES = {0: "ap", 1: "armor", 2: "ad", 3: "attackspeed", 5: "mr", 6: "movespeed",
              7: "crit", 11: "hp", 12: "currenthp", 13: "missinghp"}
FORMULA_NAMES = {0: "", 1: "_base", 2: "_bonus"}

# game-file base stats -> Data Dragon stat names (only fields on the same scale).
# Older files store plain numbers (baseDamage: 53); newer ones (16.5+) wrap them
# (baseDamageModifiable: {baseValue: 53}) and rename magic resist (baseMR, mrPerLevel).
CD_STATS = {
    "hp": ["baseHP", "baseHPModifiable"],
    "hpperlevel": ["hpPerLevel", "hpPerLevelModifiable"],
    "attackdamage": ["baseDamage", "baseDamageModifiable"],
    "attackdamageperlevel": ["damagePerLevel", "damagePerLevelModifiable"],
    "armor": ["baseArmor", "baseArmorModifiable"],
    "armorperlevel": ["armorPerLevel", "armorPerLevelModifiable"],
    "spellblock": ["baseSpellBlock", "baseMR", "baseSpellBlockModifiable", "baseMRModifiable"],
    "spellblockperlevel": ["spellBlockPerLevel", "mrPerLevel", "spellBlockPerLevelModifiable",
                           "mrPerLevelModifiable"],
    "movespeed": ["baseMoveSpeed", "baseMoveSpeedModifiable"],
    "attackrange": ["attackRange", "attackRangeModifiable"],
    "attackspeed": ["attackSpeed", "attackSpeedModifiable"],
    "attackspeedperlevel": ["attackSpeedPerLevel", "attackSpeedPerLevelModifiable"],
}


def record_stat(record, names):
    for n in names:
        v = record.get(n)
        if isinstance(v, dict):
            v = v.get("baseValue")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def field(d, *names, default=None):
    """First present field among naming variants (Riot renames fields between patches)."""
    for n in names:
        if n in d:
            return d[n]
    return default

TOOLTIP_TAGS = {"magicdamage": "magic", "physicaldamage": "physical", "truedamage": "true",
                "healing": "heal", "shield": "shield"}


# --------------------------------------------------------------------------- #
# Formula evaluation: every formula becomes  base[rank] + sum(ratio[stat][rank] * stat)
# --------------------------------------------------------------------------- #
class Lin:
    """A linear formula: flat base per rank plus per-stat ratios per rank."""

    def __init__(self, n, base=None, ratios=None, complex_=False, level=False):
        self.n = n
        self.base = base or [0.0] * n
        self.ratios = ratios or {}
        self.complex = complex_
        self.level = level

    def __add__(self, o):
        r = {k: list(v) for k, v in self.ratios.items()}
        for k, v in o.ratios.items():
            r[k] = [a + b for a, b in zip(r.get(k, [0.0] * self.n), v)]
        return Lin(self.n, [a + b for a, b in zip(self.base, o.base)], r,
                   self.complex or o.complex, self.level or o.level)

    def scaled(self, factor):
        return Lin(self.n, [a * f for a, f in zip(self.base, factor)],
                   {k: [a * f for a, f in zip(v, factor)] for k, v in self.ratios.items()},
                   self.complex, self.level)

    def is_flat(self):
        return not self.ratios


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class SpellParser:
    def __init__(self, spell, n_ranks, offset):
        self.spell = spell or {}
        self.n = n_ranks
        self.offset = offset
        self.dv = {}
        for d in field(self.spell, "mDataValues", "DataValues", "dataValues", default=None) or []:
            if not isinstance(d, dict):
                continue
            name = str(field(d, "mName", "name", default="")).lower()
            vals = field(d, "mValues", "values", default=None) or []
            self.dv[name] = self._ranked(vals)
        self.calcs = {str(k).lower(): v for k, v in
                      (self.spell.get("mSpellCalculations") or {}).items()}
        self.effects = self.spell.get("mEffectAmount") or []

    def _ranked(self, vals):
        out = []
        for r in range(self.n):
            i = r + self.offset
            out.append(_num(vals[i]) if i < len(vals) else (_num(vals[-1]) if vals else 0.0))
        return out

    def _const(self, v):
        return [v] * self.n

    def _stat_key(self, part):
        stat = int(_num(part.get("mStat", 0)))
        formula = int(_num(part.get("mStatFormula", 0)))
        return STAT_NAMES.get(stat, f"stat{stat}") + FORMULA_NAMES.get(formula, f"_f{formula}")

    def part(self, p, depth=0):
        n = self.n
        if depth > 12 or not isinstance(p, dict):
            return Lin(n, complex_=True)
        t = str(p.get("__type", ""))
        if t == "NamedDataValueCalculationPart":
            return Lin(n, list(self.dv.get(str(p.get("mDataValue", "")).lower(), self._const(0.0))))
        if t == "NumberCalculationPart":
            return Lin(n, self._const(_num(p.get("mNumber"))))
        if t == "EffectValueCalculationPart":
            idx = int(_num(p.get("mEffectIndex", 1))) - 1
            vals = (self.effects[idx].get("value") if 0 <= idx < len(self.effects)
                    and isinstance(self.effects[idx], dict) else None) or []
            return Lin(n, self._ranked(vals))
        if t == "StatByCoefficientCalculationPart":
            return Lin(n, ratios={self._stat_key(p): self._const(_num(p.get("mCoefficient")))})
        if t == "StatByNamedDataValueCalculationPart":
            vals = self.dv.get(str(p.get("mDataValue", "")).lower(), self._const(0.0))
            return Lin(n, ratios={self._stat_key(p): list(vals)})
        if t == "StatBySubPartCalculationPart":
            sub = self.part(p.get("mSubpart"), depth + 1)
            return Lin(n, ratios={self._stat_key(p): sub.base}, complex_=sub.complex)
        if t == "SumOfSubPartsCalculationPart":
            total = Lin(n)
            for sp in p.get("mSubparts") or []:
                total = total + self.part(sp, depth + 1)
            return total
        if t == "ProductOfSubPartsCalculationPart":
            a, b = self.part(p.get("mPart1"), depth + 1), self.part(p.get("mPart2"), depth + 1)
            if a.is_flat():
                return b.scaled(a.base)
            if b.is_flat():
                return a.scaled(b.base)
            return Lin(n, complex_=True)
        if t == "ByCharLevelInterpolationCalculationPart":
            lo, hi = _num(p.get("mStartValue")), _num(p.get("mEndValue"))
            return Lin(n, self._const((lo + hi) / 2), level=True)  # mid-game value
        if t == "ByCharLevelBreakpointsCalculationPart":
            return Lin(n, self._const(_num(p.get("mLevel1Value"))), complex_=True, level=True)
        if t == "GameCalculationPart" or "mGameCalculation" in p:
            return self.calc(str(p.get("mGameCalculation", "")).lower(), depth + 1)
        if "mSpellCalculationKey" in p:  # reference to another formula (hashed type name)
            return self.calc(str(p.get("mSpellCalculationKey", "")).lower(), depth + 1)
        return Lin(n, complex_=True)  # buff stacks, hashed types, ...

    def calc(self, name, depth=0):
        c = self.calcs.get(name)
        if not isinstance(c, dict) or depth > 12:
            return Lin(self.n, complex_=True)
        t = str(c.get("__type", ""))
        if t == "GameCalculationModified":
            base = self.calc(str(c.get("mModifiedGameCalculation", "")).lower(), depth + 1)
            mult = self.part(c.get("mMultiplier"), depth + 1)
            return base.scaled(mult.base) if mult.is_flat() else Lin(self.n, complex_=True)
        if t == "GameCalculationConditional":
            return self.calc(str(c.get("mDefaultGameCalculation", "")).lower(), depth + 1)
        total = Lin(self.n)
        for p in c.get("mFormulaParts") or []:
            total = total + self.part(p, depth + 1)
        if c.get("mMultiplier") is not None:
            mult = self.part(c.get("mMultiplier"), depth + 1)
            total = total.scaled(mult.base) if mult.is_flat() else Lin(self.n, complex_=True)
        return total


# --------------------------------------------------------------------------- #
# Finding each ability inside a champion's file
# --------------------------------------------------------------------------- #
def find_spells(doc, champion_id, dd_spell_ids):
    """Return {slot: spell_dict} for P/Q/W/E/R."""
    cid = champion_id.lower()
    objects = {k.lower(): v for k, v in doc.items() if isinstance(v, dict)}
    spells = {k: v for k, v in objects.items()
              if k.startswith(f"characters/{cid}/spells/") and isinstance(v.get("mSpell"), dict)}

    def by_name(name):
        if not name:
            return None
        name = name.lower()
        for k, v in spells.items():
            if k.endswith("/" + name) or str(v.get("mScriptName", "")).lower() == name:
                return v
        return None

    record = next((v for k, v in objects.items()
                   if k.startswith(f"characters/{cid}/") and k.endswith("/characterrecords/root")),
                  {})
    names = record.get("spellNames") or []
    out = {}
    for i, slot in enumerate(SLOTS):
        s = by_name(dd_spell_ids.get(slot))
        s = s or (by_name(names[i]) if i < len(names) else None)
        if s:
            out[slot] = s
    # the record names the real passive; otherwise guess from the object names
    passive = None
    ppath = str(record.get("mCharacterPassiveSpell") or "").lower()
    if ppath:
        passive = objects.get(ppath) if isinstance(objects.get(ppath), dict) else None
    if not passive and record.get("passiveLuaName"):
        passive = by_name(str(record["passiveLuaName"]))
    if not passive:
        cands = [(k, v) for k, v in spells.items() if "passive" in k]
        with_numbers = [kv for kv in cands if field(kv[1]["mSpell"], "mDataValues", "DataValues")
                        or kv[1]["mSpell"].get("mSpellCalculations")]
        passive = (with_numbers or cands or [(None, None)])[0][1]
    if passive:
        out["P"] = passive
    return out, spells, record


def _cd_matches(spell, cds, offset):
    game = ((spell or {}).get("mSpell") or {}).get("cooldownTime")
    if not game or not cds:
        return False
    seg = [_num(x) for x in game[offset:offset + len(cds)]]
    return len(seg) == len(cds) and all(abs(a - b) < 0.01 for a, b in zip(seg, cds))


def assign_by_cooldown(all_spells, dd_cooldowns, offset):
    """Slot -> spell, using only cooldown matches that are unambiguous."""
    out = {}
    for slot, cds in dd_cooldowns.items():
        if slot == "P" or not cds:
            continue
        hits = [s for s in all_spells.values() if _cd_matches(s, cds, offset)]
        # a unique match that doesn't also fit another slot's cooldowns
        if len(hits) == 1 and not any(_cd_matches(hits[0], c, offset)
                                      for o, c in dd_cooldowns.items() if o != slot and c != cds):
            out[slot] = hits[0]
    return out


def detect_offset(spells, dd_cooldowns):
    """Rank 1 is at index 1 or 0 of the game's arrays: decide by matching Data Dragon."""
    votes = {0: 0, 1: 0}
    checked = 0
    for slot, cds in dd_cooldowns.items():
        s = (spells.get(slot) or {}).get("mSpell") or {}
        game = s.get("cooldownTime")
        if not game or not cds:
            continue
        checked += 1
        for off in (1, 0):
            seg = [_num(x) for x in game[off:off + len(cds)]]
            if len(seg) == len(cds) and all(abs(a - b) < 0.01 for a, b in zip(seg, cds)):
                votes[off] += 1
                break
    offset = 1 if votes[1] >= votes[0] else 0
    return offset, votes[offset], checked


def tooltip_categories(tooltip):
    """Map formula names to damage/heal/shield using the Data Dragon tooltip tags."""
    cats = {}
    for tag, inner in re.findall(r"<(\w+)>(.*?)</\1>", tooltip or "", flags=re.S):
        cat = TOOLTIP_TAGS.get(tag.lower())
        if not cat:
            continue
        for name in re.findall(r"\{\{\s*([A-Za-z0-9_]+)", inner):
            cats.setdefault(name.lower(), cat)
    return cats


def parse_champion(doc, champion_id, dd):
    """dd: {slot: {'id', 'max_rank', 'cooldown': [...], 'tooltip'}} from Data Dragon."""
    spells, all_spells, record = find_spells(doc, champion_id,
                                             {s: d.get("id") for s, d in dd.items()})
    base_stats = [(stat, v) for stat, names in CD_STATS.items()
                  if (v := record_stat(record, names)) is not None]
    dd_cds = {s: d.get("cooldown") for s, d in dd.items() if d.get("cooldown")}
    offset, cd_match, cd_checked = detect_offset(spells, dd_cds)
    if cd_match < cd_checked:
        # some abilities don't line up with Data Dragon: re-assign by cooldown
        for off in (offset, 1 - offset):
            fixed = assign_by_cooldown(all_spells, dd_cds, off)
            if fixed:
                spells.update(fixed)
                offset, cd_match, cd_checked = detect_offset(spells, dd_cds)
                break
    values, calcs = [], []
    for slot, sp in spells.items():
        n = (dd.get(slot) or {}).get("max_rank") or (5 if slot != "R" else 3)
        if slot == "P":
            n = 1
        parser = SpellParser(sp.get("mSpell"), n, offset)
        cats = tooltip_categories((dd.get(slot) or {}).get("tooltip"))
        for name in parser.calcs:  # untagged formulas: fall back to the name
            if name not in cats:
                if "heal" in name:
                    cats[name] = "heal"
                elif "shield" in name:
                    cats[name] = "shield"
        for name, vals in parser.dv.items():
            for r, v in enumerate(vals, 1):
                values.append((slot, "data", name, r, v))
        for name in parser.calcs:
            lin = parser.calc(name)
            for r, v in enumerate(lin.base, 1):
                values.append((slot, "calc_base", name, r, v))
            for stat, vals in lin.ratios.items():
                for r, v in enumerate(vals, 1):
                    values.append((slot, "calc_ratio", f"{name}|{stat}", r, v))
            calcs.append((slot, name, cats.get(name, "other"), int(lin.complex), int(lin.level)))
    return values, calcs, offset, cd_match, cd_checked, sorted(spells), base_stats


# --------------------------------------------------------------------------- #
# Database + downloading
# --------------------------------------------------------------------------- #
def load_dd(con, version):
    dd = {}
    for cid, slot, sid, mr, cd, tip in con.execute(
            "SELECT champion_id, slot, spell_id, max_rank, cooldown, tooltip FROM abilities "
            "WHERE version = ?", (version,)):
        try:
            cds = [float(x) for x in json.loads(cd)] if cd else None
        except (ValueError, TypeError):
            cds = None
        if "_" in cid:
            continue  # special-mode variants (e.g. Jade_Ahri) - never in ranked games
        dd.setdefault(cid, {})[slot] = {"id": sid, "max_rank": mr, "cooldown": cds, "tooltip": tip}
    return dd


def cache_path(cache_dir, patch, cid):
    return os.path.join(cache_dir, patch, f"{cid.lower()}.json.gz")


_local = threading.local()


def download(base, patch, cid, cache_dir, use_cache=True):
    path = cache_path(cache_dir, patch, cid)
    if use_cache and os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return 200, json.load(f)
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    url = f"{base}/{patch}/game/data/characters/{cid.lower()}/{cid.lower()}.bin.json"
    for attempt in range(4):
        try:
            r = s.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            doc = r.json()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(doc, f)
            return 200, doc
        if r.status_code in (403, 404):
            return r.status_code, None
        time.sleep(3 * (attempt + 1))
    return 0, None


def store(con, patch, cid, status, parsed):
    con.execute("DELETE FROM ability_values WHERE patch = ? AND champion_id = ?", (patch, cid))
    con.execute("DELETE FROM ability_calcs WHERE patch = ? AND champion_id = ?", (patch, cid))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if parsed is None:
        con.execute("INSERT OR REPLACE INTO cd_files VALUES (?,?,?,?,?,?,?)",
                    (patch, cid, status, None, None, None, now))
        return
    values, calcs, offset, cd_match, cd_checked, _, base_stats = parsed
    con.execute("DELETE FROM cd_champion_stats WHERE patch = ? AND champion_id = ?", (patch, cid))
    con.executemany("INSERT OR REPLACE INTO cd_champion_stats VALUES (?,?,?,?)",
                    [(patch, cid, st, v) for st, v in base_stats])
    con.executemany("INSERT OR REPLACE INTO ability_values VALUES (?,?,?,?,?,?,?)",
                    [(patch, cid) + v for v in values])
    con.executemany("INSERT OR REPLACE INTO ability_calcs VALUES (?,?,?,?,?,?,?)",
                    [(patch, cid) + c for c in calcs])
    con.execute("INSERT OR REPLACE INTO cd_files VALUES (?,?,?,?,?,?,?)",
                (patch, cid, status, offset, cd_match, cd_checked, now))


def pkey(p):
    return tuple(int(x) for x in p.split(".")[:2])


def report(con):
    rows = con.execute("SELECT patch, COUNT(*), SUM(status = 200), SUM(cd_match), SUM(cd_checked) "
                       "FROM cd_files GROUP BY patch").fetchall()
    print("\npatch   champions  fetched  cooldown self-check")
    for p, n, ok, m, ch in sorted(rows, key=lambda r: pkey(r[0])):
        pct = f"{(m or 0) / ch:.0%} of {ch} abilities match Data Dragon" if ch else "-"
        print(f"{p:>6} {n:>10} {ok or 0:>8}  {pct}")
    cats = con.execute("SELECT category, COUNT(*), SUM(complex) FROM ability_calcs "
                       "GROUP BY category ORDER BY 2 DESC").fetchall()
    print("\nformulas by type (all patches):  " +
          ", ".join(f"{c} {n:,} ({x or 0:,} partly captured)" for c, n, x in cats))
    agree = con.execute("""
        SELECT c.stat, COUNT(*), SUM(ABS(c.value - s.value) <= 0.01 * MAX(ABS(s.value), 1)),
               SUM(s.value = 0 AND c.value != 0)
        FROM cd_champion_stats c JOIN patches p ON p.patch = c.patch
        JOIN champion_stats s ON s.version = p.version AND s.champion_id = c.champion_id
                             AND s.stat = c.stat
        GROUP BY c.stat""").fetchall()
    if agree:
        print("\nbase stats, game files vs Data Dragon:  " + ", ".join(
            f"{st} {m / n:.0%} agree" + (f" ({z} zero in Data Dragon)" if z else "")
            for st, n, m, z in agree))
    bad = con.execute("SELECT COUNT(*) FROM cd_files WHERE status = 200 AND cd_checked > 0 "
                      "AND cd_match * 2 < cd_checked").fetchone()[0]
    if bad:
        print(f"\n!!! {bad} champion files failed the cooldown self-check - run with "
              f"--inspect on one of them and send me the output.")


def inspect(con, args):
    patch = args.patch or max((r[0] for r in con.execute("SELECT patch FROM patches")), key=pkey)
    version = con.execute("SELECT version FROM patches WHERE patch = ?", (patch,)).fetchone()
    if not version:
        sys.exit(f"Patch {patch} is not in the database.")
    dd = load_dd(con, version[0])
    cid = next((c for c in dd if c.lower() == args.inspect.lower()), None)
    if not cid:
        sys.exit(f"Champion {args.inspect!r} not found (use the Data Dragon id, e.g. MonkeyKing).")
    status, doc = download(args.base_url, patch, cid, args.cache_dir)
    if doc is None:
        sys.exit(f"Download failed (HTTP {status}).")
    values, calcs, offset, m, ch, slots, base_stats = parse_champion(doc, cid, dd[cid])
    print(f"{cid} patch {patch}: abilities found {slots}, rank offset {offset}, "
          f"cooldown self-check {m}/{ch} match Data Dragon")
    print(f"base stats from game files: {dict(base_stats)}")
    by = {}
    for slot, kind, name, rank, v in values:
        by.setdefault((slot, kind, name), []).append(round(v, 4))
    for slot in ["P"] + SLOTS:
        print(f"\n[{slot}] {(dd[cid].get(slot) or {}).get('id')}")
        for (s, kind, name), vals in sorted(by.items()):
            if s == slot and kind == "data":
                print(f"   data   {name:<32} {vals}")
        for s, name, cat, cx, lv in calcs:
            if s != slot:
                continue
            base = by.get((slot, "calc_base", name), [])
            ratios = {k[2].split("|", 1)[1]: v for k, v in by.items()
                      if k[0] == slot and k[1] == "calc_ratio" and k[2].startswith(name + "|")}
            flags = (" PARTIAL" if cx else "") + (" LEVEL" if lv else "")
            print(f"   calc   {name:<32} [{cat}] base {base} ratios {ratios}{flags}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ability numbers per patch from CommunityDragon.")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--cache-dir", default="cdragon_cache")
    ap.add_argument("--patches", type=int, default=None, help="only the N most recent patches")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reparse", action="store_true", help="re-parse cached files, no downloads")
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    ap.add_argument("--inspect", default=None, help="print the parse for one champion")
    ap.add_argument("--patch", default=None, help="patch for --inspect (default: newest)")
    ap.add_argument("--base-url", default=CDRAGON, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    if args.inspect:
        inspect(con, args)
        return

    patches = sorted(con.execute("SELECT patch, version FROM patches").fetchall(),
                     key=lambda r: pkey(r[0]), reverse=True)
    if args.patches:
        patches = patches[:args.patches]
    if not patches:
        sys.exit("No balance data yet - run fetch_balance_data.py (or the collector) first.")
    done = {(p, c) for p, c in con.execute("SELECT patch, champion_id FROM cd_files WHERE status = 200")}

    jobs = []
    for patch, version in patches:
        dd = load_dd(con, version)
        for cid in sorted(dd):
            if args.reparse:
                if os.path.exists(cache_path(args.cache_dir, patch, cid)):
                    jobs.append((patch, cid, dd[cid]))
            elif args.refresh or (patch, cid) not in done:
                jobs.append((patch, cid, dd[cid]))
    print(f"{len(jobs):,} champion files to {'re-parse' if args.reparse else 'fetch'} "
          f"across {len(patches)} patches")

    def work(job):
        patch, cid, dd = job
        status, doc = download(args.base_url, patch, cid, args.cache_dir,
                               use_cache=not args.refresh)
        parsed = None
        if doc is not None:
            try:
                parsed = parse_champion(doc, cid, dd)
            except Exception as e:  # keep going; report at the end
                print(f"  parse error {patch} {cid}: {e}")
        return patch, cid, status, parsed

    t0, n_done = time.time(), 0
    with ThreadPoolExecutor(max_workers=1 if args.reparse else args.threads) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            patch, cid, status, parsed = fut.result()
            store(con, patch, cid, status, parsed)
            n_done += 1
            if n_done % 50 == 0 or n_done == len(jobs):
                con.commit()
                rate = n_done / max(time.time() - t0, 1)
                print(f"  {n_done:,}/{len(jobs):,} files | {rate:.1f}/s | "
                      f"~{(len(jobs) - n_done) / max(rate, 0.01) / 60:.0f} min left")
    con.commit()
    report(con)
    con.close()


if __name__ == "__main__":
    main()
