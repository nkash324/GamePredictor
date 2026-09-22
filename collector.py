#!/usr/bin/env python3
"""
collector.py  -  continuous data collector for the LoL prediction project.

Start it and leave it running. It:
  * keeps the balance data (Data Dragon) up to date, fetching new patches itself
  * collects ranked matches from several regions at the same time (one worker per
    region; Riot's rate limits are counted separately per region)
  * backfills history breadth-first: every patch gets a first batch of games,
    then every patch gets more, and so on, so usable data arrives early
  * stores the full raw match (and timeline for a sample of matches), compressed,
    so nothing ever has to be downloaded twice
  * extracts objectives, rune pages, and timeline stats (gold/xp/cs at 5-25 min,
    item purchase order, skill order) into tables for the model
  * pauses when the API key expires and resumes when you paste a new one into
    riot_key.txt
  * can be stopped (Ctrl+C) and restarted at any time without losing anything

Commands:
  python collector.py            start collecting
  python collector.py status     show progress per region and patch, and time left

Settings live in collector_config.json (created on first run).
Needs: fetch_balance_data.py and fetch_match_data.py in the same folder.
"""

import gzip
import hashlib
import json
import logging
import logging.handlers
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import requests

import fetch_balance_data as fbd
import fetch_match_data as fmd
from fetch_match_data import (REGIONS, PatchDater, RiotAPI, StopRequested, bucket_tiers,
                              migrate, patch_of, pkey, rank_bucket, seed_players, store_match)

CONFIG_FILE = "collector_config.json"
DEFAULT_CONFIG = {
    "regions": ["na", "euw", "kr"],
    # rank mix for the live patch: players' ranks are read from the ladder during
    # the patch, so these games get exact rank labels
    "rank_mix_current": {"MASTER+": 0.25, "DIAMOND": 0.25, "EMERALD": 0.25,
                         "PLATINUM": 0.15, "GOLD": 0.10},
    # rank mix for past patches: only rank-stable high-elo players; labelled as an
    # estimate because Riot has no historical ranks
    "rank_mix_backfill": {"MASTER+": 0.5, "DIAMOND": 0.5},
    "ladder_pages_per_division": 5,
    "patches_back": 25,
    "target_per_patch": 40000,
    "stages": [5000, 20000, 40000],
    "timeline_fraction": 0.25,
    "games_per_player_per_stage": 6,
    "min_game_seconds": 300,
    "reseed_players_days": 3,
    "new_patch_check_hours": 3,
    "db": "lol_data.db",
    "raw_dir": "raw_matches",
    "key_file": "riot_key.txt",
    "log_file": "collector.log",
}
TIMELINE_MINUTES = (5, 10, 15, 20, 25)

EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS team_objectives (
    match_id   TEXT NOT NULL,
    team_id    INTEGER NOT NULL,
    objective  TEXT NOT NULL,       -- baron, dragon, horde, riftHerald, tower, inhibitor, atakhan, champion
    first      INTEGER,
    kills      INTEGER,
    PRIMARY KEY (match_id, team_id, objective)
);
CREATE TABLE IF NOT EXISTS participant_runes (
    match_id        TEXT NOT NULL,
    participant_id  INTEGER NOT NULL,
    primary_style   INTEGER,
    sub_style       INTEGER,
    perks           TEXT,           -- all 6 rune ids, comma separated (keystone first)
    shards          TEXT,           -- offense,flex,defense stat shards
    PRIMARY KEY (match_id, participant_id)
);
CREATE TABLE IF NOT EXISTS timeline_frames (
    match_id        TEXT NOT NULL,
    participant_id  INTEGER NOT NULL,
    minute          INTEGER NOT NULL,
    gold            INTEGER,
    xp              INTEGER,
    level           INTEGER,
    cs              INTEGER,
    dmg_to_champs   INTEGER,
    PRIMARY KEY (match_id, participant_id, minute)
);
CREATE TABLE IF NOT EXISTS item_events (
    match_id        TEXT NOT NULL,
    participant_id  INTEGER,
    ts_ms           INTEGER,
    event           TEXT,           -- ITEM_PURCHASED, ITEM_SOLD, ITEM_UNDO
    item_id         INTEGER,
    after_id        INTEGER         -- for ITEM_UNDO
);
CREATE INDEX IF NOT EXISTS ix_item_events_match ON item_events(match_id);
CREATE TABLE IF NOT EXISTS skill_order (
    match_id        TEXT NOT NULL,
    participant_id  INTEGER NOT NULL,
    skills          TEXT,           -- e.g. QEWQQRQEQE...
    PRIMARY KEY (match_id, participant_id)
);
CREATE INDEX IF NOT EXISTS ix_match_platform_patch ON matches(platform, patch);
"""

RAW_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw (
    match_id      TEXT PRIMARY KEY,
    patch         TEXT,
    collected_at  INTEGER,          -- epoch seconds
    match_gz      BLOB,             -- gzip-compressed JSON from match-v5
    timeline_gz   BLOB              -- gzip-compressed timeline JSON, or NULL
);
CREATE INDEX IF NOT EXISTS ix_raw_time ON raw(collected_at);
"""

log = logging.getLogger("collector")


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg.update(json.load(f))
    else:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    bad = [r for r in cfg["regions"] if r not in REGIONS]
    if bad:
        sys.exit(f"Unknown region(s) in {CONFIG_FILE}: {bad}. Options: {sorted(REGIONS)}")
    routes = [REGIONS[r][1] for r in cfg["regions"]]
    if len(set(routes)) < len(routes):
        print("Note: some regions share a routing group (e.g. na + br are both "
              "'americas'), so they share one rate limit and won't add speed.")
    cfg["stages"] = sorted(set(min(s, cfg["target_per_patch"]) for s in cfg["stages"])
                           | {cfg["target_per_patch"]})
    valid = {"MASTER+", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"}
    for k in ("rank_mix_current", "rank_mix_backfill"):
        mix = {b.upper(): float(v) for b, v in cfg[k].items() if float(v) > 0}
        if not mix or set(mix) - valid:
            sys.exit(f"{k} in {CONFIG_FILE} must use: {sorted(valid)}")
        total = sum(mix.values())
        cfg[k] = {b: v / total for b, v in mix.items()}  # shares always add up to 1
    buckets = set(cfg["rank_mix_current"]) | set(cfg["rank_mix_backfill"])
    order = ["MASTER+", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]
    cfg["buckets"] = [b for b in order if b in buckets]
    cfg["tiers"] = [t for b in cfg["buckets"] for t in bucket_tiers(b)]
    return cfg


def bucket_targets(cfg, stage, is_open, n_regions):
    mix = cfg["rank_mix_current"] if is_open else cfg["rank_mix_backfill"]
    return {b: math.ceil(stage * share / n_regions) for b, share in mix.items()}


def setup_logging(path):
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)


def ensure_key(cfg):
    path = cfg["key_file"]
    if os.path.exists(path) and open(path, encoding="utf-8").read().strip():
        return
    env = os.environ.get("RIOT_API_KEY")
    if env:
        key = env.strip()
    else:
        print("No API key found. Get one at https://developer.riotgames.com")
        key = input("Paste your Riot API key: ").strip()
    if not key.startswith("RGAPI-"):
        sys.exit("That doesn't look like a Riot key (they start with RGAPI-).")
    with open(path, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    print(f"Key saved to {os.path.abspath(path)} (keep this file private).")


def connect(db):
    con = sqlite3.connect(db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def keep_awake():
    """Stop the computer from sleeping while the collector runs."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log.info("Windows sleep is blocked while the collector runs (screen may still turn off).")
    elif sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        log.info("macOS sleep is blocked while the collector runs.")


def recent_patches(con, n):
    rows = [r[0] for r in con.execute("SELECT DISTINCT patch FROM patches")]
    return sorted(rows, key=pkey, reverse=True)[:n]


def get_state(con, key):
    r = con.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    return r[0] if r else None


def set_state(con, key, value):
    con.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, str(value)))
    con.commit()


# --------------------------------------------------------------------------- #
# Extra parsing (objectives, runes, timelines)
# --------------------------------------------------------------------------- #
def store_extras(con, m):
    mid = m["metadata"]["matchId"]
    info = m.get("info", {})
    obj = []
    for t in info.get("teams", []):
        for name, o in (t.get("objectives") or {}).items():
            obj.append((mid, t.get("teamId"), name, int(bool(o.get("first"))), o.get("kills")))
    con.executemany("INSERT OR REPLACE INTO team_objectives VALUES (?,?,?,?,?)", obj)

    runes = []
    for p in info.get("participants", []):
        perks = p.get("perks") or {}
        styles = perks.get("styles") or []
        prim = next((s for s in styles if s.get("description") == "primaryStyle"),
                    styles[0] if styles else {})
        sub = next((s for s in styles if s.get("description") == "subStyle"),
                   styles[1] if len(styles) > 1 else {})
        ids = [sel.get("perk") for s in (prim, sub) for sel in (s.get("selections") or [])]
        sp = perks.get("statPerks") or {}
        runes.append((mid, p.get("participantId"), prim.get("style"), sub.get("style"),
                      ",".join(str(i) for i in ids),
                      f"{sp.get('offense')},{sp.get('flex')},{sp.get('defense')}"))
    con.executemany("INSERT OR REPLACE INTO participant_runes VALUES (?,?,?,?,?,?)", runes)


def store_timeline(con, mid, tl):
    info = tl.get("info", {})
    frames = info.get("frames") or []
    interval = info.get("frameInterval") or 60000
    rows = []
    for minute in TIMELINE_MINUTES:
        idx = round(minute * 60000 / interval)
        if idx >= len(frames):
            break
        for pid, f in (frames[idx].get("participantFrames") or {}).items():
            rows.append((mid, int(pid), minute, f.get("totalGold"), f.get("xp"), f.get("level"),
                         (f.get("minionsKilled") or 0) + (f.get("jungleMinionsKilled") or 0),
                         (f.get("damageStats") or {}).get("totalDamageDoneToChampions")))
    con.executemany("INSERT OR REPLACE INTO timeline_frames VALUES (?,?,?,?,?,?,?,?)", rows)

    events, skills = [], {}
    for fr in frames:
        for e in fr.get("events") or []:
            t = e.get("type")
            if t in ("ITEM_PURCHASED", "ITEM_SOLD"):
                events.append((mid, e.get("participantId"), e.get("timestamp"), t,
                               e.get("itemId"), None))
            elif t == "ITEM_UNDO":
                events.append((mid, e.get("participantId"), e.get("timestamp"), t,
                               e.get("beforeId"), e.get("afterId")))
            elif t == "SKILL_LEVEL_UP" and e.get("levelUpType", "NORMAL") == "NORMAL":
                slot = e.get("skillSlot")
                if slot in (1, 2, 3, 4):
                    skills.setdefault(e.get("participantId"), []).append("QWER"[slot - 1])
    con.execute("DELETE FROM item_events WHERE match_id = ?", (mid,))
    con.executemany("INSERT INTO item_events VALUES (?,?,?,?,?,?)", events)
    con.executemany("INSERT OR REPLACE INTO skill_order VALUES (?,?,?)",
                    [(mid, pid, "".join(s)) for pid, s in skills.items()])


def usable_match(m, min_seconds):
    """Same filters as store_match: ranked solo, not a remake / early surrender."""
    info = m.get("info", {})
    parts = info.get("participants", [])
    return (info.get("queueId") == fmd.QUEUE_ID
            and (info.get("gameDuration") or 0) >= min_seconds
            and len(parts) == 10
            and not any(p.get("gameEndedInEarlySurrender") for p in parts))


def wants_timeline(match_id, fraction):
    """Deterministic ~fraction sample, so restarts pick the same matches."""
    h = int(hashlib.md5(match_id.encode()).hexdigest()[:8], 16)
    return (h % 10000) < fraction * 10000


def gz(obj):
    return gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


# --------------------------------------------------------------------------- #
# One worker per region
# --------------------------------------------------------------------------- #
class RegionWorker(threading.Thread):
    def __init__(self, region, cfg, stop_event, host_template):
        super().__init__(name=region, daemon=True)
        self.region, self.cfg, self.stop = region, cfg, stop_event
        self.platform, self.regional = REGIONS[region]
        self.tag = f"[{region.upper()}]"
        kw = dict(key_file=cfg["key_file"],
                  host_template=host_template, status_route=self.platform,
                  log=lambda msg: log.info(f"{self.tag} {msg.strip()}"), stop_event=stop_event)
        self.api_platform = RiotAPI(**dict(kw, log=lambda msg: None if "rate limits" in msg
                                           else log.info(f"{self.tag} {msg.strip()}")))
        self.api_regional = RiotAPI(**kw)
        self.n_regions = len(cfg["regions"])
        self.recent = deque()          # timestamps of stored matches (rate)
        self.exhausted = {}            # (patch, stage, bucket) -> time we ran out of players
        self.no_window = {}            # patch -> time we failed to date it

    def say(self, msg):
        log.info(f"{self.tag} {msg}")

    def rate_per_hour(self):
        now = time.time()
        while self.recent and now - self.recent[0] > 3600:
            self.recent.popleft()
        span = min(3600, now - self.recent[0]) if self.recent else 0
        return len(self.recent) / span * 3600 if span > 60 else 0.0

    # -- main loop ---------------------------------------------------------- #
    def run(self):
        while not self.stop.is_set():
            try:
                self.loop()
            except StopRequested:
                break
            except Exception:
                log.exception(f"{self.tag} unexpected error; restarting worker in 60s")
                if self.stop.wait(60):
                    break
        self.say("stopped")

    def loop(self):
        self.con = connect(self.cfg["db"])
        os.makedirs(self.cfg["raw_dir"], exist_ok=True)
        self.raw = connect(os.path.join(self.cfg["raw_dir"], f"matches_{self.platform}.db"))
        self.raw.executescript(RAW_SCHEMA)

        self.seed_if_needed()
        self.dater = PatchDater(self.api_regional, self.con, self.platform, self.regional)
        self.have = {r[0] for r in self.con.execute(
            "SELECT match_id FROM matches WHERE platform = ?", (self.platform,))}

        while not self.stop.is_set():
            if self.seed_if_needed():
                self.dater = PatchDater(self.api_regional, self.con, self.platform,
                                        self.regional)
            patches = recent_patches(self.con, self.cfg["patches_back"])
            if not patches:
                self.say("no balance data yet, waiting")
                self.stop.wait(300)
                continue
            worked = self.one_pass(patches)
            if not worked:
                self.say("nothing to collect right now (targets reached, or no more players "
                         "for the remaining patches) - checking again in 30 min")
                if self.stop.wait(1800):
                    return

    def seed_if_needed(self):
        key = f"seeded:{self.platform}"
        last = get_state(self.con, key)
        n = self.con.execute("SELECT COUNT(*) FROM players WHERE platform = ?",
                             (self.platform,)).fetchone()[0]
        if n and last and time.time() - float(last) < self.cfg["reseed_players_days"] * 86400:
            return False
        self.say(f"loading ladder players: {', '.join(self.cfg['tiers'])}")
        added = seed_players(self.api_platform, self.con, self.platform, self.cfg["tiers"],
                             self.cfg["ladder_pages_per_division"])
        set_state(self.con, key, time.time())
        self.say(f"{added} new players added")
        return True

    def count(self, patch, bucket):
        return self.con.execute("SELECT COUNT(*) FROM matches WHERE platform = ? AND patch = ? "
                                "AND rank_bucket = ?", (self.platform, patch, bucket)).fetchone()[0]

    def window(self, patch):
        w = self.dater.window(patch)
        if w is None:
            return None
        is_open = self.con.execute("SELECT is_open FROM patch_windows WHERE platform = ? AND "
                                   "patch = ?", (self.platform, patch)).fetchone()
        return w[0], w[1], bool(is_open and is_open[0])

    def one_pass(self, patches):
        """Breadth-first: every patch (and every rank group in its mix) to stage 1,
        then stage 2, ... Returns False when there is nothing left to do."""
        now = time.time()
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
        for s_idx, stage in enumerate(self.cfg["stages"]):
            for patch in patches:
                if self.stop.is_set():
                    return True
                if now - self.no_window.get(patch, 0) < 6 * 3600:
                    continue
                w = self.window(patch)
                if w is None:
                    self.say(f"{patch}: no games found for this patch (not live yet or "
                             f"too old) - retry later")
                    self.no_window[patch] = time.time()
                    continue
                start, end, is_open = w
                targets = bucket_targets(self.cfg, stage, is_open, self.n_regions)
                for bucket, target in targets.items():
                    if now - self.exhausted.get((patch, s_idx, bucket), 0) < 12 * 3600:
                        continue
                    have = self.count(patch, bucket)
                    if have >= target:
                        continue
                    mode = "live patch, exact ranks" if is_open else "past patch, estimated ranks"
                    self.say(f"{patch} {bucket}: stage {s_idx + 1}/{len(self.cfg['stages'])}, "
                             f"{have}/{target} matches ({mode}; live {fmt(start)} to {fmt(end)})")
                    if self.collect(patch, start, end, bucket, target, s_idx, is_open):
                        self.exhausted[(patch, s_idx, bucket)] = time.time()
                    return True  # re-evaluate from the top (new patches get priority)
        return False

    def collect(self, patch, start, end, bucket, target, s_idx, is_open):
        """Collect `bucket`-rank matches of `patch` in [start, end).
        Returns True if we ran out of players for this bucket."""
        cfg, con = self.cfg, self.con
        crawl_key = f"{self.platform}:{patch}:s{s_idx}"
        per_player = min(100, cfg["games_per_player_per_stage"] * (s_idx + 1))
        tiers = bucket_tiers(bucket)
        sql = f"""
            SELECT p.puuid, p.tier, p.division, p.lp, p.seeded_at, c.crawled_at FROM players p
            LEFT JOIN crawl_log c ON c.puuid = p.puuid AND c.crawl_key = ?
            WHERE p.platform = ? AND p.tier IN ({",".join("?" * len(tiers))})"""
        args = [crawl_key, self.platform, *tiers]
        if is_open:  # exact labels: only players whose rank was read recently
            sql += " AND p.seeded_at >= ?"
            args.append(time.time() - (cfg["reseed_players_days"] + 1) * 86400)
        fresh = [r[:5] for r in con.execute(sql, args) if r[5] is None]
        random.shuffle(fresh)
        if not fresh:
            return True
        rank_source = "ladder" if is_open else "estimate"
        stored = self.count(patch, bucket)
        last_log = time.time()
        params = {"queue": fmd.QUEUE_ID, "type": "ranked", "startTime": int(start),
                  "endTime": int(end), "count": per_player}

        for puuid, tier, division, lp, seeded_at in fresh:
            rank_date = (datetime.fromtimestamp(seeded_at, timezone.utc).strftime("%Y-%m-%d")
                         if seeded_at else None)
            if self.stop.is_set():
                return False
            ids = self.api_regional.get(self.regional,
                                        f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
                                        params=params) or []
            for mid in ids:
                if mid in self.have:
                    continue
                self.have.add(mid)
                m = self.api_regional.get(self.regional, f"/lol/match/v5/matches/{mid}")
                if not m or patch_of(m.get("info", {}).get("gameVersion")) != patch:
                    continue
                if not usable_match(m, cfg["min_game_seconds"]):
                    continue
                tl = None
                if wants_timeline(mid, cfg["timeline_fraction"]):
                    tl = self.api_regional.get(self.regional,
                                               f"/lol/match/v5/matches/{mid}/timeline")
                # all downloads done: write in one short transaction so other
                # regions are never blocked while this one waits on the network
                store_match(con, m, self.platform, tier, cfg["min_game_seconds"])
                con.execute("UPDATE matches SET rank_bucket = ?, seed_division = ?, seed_lp = ?, "
                            "rank_source = ?, rank_date = ? WHERE match_id = ?",
                            (bucket, division, lp, rank_source, rank_date, mid))
                store_extras(con, m)
                if tl:
                    store_timeline(con, mid, tl)
                con.commit()
                self.raw.execute("INSERT OR REPLACE INTO raw VALUES (?,?,?,?,?)",
                                 (mid, patch, int(time.time()), gz(m), gz(tl) if tl else None))
                self.raw.commit()
                stored += 1
                self.recent.append(time.time())
                if stored >= target:
                    break
            con.execute("INSERT OR REPLACE INTO crawl_log VALUES (?,?,?)",
                        (puuid, crawl_key, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
            if time.time() - last_log > 120 or stored >= target:
                self.say(f"{patch} {bucket}: {stored}/{target} matches | "
                         f"{self.rate_per_hour():,.0f} matches/hr")
                last_log = time.time()
            if stored >= target:
                return False
        self.say(f"{patch} {bucket}: ran out of players at {stored}/{target} for this stage "
                 f"(raise ladder_pages_per_division in the config for more)")
        return True


# --------------------------------------------------------------------------- #
# Balance data upkeep
# --------------------------------------------------------------------------- #
def update_balance(cfg, n_patches):
    try:
        fbd.main(["--patches", str(n_patches), "--db", cfg["db"], "--raw-dir", "raw"])
    except SystemExit:
        pass
    except Exception as e:
        log.warning(f"balance data update failed ({e}); will retry later")


def latest_ddragon_patch():
    try:
        versions = requests.get(f"{fbd.DDRAGON}/api/versions.json", timeout=30).json()
        return fbd.pick_patch_versions(versions, 1)[0]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Status report
# --------------------------------------------------------------------------- #
def status(cfg):
    if not os.path.exists(cfg["db"]):
        sys.exit("No database yet - start the collector first.")
    con = sqlite3.connect(f"file:{cfg['db']}?mode=ro", uri=True, timeout=30)
    patches = recent_patches(con, cfg["patches_back"])
    n_reg = len(cfg["regions"])
    plats = [REGIONS[r][0] for r in cfg["regions"]]
    has_bucket = "rank_bucket" in {r[1] for r in con.execute("PRAGMA table_info(matches)")}
    bcol = "rank_bucket" if has_bucket else "NULL"
    counts = {}
    for p, pa, b, n in con.execute(f"SELECT platform, patch, {bcol}, COUNT(*) FROM matches "
                                   f"GROUP BY platform, patch, {bcol}"):
        counts[(p, pa, b)] = n
    tl = dict(con.execute("SELECT m.patch, COUNT(*) FROM timeline_frames t JOIN matches m "
                          "USING (match_id) WHERE t.minute = 5 AND t.participant_id = 1 "
                          "GROUP BY m.patch"))
    windows = {}
    for p, pa, o in con.execute("SELECT platform, patch, is_open FROM patch_windows"):
        windows.setdefault(pa, set()).add(bool(o))
    buckets = cfg["buckets"]
    short = {"MASTER+": "M+", "DIAMOND": "D", "EMERALD": "E", "PLATINUM": "P", "GOLD": "G",
             "SILVER": "S", "BRONZE": "B", "IRON": "I"}

    print(f"\nTarget: {cfg['target_per_patch']:,} matches per patch across "
          f"{', '.join(plats)}, last {len(patches)} patches")
    print("Rank mix  live patch: " + ", ".join(f"{b} {v:.0%}" for b, v in
          cfg["rank_mix_current"].items()) + "  |  past patches: " +
          ", ".join(f"{b} {v:.0%}" for b, v in cfg["rank_mix_backfill"].items()) + "\n")
    head = (f"{'patch':>7} {'type':<9}" + "".join(f"{p:>9}" for p in plats) +
            f"{'total':>9}{'tl':>8}  " + " ".join(f"{short[b]:>11}" for b in buckets) +
            "  progress")
    print(head)
    print("-" * len(head))
    total_have = total_need = 0
    for pa in patches:
        live = True in windows.get(pa, set())
        kind = "live" if live else ("past" if pa in windows else "not live")
        targets = bucket_targets(cfg, cfg["target_per_patch"], live, n_reg)
        have = sum(min(counts.get((p, pa, b), 0), t) for p in plats for b, t in targets.items())
        need = sum(targets.values()) * n_reg
        if kind != "not live":
            total_have += have
            total_need += need
        pct = have / need if need else 0
        bar = "#" * int(pct * 10) + "." * (10 - int(pct * 10))
        per_plat = [sum(n for (p, x, _), n in counts.items() if p == pl and x == pa)
                    for pl in plats]
        cells = []
        for b in buckets:
            got = sum(counts.get((p, pa, b), 0) for p in plats)
            t = targets.get(b, 0) * n_reg
            cells.append(f"{got:>5}/{t:<5}" if t else f"{got:>5}/-    ")
        print(f"{pa:>7} {kind:<9}" + "".join(f"{n:>9,}" for n in per_plat) +
              f"{sum(per_plat):>9,}{tl.get(pa, 0):>8,}  " + " ".join(cells) +
              f"  {bar} {pct:4.0%}")
    print("tl = matches with timelines;  rank columns = collected/target over all regions")

    need = total_need - total_have
    rate = 0
    for p in plats:
        path = os.path.join(cfg["raw_dir"], f"matches_{p}.db")
        if os.path.exists(path):
            rc = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
            rate += rc.execute("SELECT COUNT(*) FROM raw WHERE collected_at > ?",
                               (int(time.time()) - 3600,)).fetchone()[0]
            rc.close()
    print(f"\nCollected in the last hour: {rate:,} matches")
    if need <= 0:
        print("All patches are at target.")
    elif rate:
        hrs = need / rate
        print(f"Remaining: {need:,} matches  ->  about {hrs:,.1f} hours ({hrs / 24:.1f} days) "
              f"at the current speed")
    else:
        print(f"Remaining: {need:,} matches (collector not running, or just started)")
    size = sum(os.path.getsize(os.path.join(d, f)) for d in (cfg["raw_dir"],)
               if os.path.isdir(d) for f in os.listdir(d))
    print(f"Disk used: database {os.path.getsize(cfg['db']) / 1e9:.2f} GB, "
          f"raw matches {size / 1e9:.2f} GB\n")


# --------------------------------------------------------------------------- #
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    host_template = "https://{route}.api.riotgames.com"
    if "--host-template" in argv:  # testing only
        i = argv.index("--host-template")
        host_template = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if argv[:1] == ["--ddragon"]:  # testing only
        fbd.DDRAGON = argv[1]
        argv = argv[2:]

    cfg = load_config()
    if argv[:1] == ["status"]:
        status(cfg)
        return
    if argv:
        sys.exit(f"Unknown command {argv[0]!r}. Use:  python collector.py  or  "
                 f"python collector.py status")

    setup_logging(cfg["log_file"])
    ensure_key(cfg)
    con = connect(cfg["db"])
    con.executescript(fbd.SCHEMA)
    con.executescript(fmd.SCHEMA)
    migrate(con)
    con.executescript(EXTRA_SCHEMA)

    log.info("Rank mix, live patch: " + ", ".join(f"{b} {v:.0%}" for b, v in
             cfg["rank_mix_current"].items()) + " | past patches: " +
             ", ".join(f"{b} {v:.0%}" for b, v in cfg["rank_mix_backfill"].items()))
    log.info(f"Collector starting: regions {', '.join(cfg['regions'])}, "
             f"{cfg['target_per_patch']:,} matches/patch over the last {cfg['patches_back']} "
             f"patches, stages {cfg['stages']}, timelines for {cfg['timeline_fraction']:.0%}")
    have_patches = len(recent_patches(con, cfg["patches_back"]))
    if have_patches < cfg["patches_back"]:
        log.info("Fetching balance data first ...")
        update_balance(cfg, cfg["patches_back"])
    keep_awake()

    stop = threading.Event()
    workers = [RegionWorker(r, cfg, stop, host_template) for r in cfg["regions"]]
    for w in workers:
        w.start()

    last_patch_check = time.time()
    last_summary = time.time()
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(1)
            if time.time() - last_patch_check > cfg["new_patch_check_hours"] * 3600:
                last_patch_check = time.time()
                newest = latest_ddragon_patch()
                known = {r[0] for r in con.execute("SELECT version FROM patches")}
                if newest and newest not in known:
                    log.info(f"New patch {newest} found - fetching its balance data")
                    update_balance(cfg, 3)
            if time.time() - last_summary > 900:
                last_summary = time.time()
                total = sum(w.rate_per_hour() for w in workers)
                log.info(f"== overall speed {total:,.0f} matches/hr "
                         f"({', '.join(f'{w.region} {w.rate_per_hour():,.0f}' for w in workers)})"
                         f" - run 'python collector.py status' in another window for details")
    except KeyboardInterrupt:
        log.info("Stopping (finishing the current request) ...")
        stop.set()
        for w in workers:
            w.join(timeout=60)
    log.info("Collector stopped. Everything collected so far is saved.")


if __name__ == "__main__":
    main()
