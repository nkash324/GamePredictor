#!/usr/bin/env python3
"""
fetch_match_data.py  -  Part 2 of the LoL prediction project.

Collects ranked solo/duo matches from the Riot API and stores them in the same
SQLite file as the balance data (lol_data.db). From the raw matches it builds
views with the outcome statistics the model will learn to predict:

  champion_outcomes       win / pick / ban rate + avg stats per champion per patch
  champion_role_outcomes  same, split by position (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
  matchups_lane           champion vs its direct lane opponent, per patch
  matchups_any            champion vs every champion on the enemy team, per patch

How the crawl works:
  1. take players from the ranked ladder for the tiers you choose
  2. get each player's ranked matches in a time window (skipping ones already stored)
  3. download each match and store all 10 participants + bans
The crawl is resumable: stop it any time (Ctrl+C) and run it again later.

Two modes:
  recent    (default) matches from the last --days days
  backfill  matches from specific past patches. The script works out when each
            patch was live by sampling real games (Riot doesn't publish patch
            dates), saves those dates, then only collects games in that window.
            --max-matches becomes a per-patch target, so re-running tops up.

Setup (the key is read from an environment variable, never saved to a file):
  pip install requests
  export RIOT_API_KEY=RGAPI-xxxxxxxx          # macOS / Linux
  python fetch_match_data.py --max-matches 500
  python fetch_match_data.py --region euw --tiers CHALLENGER,GRANDMASTER,MASTER,DIAMOND
  python fetch_match_data.py --backfill-last 10 --max-matches 3000   # patches from balance data
  python fetch_match_data.py --backfill 15.10,15.11 --max-matches 3000

Development keys allow 100 requests per 2 minutes, i.e. roughly 2,500-3,000
matches per hour. Large datasets need a personal/production key from Riot.
"""

import argparse
import collections
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

# platform routing (league / summoner endpoints) -> regional routing (match endpoints)
REGIONS = {
    "na": ("na1", "americas"), "br": ("br1", "americas"), "lan": ("la1", "americas"),
    "las": ("la2", "americas"), "euw": ("euw1", "europe"), "eune": ("eun1", "europe"),
    "tr": ("tr1", "europe"), "ru": ("ru", "europe"), "me": ("me1", "europe"),
    "kr": ("kr", "asia"), "jp": ("jp1", "asia"), "oce": ("oc1", "sea"),
    "sg": ("sg2", "sea"), "tw": ("tw2", "sea"), "vn": ("vn2", "sea"),
}
APEX = {"CHALLENGER": "challengerleagues", "GRANDMASTER": "grandmasterleagues",
        "MASTER": "masterleagues"}
DIVISIONS = ["I", "II", "III", "IV"]
QUEUE_NAME, QUEUE_ID = "RANKED_SOLO_5x5", 420
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid         TEXT PRIMARY KEY,
    platform      TEXT,
    tier          TEXT,
    division      TEXT,
    lp            INTEGER,
    last_crawled  TEXT,
    seeded_at     REAL             -- when tier/division/lp were last read from the ladder
);
CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,
    platform      TEXT,
    patch         TEXT,            -- e.g. 14.23 (joins to patches.patch)
    game_version  TEXT,
    game_start    INTEGER,         -- epoch ms
    duration_s    INTEGER,
    queue_id      INTEGER,
    source_tier   TEXT,            -- tier of the ladder player we found it through
    winning_team  INTEGER,
    rank_bucket   TEXT,            -- MASTER+, DIAMOND, EMERALD, PLATINUM, GOLD ...
    seed_division TEXT,
    seed_lp       INTEGER,
    rank_source   TEXT,            -- 'ladder' (rank read during this patch) or
                                   -- 'estimate' (past patch: player's rank today)
    rank_date     TEXT             -- date the seed player's rank was read
);
CREATE TABLE IF NOT EXISTS participants (
    match_id        TEXT NOT NULL,
    participant_id  INTEGER NOT NULL,
    puuid           TEXT,
    team_id         INTEGER,       -- 100 blue, 200 red
    champion_id     INTEGER,       -- joins to champions.champ_key
    champion_name   TEXT,          -- joins to champions.champion_id
    position        TEXT,          -- TOP JUNGLE MIDDLE BOTTOM UTILITY
    win             INTEGER,
    champ_level     INTEGER,
    kills           INTEGER,
    deaths          INTEGER,
    assists         INTEGER,
    damage_to_champs     INTEGER,
    damage_taken         INTEGER,
    damage_mitigated     INTEGER,
    healing              INTEGER,
    damage_to_objectives INTEGER,
    gold_earned     INTEGER,
    cs              INTEGER,
    vision_score    INTEGER,
    cc_time         INTEGER,
    first_blood     INTEGER,
    items           TEXT,          -- final items, comma separated ids
    summoner_spells TEXT,
    keystone        INTEGER,
    PRIMARY KEY (match_id, participant_id)
);
CREATE TABLE IF NOT EXISTS bans (
    match_id     TEXT NOT NULL,
    team_id      INTEGER,
    pick_turn    INTEGER,
    champion_id  INTEGER,
    PRIMARY KEY (match_id, team_id, pick_turn)
);
CREATE TABLE IF NOT EXISTS crawl_log (   -- which player was crawled for which window
    puuid       TEXT NOT NULL,
    crawl_key   TEXT NOT NULL,           -- 'recent' or a patch like '15.10'
    crawled_at  TEXT,
    PRIMARY KEY (puuid, crawl_key)
);
CREATE TABLE IF NOT EXISTS patch_probes (  -- (time, patch) observations used to date patches
    platform  TEXT NOT NULL,
    ts        INTEGER NOT NULL,             -- epoch seconds of a real game start
    patch     TEXT NOT NULL,
    PRIMARY KEY (platform, ts)
);
CREATE TABLE IF NOT EXISTS patch_windows (
    platform  TEXT NOT NULL,
    patch     TEXT NOT NULL,
    start_ts  INTEGER,                      -- epoch seconds, approx. (+-2h)
    end_ts    INTEGER,
    is_open   INTEGER,                      -- 1 = still the live patch when computed
    PRIMARY KEY (platform, patch)
);
CREATE INDEX IF NOT EXISTS ix_part_champ ON participants(champion_id);
CREATE INDEX IF NOT EXISTS ix_match_patch ON matches(patch);

DROP VIEW IF EXISTS champion_outcomes;
CREATE VIEW champion_outcomes AS
WITH totals AS (SELECT patch, COUNT(*) AS n_matches FROM matches GROUP BY patch),
banned AS (
    SELECT m.patch, b.champion_id, COUNT(DISTINCT b.match_id) AS n_bans
    FROM bans b JOIN matches m USING (match_id)
    WHERE b.champion_id > 0 GROUP BY m.patch, b.champion_id)
SELECT m.patch, p.champion_id, p.champion_name,
       COUNT(*)                               AS games,
       SUM(p.win)                             AS wins,
       ROUND(AVG(p.win), 4)                   AS winrate,
       ROUND(1.0 * COUNT(*) / t.n_matches, 4) AS pickrate,
       ROUND(1.0 * COALESCE(bn.n_bans, 0) / t.n_matches, 4) AS banrate,
       ROUND(AVG(p.kills), 2)   AS avg_kills,
       ROUND(AVG(p.deaths), 2)  AS avg_deaths,
       ROUND(AVG(p.assists), 2) AS avg_assists,
       ROUND((SUM(p.kills) + SUM(p.assists)) * 1.0 / MAX(SUM(p.deaths), 1), 2) AS kda,
       ROUND(AVG(p.damage_to_champs * 60.0 / m.duration_s), 1) AS dmg_per_min,
       ROUND(AVG(p.gold_earned * 60.0 / m.duration_s), 1)      AS gold_per_min,
       ROUND(AVG(p.cs * 60.0 / m.duration_s), 2)               AS cs_per_min,
       ROUND(AVG(p.vision_score * 60.0 / m.duration_s), 2)     AS vision_per_min,
       ROUND(AVG(m.duration_s) / 60.0, 1)                      AS avg_game_min
FROM participants p
JOIN matches m USING (match_id)
JOIN totals t ON t.patch = m.patch
LEFT JOIN banned bn ON bn.patch = m.patch AND bn.champion_id = p.champion_id
GROUP BY m.patch, p.champion_id;

DROP VIEW IF EXISTS champion_role_outcomes;
CREATE VIEW champion_role_outcomes AS
WITH totals AS (SELECT patch, COUNT(*) AS n_matches FROM matches GROUP BY patch)
SELECT m.patch, p.position, p.champion_id, p.champion_name,
       COUNT(*)                               AS games,
       ROUND(AVG(p.win), 4)                   AS winrate,
       ROUND(1.0 * COUNT(*) / t.n_matches, 4) AS pickrate,
       ROUND((SUM(p.kills) + SUM(p.assists)) * 1.0 / MAX(SUM(p.deaths), 1), 2) AS kda,
       ROUND(AVG(p.damage_to_champs * 60.0 / m.duration_s), 1) AS dmg_per_min,
       ROUND(AVG(p.gold_earned * 60.0 / m.duration_s), 1)      AS gold_per_min,
       ROUND(AVG(p.cs * 60.0 / m.duration_s), 2)               AS cs_per_min
FROM participants p
JOIN matches m USING (match_id)
JOIN totals t ON t.patch = m.patch
WHERE p.position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
GROUP BY m.patch, p.position, p.champion_id;

DROP VIEW IF EXISTS matchups_lane;
CREATE VIEW matchups_lane AS
SELECT m.patch, a.position,
       a.champion_id, a.champion_name,
       b.champion_id AS opp_champion_id, b.champion_name AS opp_champion_name,
       COUNT(*)             AS games,
       ROUND(AVG(a.win), 4) AS winrate,
       ROUND(AVG(a.gold_earned - b.gold_earned), 0) AS avg_gold_diff,
       ROUND(AVG(a.cs - b.cs), 1)                   AS avg_cs_diff
FROM participants a
JOIN participants b
  ON b.match_id = a.match_id AND b.team_id <> a.team_id AND b.position = a.position
JOIN matches m ON m.match_id = a.match_id
WHERE a.position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')
GROUP BY m.patch, a.position, a.champion_id, b.champion_id;

DROP VIEW IF EXISTS matchups_any;
CREATE VIEW matchups_any AS
SELECT m.patch,
       a.champion_id, a.champion_name,
       b.champion_id AS opp_champion_id, b.champion_name AS opp_champion_name,
       COUNT(*)             AS games,
       ROUND(AVG(a.win), 4) AS winrate
FROM participants a
JOIN participants b ON b.match_id = a.match_id AND b.team_id <> a.team_id
JOIN matches m ON m.match_id = a.match_id
GROUP BY m.patch, a.champion_id, b.champion_id;
"""


APEX_TIERS = ("CHALLENGER", "GRANDMASTER", "MASTER")


def rank_bucket(tier):
    """Group apex tiers together: CHALLENGER/GRANDMASTER/MASTER -> MASTER+."""
    return "MASTER+" if tier in APEX_TIERS else tier


def bucket_tiers(bucket):
    return list(APEX_TIERS) if bucket == "MASTER+" else [bucket]


def migrate(con):
    """Add columns introduced after a database was first created."""
    want = {
        "players": [("seeded_at", "REAL")],
        "matches": [("rank_bucket", "TEXT"), ("seed_division", "TEXT"), ("seed_lp", "INTEGER"),
                    ("rank_source", "TEXT"), ("rank_date", "TEXT")],
    }
    for table, cols in want.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, typ in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    # older matches: derive the bucket from the tier we found them through
    con.execute("UPDATE matches SET rank_bucket = CASE WHEN source_tier IN "
                "('CHALLENGER','GRANDMASTER','MASTER') THEN 'MASTER+' ELSE source_tier END, "
                "rank_source = 'legacy' WHERE rank_bucket IS NULL")
    con.execute("CREATE INDEX IF NOT EXISTS ix_match_bucket ON matches(platform, patch, rank_bucket)")
    con.commit()


# --------------------------------------------------------------------------- #
# Rate-limited Riot API client
# --------------------------------------------------------------------------- #
class StopRequested(Exception):
    """Raised inside API calls when the program is shutting down."""


class RiotAPI:
    """
    Rate-limited Riot API client.
      - starts with development-key limits and automatically adopts the real
        limits Riot reports in response headers (so a personal/production key
        is used at its full speed with no code changes)
      - with key_file: re-reads the key from that file, and when the key is
        rejected it pauses until a new key is pasted into the file
      - one instance per routing value (na1, americas, ...), since Riot counts
        limits separately for each
    """
    LIMITS = [(20, 1.0), (100, 120.0)]  # development key: 20/1s and 100/2min

    def __init__(self, key=None, host_template="https://{route}.api.riotgames.com",
                 limits=None, key_file=None, status_route=None, log=print,
                 stop_event=None):
        self.s = requests.Session()
        self.key_file = key_file
        self.key = key or self._read_key_file()
        if not self.key:
            raise ValueError("no API key given")
        self.s.headers["X-Riot-Token"] = self.key
        self.host = host_template
        self.limits = list(limits or self.LIMITS)
        self.calls = collections.deque()
        self.method_limits = {}   # endpoint -> [(n, window)], from X-Method-Rate-Limit
        self.method_calls = collections.defaultdict(collections.deque)
        self.forbidden = {}       # endpoint -> time it was refused (retried after 1h)
        self.n_requests = 0
        self.stats = collections.Counter()  # diagnostics, read by the collector
        self.status_route = status_route
        self.log = log
        self.stop_event = stop_event

    # -- helpers ----------------------------------------------------------- #
    def _read_key_file(self):
        if not self.key_file:
            return None
        try:
            with open(self.key_file, encoding="utf-8") as f:
                return f.read().strip() or None
        except OSError:
            return None

    def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise StopRequested()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 1.0))

    @staticmethod
    def endpoint(path):
        """Name of the API method a path belongs to (Riot limits each separately)."""
        for pattern, name in _ENDPOINTS:
            if pattern.search(path):
                return name
        return path

    @staticmethod
    def _needed_wait(calls, limits, now):
        longest = max(w for _, w in limits)
        while calls and now - calls[0] > longest:
            calls.popleft()
        wait = 0.0
        for n, window in limits:
            recent = [t for t in calls if now - t < window]
            if len(recent) >= n:
                wait = max(wait, window - (now - recent[-n]) + 0.05)
        return wait

    def _wait_for_slot(self, method=None):
        while True:
            now = time.monotonic()
            wait = self._needed_wait(self.calls, self.limits, now)
            if method in self.method_limits:
                wait = max(wait, self._needed_wait(self.method_calls[method],
                                                   self.method_limits[method], now))
            if wait <= 0:
                self.calls.append(now)
                if method in self.method_limits:
                    self.method_calls[method].append(now)
                return
            self.stats["wait_limit_s"] += wait
            self._sleep(wait)

    @staticmethod
    def _parse_limits(raw):
        try:
            return sorted((int(n), float(w)) for n, w in
                          (part.split(":") for part in raw.split(",")))
        except (ValueError, AttributeError):
            return None

    def _adopt_header_limits(self, headers, method):
        lim = self._parse_limits(headers.get("X-App-Rate-Limit"))
        if lim and lim != sorted(self.limits):
            self.limits = lim
            self.log("    rate limits from Riot for this key: " +
                     ", ".join(f"{n} per {w:g}s" for n, w in lim))
        mlim = self._parse_limits(headers.get("X-Method-Rate-Limit"))
        if mlim and mlim != self.method_limits.get(method):
            self.method_limits[method] = mlim
            if self.calls:  # count this call in the method window too
                self.method_calls[method].append(self.calls[-1])

    def _key_is_valid(self):
        """Distinguish 'bad key' from 'this endpoint is forbidden'."""
        if not self.status_route:
            return False
        try:
            r = self.s.get(self.host.format(route=self.status_route) +
                           "/lol/status/v4/platform-data", timeout=30)
            return r.status_code == 200
        except requests.RequestException:
            return True  # network trouble, not a key problem

    def _wait_for_new_key(self):
        rejected = self.key
        if not self.key_file:
            sys.exit("Riot API rejected the key (401/403). Development keys expire "
                     "every 24h: get a fresh one at developer.riotgames.com")
        self.log(f"!!! API key rejected (expired?). Paste a new key into "
                 f"{os.path.abspath(self.key_file)} - collection resumes automatically.")
        while True:
            self.stats["wait_key_s"] += 20
            self._sleep(20)
            k = self._read_key_file()
            if k and k != rejected:
                self.key = k
                self.s.headers["X-Riot-Token"] = k
                self.log("    new API key found, resuming")
                return

    # -- main call --------------------------------------------------------- #
    def get(self, route, path, params=None, retries=5):
        url = self.host.format(route=route) + path
        method = self.endpoint(path)
        if time.time() - self.forbidden.get(method, 0) < 3600:
            return None
        attempt = 0
        while attempt < retries:
            if self.key_file:  # pick up a key replaced while running
                k = self._read_key_file()
                if k and k != self.key:
                    self.key = k
                    self.s.headers["X-Riot-Token"] = k
                    self.forbidden.clear()
            self._wait_for_slot(method)
            t_req = time.monotonic()
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                self.stats["request_s"] += time.monotonic() - t_req
                self.log(f"    network error ({e.__class__.__name__}), retrying")
                self.stats["network_errors"] += 1
                attempt += 1
                self._sleep(2 + attempt * 3)
                continue
            self.stats["request_s"] += time.monotonic() - t_req
            self.n_requests += 1
            self.stats["requests"] += 1
            self._adopt_header_limits(r.headers, method)
            if r.status_code == 200:
                data = r.json()
                self.stats["empty" if data in ([], {}, None) else "ok"] += 1
                return data
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                kind = r.headers.get("X-Rate-Limit-Type", "unknown")
                self.stats[f"429_{kind}"] += 1
                self.stats["wait_429_s"] += wait
                self.log(f"    rate limited by Riot ({kind} limit on {method}), waiting {wait}s")
                self._sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                self.stats["server_errors"] += 1
                attempt += 1
                self._sleep(2 + attempt * 3)
                continue
            if r.status_code in (400, 404):
                self.stats["empty"] += 1
                return None
            if r.status_code in (401, 403):
                if self.key_file and self._key_is_valid():
                    # the key works, but may not use this endpoint
                    self.forbidden[method] = time.time()
                    self.stats["forbidden"] += 1
                    self.log(f"!!! This API key is not allowed to use {method} "
                             f"(HTTP {r.status_code}). Check which APIs your product has "
                             f"access to at developer.riotgames.com. Skipping it for now.")
                    return None
                self._wait_for_new_key()
                continue
            attempt += 1
        return None


_ENDPOINTS = [
    (re.compile(r"/lol/match/v5/matches/by-puuid/[^/]+/ids"), "match-v5 match ids"),
    (re.compile(r"/lol/match/v5/matches/[^/]+/timeline"), "match-v5 timeline"),
    (re.compile(r"/lol/match/v5/matches/[^/]+$"), "match-v5 match"),
    (re.compile(r"/lol/league/v4/entries/"), "league-v4 entries"),
    (re.compile(r"/lol/league/v4/challengerleagues/"), "league-v4 challenger"),
    (re.compile(r"/lol/league/v4/grandmasterleagues/"), "league-v4 grandmaster"),
    (re.compile(r"/lol/league/v4/masterleagues/"), "league-v4 master"),
    (re.compile(r"/lol/summoner/v4/"), "summoner-v4"),
    (re.compile(r"/lol/status/v4/"), "lol-status-v4"),
]


# --------------------------------------------------------------------------- #
# Crawl steps
# --------------------------------------------------------------------------- #
def seed_players(api, con, platform, tiers, pages_per_division):
    """Pull ladder entries for the requested tiers into the players table."""
    added = 0
    for tier in tiers:
        entries = []
        if tier in APEX:
            data = api.get(platform, f"/lol/league/v4/{APEX[tier]}/by-queue/{QUEUE_NAME}")
            entries = [dict(e, tier=tier, rank="I") for e in (data or {}).get("entries", [])]
        else:
            for div in DIVISIONS:
                for page in range(1, pages_per_division + 1):
                    data = api.get(platform, f"/lol/league/v4/entries/{QUEUE_NAME}/{tier}/{div}",
                                   params={"page": page})
                    if not data:
                        break
                    entries.extend(data)
        rows = []
        for e in entries:  # resolve ids first, write afterwards (no lock held during calls)
            puuid = e.get("puuid")
            if not puuid and e.get("summonerId"):  # older API responses
                s = api.get(platform, f"/lol/summoner/v4/summoners/{e['summonerId']}")
                puuid = (s or {}).get("puuid")
            if puuid:
                rows.append((puuid, platform, e.get("tier", tier), e.get("rank"),
                             e.get("leaguePoints")))
        before = con.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        now = time.time()
        con.executemany(
            "INSERT INTO players (puuid, platform, tier, division, lp, seeded_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(puuid) DO UPDATE SET platform=excluded.platform, "
            "tier=excluded.tier, division=excluded.division, lp=excluded.lp, "
            "seeded_at=excluded.seeded_at", [r + (now,) for r in rows])
        added += con.execute("SELECT COUNT(*) FROM players").fetchone()[0] - before
        con.commit()
        print(f"  {tier}: {len(entries)} ladder entries")
    return added


def patch_of(game_version):
    parts = (game_version or "").split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else None


def store_match(con, m, platform, source_tier, min_duration):
    info = m.get("info", {})
    duration = info.get("gameDuration") or 0
    if info.get("queueId") != QUEUE_ID or duration < min_duration:
        return False  # remakes / wrong queue
    parts = info.get("participants", [])
    if len(parts) != 10 or any(p.get("gameEndedInEarlySurrender") for p in parts):
        return False
    mid = m["metadata"]["matchId"]
    win_team = next((t["teamId"] for t in info.get("teams", []) if t.get("win")), None)
    con.execute("INSERT OR REPLACE INTO matches (match_id, platform, patch, game_version, "
                "game_start, duration_s, queue_id, source_tier, winning_team) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (
        mid, platform, patch_of(info.get("gameVersion")), info.get("gameVersion"),
        info.get("gameStartTimestamp"), duration, info.get("queueId"), source_tier, win_team))

    rows = []
    for p in parts:
        perks = (p.get("perks") or {}).get("styles") or [{}]
        keystone = ((perks[0].get("selections") or [{}])[0]).get("perk")
        rows.append((
            mid, p.get("participantId"), p.get("puuid"), p.get("teamId"),
            p.get("championId"), p.get("championName"),
            p.get("teamPosition") or p.get("individualPosition"), int(bool(p.get("win"))),
            p.get("champLevel"), p.get("kills"), p.get("deaths"), p.get("assists"),
            p.get("totalDamageDealtToChampions"), p.get("totalDamageTaken"),
            p.get("damageSelfMitigated"), p.get("totalHeal"),
            p.get("damageDealtToObjectives"), p.get("goldEarned"),
            (p.get("totalMinionsKilled") or 0) + (p.get("neutralMinionsKilled") or 0),
            p.get("visionScore"), p.get("timeCCingOthers"),
            int(bool(p.get("firstBloodKill"))),
            ",".join(str(p.get(f"item{i}", 0)) for i in range(7)),
            f"{p.get('summoner1Id')},{p.get('summoner2Id')}", keystone,
        ))
    con.executemany("INSERT OR REPLACE INTO participants VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    bans = [(mid, t["teamId"], b.get("pickTurn"), b.get("championId"))
            for t in info.get("teams", []) for b in t.get("bans", [])]
    con.executemany("INSERT OR REPLACE INTO bans VALUES (?,?,?,?)", bans)
    return True


def pkey(patch):
    """'15.9' -> (15, 9) so patches sort numerically, not as text."""
    try:
        a, b = patch.split(".")[:2]
        return int(a), int(b)
    except (AttributeError, ValueError):
        return (-1, -1)


def now_ts():
    return int(time.time())


# --------------------------------------------------------------------------- #
# Working out when each patch was live (Riot's API does not publish this)
# --------------------------------------------------------------------------- #
class PatchDater:
    """
    Finds patch start/end times by looking at real games: 'probe' a time t by
    fetching one ranked game that started shortly after t and reading its patch.
    Patches only move forward in time, so a binary search on these probes finds
    each boundary to within `precision` seconds. Every probe is saved, so later
    searches (and later runs) reuse what earlier ones learned.
    """

    def __init__(self, api, con, platform, regional, precision=2 * 3600):
        self.api, self.con = api, con
        self.platform, self.regional = platform, regional
        self.precision = precision
        self.obs = dict(con.execute(
            "SELECT ts, patch FROM patch_probes WHERE platform = ?", (platform,)).fetchall())
        self.players = [r[0] for r in con.execute(
            "SELECT puuid FROM players WHERE platform = ?", (platform,))]

    def _record(self, ts, patch):
        self.obs[ts] = patch
        self.con.execute("INSERT OR REPLACE INTO patch_probes VALUES (?,?,?)",
                         (self.platform, ts, patch))
        self.con.commit()

    def probe(self, t, t_end, tries=8):
        """Patch of some ranked game starting in [t, t_end); None if none found."""
        for puuid in random.sample(self.players, min(tries, len(self.players))):
            ids = self.api.get(self.regional, f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
                               params={"queue": QUEUE_ID, "startTime": int(t),
                                       "endTime": int(t_end), "count": 1})
            if not ids:
                continue
            m = self.api.get(self.regional, f"/lol/match/v5/matches/{ids[0]}")
            info = (m or {}).get("info", {})
            patch = patch_of(info.get("gameVersion"))
            start = info.get("gameStartTimestamp")
            if patch and start:
                ts = int(start // 1000)
                self._record(ts, patch)
                return ts, patch
        return None

    def first_time(self, pred, max_lookback_days=800):
        """Earliest time at which pred(patch) becomes True (pred must flip once)."""
        def bracket():
            lo = max((t for t, p in self.obs.items() if not pred(p)), default=None)
            hi = min((t for t, p in self.obs.items() if pred(p) and (lo is None or t > lo)),
                     default=None)
            return lo, hi

        lo, hi = bracket()
        if hi is None:  # make sure we know what "now" looks like
            r = self.probe(now_ts() - 6 * 3600, now_ts())
            if r is None:
                r = self.probe(now_ts() - 3 * 86400, now_ts())
            lo, hi = bracket()
            if hi is None:
                return None  # condition not true even now (patch still live / future)

        step = 7 * 86400
        while lo is None:  # step back in time until pred is False
            t = hi - step
            if t < now_ts() - max_lookback_days * 86400:
                return hi
            r = self.probe(t, t + 12 * 3600)
            if r is None:
                r = self.probe(t, t + 3 * 86400)
            if r is None:
                return hi  # no games found that far back
            lo, hi = bracket()
            step *= 2

        while hi - lo > self.precision:  # binary search
            mid = (lo + hi) // 2
            # Riot lists newest games first, so probe a short window just after
            # mid (widening if empty) to get a game close to mid.
            width = max(3600, (hi - mid) // 8)
            while True:
                w_end = min(mid + width, hi)
                r = self.probe(mid, w_end, tries=4)
                if r is not None or w_end >= hi:
                    break
                width *= 3
            if r is None:
                break  # no games between mid and hi; boundary is ~hi
            before = (lo, hi)
            lo, hi = bracket()
            if (lo, hi) == before:
                break  # no progress possible (safety guard)
        return hi

    def window(self, patch, refresh=False):
        row = self.con.execute(
            "SELECT start_ts, end_ts, is_open FROM patch_windows WHERE platform=? AND patch=?",
            (self.platform, patch)).fetchone()
        if row and not refresh and not row[2]:
            return row[0], row[1]
        target = pkey(patch)
        start = self.first_time(lambda p: pkey(p) >= target)
        if start is None:
            return None
        end = self.first_time(lambda p: pkey(p) > target)
        is_open = end is None
        end = end if end is not None else now_ts()
        self.con.execute("INSERT OR REPLACE INTO patch_windows VALUES (?,?,?,?,?)",
                         (self.platform, patch, start, end, int(is_open)))
        self.con.commit()
        return start, end


# --------------------------------------------------------------------------- #
# Crawling
# --------------------------------------------------------------------------- #
def crawl(api, con, platform, regional, start_time, end_time, per_player, target,
          min_duration, patch_filter, crawl_key, count_existing=False):
    """
    Collect matches that started in [start_time, end_time).
    target: stop once this many matches are stored (new ones, or total for the
            patch when count_existing=True, which makes backfills resumable).
    """
    rows = con.execute("""
        SELECT p.puuid, p.tier, c.crawled_at FROM players p
        LEFT JOIN crawl_log c ON c.puuid = p.puuid AND c.crawl_key = ?
        WHERE p.platform = ?""", (crawl_key, platform)).fetchall()
    fresh = [(p, t) for p, t, c in rows if c is None]
    random.shuffle(fresh)  # spread uncrawled players across the ladder
    done = sorted([(c, p, t) for p, t, c in rows if c is not None])
    players = fresh + [(p, t) for _, p, t in done]  # then least-recently crawled

    have = {r[0] for r in con.execute("SELECT match_id FROM matches")}
    stored = 0
    if count_existing and patch_filter:
        stored = con.execute("SELECT COUNT(*) FROM matches WHERE patch = ? AND platform = ?",
                             (patch_filter, platform)).fetchone()[0]
    if stored >= target:
        print(f"  already have {stored} matches, target {target} reached")
        return 0
    start_count, skipped = stored, 0

    params = {"queue": QUEUE_ID, "type": "ranked", "startTime": int(start_time),
              "count": per_player}
    if end_time:
        params["endTime"] = int(end_time)

    for i, (puuid, tier) in enumerate(players, 1):
        ids = api.get(regional, f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
                      params=params) or []
        for mid in ids:
            if mid in have:
                continue
            have.add(mid)
            m = api.get(regional, f"/lol/match/v5/matches/{mid}")
            if not m:
                continue
            if patch_filter and patch_of(m.get("info", {}).get("gameVersion")) != patch_filter:
                skipped += 1
                continue
            if store_match(con, m, platform, tier, min_duration):
                stored += 1
            else:
                skipped += 1
            con.commit()
            if stored >= target:
                break
        con.execute("INSERT OR REPLACE INTO crawl_log VALUES (?,?,?)",
                    (puuid, crawl_key, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()
        if i % 10 == 0 or stored >= target:
            print(f"  players {i}/{len(players)} | matches {stored}/{target} | "
                  f"skipped {skipped} | requests {api.n_requests}")
        if stored >= target:
            break
    else:
        print(f"  ran out of players ({stored}/{target}); add more with "
              f"--reseed and extra --tiers / --pages")
    return stored - start_count


def backfill_targets(con, args):
    if args.backfill:
        return sorted({p.strip() for p in args.backfill.split(",") if p.strip()},
                      key=pkey, reverse=True)
    rows = [r[0] for r in con.execute("SELECT DISTINCT patch FROM patches")] \
        if con.execute("SELECT name FROM sqlite_master WHERE name='patches'").fetchone() else []
    if not rows:
        sys.exit("--backfill-last needs the patches table: run fetch_balance_data.py first, "
                 "or list patches with --backfill 15.17,15.16,...")
    return sorted(rows, key=pkey, reverse=True)[: args.backfill_last]


def run_backfill(api, con, platform, regional, args):
    patches = backfill_targets(con, args)
    print(f"Backfilling {len(patches)} patches, target {args.max_matches} matches each: "
          + ", ".join(patches))
    dater = PatchDater(api, con, platform, regional)
    new_total = 0
    for patch in patches:
        print(f"\nPatch {patch}: finding when it was live ...")
        w = dater.window(patch)
        if w is None:
            print("  could not find games from this patch (too old, or not released) - skipping")
            continue
        start, end = w
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"  live from {fmt(start)} to {fmt(end)}  ({api.n_requests} requests so far)")
        if end - start < 3600:
            print("  window too short - skipping")
            continue
        new_total += crawl(api, con, platform, regional, start, end, args.per_player,
                           args.max_matches, args.min_duration, patch, patch,
                           count_existing=True)
    return new_total


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Collect ranked match outcomes from the Riot API.")
    ap.add_argument("--region", default="na", choices=sorted(REGIONS))
    ap.add_argument("--tiers", default="CHALLENGER,GRANDMASTER,MASTER",
                    help="comma list: CHALLENGER,GRANDMASTER,MASTER,DIAMOND,EMERALD,"
                         "PLATINUM,GOLD,SILVER,BRONZE,IRON")
    ap.add_argument("--pages", type=int, default=1,
                    help="ladder pages per division for non-apex tiers (~205 players each)")
    ap.add_argument("--max-matches", type=int, default=1000,
                    help="recent mode: stop after this many new matches. "
                         "backfill mode: target number of matches per patch")
    ap.add_argument("--days", type=int, default=14,
                    help="recent mode: only matches from the last N days")
    ap.add_argument("--backfill", default=None,
                    help="backfill mode: comma list of patches to collect, e.g. 15.10,15.11")
    ap.add_argument("--backfill-last", type=int, default=None,
                    help="backfill mode: the N most recent patches in the balance data")
    ap.add_argument("--per-player", type=int, default=20, help="match ids to request per player")
    ap.add_argument("--patch", default=None,
                    help="recent mode: only keep matches from this patch, e.g. 15.18")
    ap.add_argument("--min-duration", type=int, default=300,
                    help="drop games shorter than this many seconds (remakes)")
    ap.add_argument("--reseed", action="store_true", help="refresh the ladder player list")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--api-key", default=None, help="defaults to $RIOT_API_KEY")
    ap.add_argument("--host-template", default="https://{route}.api.riotgames.com",
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    key = args.api_key or os.environ.get("RIOT_API_KEY")
    if not key:
        sys.exit("No API key. Run:  export RIOT_API_KEY=RGAPI-...   then try again.")
    platform, regional = REGIONS[args.region]
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]

    api = RiotAPI(key, args.host_template)
    con = sqlite3.connect(args.db, timeout=120)
    con.executescript(SCHEMA)
    migrate(con)

    n_players = con.execute("SELECT COUNT(*) FROM players WHERE platform = ?",
                            (platform,)).fetchone()[0]
    if args.reseed or n_players == 0:
        print(f"Seeding players from the {platform} ladder: {', '.join(tiers)}")
        added = seed_players(api, con, platform, tiers, args.pages)
        print(f"  {added} new players added")

    try:
        if args.backfill or args.backfill_last:
            stored = run_backfill(api, con, platform, regional, args)
        else:
            start = now_ts() - args.days * 86400
            print(f"Collecting matches from the last {args.days} days ...")
            stored = crawl(api, con, platform, regional, start, None, args.per_player,
                           args.max_matches, args.min_duration, args.patch, "recent")
    except KeyboardInterrupt:
        print("\nStopped by user; everything collected so far is saved.")
        stored = None

    by_patch = sorted(con.execute("SELECT patch, COUNT(*) FROM matches GROUP BY patch"),
                      key=lambda r: pkey(r[0]), reverse=True)
    if stored is not None:
        print(f"\nStored {stored} new matches ({api.n_requests} API requests).")
    print(f"Database now holds {sum(n for _, n in by_patch)} matches: " +
          ", ".join(f"{p} ({n})" for p, n in by_patch))
    con.close()


if __name__ == "__main__":
    main()
