# Running the collector on Windows

## One-time setup (about 15 minutes)

1. **Install Python** from https://www.python.org/downloads/ (3.10 or newer).
   On the first installer screen, tick **"Add python.exe to PATH"**.
2. **Copy the project folder** to the desktop PC, e.g. `C:\Pred`. Bring everything:
   `collector.py`, `fetch_balance_data.py`, `fetch_match_data.py`, the two `.bat` files,
   and `lol_data.db` + the `raw` folder if you want to keep the data already collected.
   - Don't put it inside a OneDrive / Dropbox / Google Drive folder. Syncing a
     database that changes every second can corrupt it or slow everything down.
   - Needs about **25-30 GB** free disk space for the full dataset.
3. **Install the one library it needs.** Open PowerShell (Start menu, type "PowerShell") and run:
   ```
   pip install requests
   ```
4. **Add your API key.** Create a file called `riot_key.txt` in the folder and paste your
   key into it (just the key, e.g. `RGAPI-xxxx...`). If you skip this, the collector asks
   for the key the first time it starts and creates the file for you.
   Keep this file private.

## Running it

- **Start:** double-click `run_collector.bat`. A window opens and shows what it's doing.
  Leave it open. It fetches any missing balance data first, then collects NA, EUW and KR
  at the same time.
- **Check progress:** double-click `status.bat` at any time. It shows games per patch and
  region, how many have timelines, speed, time left and disk space used.
- **Stop:** press Ctrl+C in the collector window, or close it. Nothing is lost, and the next
  start continues where it stopped.

## Starting over / changing API keys

Stop the collector first (Ctrl+C), then in PowerShell in the project folder:

| Command | What it does |
|---|---|
| `python collector.py reset-players` | Clears stored players, patch dates and search progress. **Keeps all collected games.** Use this if you ever switch to a key from a different app. Riot encrypts player IDs separately for every app, so old IDs stop working. (The collector also detects this and does it automatically.) |
| `python collector.py reset-all` | Deletes **all** collected games, timelines and raw backups, and starts from zero. Balance data is kept. Asks you to type `yes` first. |

Then start `run_collector.bat` again.

## When the development key expires (every 24 hours)

The window shows `!!! API key rejected`. Get a new key at https://developer.riotgames.com,
paste it into `riot_key.txt` (replacing the old one) and save. Collection resumes by itself
within about 20 seconds, with no restart needed. A Personal key removes this step.

## If collection seems slow: read the HEALTH lines

Every 15 minutes each region prints a `HEALTH` line, for example:

```
[NA] HEALTH last 15 min: 1,480 requests -> 890 games (1.7 requests/game) | empty replies 6% | time: 95% pacing to the rate limit, 4% waiting for Riot's replies
[NA] HEALTH ok: running at the full speed this API key allows
```

| What you see | What it means |
|---|---|
| `HEALTH ok` | Working at the maximum speed your key allows. With a Personal or development key that's ~2,000 games/hr per region |
| `!!! API key rejected` / `waiting for a new API key` | Key expired or wrong: paste a valid key into `riot_key.txt` |
| `!!! This API key is not allowed to use ...` | Your product key doesn't have access to that API: check your product's APIs on the developer portal |
| `Riot 429s: ...` | Riot is blocking requests for going too fast: usually another program or a second collector window is using the same key |
| `empty replies` above 50% | Most players had no games in that patch window (common for old patches): raise `ladder_pages_per_division` |
| `idle for much of this time` | Nothing left to collect, or something is stuck: read the lines above it |

## Keeping it running

- While the collector runs it **stops Windows from sleeping** (the screen can still turn
  off). Just keep the PC plugged in and on.
- `run_collector.bat` restarts the collector automatically if it ever crashes.
- **Start automatically after a reboot** (e.g. Windows Update): press `Win + R`, type
  `shell:startup`, press Enter, and put a *shortcut* to `run_collector.bat` in that folder.
- Optional: in Settings > Windows Update > Advanced options, set **Active hours** so
  updates don't restart the PC in the middle of the day.

## Settings (`collector_config.json`, created on first run)

| Setting | Default | Meaning |
|---|---|---|
| `regions` | `["na","euw","kr"]` | Regions to collect. Each extra *routing group* adds speed: americas (na, br, lan, las), europe (euw, eune, tr, ru, me), asia (kr, jp), sea (oce, sg, tw, vn). Adding `"vn"` gives about +33% speed |
| `rank_mix` | M+ 50%, Diamond 50% | Rank mix used for **every** patch, so all patches are comparable. High-elo players are the most rank-stable, which keeps older patches reliable |
| `ladder_pages_per_division` | 5 | Players loaded per division below Master (~205 per page, 4 divisions per tier) |
| `reseed_players_days` | 3 | How often the ladder is reloaded, which keeps players' ranks current |
| `patches_back` | 25 | How many past patches to cover |
| `target_per_patch` | 40000 | Games per patch, all regions together |
| `stages` | 5000, 20000, 40000 | Fill every patch to 5k first, then 20k, then 40k |
| `timeline_fraction` | 0.25 | Share of games that also get a timeline (build order, gold at 15 min) |
| `games_per_player_per_stage` | 6 | Limits how much one player (e.g. a one-trick) can skew a patch |

Stop the collector before editing the settings, then start it again.

## Ability numbers and patch changes (model inputs)

These two scripts turn the balance data into numbers for the model. Neither uses your
Riot API key, so they can run in a second PowerShell window while the collector runs.

```
python fetch_ability_data.py          # ~1 hour the first time: ability numbers per patch
python build_patch_state.py           # a few seconds: champion/item features + patch changes
```

- `python fetch_ability_data.py --inspect Ahri` shows exactly what was read for one
  champion (useful for checking, or to send to me).
- `python build_patch_state.py --report 16.18` lists everything that changed in a patch,
  like patch notes in numbers.
- Re-run both after a new patch. `fetch_ability_data.py` only downloads what's new.
- `fetch_ability_data.py --reparse` re-reads the downloaded files without downloading again
  (after a parser fix).

## Training tables and baselines

Needs a few extra libraries once: `pip install pandas numpy pyarrow`

```
python build_training_tables.py   # seconds: games + champion-patch tables into training/
python baselines.py               # scores simple predictions per patch
```

- `training/games.parquet`: one row per game with both teams' champion numbers
  (the main model's training data).
- `training/champion_patch.parquet`: one row per champion per patch with winrate, pick
  and ban rate, the champion's numbers and what changed from the last patch.
- `baselines.py` reports how close each simple prediction gets to real winrates, next to
  the **noise floor** (the best any prediction could do with this many games).

Re-run both whenever the collector has added games or a new patch arrives (after
`build_patch_state.py`).

## The main model

Needs once: `pip install lightgbm`

```
python train_game_model.py                  # ~30 s now, longer as data grows
python train_game_model.py --predict 16.18  # predicted effect of a patch's balance changes
```

- Trains a LightGBM "which team wins" model on every game, tests it on patches it never
  saw (training only on earlier patches), and compares it with a coin flip and the
  baselines. It then trains final models on all games and saves them to `models/`.
- **AUC** (0.5 = coin flip) shows how well it predicts single games. In high elo, games
  are close to coin flips, so small numbers (0.52-0.55) are normal. It grows with more
  games.
- Re-run it whenever more games have been collected.

## What gets saved

- `lol_data.db`: everything the model uses (balance data, matches, players, objectives,
  runes, timeline stats, item purchase order, skill order, plus summary views).
  Every match has a rank label: `rank_bucket` (MASTER+ or DIAMOND), and `rank_source`
  is `ladder` (rank read while that patch was live) or `estimate` (past patch). This is
  information only: every patch is collected the same way.
- `raw_matches\`: the full original match downloads, compressed, one file per region, so
  nothing ever needs downloading again.
- `collector.log`: the activity log.
