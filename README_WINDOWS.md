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

## Updating the model: one click

Double-click **`update_model.bat`** (or run `python run_pipeline.py`). It runs every step
below in order: new ability data, patch numbers, training tables, baselines, the
patch-change model, the game model, and finally the diagnostics report (`report.html`,
which opens in your browser when it's done). It stops with a clear message if anything fails, and saves all output to
`pipeline.log`. It's safe to run while the collector keeps going.

- `python run_pipeline.py --quick` retrains the model without testing it patch by patch
  (much faster once there's a lot of data).
- `python run_pipeline.py --skip-fetch` skips checking CommunityDragon for new patches.
- `python run_pipeline.py --only-data` updates the data and tables, but not the model.
- `update_model.bat --max-ram-gb 6` changes the RAM limit (see below).

### Sharing results

After every run, `update_model.bat` copies the small result files (pipeline.log,
report.html, the model CSVs) into the `results` folder. To send them to GitHub,
double-click **`share_results.bat`** (or run `git add results`, `git commit -m results`,
`git push`). Then `git pull` on the other computer. The database, games table and models
stay out of git.

### RAM limit

The update is limited to **7 GB of RAM** by default (all steps together), so it can't
freeze the PC. If a step needs more, only that step stops, with a message saying it ran
out of memory, and the rest of the computer keeps working. After each step the window
shows the peak RAM used so far, so you can see how close it gets.

- Change the limit: `update_model.bat --max-ram-gb 8` (or `python run_pipeline.py --max-ram-gb 8`).
  `0` turns the limit off.
- If it runs out of memory: close other programs (browsers use a lot), raise the limit a
  little, or use `--quick` (skips the patch-by-patch testing, the heaviest part).
- Running one script on its own (e.g. `python train_game_model.py`) uses the same 7 GB
  limit. To change it for that PowerShell window: `$env:LOL_MAX_RAM_GB = "6"`.

The sections below describe each step, if you ever want to run one by itself.

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

- `training/games.parquet`: one row per game with the blue-minus-red differences of the
  teams' champion numbers, lane-vs-lane differences and the 10 champions (the main
  model's training data).
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

## The patch-change model

```
python train_change_model.py                  # tests it on past patches (seconds)
python train_change_model.py --predict 16.19  # predictions for every champion on a patch
```

Two parts:

- **Strength model**: how strong each champion is before balance changes, from recent
  winrate, long-term winrate and popularity. Champions with lots of games keep their real
  winrate (e.g. Ezreal around 47%). Champions with few games are pulled toward 50%, because
  their winrate is mostly luck.
- **Change model**: how much this patch's changes move the champion, learned from past
  patches. Buffs and nerfs are scored separately (nerfs can hurt more than buffs help).
  Reworks (25+ changed numbers) are flagged rather than read as % changes.

Every changed champion gets two predictions:

- **best guess**: the most accurate on average. Held back when the data is noisy, and
  switched off for a patch if it didn't beat "no effect" on the latest patches.
- **bold**: the same direction, sized like real patch swings. Less accurate on average,
  but shows who is likely to move a lot. The test reports how often its direction is
  right (50% = coin flip), overall and for the 3 biggest predicted movers per patch.

There's also a likely range for each champion's true winrate. `--predict` lists changed
champions plus the strongest and weakest predicted champions, and saves every champion to
`training/predicted_changes_<patch>.csv`. It works for a brand-new patch before any games
are played, as soon as its balance data is loaded.

### Game model options

```
python train_game_model.py --engine xgboost --gpu   # XGBoost on the graphics card
python train_game_model.py --test-last 5            # only test the latest 5 patches (faster)
```

- `--engine lightgbm` (default) or `--engine xgboost` (needs `pip install xgboost`). Both
  use the same technique and give similar accuracy. XGBoost is faster on an NVIDIA GPU.
- `--gpu` trains on the graphics card and falls back to the CPU if it can't.
- The same options work on the whole update: `update_model.bat --engine xgboost --gpu --test-last 5`.

## Champion strength model

```
python train_strength_model.py                  # tests on the latest 6 patches + final model
python train_strength_model.py --predict 16.19  # predicted winrate for every champion
```

A small neural network turns each champion's numbers (and role) into a strength score.
The same network scores all 10 champions in a game, so it learns what numbers are worth
from every game, not just from champions that were changed. Lane matchups and team mix
are added on top. A nerf changes that champion's score directly, so patch effects aren't
diluted. `--predict` shows which changed numbers caused each champion's predicted change.

It runs as part of `update_model.bat` (skipped with a note if PyTorch isn't installed;
`--skip-strength` leaves it out). Results appear in report.html next to the other models.

**Installing PyTorch with GPU support (NVIDIA, e.g. RTX 3060 Ti).** Plain `pip install torch`
installs a CPU-only version on Windows. Instead, go to https://pytorch.org/get-started/locally/,
choose Stable / Windows / Pip / Python / the newest CUDA version, and run the command it
shows, which looks like:
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Check it worked (should print True and your card's name):
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Keep your NVIDIA driver up to date. Without a working GPU it still runs on the CPU, just
slower (`--cpu` forces the CPU).

`python train_game_model.py --linear-trees` tries LightGBM linear trees (smoother
reaction to small stat changes). Off by default because it didn't help on 82k games.

## The diagnostics report (report.html)

`python diagnostics.py` (run automatically by `update_model.bat`) builds `report.html`
from what the other steps saved. It takes seconds and retrains nothing. Sections:

1. **Luck or real mistakes?** Every miss compared with how much that champion's winrate
   wobbles by chance (fewer games = more wobble). About 95% of misses should be within 2
   wobbles. The extra is the model's **own error**.
2. **Where the misses happen**, by changed/unchanged/rework, buff or nerf size, role,
   class, number of games and how strong the champion was. Also shows **bias**: + means
   predicted too high.
3. **Champions the model keeps getting wrong**, missed in the same direction patch after
   patch. That's a real pattern, not luck.
4. **Calibration**: when it says 52% (or +2 points), is that what happens on average?
5. **Bold picks**: how often bold predictions get the direction right, by prediction
   size, plus the biggest ones and what caused them.
6. **What the models learned**: the strength model's settings and the change model's
   weights. Weights learned from fewer than 10 champions are marked shaky.
7. **Patch by patch**, every method next to the luck floor.
8. **Game model**: accuracy per patch, region and rank, calibration, champions it
   misjudges, and what it relies on.
9. **Champion strength model**: accuracy per patch next to the other models, a head to
   head on the same changed champions, its biggest predicted effects and which numbers
   caused them, and the champions the numbers explain least.
10. **Data checks**: games per patch, suspiciously large changes (possible reading
   errors) and reworks.

The same group numbers are also saved to `training/diagnostics_breakdown.csv`, and every
single prediction (with its miss, luck wobble and reasons) to
`training/change_model_predictions.csv`, for Excel.

## What gets saved

- `lol_data.db`: everything the model uses (balance data, matches, players, objectives,
  runes, timeline stats, item purchase order, skill order, plus summary views).
  Every match has a rank label: `rank_bucket` (MASTER+ or DIAMOND), and `rank_source`
  is `ladder` (rank read while that patch was live) or `estimate` (past patch). This is
  information only: every patch is collected the same way.
- `raw_matches\`: the full original match downloads, compressed, one file per region, so
  nothing ever needs downloading again.
- `collector.log`: the activity log.
