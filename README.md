# GamePredictor

Data pipeline and models behind **[PatchCast](https://nkash324.github.io/patchcast/)**, which forecasts each character's winrate in a competitive online game on a new patch before any games are played on it.

## What it does
1. **Collects matches.** `collector.py` pulls ranked games from the Riot API across NA, EUW and KR in parallel (one worker per region, respecting per-region rate limits) and backfills every patch breadth-first into SQLite. It resumes where it left off.
2. **Turns patches into numbers.** `fetch_balance_data.py`, `fetch_ability_data.py` and `fetch_item_data.py` read Data Dragon and CommunityDragon. `build_patch_state.py` converts each champion's stats, ability damage/ratios, cooldowns and item changes into per-patch features (e.g. "ability damage +8%").
3. **Builds training tables.** `build_training_tables.py` writes one row per game plus per-champion, per-patch tables.
4. **Trains and tests forward in time.** `train_strength_model.py`, `train_change_model.py` and `train_game_model.py` only learn from earlier patches and are scored on the next one, exactly how they're used. `baselines.py` checks every model against simple baselines (last patch's winrate, shrunk recent winrate, coin flip), and results are compared with the "luck floor": the error you'd get even with a perfect forecast.

## Results
Tested forward in time on 24 patches (15.20 to 16.19), ~500k ranked games. Full numbers are in `results/`.

| Champion winrate forecast | Average miss (winrate pts) |
|---|---|
| Luck floor (best possible) | 1.11 |
| **Change model** | **1.28** |
| Shrunk last-patch winrate | 1.36 |
| Coin flip (50%) | 1.47 |
| Last patch's winrate as-is | 1.61 |

- Called the direction of buffed/nerfed champions right **61%** of the time (65% for each patch's three biggest changes).
- Predicting single games from the draft alone tops out around **53% accuracy (AUC 0.54)**. Individual games are mostly decided by the players, so the model is aimed at champion-level winrates instead.

## Run it
```
pip install requests pandas pyarrow scikit-learn
export RIOT_API_KEY=RGAPI-...        # read from the environment, never committed
python collector.py                  # leave running to collect games
python run_pipeline.py               # build features, train, write results/
```
Windows setup: see `README_WINDOWS.md`.

## Tech
Python, SQLite, pandas, scikit-learn, Riot Games API, Data Dragon / CommunityDragon.
