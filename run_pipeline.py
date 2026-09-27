#!/usr/bin/env python3
"""
run_pipeline.py  -  update everything, from new balance data to a retrained model, in one go.

Runs, in order:
  1. fetch_ability_data.py     ability numbers for any new patch (instant if nothing is new)
     fetch_item_data.py        item passive numbers + item tooltips for any new patch
  2. build_patch_state.py      champion / item numbers per patch + what changed
  3. build_training_tables.py  games table + champion-patch table
  4. baselines.py              simple predictions to compare against
  5. train_change_model.py     champion-level model of how balance changes move winrates
  6. train_game_model.py       game model (LightGBM or XGBoost): test on unseen patches,
                               train final models
  7. train_strength_model.py   champion strength model (neural network; skipped if PyTorch
                               isn't installed)
  8. diagnostics.py            report.html: where the models miss, and why

Safe to run while the collector is running. Output is shown and also saved to
pipeline.log. If a step fails, the pipeline stops and says which one.

Usage:
  python run_pipeline.py                 everything
  python run_pipeline.py --skip-fetch    don't check CommunityDragon for new patches
  python run_pipeline.py --quick         skip the per-patch model testing (just retrain)
  python run_pipeline.py --only-data     steps 1-3 only (no baselines / models)
  python run_pipeline.py --max-ram-gb 6  keep RAM use under 6 GB (default 7, 0 = no limit)
  python run_pipeline.py --engine xgboost   game model with XGBoost instead of LightGBM
  python run_pipeline.py --gpu           use the graphics card where possible
  python run_pipeline.py --test-last 5   only test the latest 5 patches (faster)
  python run_pipeline.py --skip-strength / --skip-game-model   leave one model out
  python run_pipeline.py --skip-data     reuse the existing training tables (no new games)
  python run_pipeline.py --patch-charts 8   per-patch charts in the report (default 5)
(or double-click update_model.bat)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memlimit  # noqa: E402
LOG = os.path.join(HERE, "pipeline.log")


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def run_step(n, total, title, args, fh, optional=False):
    log(f"\n{'=' * 70}\n[{n}/{total}] {title}\n{'=' * 70}", fh)
    t0 = time.time()
    mem_error = False
    proc = subprocess.Popen([sys.executable, "-u"] + args, cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    for line in proc.stdout:
        # the fetch step prints a progress line every 50 files - keep the screen readable
        if "files |" in line and "/s |" in line:
            continue
        mem_error = mem_error or memlimit.is_memory_error(line)
        log(line.rstrip("\n"), fh)
    code = proc.wait()
    peak = memlimit.peak_gb()
    try:
        near_limit = peak is not None and peak >= 0.95 * float(os.environ.get("LOL_MAX_RAM_GB", 0))
    except ValueError:
        near_limit = False
    if code != 0 and (mem_error or near_limit):   # LightGBM can stop silently when out of RAM
        limit = os.environ.get("LOL_MAX_RAM_GB", "?")
        log(f"\n!!! {title} ran out of memory (RAM limit {limit} GB). Your PC is fine - only this "
            f"step stopped. Options: close other programs and run again, raise the limit "
            f"(update_model.bat --max-ram-gb 8), or use --quick to skip the per-patch testing.",
            fh)
    took = time.time() - t0
    if code != 0:
        if optional:
            log(f"\n!!! {title} failed (exit code {code}) - continuing with the data already "
                f"stored.", fh)
            return True
        log(f"\n!!! {title} FAILED (exit code {code}) after {took:.0f}s - stopping here. "
            f"See the messages above (also saved in pipeline.log).", fh)
        return False
    peak_txt = f", peak RAM so far {peak:.1f} GB" if peak else ""
    log((f"-- done in {took / 60:.1f} min" if took >= 60 else f"-- done in {took:.0f}s") + peak_txt,
        fh)
    return True


# small outputs copied to results/ after every run, so they can be shared through git
# (the big files - database, games table, models - stay out of git)
RESULT_FILES = [
    "pipeline.log", "report.html",
    "training/change_model_predictions.csv", "training/change_model_results.csv",
    "training/change_model_weights.csv", "training/baseline_results.csv",
    "training/model_results.csv", "training/game_model_effects.csv",
    "training/diagnostics_breakdown.csv", "training/predicted_changes_*.csv",
    "training/game_test_predictions.parquet", "models/importance_numbers.csv",
    "training/strength_results.csv", "training/strength_predictions.csv",
    "training/item_changes.csv", "training/item_users.csv", "training/perf_step1_check.csv",
    "training/champion_patch.parquet",
    "training/strength_predicted_*.csv", "models/strength_corrections.csv",
]
MAX_RESULT_MB = 25


def export_results(fh):
    import glob
    import shutil
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    copied, skipped = 0, []
    for pattern in RESULT_FILES:
        for src in glob.glob(os.path.join(HERE, pattern)):
            size_mb = os.path.getsize(src) / 1024 ** 2
            if size_mb > MAX_RESULT_MB:
                skipped.append(f"{os.path.basename(src)} ({size_mb:.0f} MB)")
                continue
            try:
                shutil.copy2(src, os.path.join(out, os.path.basename(src)))
                copied += 1
            except OSError as e:
                skipped.append(f"{os.path.basename(src)} ({e})")
    log(f"Copied {copied} result files to results\\ (to share: git add results, commit, push)"
        + (f"; skipped: {', '.join(skipped)}" if skipped else ""), fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the whole data + model pipeline.")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="don't check CommunityDragon for new patches")
    ap.add_argument("--quick", action="store_true",
                    help="skip testing the model patch by patch; just train the final models")
    ap.add_argument("--only-data", action="store_true", help="stop after the training tables")
    ap.add_argument("--engine", choices=["lightgbm", "xgboost"], default="lightgbm",
                    help="game model library (default lightgbm)")
    ap.add_argument("--gpu", action="store_true",
                    help="game model on the graphics card (the strength model uses it automatically)")
    ap.add_argument("--test-last", type=int, default=None,
                    help="only test the latest N patches (game model default: all; "
                         "strength model default: 6)")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse the existing training tables (skip steps 1-3)")
    ap.add_argument("--patch-charts", type=int, default=5,
                    help="predicted-vs-actual charts in the report for the latest N patches")
    ap.add_argument("--skip-game-model", action="store_true")
    ap.add_argument("--skip-strength", action="store_true")
    ap.add_argument("--max-ram-gb", type=float, default=memlimit.DEFAULT_GB,
                    help=f"RAM limit for all steps together (default {memlimit.DEFAULT_GB:g}, 0 = none)")
    args = ap.parse_args(argv)

    for f in ("lol_data.db", "build_patch_state.py", "build_training_tables.py"):
        if not os.path.exists(os.path.join(HERE, f)):
            sys.exit(f"{f} not found - run this from the project folder.")

    steps = []
    if args.skip_data:
        for f in ("games.parquet", "champion_patch.parquet"):
            if not os.path.exists(os.path.join(HERE, "training", f)):
                sys.exit(f"training/{f} not found - run once without --skip-data first.")
    else:
        if not args.skip_fetch:
            steps.append(("Ability numbers for new patches (CommunityDragon)",
                          ["fetch_ability_data.py"], True))
            steps.append(("Item passives and tooltips for new patches",
                          ["fetch_item_data.py"], True))
        steps.append(("Champion / item numbers and patch changes", ["build_patch_state.py"], False))
        steps.append(("Training tables", ["build_training_tables.py"], False))
    if not args.only_data:
        steps.append(("Baselines", ["baselines.py"], False))
        steps.append(("Patch-change model", ["train_change_model.py"], False))
        if not args.skip_game_model:
            cmd = ["train_game_model.py", "--engine", args.engine]
            cmd += ["--gpu"] if args.gpu else []
            cmd += ["--skip-eval"] if args.quick else []
            cmd += ["--test-last", str(args.test_last)] if args.test_last is not None else []
            steps.append((f"Game model ({args.engine}{', GPU' if args.gpu else ''})", cmd, False))
        if not args.skip_strength:
            cmd = ["train_strength_model.py"]
            cmd += ["--skip-eval"] if args.quick else []
            cmd += ["--test-last", str(args.test_last)] if args.test_last is not None else []
            # optional: a problem here (e.g. PyTorch/GPU setup) doesn't stop the report
            steps.append(("Champion strength model", cmd, True))
        steps.append(("Diagnostics report",
                      ["diagnostics.py", "--patch-charts", str(args.patch_charts)], True))

    t_start = time.time()
    with open(LOG, "w", encoding="utf-8") as fh:
        log(f"Pipeline started {datetime.now():%Y-%m-%d %H:%M}  ({len(steps)} steps)", fh)
        lim = memlimit.apply(args.max_ram_gb, quiet=True)
        log(f"RAM limit: {lim:g} GB for all steps (change with --max-ram-gb)" if lim else
            "RAM limit: none", fh)
        for i, (title, cmd, optional) in enumerate(steps, 1):
            if not run_step(i, len(steps), title, cmd, fh, optional):
                export_results(fh)          # share the log of a failed run too
                sys.exit(1)
        total = time.time() - t_start
        log(f"\nAll {len(steps)} steps finished in {total / 60:.1f} min. "
            f"Full output saved to pipeline.log", fh)
        if os.path.exists(os.path.join(HERE, "report.html")) and not args.only_data:
            log("Diagnostics: open report.html in your browser.", fh)
        export_results(fh)


if __name__ == "__main__":
    main()
