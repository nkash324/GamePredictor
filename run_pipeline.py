#!/usr/bin/env python3
"""
run_pipeline.py  -  update everything, from new balance data to a retrained model, in one go.

Runs, in order:
  1. fetch_ability_data.py     ability numbers for any new patch (instant if nothing is new)
  2. build_patch_state.py      champion / item numbers per patch + what changed
  3. build_training_tables.py  games table + champion-patch table
  4. baselines.py              simple predictions to compare against
  5. train_change_model.py     champion-level model of how balance changes move winrates
  6. train_game_model.py       test the main model on unseen patches, train final models
  7. diagnostics.py            report.html: where the models miss, and why

Safe to run while the collector is running. Output is shown and also saved to
pipeline.log. If a step fails, the pipeline stops and says which one.

Usage:
  python run_pipeline.py                 everything
  python run_pipeline.py --skip-fetch    don't check CommunityDragon for new patches
  python run_pipeline.py --quick         skip the per-patch model testing (just retrain)
  python run_pipeline.py --only-data     steps 1-3 only (no baselines / models)
(or double-click update_model.bat)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "pipeline.log")


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def run_step(n, total, title, args, fh, optional=False):
    log(f"\n{'=' * 70}\n[{n}/{total}] {title}\n{'=' * 70}", fh)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u"] + args, cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    for line in proc.stdout:
        # the fetch step prints a progress line every 50 files - keep the screen readable
        if "files |" in line and "/s |" in line:
            continue
        log(line.rstrip("\n"), fh)
    code = proc.wait()
    took = time.time() - t0
    if code != 0:
        if optional:
            log(f"\n!!! {title} failed (exit code {code}) - continuing with the data already "
                f"stored.", fh)
            return True
        log(f"\n!!! {title} FAILED (exit code {code}) after {took:.0f}s - stopping here. "
            f"See the messages above (also saved in pipeline.log).", fh)
        return False
    log(f"-- done in {took / 60:.1f} min" if took >= 60 else f"-- done in {took:.0f}s", fh)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the whole data + model pipeline.")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="don't check CommunityDragon for new patches")
    ap.add_argument("--quick", action="store_true",
                    help="skip testing the model patch by patch; just train the final models")
    ap.add_argument("--only-data", action="store_true", help="stop after the training tables")
    args = ap.parse_args(argv)

    for f in ("lol_data.db", "build_patch_state.py", "build_training_tables.py"):
        if not os.path.exists(os.path.join(HERE, f)):
            sys.exit(f"{f} not found - run this from the project folder.")

    steps = []
    if not args.skip_fetch:
        steps.append(("Ability numbers for new patches (CommunityDragon)",
                      ["fetch_ability_data.py"], True))
    steps.append(("Champion / item numbers and patch changes", ["build_patch_state.py"], False))
    steps.append(("Training tables", ["build_training_tables.py"], False))
    if not args.only_data:
        steps.append(("Baselines", ["baselines.py"], False))
        steps.append(("Patch-change model", ["train_change_model.py"], False))
        steps.append(("Main model", ["train_game_model.py"] + (["--skip-eval"] if args.quick else []),
                      False))
        steps.append(("Diagnostics report", ["diagnostics.py"], True))

    t_start = time.time()
    with open(LOG, "w", encoding="utf-8") as fh:
        log(f"Pipeline started {datetime.now():%Y-%m-%d %H:%M}  ({len(steps)} steps)", fh)
        for i, (title, cmd, optional) in enumerate(steps, 1):
            if not run_step(i, len(steps), title, cmd, fh, optional):
                sys.exit(1)
        total = time.time() - t_start
        log(f"\nAll {len(steps)} steps finished in {total / 60:.1f} min. "
            f"Full output saved to pipeline.log", fh)
        if os.path.exists(os.path.join(HERE, "report.html")) and not args.only_data:
            log("Diagnostics: open report.html in your browser.", fh)


if __name__ == "__main__":
    main()
