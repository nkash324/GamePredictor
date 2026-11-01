#!/usr/bin/env python3
"""
build_site.py  -  builds the public PatchCast website from the latest model results.

Writes into the website folder (default: ..\\PatchCastSite, next to this project):
  index.html   the page (copied from site\\index.html in this project)
  data.js      the latest forecast, patch changes and accuracy numbers
  .nojekyll    tells GitHub Pages to serve the files as they are

Nothing here calls the Riot API and no API key is involved: the site only shows
results that were already computed on this PC. Run it after update_model.bat
(publish_site.bat does this and then uploads the site to GitHub).

Usage:
  python build_site.py                    forecast for the newest patch in the data
  python build_site.py --patch 16.20      forecast for a specific patch
  python build_site.py --site-dir D:\\x    write the site somewhere else
"""

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_change_model as tcm  # noqa: E402

DEFAULT_SITE_DIR = os.path.join(os.path.dirname(HERE), "PatchCastSite")
MIN_GAMES_SHOWN = 200          # champions need this many games on a patch to show "actual"

# plain-English names for the model's inputs (used for "why" on the site)
REASON_NAMES = {
    "rel_total_dmg_base_rmax": "ability damage", "rel_total_dmg_ratio_ap": "AP scaling",
    "rel_total_dmg_ratio_ad_all": "AD scaling", "rel_hp_l9": "health", "rel_armor_l9": "armor",
    "rel_mr_l9": "magic resist", "rel_ad_l9": "attack damage", "rel_as_l9": "attack speed",
    "rel_basic_cd_rmax_mean": "ability cooldowns", "rel_ult_cd_rmax": "ultimate cooldown",
    "rel_total_heal_base_rmax": "healing", "rel_total_shield_base_rmax": "shielding",
    "buff_part": "overall buffs", "nerf_part": "overall nerfs", "dmg_power": "damage output",
    "surv_power": "survivability", "power_change": "overall power",
    "item_stat_buff": "item stat changes", "item_passive_buff": "item passive changes",
    "item_changed_share": "changed items in their builds", "raw_buff": "ability numbers",
    "raw_pve_buff": "jungle / minion numbers", "flags_gained": "new effects",
    "flags_lost": "removed effects", "prev_strength": "recent strength",
    "prev_presence": "popularity", "n_changes": "size of the change", "rework": "rework",
    "buff_x_strength": "buffs on an already strong champion",
}


STAT_NAMES = {
    "FlatHPPoolMod": "health", "FlatMPPoolMod": "mana", "FlatPhysicalDamageMod": "attack damage",
    "FlatMagicDamageMod": "ability power", "FlatArmorMod": "armor", "FlatSpellBlockMod": "magic resist",
    "PercentAttackSpeedMod": "attack speed", "FlatCritChanceMod": "crit chance",
    "FlatMovementSpeedMod": "move speed", "PercentMovementSpeedMod": "move speed %",
    "PercentLifeStealMod": "life steal", "FlatHPRegenMod": "health regen", "cost": "price",
}


def plain_changes(text):
    for k, v in STAT_NAMES.items():
        text = text.replace(k, v)
    return text.replace("->", " → ")


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


def reason_name(col):
    if col in REASON_NAMES:
        return REASON_NAMES[col]
    m = re.match(r"^(\w+)_x_(buff|item)$", col)
    if m:
        return f"{'item changes' if m.group(2) == 'item' else 'buffs'} for {m.group(1)}s"
    if col.startswith("via "):
        return "expected change in " + col[4:]
    return col.replace("_", " ")


def champion_names(db):
    """Display names (e.g. MonkeyKing -> Wukong) from Data Dragon data in the database."""
    names, version = {}, {}
    if not os.path.exists(db):
        return names, version
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
        version = dict(con.execute("SELECT patch, version FROM patches").fetchall())
        latest = max(version.values(), key=lambda v: tuple(int(x) for x in v.split(".")[:3]))
        names = dict(con.execute("SELECT champion_id, name FROM champions WHERE version = ?",
                                 (latest,)).fetchall())
        con.close()
    except Exception as e:
        print(f"  (couldn't read champion names from {db}: {e})")
    return names, version


def pretty(cid, names):
    if cid in names:
        return names[cid]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(cid))


def r2(v, nd=4):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


# --------------------------------------------------------------------------- #
def build_forecast(cp, X, target, names):
    ev = tcm.eval_rows(cp)
    train = ev[ev["patch"].map(pkey) < pkey(target)]
    past = sorted(train["patch"].unique(), key=pkey)
    if len(past) < tcm.MIN_TRAIN_PATCHES:
        sys.exit("Not enough earlier patches with games to make a forecast.")
    pr = tcm.Predictor(train, X, past)
    rows = cp[(cp["patch"] == target) & (cp["recent_games"].fillna(0) > 0)]
    base, best, bold = pr.parts(rows, X)
    contrib = pr.contributions(rows, X)
    # last patch's real winrate, for "predicted change" vs "actual change"
    prev = {(r.champion_id, r.patch): (r.winrate, r.games) for r in cp[["champion_id", "patch", "winrate", "games"]].itertuples()}
    out = []
    for i in rows.index:
        r = rows.loc[i]
        affected = int(r.get("affected", 0) == 1)
        reasons = []
        if affected and abs(bold[i]) > 0.002 and len(contrib.columns):
            c = contrib.loc[i]
            c = c[c.abs() > 1e-5]
            for col, v in c.reindex(c.abs().sort_values(ascending=False).index).head(3).items():
                reasons.append({"what": reason_name(col), "dir": 1 if v > 0 else -1})
        games = r.get("games")
        has_actual = pd.notna(games) and games >= MIN_GAMES_SHOWN
        last_wr, last_games = prev.get((r["champion_id"], r.get("prev_patch")), (None, None))
        has_last = last_wr is not None and pd.notna(last_wr) and pd.notna(last_games) and last_games >= MIN_GAMES_SHOWN
        out.append({
            "id": r["champion_id"], "name": pretty(r["champion_id"], names),
            "role": r.get("role") if isinstance(r.get("role"), str) else None,
            "changed": int(r.get("changed", 0) == 1), "items": int(affected and r.get("changed", 0) != 1),
            "expected": r2(base[i]), "predicted": r2(base[i] + best[i]),
            "impact": r2(bold[i]),
            "low": r2(base[i] + best[i] - 1.96 * pr.s["sd_true"]),
            "high": r2(base[i] + best[i] + 1.96 * pr.s["sd_true"]),
            "actual": r2(r["winrate"]) if has_actual else None,
            "games": int(games) if has_actual else None,
            "last": r2(last_wr) if has_last else None,
            "reasons": reasons,
        })
    return out, len(past), pr


def build_accuracy(pred_path):
    if not os.path.exists(pred_path):
        return None
    d = pd.read_csv(pred_path, dtype={"patch": str})
    d = d.dropna(subset=["pred_winrate", "winrate"])
    per = []
    for p, g in sorted(d.groupby("patch"), key=lambda kv: pkey(kv[0])):
        if len(g) < 20:
            continue
        r = float(np.corrcoef(g["pred_winrate"], g["winrate"])[0, 1])
        gl = g.dropna(subset=["last_patch_pred"])
        rl = float(np.corrcoef(gl["last_patch_pred"], gl["winrate"])[0, 1]) if len(gl) > 20 else None
        w = g["games"]
        per.append({"patch": p, "r": r2(r, 3), "r_last": r2(rl, 3),
                    "miss": r2(np.average((g["pred_winrate"] - g["winrate"]).abs(), weights=w) * 100, 2),
                    "miss_last": r2(np.average((gl["last_patch_pred"] - gl["winrate"]).abs(),
                                               weights=gl["games"]) * 100, 2) if len(gl) else None,
                    "luck": r2(np.average(g["noise_sd"], weights=w) * math.sqrt(2 / math.pi) * 100, 2)})
    last = sorted(d["patch"].unique(), key=pkey)[-1]
    lp = d[d["patch"] == last]
    scatter = [{"n": c, "p": r2(p), "a": r2(a), "g": int(g)} for c, p, a, g in
               zip(lp["champion_id"], lp["pred_winrate"], lp["winrate"], lp["games"])]
    # calibration: champions grouped by predicted winrate (1000+ games)
    big = d[d["games"] >= 1000].copy()
    if len(big) < 120:
        big = d.copy()
    nb = 12 if len(big) >= 120 else max(3, len(big) // 10)
    big["b"] = pd.qcut(big["pred_winrate"].rank(method="first"), nb, labels=False)
    calib = []
    for _, g in big.groupby("b"):
        w = g["games"]
        calib.append({"p": r2(np.average(g["pred_winrate"], weights=w)),
                      "a": r2(np.average(g["winrate"], weights=w)),
                      "se": r2(math.sqrt(np.sum(w ** 2 * g["noise_sd"] ** 2)) / w.sum()), "n": len(g)})
    tot_r = float(np.corrcoef(d["pred_winrate"], d["winrate"])[0, 1])
    dl = d.dropna(subset=["last_patch_pred"])
    ch = d[(d["changed"] == 1) & (d["games"] >= 100) & (d["bold"].abs() > 0.005)] if "bold" in d else d.iloc[:0]
    return {
        "patches_tested": len(per), "r": r2(tot_r, 3),
        "r_last": r2(float(np.corrcoef(dl["last_patch_pred"], dl["winrate"])[0, 1]), 3),
        "miss": r2(np.average((d["pred_winrate"] - d["winrate"]).abs(), weights=d["games"]) * 100, 2),
        "miss_last": r2(np.average((dl["last_patch_pred"] - dl["winrate"]).abs(), weights=dl["games"]) * 100, 2),
        "luck": r2(np.average(d["noise_sd"], weights=d["games"]) * math.sqrt(2 / math.pi) * 100, 2),
        "beat_last": int(sum(1 for x in per if x["miss_last"] is not None and x["miss"] < x["miss_last"])),
        "direction_hits": int((np.sign(ch["bold"]) == np.sign(ch["actual"])).sum()) if len(ch) else None,
        "direction_n": int(len(ch)),
        "per_patch": per, "scatter_patch": last, "scatter": scatter, "calibration": calib,
    }


def build_items(path, target):
    if not os.path.exists(path):
        return []
    ic = pd.read_csv(path, dtype={"patch": str, "item_id": str})
    ic = ic[ic["patch"] == target]
    out = []
    for _, r in ic.iterrows():
        size = (r.get("stat_pct", 0) or 0) + (r.get("passive_pct", 0) or 0)
        out.append({"name": r.get("name", r["item_id"]), "id": r["item_id"],
                    "size": r2(size * 100, 1), "changes": plain_changes(r["changes"]) if isinstance(r.get("changes"), str) else "",
                    "users": r.get("main_users") if isinstance(r.get("main_users"), str) else ""})
    return sorted(out, key=lambda x: -abs(x["size"] or 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the public PatchCast website files.")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--db", default="lol_data.db")
    ap.add_argument("--site-dir", default=DEFAULT_SITE_DIR)
    ap.add_argument("--patch", default=None, help="forecast this patch (default: newest)")
    args = ap.parse_args(argv)
    os.chdir(HERE)

    cp, X = tcm.build_features(tcm.load(args.dir))
    patches = sorted(cp["patch"].unique(), key=pkey)
    target = args.patch or patches[-1]
    names, versions = champion_names(args.db)
    ddragon = versions.get(target) or versions.get(patches[-1]) or f"{target}.1"
    print(f"Building the forecast for patch {target} ...")
    forecast, n_past, pr = build_forecast(cp, X, target, names)
    data = {
        "patch": target, "ddragon": ddragon,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "trained_patches": n_past,
        "impact_on": bool(pr.s["on"]),
        "forecast": forecast,
        "items": build_items(os.path.join(args.dir, "item_changes.csv"), target),
        "accuracy": build_accuracy(os.path.join(args.dir, "change_model_predictions.csv")),
    }
    os.makedirs(args.site_dir, exist_ok=True)
    with open(os.path.join(args.site_dir, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.PATCHCAST = ")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    shutil.copyfile(os.path.join(HERE, "site", "index.html"), os.path.join(args.site_dir, "index.html"))
    open(os.path.join(args.site_dir, ".nojekyll"), "w").close()
    n_ch = sum(1 for c in forecast if c["changed"])
    print(f"Saved the website to {os.path.abspath(args.site_dir)}")
    print(f"  {len(forecast)} champions ({n_ch} changed this patch), {len(data['items'])} item changes, "
          f"accuracy from {data['accuracy']['patches_tested'] if data['accuracy'] else 0} tested patches")
    print("  Open index.html there to preview it, then run publish_site.bat to put it online.")


if __name__ == "__main__":
    main()
