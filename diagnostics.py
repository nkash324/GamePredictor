#!/usr/bin/env python3
"""
diagnostics.py  -  builds report.html: where the models are right, where they miss, and why.

Reads what the other steps saved (nothing is retrained, takes seconds):
  training/champion_patch.parquet        champion winrates per patch
  training/change_model_predictions.csv  every champion-patch prediction of the change model
  training/change_model_results.csv      per-patch scores of the change model
  training/change_model_weights.csv      what the change model learned
  training/baseline_results.csv          simple predictions' scores
  training/model_results.csv             game model per-patch scores
  training/game_test_predictions.parquet every test game's predicted win chance
  training/game_model_effects.csv        game model's predicted patch effect per champion
  models/importance_numbers.csv          what the game model relies on

Key idea used throughout - separating LUCK from REAL mistakes:
  A champion with 300 games has a winrate that wobbles about +-2.9 points by pure chance.
  So a 3-point miss there is normal. For every prediction:
     z = miss / expected luck wobble
  |z| below 1 happens ~68% of the time and below 2 ~95% of the time from luck alone.
  If the model misses more than that, the extra is the model's own error.
  "Own error" = sqrt(average squared miss - average squared luck wobble).

Usage:  python diagnostics.py     then open report.html (update_model.bat does both)
"""

import argparse
import html
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read(path):
    for p in (path, path.replace(".parquet", ".csv.gz")):
        if os.path.exists(p):
            try:
                # patch must stay text: as a number, 16.10 would turn into 16.1
                return (pd.read_parquet(p) if p.endswith(".parquet")
                        else pd.read_csv(p, dtype={"patch": str}))
            except Exception as e:  # unreadable file: report it, keep going
                print(f"  could not read {p}: {e}")
                return None
    return None


def pkey(p):
    return tuple(int(x) for x in str(p).split(".")[:2])


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
def wavg(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    m = ~np.isnan(v)
    return float(np.sum(v[m] * w[m]) / np.sum(w[m])) if m.any() and np.sum(w[m]) > 0 else np.nan


def error_summary(d):
    """Miss, luck, own error, bias for a group of predictions (points)."""
    w = d["games"]
    miss = wavg(d["error"].abs(), w)
    mse = wavg(d["error"] ** 2, w)
    luck2 = wavg(d["noise_sd"] ** 2, w)
    return pd.Series({
        "n": len(d),
        "games": int(w.sum()),
        "miss": miss * 100,
        "luck": wavg(d["noise_sd"], w) * math.sqrt(2 / math.pi) * 100,   # expected miss from luck
        "own error": math.sqrt(max(mse - luck2, 0)) * 100,
        "bias": wavg(d["error"], w) * 100,
        "within 2 sd": float(np.mean(d["z"].abs() < 2)) * 100,
    })


def group_table(d, by, order=None):
    t = d.groupby(by, observed=True).apply(error_summary)
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    return t


# --------------------------------------------------------------------------- #
# SVG charts (no extra libraries needed)
# --------------------------------------------------------------------------- #
# every chart in the report is the same size and shape (square), so they're easy to compare
W, H, PAD = 460, 460, 58


def _scale(lo, hi, a, b):
    span = (hi - lo) or 1
    return lambda v: a + (v - lo) / span * (b - a)


def _axes(xlo, xhi, ylo, yhi, xlab, ylab, xt=None, yt=None, fmt="{:g}", w=None, h=None):
    W, H = (w or globals()["W"]), (h or globals()["H"])
    sx, sy = _scale(xlo, xhi, PAD, W - 12), _scale(ylo, yhi, H - PAD + 10, 12)
    out = [f'<line x1="{PAD}" y1="{H - PAD + 10}" x2="{W - 12}" y2="{H - PAD + 10}" class="ax"/>',
           f'<line x1="{PAD}" y1="12" x2="{PAD}" y2="{H - PAD + 10}" class="ax"/>']
    for v in (xt if xt is not None else np.linspace(xlo, xhi, 5)):
        out.append(f'<text x="{sx(v):.1f}" y="{H - PAD + 26}" class="tk" text-anchor="middle">'
                   f'{fmt.format(v)}</text>')
    for v in (yt if yt is not None else np.linspace(ylo, yhi, 5)):
        out.append(f'<line x1="{PAD}" y1="{sy(v):.1f}" x2="{W - 12}" y2="{sy(v):.1f}" class="gr"/>')
        out.append(f'<text x="{PAD - 6}" y="{sy(v) + 4:.1f}" class="tk" text-anchor="end">'
                   f'{fmt.format(v)}</text>')
    out.append(f'<text x="{(PAD + W) / 2}" y="{H - 4}" class="lb" text-anchor="middle">'
               f'{html.escape(xlab)}</text>')
    out.append(f'<text x="12" y="{H / 2}" class="lb" text-anchor="middle" '
               f'transform="rotate(-90 12 {H / 2})">{html.escape(ylab)}</text>')
    return sx, sy, out


def svg_wrap(parts, w=None, h=None):
    return (f'<svg viewBox="0 0 {w or W} {h or H}" class="chart{" sq" if w else ""}" role="img">'
            f'{"".join(parts)}</svg>')


def svg_scatter(x, y, size, labels, xlab, ylab, diag=True, lim=None, same_scale=True):
    x, y, size = np.asarray(x, float), np.asarray(y, float), np.asarray(size, float)
    if not len(x):
        return "<p class='muted'>No data yet.</p>"
    if same_scale:
        m = float(np.nanpercentile(np.abs(np.concatenate([x, y])), 99)) if lim is None else lim
        mx = my = max(math.ceil(m), 1)
    else:          # different units on the two axes: each axis fits its own data
        mx = max(math.ceil(float(np.nanpercentile(np.abs(x), 99))), 1)
        my = max(math.ceil(float(np.nanpercentile(np.abs(y), 99))), 1)
    x, y = np.clip(x, -mx, mx), np.clip(y, -my, my)
    lo, hi = -mx, mx
    sx, sy, parts = _axes(-mx, mx, -my, my, xlab, ylab, xt=np.linspace(-mx, mx, 5),
                          yt=np.linspace(-my, my, 5), fmt="{:+.0f}")
    if diag:
        parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" '
                     f'y2="{sy(hi):.1f}" class="diag"/>')
    parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(0):.1f}" x2="{sx(hi):.1f}" y2="{sy(0):.1f}" '
                 f'class="zero"/>' if lo < 0 < hi else "")
    r = 2.5 + 5 * np.sqrt(size / (size.max() or 1))
    for xi, yi, ri, lab in zip(x, y, r, labels):
        parts.append(f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="{ri:.1f}" class="pt">'
                     f'<title>{html.escape(str(lab))}</title></circle>')
    return svg_wrap(parts)


def svg_calibration(px, py, perr, xlab, ylab, lo, hi, fmt="{:g}"):
    """Binned predicted (x) vs actual (y) with +-2 sd error bars; perfect = diagonal."""
    sx, sy, parts = _axes(lo, hi, lo, hi, xlab, ylab, fmt=fmt)
    parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" '
                 f'class="diag"/>')
    for x, y, e in zip(px, py, perr):
        yl, yh = max(y - 2 * e, lo), min(y + 2 * e, hi)
        parts.append(f'<line x1="{sx(x):.1f}" y1="{sy(yl):.1f}" x2="{sx(x):.1f}" y2="{sy(yh):.1f}" '
                     f'class="eb"/>')
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(min(max(y, lo), hi)):.1f}" r="5" class="pt2">'
                     f'<title>predicted {fmt.format(x)}, actual {fmt.format(y)} (+-{fmt.format(2 * e)})'
                     f'</title></circle>')
    return svg_wrap(parts)


def svg_z_hist(z):
    z = np.clip(np.asarray(z, float), -4.99, 4.99)
    if not len(z):
        return "<p class='muted'>No data yet.</p>"
    bins = np.linspace(-5, 5, 21)
    cnt, _ = np.histogram(z, bins)
    share = cnt / len(z)
    normal = np.array([0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
                       for a, b in zip(bins[:-1], bins[1:])])
    top = max(share.max(), normal.max()) * 1.1
    sx, sy, parts = _axes(-5, 5, 0, top, "miss divided by luck wobble (z)", "share of predictions",
                          xt=[-4, -2, 0, 2, 4], yt=np.linspace(0, top, 5), fmt="{:.2g}")
    for a, b, s in zip(bins[:-1], bins[1:], share):
        cls = "bar" if abs((a + b) / 2) < 2 else "bar bad"
        parts.append(f'<rect x="{sx(a) + 1:.1f}" y="{sy(s):.1f}" width="{sx(b) - sx(a) - 2:.1f}" '
                     f'height="{sy(0) - sy(s):.1f}" class="{cls}"><title>{s:.1%}</title></rect>')
    pts = " ".join(f"{sx((a + b) / 2):.1f},{sy(n):.1f}" for a, b, n in zip(bins[:-1], bins[1:], normal))
    parts.append(f'<polyline points="{pts}" class="curve"/>')
    return svg_wrap(parts)


def svg_bars(labels, values, xlab, fmt="{:.1f}", highlight=None):
    if not len(values):
        return "<p class='muted'>No data yet.</p>"
    n = len(values)
    h = H                                   # same size as every other chart
    row = (h - 40) / max(n, 1)              # rows shrink to fit
    bar_h = max(min(row * 0.72, 16), 3)
    vmax = max(max(values), 1e-9)
    left = 190
    parts = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = 10 + i * row
        wbar = (W - left - 50) * v / vmax
        cls = "bar bad" if highlight and highlight(lab) else "bar"
        parts.append(f'<text x="{left - 6}" y="{y + bar_h - 2:.1f}" class="tk" text-anchor="end">'
                     f'{html.escape(str(lab))[:28]}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{max(wbar, 1):.1f}" height="{bar_h:.1f}" '
                     f'class="{cls}"/>')
        parts.append(f'<text x="{left + wbar + 4:.1f}" y="{y + bar_h - 2:.1f}" class="tk">'
                     f'{fmt.format(v)}</text>')
    parts.append(f'<text x="{(left + W) / 2}" y="{h - 6}" class="lb" text-anchor="middle">'
                 f'{html.escape(xlab)}</text>')
    return f'<svg viewBox="0 0 {W} {h}" class="chart" role="img">{"".join(parts)}</svg>'


def svg_lines(x_labels, series, ylab, ref=None):
    """series: {name: values}; x is categorical (patches)."""
    vals = np.concatenate([np.asarray(v, float) for v in series.values()] +
                          ([np.asarray(ref[1], float)] if ref else []))
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return "<p class='muted'>No data yet.</p>"
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.1 or 0.1
    lo, hi = lo - pad, hi + pad
    n = len(x_labels)
    sx, sy, parts = _axes(0, max(n - 1, 1), lo, hi, "patch", ylab, xt=[], fmt="{:.2f}")
    step = max(1, n // 10)
    for i, lab in enumerate(x_labels):
        if i % step == 0:
            parts.append(f'<text x="{sx(i):.1f}" y="{H - PAD + 26}" class="tk" '
                         f'text-anchor="middle">{html.escape(str(lab))}</text>')
    items = list(series.items()) + ([(ref[0], ref[1])] if ref else [])
    for j, (name, v) in enumerate(items):
        is_ref = ref and name == ref[0]
        pts = " ".join(f"{sx(i):.1f},{sy(val):.1f}" for i, val in enumerate(v) if not np.isnan(val))
        parts.append(f'<polyline points="{pts}" class="{"refl" if is_ref else f"ln ln{j}"}"/>')
        parts.append(f'<text x="{W - 14}" y="{20 + 14 * j}" text-anchor="end" '
                     f'class="{"tk refc" if is_ref else f"tk c{j}"}">{html.escape(name)}</text>')
    return svg_wrap(parts)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def table(df, fmts=None, color=None, index=True, max_rows=60):
    """color: {column: fn(value) -> css class}"""
    if df is None or not len(df):
        return "<p class='muted'>No data yet.</p>"
    fmts, color = fmts or {}, color or {}
    df = df.head(max_rows)
    cols = list(df.columns)
    isnum = {c: pd.api.types.is_numeric_dtype(df[c]) for c in cols}
    head = (f"<th class='l'>{html.escape(str(df.index.name or ''))}</th>" if index else "") + "".join(
        f"<th class='{'' if isnum[c] else 'l'}'>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<td class='l'>{html.escape(str(idx))}</td>"] if index else []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                s = "-"
            elif c in fmts:
                s = fmts[c].format(v)
            elif isinstance(v, (float, np.floating)):
                s = f"{v:.2f}"
            else:
                s = str(v)
            cls = color[c](v) if c in color and not (isinstance(v, float) and np.isnan(v)) else ""
            cells.append(f"<td class='{cls} {'' if isnum[c] else 'l'}'>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


ERR_FMT = {"n": "{:,.0f}", "games": "{:,.0f}", "miss": "{:.2f}", "luck": "{:.2f}",
           "own error": "{:.2f}", "bias": "{:+.2f}", "within 2 sd": "{:.0f}%"}


def own_color(v):
    return "bad" if v >= 1.5 else ("warn" if v >= 0.75 else "good")


def within_color(v):
    return "bad" if v < 88 else ("warn" if v < 93 else "good")


def bias_color(v):
    return "bad" if abs(v) >= 1 else ("warn" if abs(v) >= 0.5 else "")


ERR_COLOR = {"own error": own_color, "within 2 sd": within_color, "bias": bias_color}


def card(label, value, sub=""):
    return (f"<div class='card'><div class='cl'>{html.escape(label)}</div>"
            f"<div class='cv'>{html.escape(value)}</div><div class='cs'>{html.escape(sub)}</div></div>")


def section(sid, title, intro, body):
    return (f"<section id='{sid}'><h2>{html.escape(title)}</h2>"
            f"<p class='intro'>{intro}</p>{body}</section>")


def note(text, kind="info"):
    return f"<div class='note {kind}'>{text}</div>"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def games_bucket(g):
    return pd.cut(g, [0, 99, 299, 999, 1e9], labels=["30-99", "100-299", "300-999", "1000+"])


def strength_bucket(r):
    return pd.cut(r, [0, 0.48, 0.495, 0.505, 0.52, 1],
                  labels=["weak (<48%)", "a bit weak", "average", "a bit strong", "strong (>52%)"])


STRENGTH_EFFECT_SCALE = None   # None = learn how much to trust the strength model's effects


def svg_pred_actual(x, y, size, labels, bins=None, faint=False):
    """Predicted (x) vs actual (y) winrate %. Axes fit the data; dashed = perfect."""
    x, y, size = (np.asarray(v, float) for v in (x, y, size))
    if len(x) < 3:
        return "<p class='muted'>No data yet.</p>"
    # SAME range and scale on both axes (square chart), so nothing is stretched: 1 point
    # up = 1 point across. Predictions spread less than actual winrates (luck adds spread
    # to the actual ones), so the cloud is naturally taller than wide.
    lo = min(np.nanpercentile(x, 0.5), np.nanpercentile(y, 0.5)) - 0.5
    hi = max(np.nanpercentile(x, 99.5), np.nanpercentile(y, 99.5)) + 0.5
    mid, half = (lo + hi) / 2, max(hi - lo, 4.0) / 2
    lo, hi = mid - half, mid + half
    xlo, xhi, ylo, yhi = lo, hi, lo, hi
    step = 2.0 if hi - lo > 8 else 1.0
    ticks = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
    side = W
    sx, sy, parts = _axes(lo, hi, lo, hi, "predicted winrate %", "actual winrate %",
                          xt=ticks, yt=ticks, fmt="{:.0f}", w=side, h=side)
    parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" '
                 f'class="diag"><title>perfect prediction</title></line>')
    keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    r = 2.0 + 6 * np.sqrt(size / (np.nanmax(size) or 1))
    for xi, yi, ri, lab, k in zip(x, y, r, labels, keep):
        if k:
            cls = "ptf" if faint else "pt"
            rad = 1.6 if faint else ri
            parts.append(f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="{rad:.1f}" class="{cls}">'
                         f'<title>{html.escape(str(lab))}</title></circle>')
    if bins is not None:
        for bx, by, be in bins:
            parts.append(f'<line x1="{sx(bx):.1f}" y1="{sy(by - 2 * be):.1f}" x2="{sx(bx):.1f}" '
                         f'y2="{sy(by + 2 * be):.1f}" class="eb"/>')
            parts.append(f'<circle cx="{sx(bx):.1f}" cy="{sy(by):.1f}" r="5" class="pt2"><title>'
                         f'predicted {bx:.2f}%, actual {by:.2f}% (+-{2 * be:.2f})</title></circle>')
    else:     # trend line (games-weighted fit), over the range of the predictions
        w = np.sqrt(size)
        b1, b0 = np.polyfit(x, y, 1, w=w)
        x0, x1 = np.nanmin(x), np.nanmax(x)
        parts.append(f'<line x1="{sx(x0):.1f}" y1="{sy(b0 + b1 * x0):.1f}" x2="{sx(x1):.1f}" '
                     f'y2="{sy(b0 + b1 * x1):.1f}" class="trend"><title>trend</title></line>')
    return svg_wrap(parts, side, side)


def prediction_table(d, strength_det):
    """One predicted winrate per champion-patch: expected winrate + the patch effect.
    The effect comes from the strength model (scaled) when it's available."""
    m = d.copy()
    m["effect_used"] = m["predicted"]
    source = "change model"
    if strength_det is not None and len(strength_det) and "effect" in strength_det:
        s = strength_det[["patch", "champion_id", "effect"]].copy()
        s["patch"] = s["patch"].astype(str)
        m = m.merge(s, on=["patch", "champion_id"], how="left")
        has = m["effect"].notna() & (m["changed"] == 1)
        scale = STRENGTH_EFFECT_SCALE
        if scale is None:     # how big the strength model's effects turn out, on average
            t = m[has]
            den = float(np.sum(t["games"] * t["effect"] ** 2))
            scale = float(np.clip(np.sum(t["games"] * t["effect"] * t["actual"]) / den, 0, 1)) if den > 0 else 0.0
        m.loc[has, "effect_used"] = scale * m.loc[has, "effect"]
        source = f"strength model effects x{scale:.2f}"
    m["final_pred"] = m["base_pred"] + m["effect_used"]
    return m, source


def sec_pred_vs_actual(d, strength_det, n_charts=5):
    m, source = prediction_table(d, strength_det)
    m["p"], m["a"] = m["final_pred"] * 100, m["winrate"] * 100
    corr = lambda t: float(np.corrcoef(t["p"], t["a"])[0, 1]) if len(t) > 2 else np.nan
    patches = sorted(m["patch"].unique(), key=pkey)
    last = patches[-1]
    x = m[m["patch"] == last]
    lab = lambda t: (t["champion_id"] + " " + t["patch"] + ": predicted " + t["p"].round(1).astype(str)
                     + "%, actual " + t["a"].round(1).astype(str) + "% (" + t["games"].astype(int).astype(str)
                     + " games)")
    left = svg_pred_actual(x["p"], x["a"], x["games"], lab(x))
    y = m[m["games"] >= 1000]
    if len(y) < 50:
        y = m
    nb = 12 if len(y) >= 120 else max(3, len(y) // 10)
    y = y.assign(b=pd.qcut(y["p"].rank(method="first"), nb, labels=False))
    bins = []
    for _, t in y.groupby("b"):
        w = t["games"]
        bins.append((np.average(t["p"], weights=w), np.average(t["a"], weights=w),
                     math.sqrt(np.sum(w ** 2 * (t["noise_sd"] * 100) ** 2)) / w.sum()))
    right = svg_pred_actual(y["p"], y["a"], y["games"], lab(y), bins=bins, faint=True)
    per = pd.DataFrame({"champions": m.groupby("patch").size(),
                        "r (all)": m.groupby("patch").apply(corr),
                        "r (1000+ games)": m[m["games"] >= 1000].groupby("patch").apply(corr)})
    per = per.loc[sorted(per.index, key=pkey)]
    s_cur = m.dropna(subset=["last_patch_pred"])
    base_r = float(np.corrcoef(s_cur["last_patch_pred"] * 100, s_cur["a"])[0, 1])
    body = (f"<div class='grid2'><div><h3>Patch {last}: every champion "
            f"(r = {corr(x):.2f})</h3>{left}<p class='cap'>One dot per champion, size = games. "
            f"Dashed = perfect prediction, orange = trend. Hover a dot for the champion.</p></div>"
            f"<div><h3>All tested patches, champions with 1000+ games (r = {corr(y):.2f})</h3>{right}"
            f"<p class='cap'>Grey = each champion on each patch. Blue = the champions split into "
            f"{nb} groups by predicted winrate, with their average actual winrate (bars = 2 luck "
            f"wobbles). Blue dots on the dashed line = the predictions are right on average.</p>"
            f"</div></div>")
    body += note(f"<b>r</b> measures how well predicted and actual winrates line up: 0 = not at all, "
                 f"1 = perfectly. Here {corr(m):.2f} for all champions (last patch's winrate alone: "
                 f"{base_r:.2f}). It can't reach 1: a champion's actual winrate on one patch includes "
                 f"luck, which no prediction can know. More games per patch raise the ceiling. "
                 f"Prediction = expected winrate + patch effect ({source}).")
    if n_charts > 0:
        charts = []
        for p in patches[-n_charts:][::-1]:
            t = m[m["patch"] == p]
            charts.append(f"<div><h3>Patch {p} (r = {corr(t):.2f})</h3>"
                          f"{svg_pred_actual(t['p'], t['a'], t['games'], lab(t))}</div>")
        body += (f"<h3 style='margin-top:26px'>Last {len(charts)} patches</h3>"
                 f"<div class='grid2'>{''.join(charts)}</div><p class='cap'>Change how many with "
                 f"<code>python diagnostics.py --patch-charts 8</code> (or update_model.bat "
                 f"--patch-charts 8).</p>")
    body += "<h3>Per patch</h3>" + table(per.rename_axis("patch"),
                                         {"champions": "{:.0f}", "r (all)": "{:.2f}",
                                          "r (1000+ games)": "{:.2f}"})
    return section("pva", "Predicted vs actual winrate",
                   "Every champion's predicted winrate on a patch (made before the patch, from "
                   "earlier patches only) against what actually happened.", body)


def sec_luck(d):
    cov1, cov2 = float(np.mean(d["z"].abs() < 1)) * 100, float(np.mean(d["z"].abs() < 2)) * 100
    s = error_summary(d)
    verdict = ("The misses are about as big as luck alone explains - with this many games per "
               "champion, the model can't do much better. More games per patch is the fix."
               if s["own error"] < 0.75 else
               "The misses are bigger than luck explains: part of the error is the model's own. "
               "The sections below show where.")
    body = (f"<div class='grid2'><div>{svg_z_hist(d['z'])}"
            f"<p class='cap'>Bars: how far off each prediction was, in units of luck wobble. The "
            f"line is what pure luck would give. Red bars (beyond 2) are misses luck rarely "
            f"explains.</p></div><div>"
            + table(pd.DataFrame({"luck alone": ["68%", "95%"],
                                  "this model": [f"{cov1:.0f}%", f"{cov2:.0f}%"]},
                                 index=pd.Index(["within 1 wobble", "within 2 wobbles"], name="")))
            + note(verdict, "good" if s["own error"] < 0.75 else "warn")
            + "</div></div>")
    worst = d.reindex(d["z"].abs().sort_values(ascending=False).index).head(25)
    worst = worst.assign(**{"pred %": worst["pred_winrate"] * 100, "actual %": worst["winrate"] * 100,
                            "miss": worst["error"] * 100})
    cols = ["patch", "champion_id", "role", "games", "changed", "pred %", "actual %", "miss", "z", "why"]
    body += ("<h3>Biggest misses beyond luck</h3><p class='cap'>Sorted by z. A champion listed "
             "here several times is a pattern (see 'Champions the model keeps getting wrong').</p>"
             + table(worst[cols].set_index("patch"), {"games": "{:,.0f}", "changed": "{:.0f}",
                                                      "pred %": "{:.1f}", "actual %": "{:.1f}",
                                                      "miss": "{:+.1f}", "z": "{:+.1f}"},
                     {"z": lambda v: "bad" if abs(v) >= 3 else "warn"}))
    return section("luck", "1. Luck or real mistakes?",
                   "Every champion's winrate on a patch wobbles by chance, more when it has few "
                   "games. Here each miss is compared with that wobble.", body)


def sec_breakdown(d, X_info):
    d = d.copy()
    d["games bucket"] = games_bucket(d["games"])
    d["strength"] = strength_bucket(d["recent_winrate"])
    d["kind"] = np.where(d["rework"] > 0, "rework", np.where(d["changed"] == 1, "changed", "unchanged"))
    b = d["buff_score"].fillna(0)
    d["change size"] = np.select([d["changed"] != 1, b <= -0.15, b < -0.02, b <= 0.02, b < 0.15],
                                 ["not changed", "big nerf", "small nerf", "mixed / neutral",
                                  "small buff"], "big buff")
    parts = []
    for by, title, order in [
        ("kind", "Changed or not", ["unchanged", "changed", "rework"]),
        ("change size", "Size of the balance change (buff score)",
         ["big nerf", "small nerf", "mixed / neutral", "small buff", "big buff", "not changed"]),
        ("role", "Role", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]),
        ("class", "Class", None),
        ("games bucket", "Games played on the patch", ["30-99", "100-299", "300-999", "1000+"]),
        ("strength", "How strong the champion was coming in",
         ["weak (<48%)", "a bit weak", "average", "a bit strong", "strong (>52%)"]),
    ]:
        t = group_table(d, by, order)
        parts.append(f"<h3>{html.escape(title)}</h3>" + table(t, ERR_FMT, ERR_COLOR))
    legend = ("<b>miss</b>: average miss in winrate points. <b>luck</b>: the miss a perfect model "
              "would still have. <b>own error</b>: the model's part after removing luck (green "
              "under 0.75, red over 1.5). <b>bias</b>: + means predicted too high on average, - too "
              "low. <b>within 2 sd</b>: should be about 95%.")
    return section("where", "2. Where the misses happen",
                   "The same predictions, split into groups. A group with high own error or "
                   "strong bias is where the model is weakest. " + legend, "".join(parts))


def sec_champions(d):
    g = d.groupby("champion_id")
    t = pd.DataFrame({
        "patches": g.size(),
        "games": g["games"].sum(),
        "avg z": g["z"].mean(),
        "bias": g.apply(lambda x: wavg(x["error"], x["games"]) * 100),
        "times changed": g["changed"].sum(),
    })
    t["pattern strength"] = t["avg z"] * np.sqrt(t["patches"])
    t = t[t["patches"] >= 3]
    t = t.reindex(t["pattern strength"].abs().sort_values(ascending=False).index)
    t["verdict"] = np.where(t["pattern strength"].abs() >= 3,
                            np.where(t["bias"] > 0, "rated too high", "rated too low"),
                            np.where(t["pattern strength"].abs() >= 2, "maybe", "ok"))
    t = t.rename_axis("champion")
    body = table(t.head(30), {"patches": "{:.0f}", "games": "{:,.0f}", "avg z": "{:+.2f}",
                              "bias": "{:+.2f}", "times changed": "{:.0f}",
                              "pattern strength": "{:+.1f}"},
                 {"pattern strength": lambda v: "bad" if abs(v) >= 3 else
                  ("warn" if abs(v) >= 2 else "")})
    body += note("A champion missed in the <b>same direction</b> patch after patch is a real "
                 "pattern, not luck. The usual causes: something the numbers don't capture (a "
                 "unique mechanic, an item or rune that suits them, a pro-play trend), or a "
                 "parsing problem in their ability data. Check their numbers with "
                 "<code>python fetch_ability_data.py --inspect NAME</code>.")
    return section("champs", "3. Champions the model keeps getting wrong",
                   "Champions with 3+ patches, sorted by how consistent their misses are. "
                   "<b>pattern strength</b> above 3 (or below -3) is very unlikely to be luck.",
                   body)


def calib_bins(pred, actual, w, noise_var, nbins=8):
    df = pd.DataFrame({"p": pred, "a": actual, "w": w, "v": noise_var}).dropna()
    if len(df) < nbins * 3:
        nbins = max(2, len(df) // 3)
    if len(df) < 4:
        return [], [], []
    df["b"] = pd.qcut(df["p"].rank(method="first"), nbins, labels=False)
    px, py, pe = [], [], []
    for _, g in df.groupby("b"):
        px.append(wavg(g["p"], g["w"]))
        py.append(wavg(g["a"], g["w"]))
        # standard error of a games-weighted mean of noisy winrates
        pe.append(math.sqrt(np.sum(g["w"] ** 2 * g["v"]) / np.sum(g["w"]) ** 2 + 1e-12))
    return px, py, pe


def sec_calibration(d):
    parts = []
    px, py, pe = calib_bins(d["base_pred"] * 100, d["winrate"] * 100, d["games"],
                            (d["noise_sd"] * 100) ** 2)
    lo = min(px + py) - 1 if px else 45
    hi = max(px + py) + 1 if px else 55
    parts.append("<div class='grid2'><div><h3>Starting point: expected strength</h3>"
                 + svg_calibration(px, py, pe, "predicted winrate %", "actual winrate %", lo, hi,
                                   "{:.1f}")
                 + "<p class='cap'>Champions grouped by predicted winrate (before balance "
                   "changes). On the diagonal = right on average. Flatter than the diagonal = the "
                   "model spreads champions too far from 50%; steeper = not far enough.</p></div>")
    eff = "bold" if "bold" in d else "predicted"
    ch = d[(d["changed"] == 1) & (d[eff].abs() > 0)]
    px, py, pe = calib_bins(ch[eff] * 100, ch["actual"] * 100, ch["games"],
                            (ch["noise_sd"] * 100) ** 2, nbins=5)
    if px:
        m = max(abs(v) for v in px + py) + 1
        parts.append(f"<div><h3>Balance-change effect ({'bold' if eff == 'bold' else 'best guess'})</h3>"
                     + svg_calibration(px, py, pe, "predicted effect (points)",
                                       "actual change beyond expected strength", -m, m, "{:+.1f}")
                     + "<p class='cap'>Changed champions grouped by predicted effect. Bars = 2 "
                       "luck wobbles. Points rising left to right = the direction is right. Bold "
                       "predictions are sized like real swings, so they're expected to overshoot "
                       "(flatter than the diagonal) while the signal is weak.</p></div></div>")
    else:
        parts.append("<div>" + note("No balance-change effects predicted yet.") + "</div></div>")
    allch = d[d["changed"] == 1]
    parts.append("<h3>Every changed champion: predicted vs actual change</h3>"
                 + svg_scatter(allch[eff] * 100, allch["actual"] * 100, allch["games"],
                               allch["patch"].astype(str) + " " + allch["champion_id"] + ": predicted "
                               + (allch[eff] * 100).round(1).astype(str) + ", actual "
                               + (allch["actual"] * 100).round(1).astype(str),
                               "predicted effect (points)", "actual change beyond expected (points)")
                 + "<p class='cap'>Dot size = games. Hover a dot for the champion. Points on the "
                   "diagonal are perfect. Top-right and bottom-left = right direction.</p>")
    return section("calib", "4. Are the predictions the right size?",
                   "Calibration: when the model says 52%, does it end up 52% on average? When it "
                   "says +2 points, do champions really gain 2?", "".join(parts))


def sec_bold(d, cres):
    if "bold" not in d:
        return ""
    ch = d[(d["changed"] == 1) & (d["bold"].abs() > 0.005)].copy()
    if not len(ch):
        return section("bold", "5. Bold predictions", "", note("No bold predictions yet."))
    ch["right"] = np.sign(ch["bold"]) == np.sign(ch["actual"])
    ch["size"] = pd.cut(ch["bold"].abs() * 100, [0, 1, 2, 4, 100],
                        labels=["under 1 pt", "1-2 pts", "2-4 pts", "4+ pts"])
    ch["games bucket"] = games_bucket(ch["games"])
    parts = []
    rows = []
    for label, g in [("all changed champions", ch),
                     ("with 300+ games", ch[ch["games"] >= 300])] + \
                    [(f"predicted {s}", g) for s, g in ch.groupby("size", observed=True)]:
        if len(g):
            rows.append({"group": label, "n": len(g), "direction right": g["right"].mean() * 100,
                         "avg predicted": g["bold"].abs().mean() * 100,
                         "avg actual move": g["actual"].abs().mean() * 100})
    t = pd.DataFrame(rows).set_index("group")
    parts.append(table(t, {"n": "{:.0f}", "direction right": "{:.0f}%",
                           "avg predicted": "{:.1f}", "avg actual move": "{:.1f}"},
                       {"direction right": lambda v: "good" if v >= 60 else
                        ("warn" if v >= 52 else "bad")}))
    parts.append("<p class='cap'>50% = coin flip. If bigger predictions are right more often, "
                 "the model knows which changes matter. <b>avg actual move</b> includes luck, so "
                 "it's always a bit larger than the real effect.</p>")
    if cres is not None and "top3 direction" in cres:
        n = cres["top3 n"].sum()
        rate = np.nansum(cres["top3 direction"] * cres["top3 n"]) / max(n, 1) * 100
        parts.append(note(f"The 3 champions the model expected to move most each patch (100+ "
                          f"games): direction right <b>{rate:.0f}%</b> of {int(n)}.",
                          "good" if rate >= 60 else "warn"))
    top = ch.reindex(ch["bold"].abs().sort_values(ascending=False).index).head(25)
    top = top.assign(**{"bold": top["bold"] * 100, "best": top["predicted"] * 100,
                        "actual": top["actual"] * 100,
                        "result": np.where(top["right"], "right", "wrong")})
    parts.append("<h3>Biggest bold predictions, and what caused them</h3>"
                 + table(top[["patch", "champion_id", "role", "class", "games", "bold", "best",
                              "actual", "result", "why"]].set_index("patch"),
                         {"games": "{:,.0f}", "bold": "{:+.1f}", "best": "{:+.1f}",
                          "actual": "{:+.1f}"},
                         {"result": lambda v: "good" if v == "right" else "bad"}))
    parts.append("<p class='cap'><b>why</b>: the three inputs that pushed the prediction most "
                 "(before bold scaling), in points. <code>rel_</code> = % change of a number, "
                 "<code>buff_part</code> / <code>nerf_part</code> = all buffs / nerfs added up, "
                 "<code>mage_x_buff</code> = buffs on a mage.</p>")
    return section("bold", "5. Bold predictions: does the model pick the right movers?",
                   "Bold predictions keep the model's direction but are sized like real patch "
                   "swings, so they show who is likely to move a lot. The question here is "
                   "whether the direction is right.", "".join(parts))


def sec_why(d, weights):
    parts = []
    if weights is not None and len(weights):
        w = weights.copy()
        g = lambda c, default=np.nan: w[c].iloc[0] if c in w else default
        if "k_long" in w:
            parts.append("<h3>Strength model (before balance changes)</h3>" + table(pd.DataFrame({
                "value": [f"{g('trust_games'):,.0f} games", f"{g('k_recent'):.2f}",
                          f"{g('k_long'):.2f}", f"{g('k_popularity'):+.2f}"],
                "meaning": ["games needed to half-trust a champion's winrate (fewer = pulled "
                            "toward 50%)", "weight on recent winrate (last few patches)",
                            "weight on long-term winrate", "effect of popularity (negative = "
                            "popular champions win less)"]},
                index=pd.Index(["trust", "recent", "long-term", "popularity"], name=""))))
        w["weight (pts per typical change)"] = w["weight"] * 100
        w["shaky"] = np.where(w["used_by_changed"] < 10, "yes - learned from few examples", "")
        parts.append("<h3>What the change model has learned (all patches)</h3>"
                     + table(w.set_index("feature")[["weight (pts per typical change)",
                                                     "used_by_changed", "shaky"]].head(30),
                             {"weight (pts per typical change)": "{:+.2f}",
                              "used_by_changed": "{:.0f}"},
                             {"shaky": lambda v: "warn" if v else ""}))
        on = g("switched_on", 1)
        parts.append(note(
            f"Ridge strength alpha = {g('alpha'):g}. Best guess currently "
            f"<b>{'ON' if on else 'OFF'}</b>. Bold = {g('bold_scale', 1):.1f}x the raw effect. "
            f"Likely range: +-{1.96 * g('sd_true', 0) * 100:.1f} pts around the best guess. "
            "Weights marked <b>shaky</b> come from fewer than 10 changed champions and are likely "
            "noise - these are the first to distrust."))
    else:
        parts.append("<p class='muted'>No data yet.</p>")
    return section("why", "6. What the models learned",
                   "Both parts are sums of simple pieces, so every prediction can be traced back "
                   "to its inputs.", "".join(parts))


def sec_patches(cres, bres):
    parts = []
    if cres is not None and len(cres):
        c = cres.copy()
        c["best on"] = np.where(np.isfinite(c["alpha"]), "on", "off")
        order = sorted(c["patch"].astype(str), key=pkey)
        c = c.set_index(c["patch"].astype(str)).loc[order]
        series = {"last patch": (c["last"] * 100).values, "base": (c["shrunk"] * 100).values,
                  "best": (c["model"] * 100).values}
        parts.append(svg_lines(order, series, "avg miss (points)",
                               ref=("luck floor", (c["floor"] * 100).values)))
        cols = ["train patches", "best on", "floor", "last", "shrunk", "model", "changed",
                "shrunk ch", "model ch", "bold ch", "direction", "top3 direction"]
        t = c[[x for x in cols if x in c]].rename(columns={
            "shrunk": "base", "model": "best", "shrunk ch": "base ch", "model ch": "best ch",
            "direction": "bold dir", "top3 direction": "top-3 dir"})
        for col in ["floor", "last", "base", "best", "base ch", "best ch", "bold ch",
                    "bold dir", "top-3 dir"]:
            if col in t:
                t[col] = t[col] * 100
        parts.append(table(t.rename_axis("patch"),
                           {"train patches": "{:.0f}", "changed": "{:.0f}", "bold dir": "{:.0f}%",
                            "top-3 dir": "{:.0f}%"}))
        parts.append("<p class='cap'><b>base</b> = strength model only, <b>best</b> = + best-guess "
                     "effect, <b>bold ch</b> = miss of bold predictions on changed champions, "
                     "<b>bold dir</b> = bold direction right (100+ games), <b>top-3 dir</b> = same "
                     "for the 3 biggest predicted movers.</p>")
    if bres is not None and len(bres):
        b = bres.copy()
        b["patch"] = b["patch"].astype(str)
        piv = b.pivot_table(index="patch", columns="baseline", values="MAE") * 100
        piv = piv.loc[sorted(piv.index, key=pkey)]
        parts.append("<h3>Simple baselines per patch (avg miss, points)</h3>"
                     + table(piv.rename_axis("patch"), {c: "{:.2f}" for c in piv.columns}))
    return section("patches", "7. Patch by patch",
                   "Each patch predicted using only earlier patches. <b>floor</b> = the miss "
                   "that luck alone causes. A patch far above the floor for every method means "
                   "something happened that no input captures (for example a big item or rune "
                   "change).", "".join(parts) or "<p class='muted'>No data yet.</p>")


def sec_game(gres, gp, imp, effects, cp):
    parts = []
    if gres is not None and len(gres):
        g = gres.copy()
        g["patch"] = g["patch"].astype(str)
        g = g.set_index("patch").loc[sorted(g["patch"], key=pkey)]
        parts.append(svg_lines(list(g.index), {"numbers AUC": g["numbers AUC"].values,
                                               "identity AUC": g["identity AUC"].values},
                               "AUC", ref=("coin flip", np.full(len(g), 0.5))))
        cols = [c for c in ["train games", "test games", "coin logloss", "numbers logloss",
                            "numbers AUC", "identity logloss", "identity AUC"] if c in g]
        parts.append(table(g[cols].rename_axis("patch"),
                           {"train games": "{:,.0f}", "test games": "{:,.0f}",
                            "coin logloss": "{:.4f}", "numbers logloss": "{:.4f}",
                            "identity logloss": "{:.4f}", "numbers AUC": "{:.3f}",
                            "identity AUC": "{:.3f}"},
                           {"numbers AUC": lambda v: "bad" if v < 0.5 else "",
                            "identity AUC": lambda v: "bad" if v < 0.5 else ""}))
    if gp is not None and len(gp):
        from train_game_model import auc, logloss
        y = gp["blue_win"].values.astype(float)

        def grp(by):
            rows = {}
            for k, x in gp.groupby(by):
                yy = x["blue_win"].values.astype(float)
                rows[k] = {"games": len(x), "blue win % actual": yy.mean() * 100,
                           "blue win % predicted": x["p_numbers"].mean() * 100,
                           "numbers AUC": auc(yy, x["p_numbers"].values),
                           "identity AUC": auc(yy, x["p_identity"].values)
                           if "p_identity" in x else np.nan,
                           "logloss vs coin": logloss(yy, x["p_numbers"].values)
                           - logloss(yy, np.full(len(yy), yy.mean()))}
            return pd.DataFrame(rows).T
        fm = {"games": "{:,.0f}", "blue win % actual": "{:.1f}", "blue win % predicted": "{:.1f}",
              "numbers AUC": "{:.3f}", "identity AUC": "{:.3f}", "logloss vs coin": "{:+.4f}"}
        colr = {"logloss vs coin": lambda v: "good" if v < 0 else "bad"}
        parts.append("<h3>By region</h3>" + table(grp("platform").rename_axis("region"), fm, colr))
        parts.append("<h3>By rank</h3>" + table(grp("rank_bucket").rename_axis("rank"), fm, colr))
        parts.append("<p class='cap'><b>logloss vs coin</b>: negative = better than guessing "
                     "(green).</p>")
        px, py, pe = [], [], []
        df = pd.DataFrame({"p": gp["p_numbers"], "y": y})
        df["b"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
        for _, x in df.groupby("b"):
            px.append(x["p"].mean() * 100)
            py.append(x["y"].mean() * 100)
            pe.append(math.sqrt(max(x["y"].mean() * (1 - x["y"].mean()), 1e-6) / len(x)) * 100)
        lo, hi = min(px + py) - 2, max(px + py) + 2
        # champion residuals: actual win rate of the champion's team vs predicted
        rows = []
        for side, sign in (("blue", 1), ("red", -1)):
            pw = gp["p_numbers"] if sign == 1 else 1 - gp["p_numbers"]
            yw = gp["blue_win"] if sign == 1 else 1 - gp["blue_win"]
            for c in [c for c in gp.columns if c.startswith(f"{side}_") and c.endswith("_champ")]:
                rows.append(pd.DataFrame({"champ_key": gp[c].values, "p": pw.values, "y": yw.values}))
        r = pd.concat(rows)
        r = r.groupby("champ_key").agg(games=("y", "size"), actual=("y", "mean"), predicted=("p", "mean"))
        r = r[r["games"] >= 100]
        r["gap"] = (r["actual"] - r["predicted"]) * 100
        r["z"] = r["gap"] / (np.sqrt(r["actual"].clip(0.05, 0.95) * (1 - r["actual"].clip(0.05, 0.95))
                                     / r["games"]) * 100)
        names = dict(zip(cp["champ_key"], cp["champion_id"]))
        r.index = [names.get(k, str(k)) for k in r.index]
        r = r.reindex(r["z"].abs().sort_values(ascending=False).index)
        r[["actual", "predicted"]] *= 100
        parts.append("<div class='grid2'><div><h3>Calibration</h3>"
                     + svg_calibration(px, py, pe, "predicted blue win %", "actual blue win %", lo, hi,
                                       "{:.0f}")
                     + "<p class='cap'>Test games grouped by predicted win chance. Near the "
                       "diagonal = the model's confidence is honest.</p></div><div>"
                     + "<h3>Champions the game model misjudges</h3>"
                     + table(r.head(15).rename_axis("champion"),
                             {"games": "{:,.0f}", "actual": "{:.1f}", "predicted": "{:.1f}",
                              "gap": "{:+.1f}", "z": "{:+.1f}"},
                             {"z": lambda v: "bad" if abs(v) >= 3 else ("warn" if abs(v) >= 2 else "")})
                     + "<p class='cap'>Win % of games with this champion vs the game model's "
                       "prediction for those games (test patches only, numbers model).</p></div></div>")
    if imp is not None and len(imp):
        top = imp.head(25)
        parts.append("<h3>What the game model relies on most</h3>"
                     + svg_bars(top["feature"].tolist(), (top["share"] * 100).tolist(),
                                "share of the model's total gain (%)", "{:.1f}%",
                                highlight=lambda f: str(f).startswith("ctx_")))
        parts.append("<p class='cap'>Red = context (side, region, rank), not champion numbers. "
                     "If context dominates, the champion numbers carry little signal yet.</p>")
    if effects is not None and len(effects):
        e = effects[effects["variant"] == "numbers"]
        spread = e.groupby("patch")["effect"].agg(lambda s: s.abs().max() * 100)
        spread = spread.loc[sorted(spread.index.astype(str), key=pkey)] if len(spread) else spread
        parts.append("<h3>How big are the game model's patch effects?</h3>"
                     + svg_bars(list(spread.index), list(spread.values),
                                "largest predicted champion effect on the patch (points)", "{:.2f}"))
    if not parts:
        parts.append("<p class='muted'>Run train_game_model.py (without --skip-eval) to fill "
                     "this section.</p>")
    return section("game", "8. Game model (who wins each game)",
                   "Tested on patches it never saw. AUC 0.5 = coin flip. In high elo, games are "
                   "close to coin flips, so 0.52-0.56 is normal and rises with more games.",
                   "".join(parts))


def sec_strength(sres, sdet, scorr, d, gres):
    if sres is None or not len(sres):
        return section("strength", "9. Champion strength model",
                       "The neural network that scores each champion from their numbers.",
                       note("No results yet - it runs as part of update_model.bat once PyTorch "
                            "is installed (see README_WINDOWS.md, 'Champion strength model')."))
    parts = []
    r = sres.copy()
    r["patch"] = r["patch"].astype(str)
    r = r.set_index("patch").loc[sorted(r["patch"], key=pkey)]
    t = pd.DataFrame(index=r.index)
    t["game AUC"] = r["AUC"]
    if gres is not None and "numbers AUC" in gres:
        g = gres.assign(patch=gres["patch"].astype(str)).set_index("patch")
        t["tree AUC"] = g["numbers AUC"].reindex(t.index)
    for c, lab in [("floor", "floor"), ("champ MAE", "miss"), ("floor ch", "floor ch"),
                   ("champ MAE ch", "miss ch")]:
        if c in r:
            t[lab] = r[c] * 100
    # the change model on the same patches, for comparison
    if d is not None and len(d):
        dd = d[d["patch"].isin(t.index)]
        wm = lambda x, col: np.average((x[col]).abs(), weights=x["games"]) * 100 if len(x) else np.nan
        t["change model miss ch"] = [wm(dd[(dd["patch"] == p) & (dd["changed"] == 1)], "error")
                                     for p in t.index]
    if "direction" in r:
        t["dir"] = r["direction"] * 100
    if "max effect" in r:
        t["biggest effect"] = r["max effect"]
    parts.append(table(t.rename_axis("patch"),
                       {"game AUC": "{:.3f}", "tree AUC": "{:.3f}", "dir": "{:.0f}%",
                        "biggest effect": "{:.1f}"}))
    parts.append("<p class='cap'><b>game AUC</b> vs <b>tree AUC</b> = this model vs the tree game "
                 "model on the same patch. <b>miss</b> = average champion winrate miss (points), "
                 "<b>ch</b> = changed champions only, <b>change model miss ch</b> = the patch-change "
                 "model's best guess on the same champions. <b>biggest effect</b> = largest "
                 "predicted patch effect (points).</p>")
    if sdet is not None and len(sdet):
        s = sdet.copy()
        s["patch"] = s["patch"].astype(str)
        ch = s[s["changed"] == 1]
        # head to head on the same changed champions
        if d is not None and len(d):
            m = ch.merge(d[["patch", "champion_id", "error", "bold", "actual"]]
                         if "bold" in d else d[["patch", "champion_id", "error", "actual"]],
                         on=["patch", "champion_id"], suffixes=("", "_cm"))
            if len(m):
                w = m["games"]
                rows = {"strength model": [np.average(m["error"].abs(), weights=w) * 100,
                                           np.mean(np.sign(m["effect"]) == np.sign(m["actual_change"])) * 100],
                        "change model (best guess)": [np.average(m["error_cm"].abs(), weights=w) * 100, np.nan]}
                if "bold" in m:
                    rows["change model (bold)"] = [np.nan, np.mean(np.sign(m["bold"]) == np.sign(m["actual"])) * 100]
                hh = pd.DataFrame(rows, index=["avg miss (points)", "direction right"]).T
                parts.append(f"<h3>Head to head: the same {len(m)} changed champions</h3>"
                             + table(hh.rename_axis("model"),
                                     {"avg miss (points)": "{:.2f}", "direction right": "{:.0f}%"}))
        parts.append("<h3>Predicted vs actual patch effect (changed champions)</h3>"
                     + svg_scatter(ch["effect"] * 100, ch["actual_change"] * 100, ch["games"],
                                   ch["patch"] + " " + ch["champion_id"].astype(str) + ": predicted "
                                   + (ch["effect"] * 100).round(1).astype(str) + ", actual "
                                   + (ch["actual_change"] * 100).round(1).astype(str),
                                   "predicted effect (points)", "actual change (points)"))
        top = ch.reindex(ch["effect"].abs().sort_values(ascending=False).index).head(20)
        top = top.assign(predicted=top["effect"] * 100, actual=top["actual_change"] * 100,
                         result=np.where(np.sign(top["effect"]) == np.sign(top["actual_change"]),
                                         "right", "wrong"))
        parts.append("<h3>Biggest predicted effects, and which numbers caused them</h3>"
                     + table(top[["patch", "champion_id", "games", "predicted", "actual", "result",
                                  "why"]].set_index("patch"),
                             {"games": "{:,.0f}", "predicted": "{:+.1f}", "actual": "{:+.1f}"},
                             {"result": lambda v: "good" if v == "right" else "bad"}))
        parts.append("<p class='cap'><b>why</b>: how much each changed number moved the "
                     "champion's strength score, in winrate points.</p>")
    if scorr is not None and len(scorr):
        parts.append("<h3>Champions the numbers explain least</h3>"
                     + table(scorr.head(15).set_index("champion_id"), {"correction_pts": "{:+.1f}"})
                     + "<p class='cap'>The model's own correction per champion (points), on top of "
                       "what their numbers predict. Big ones point to something the numbers miss: "
                       "a unique mechanic, popularity, items, or a reading error in their data.</p>")
    return section("strength", "9. Champion strength model",
                   "A neural network that turns each champion's numbers into a strength score, "
                   "learned from every game. Tested forward in time on the latest patches, next "
                   "to the other models.", "".join(parts))


def _item_frame(d, cp):
    """Change model predictions with the champions' item scores attached."""
    cols = ["item_stat_buff", "item_passive_buff", "item_changed_share"]
    m = d.copy()
    if not all(c in m for c in cols) and cp is not None and all(c in cp for c in cols):
        c2 = cp[["patch", "champion_id"] + cols].copy()
        c2["patch"] = c2["patch"].astype(str)
        m = m.drop(columns=[c for c in cols if c in m]).merge(c2, on=["patch", "champion_id"], how="left")
    if not all(c in m for c in cols):
        return None
    for c in cols:
        m[c] = m[c].fillna(0.0)
    m["item_buff"] = m["item_stat_buff"] + m["item_passive_buff"]
    return m


def _group_effect(t):
    """Average change beyond expected (points) and its luck wobble, games-weighted."""
    w = t["games"]
    mean = np.average(t["actual"], weights=w) * 100
    se = math.sqrt(np.sum(w ** 2 * t["noise_sd"] ** 2)) / w.sum() * 100
    return mean, se


def sec_items(d, cp, ichg, iusers):
    m = _item_frame(d, cp)
    if m is None:
        return section("items", "Item changes",
                       "Which champions, roles and classes gain when items change.",
                       note("No item scores yet - run update_model.bat (it rebuilds the tables with "
                            "item changes). For item passives, fetch_item_data.py runs as part of it."))
    parts = []
    # only champions touched by item changes but NOT changed themselves: a clean look at items
    t = m[(m["item_changed_share"] >= 0.10) & (m["changed"] != 1)].copy()
    if len(t) >= 10:
        t["kind"] = np.select([t["item_buff"] > 0.01, t["item_buff"] < -0.01],
                              ["items buffed (1%+ of cost)", "items nerfed (1%+ of cost)"],
                              "items changed, roughly even")
        rows = []
        for label, g in [("all", t)] + list(t.groupby("kind")):
            mean, se = _group_effect(g)
            rows.append({"group": label, "champion-patches": len(g),
                         "avg item change (% of cost)": np.average(g["item_buff"], weights=g["games"]) * 100,
                         "avg winrate change (pts)": mean, "+- luck": 2 * se,
                         "clear?": "yes" if abs(mean) > 2 * se else "not yet"})
        parts.append("<h3>Champions whose items changed (but who weren't changed themselves)</h3>"
                     + table(pd.DataFrame(rows).set_index("group"),
                             {"champion-patches": "{:.0f}", "avg item change (% of cost)": "{:+.1f}",
                              "avg winrate change (pts)": "{:+.2f}", "+- luck": "{:.2f}"},
                             {"clear?": lambda v: "good" if v == "yes" else ""}))
        r_all = float(np.corrcoef(t["item_buff"], t["actual"])[0, 1]) if t["item_buff"].std() > 0 else np.nan
        parts.append(f"<p class='cap'><b>winrate change</b> = actual winrate minus the expected "
                     f"winrate (before any patch effect). <b>+- luck</b> = 2 luck wobbles: a "
                     f"change bigger than that is unlikely to be chance. Line-up of item change "
                     f"and winrate change: r = {r_all:.2f}.</p>")
        # by role and class: buffed vs nerfed
        for by, title in (("role", "By role"), ("class", "By class")):
            rows = []
            for k, g in t.groupby(by):
                if len(g) < 5:
                    continue
                b, n = g[g["item_buff"] > 0.01], g[g["item_buff"] < -0.01]
                mb, sb = _group_effect(b) if len(b) else (np.nan, np.nan)
                mn, sn = _group_effect(n) if len(n) else (np.nan, np.nan)
                rr = float(np.corrcoef(g["item_buff"], g["actual"])[0, 1]) if g["item_buff"].std() > 0 else np.nan
                rows.append({by: k, "n": len(g), "after item buffs (pts)": mb, "+-": 2 * sb,
                             "after item nerfs (pts)": mn, "+- ": 2 * sn, "r": rr})
            if rows:
                parts.append(f"<h3>{title}</h3>" + table(
                    pd.DataFrame(rows).set_index(by),
                    {"n": "{:.0f}", "after item buffs (pts)": "{:+.2f}", "+-": "{:.2f}",
                     "after item nerfs (pts)": "{:+.2f}", "+- ": "{:.2f}", "r": "{:+.2f}"},
                    {"r": lambda v: "good" if v >= 0.2 else ("bad" if v <= -0.2 else "")}))
        parts.append("<h3>Item change vs winrate change</h3>"
                     + svg_scatter(t["item_buff"] * 100, t["actual"] * 100, t["games"],
                                   t["patch"] + " " + t["champion_id"] + ": items "
                                   + (t["item_buff"] * 100).round(1).astype(str) + "% of cost, winrate "
                                   + (t["actual"] * 100).round(1).astype(str) + " pts",
                                   "how much their items changed (% of item cost)",
                                   "winrate change beyond expected (pts)", diag=False,
                                   same_scale=False)
                     + "<p class='cap'>One dot per champion and patch (size = games). Dots rising "
                       "to the right = champions gain when their items get stronger.</p>")
    else:
        parts.append(note("Not enough item changes with games yet to measure."))
    # per item
    if ichg is not None and len(ichg):
        ic = ichg.copy()
        ic["patch"] = ic["patch"].astype(str)
        ic["item_id"] = ic["item_id"].astype(str)
        eff = {}
        if iusers is not None and len(iusers):
            iu = iusers.assign(patch=iusers["patch"].astype(str), item_id=iusers["item_id"].astype(str))
            j = iu.merge(m[["patch", "champion_id", "games", "actual", "noise_sd"]],
                         on=["patch", "champion_id"], how="inner")
            j = j[j["share"] >= 0.05]
            for (p, i), g in j.groupby(["patch", "item_id"]):
                w = g["share"] * g["games"]
                if w.sum() > 0:
                    eff[(p, i)] = (np.average(g["actual"], weights=w) * 100,
                                   math.sqrt(np.sum(w ** 2 * g["noise_sd"] ** 2)) / w.sum() * 100)
        ic["users' change (pts)"] = [eff.get((p, i), (np.nan, np.nan))[0] for p, i in zip(ic["patch"], ic["item_id"])]
        ic["+-"] = [2 * eff.get((p, i), (np.nan, np.nan))[1] for p, i in zip(ic["patch"], ic["item_id"])]
        ic["size (% of cost)"] = (ic["stat_pct"] + ic["passive_pct"]) * 100
        sign = np.sign(ic["size (% of cost)"])
        ic["verdict"] = np.where(ic["users' change (pts)"].isna(), "",
                        np.where(ic["users' change (pts)"].abs() <= ic["+-"], "no clear change",
                        np.where(np.sign(ic["users' change (pts)"]) == sign, "as expected", "opposite")))
        ic = ic.reindex(ic["size (% of cost)"].abs().sort_values(ascending=False).index)
        ic = ic.sort_values("patch", key=lambda s: s.map(pkey), ascending=False, kind="stable")
        show = ic[["patch", "name", "changes", "size (% of cost)", "main_users",
                   "users' change (pts)", "+-", "verdict"]].head(40)
        parts.append("<h3>Item changes, newest first</h3>" + table(
            show.set_index("patch"),
            {"size (% of cost)": "{:+.1f}", "users' change (pts)": "{:+.2f}", "+-": "{:.2f}"},
            {"verdict": lambda v: "good" if v == "as expected" else ("bad" if v == "opposite" else "")}))
        parts.append("<p class='cap'><b>size</b> = stat value change minus price change, as a share "
                     "of the item's price, plus the average change of its passive numbers. "
                     "<b>users' change</b> = winrate change beyond expected of the champions who "
                     "build it, weighted by how often they build it. Passive changes that aren't "
                     "numbers (new effects) only show up through the flags.</p>")
    # flags: how often each changes (too rare = can't be learned yet)
    if cp is not None:
        fl = [c for c in cp.columns if c.startswith("delta_flag_")]
        if fl:
            rows = []
            for c in fl:
                name = c.replace("delta_flag_", "")
                ch = int((cp[c].fillna(0) != 0).sum())
                has = cp.get(f"flag_{name}")
                share = float(has.mean()) if has is not None else np.nan
                it = 0
                if ichg is not None and len(ichg):
                    it = int(ichg["flags_gained"].fillna("").str.contains(name).sum()
                             + ichg["flags_lost"].fillna("").str.contains(name).sum())
                rows.append({"flag": name, "champions with it": share * 100,
                             "champion changes": ch, "item changes": it,
                             "usable?": "yes" if ch + it >= 10 else ("few" if ch + it >= 4 else "too rare")})
            parts.append("<h3>Effect flags: how often they change</h3>" + table(
                pd.DataFrame(rows).set_index("flag"),
                {"champions with it": "{:.0f}%", "champion changes": "{:.0f}", "item changes": "{:.0f}"},
                {"usable?": lambda v: "good" if v == "yes" else ("warn" if v == "few" else "bad")}))
            parts.append("<p class='cap'>A flag has to change a number of times across patches before "
                         "its effect can be learned. Rare ones still help the strength model tell "
                         "champions apart (e.g. which champions have healing), but their changes are "
                         "noise until more patches arrive.</p>")
    return section("items", "Item changes",
                   "Do champions gain when the items they build get stronger? Split by role and "
                   "class, plus every item change and how its users did.", "".join(parts))


def sec_data(cp):
    parts = []
    cp = cp.copy()
    cp["patch"] = cp["patch"].astype(str)
    per = cp.groupby("patch").agg(games=("games", lambda s: s.sum() / 10),
                                  champions=("games", lambda s: int((s >= 30).sum())),
                                  changed=("changed", "sum"),
                                  reworks=("n_changes", lambda s: int((s.fillna(0) >= 25).sum())))
    per = per.loc[sorted(per.index, key=pkey)]
    per["games"] = per["games"].round()
    typical = per.loc[per["games"] > 0, "games"].median() if (per["games"] > 0).any() else 0
    parts.append("<h3>Data per patch</h3>"
                 + table(per.rename_axis("patch"), {"games": "{:,.0f}", "champions": "{:.0f}",
                                                    "changed": "{:.0f}", "reworks": "{:.0f}"},
                         {"games": lambda v: "bad" if v == 0 else
                          ("warn" if v < 0.5 * typical else "")})
                 + "<p class='cap'>Red = no games yet (the collector fills older patches over "
                   "time). Yellow = under half the usual amount.</p>")
    try:
        import train_change_model as tcm
        c2, X = tcm.build_features(cp)
        rel = [c for c in c2.columns if c.startswith("chg_")]
        big = []
        for c in rel:
            m = c2[c].abs() >= tcm.MAX_REL - 1e-9
            for i in c2.index[m]:
                big.append({"patch": c2.at[i, "patch"], "champion": c2.at[i, "champion_id"],
                            "number": c.replace("chg_", ""), "change": f"{c2.at[i, c]:+.0%} (capped)",
                            "changes that patch": int(c2.at[i, "n_changes"] or 0)})
        big = pd.DataFrame(big)
        if len(big):
            big = big.sort_values("patch", key=lambda s: s.map(pkey), ascending=False)
            parts.append("<h3>Suspiciously large changes (50%+)</h3>"
                         + table(big.set_index("patch"), max_rows=40)
                         + "<p class='cap'>Real patch changes are rarely this big. Check these "
                           "against the patch notes with <code>python build_patch_state.py "
                           "--report PATCH</code> - they may be reading errors.</p>")
        rw = c2[c2["n_changes"].fillna(0) >= tcm.REWORK_CHANGES][["patch", "champion_id", "n_changes"]]
        if len(rw):
            parts.append("<h3>Treated as reworks</h3>"
                         + table(rw.set_index("patch").sort_index(key=lambda s: s.map(pkey)),
                                 {"n_changes": "{:.0f}"}))
    except Exception as e:  # never let a data check break the report
        parts.append(note(f"Data checks skipped: {html.escape(str(e))}", "warn"))
    return section("data", "10. Data checks",
                   "Problems in the inputs show up as model mistakes, so check these first "
                   "when something looks odd.", "".join(parts))


# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;--acc:#2f6fdf;
--good:#e3f4e8;--goodf:#1c6b36;--warn:#fff3d6;--warnf:#8a5a00;--bad:#fde4e1;--badf:#a3261b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9d9c96;--card:#1f1f1d;
--line:#34332f;--acc:#7aa7ff;--good:#16311f;--goodf:#8fd6a5;--warn:#3a2e10;--warnf:#f0c46a;
--bad:#3d1a17;--badf:#ff9d92}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:0 0 6px}h3{font-size:15px;margin:22px 0 8px}
.sub,.muted,.cap{color:var(--mut)}.cap{font-size:13px;margin:4px 0 0}.intro{margin:0 0 12px}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0 8px}nav a{color:var(--acc);text-decoration:none;
border:1px solid var(--line);border-radius:14px;padding:3px 10px;font-size:13px;background:var(--card)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.cl{font-size:12px;color:var(--mut)}.cv{font-size:22px;font-weight:600}.cs{font-size:12px;color:var(--mut)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:800px){.grid2{grid-template-columns:1fr}}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--mut);position:sticky;top:0;background:var(--card)}td.l,th.l{text-align:left}
td.good{background:var(--good);color:var(--goodf)}td.warn{background:var(--warn);color:var(--warnf)}
td.bad{background:var(--bad);color:var(--badf)}
.note{border-left:4px solid var(--acc);padding:8px 12px;margin:10px 0;background:var(--bg);border-radius:4px;font-size:14px}
.note.warn{border-color:var(--warnf)}.note.good{border-color:var(--goodf)}
code{font-size:12px;background:var(--bg);padding:1px 4px;border-radius:3px}
.chart{display:block;width:100%;height:auto;max-width:460px;aspect-ratio:1/1}.ax{stroke:var(--mut)}.gr{stroke:var(--line)}
.tk{fill:var(--mut);font-size:11px}.lb{fill:var(--fg);font-size:12px}
.diag{stroke:var(--mut);stroke-dasharray:4 4}.zero{stroke:var(--line)}
.pt{fill:var(--acc);fill-opacity:.45;stroke:var(--acc)}.pt2{fill:var(--acc)}.ptf{fill:var(--mut);fill-opacity:.25}
.trend{stroke:#e8a33d;stroke-width:2}.eb{stroke:var(--acc);stroke-width:2}
.bar{fill:var(--acc);fill-opacity:.75}.bar.bad{fill:var(--badf)}.curve{fill:none;stroke:var(--fg);stroke-width:2}
.ln{fill:none;stroke-width:2}.ln0{stroke:#9aa0a6}.ln1{stroke:#e8a33d}.ln2{stroke:var(--acc)}
.c0{fill:#9aa0a6}.c1{fill:#e8a33d}.c2{fill:var(--acc)}.refl{fill:none;stroke:var(--goodf);stroke-dasharray:5 4;stroke-width:2}
.refc{fill:var(--goodf)}
"""


def build(dir_training, dir_models, out, patch_charts=5):
    T = lambda n: os.path.join(dir_training, n)
    cp = read(T("champion_patch.parquet"))
    if cp is None:
        sys.exit("training/champion_patch not found - run build_training_tables.py first.")
    d = read(T("change_model_predictions.csv"))
    if d is None or "z" not in d:
        sys.exit("training/change_model_predictions.csv missing or old - run "
                 "train_change_model.py first.")
    d["patch"] = d["patch"].astype(str)
    d["why"] = d["why"].fillna("")
    for c in ("role", "class"):
        d[c] = d[c].fillna("?")
    cres, bres = read(T("change_model_results.csv")), read(T("baseline_results.csv"))
    weights = read(T("change_model_weights.csv"))
    gres, gp = read(T("model_results.csv")), read(T("game_test_predictions.parquet"))
    sres, sdet = read(T("strength_results.csv")), read(T("strength_predictions.csv"))
    scorr = read(os.path.join(dir_models, "strength_corrections.csv"))
    effects, imp = read(T("game_model_effects.csv")), read(os.path.join(dir_models,
                                                                        "importance_numbers.csv"))

    s_all = error_summary(d)
    s_ch = error_summary(d[d["changed"] == 1]) if (d["changed"] == 1).any() else None
    n_games = int(cp["games"].sum() / 10)
    patches_with = int((cp.groupby("patch")["games"].sum() > 0).sum())
    on = f"{int(cres['alpha'].apply(np.isfinite).sum())} of {len(cres)}" if cres is not None else "-"
    bold_txt = (f"{np.nanmean(cres['direction']) * 100:.0f}%"
                if cres is not None and "direction" in cres and cres["direction"].notna().any() else "-")
    auc_txt = f"{gres['numbers AUC'].mean():.3f}" if gres is not None and "numbers AUC" in gres else "-"
    cards = "".join([
        card("Games", f"{n_games:,}", f"{patches_with} patches with games"),
        card("Avg miss (all champions)", f"{s_all['miss']:.2f} pts",
             f"luck alone: {s_all['luck']:.2f}"),
        card("Model's own error", f"{s_all['own error']:.2f} pts", "after removing luck"),
        card("Changed champions", f"{s_ch['miss']:.2f} pts" if s_ch is not None else "-",
             f"own error {s_ch['own error']:.2f}" if s_ch is not None else ""),
        card("Best guess on", on, "patches where it beat 'no effect'"),
        card("Bold direction right", bold_txt, "changed champions, 100+ games (50% = coin flip)"),
        card("Game model AUC", auc_txt, "0.5 = coin flip"),
    ])
    nav = "".join(f"<a href='#{a}'>{t}</a>" for a, t in [
        ("pva", "Predicted vs actual"), ("luck", "Luck or mistakes"), ("where", "Where"), ("champs", "Champions"),
        ("calib", "Calibration"), ("bold", "Bold picks"), ("why", "What it learned"),
        ("patches", "Per patch"),
        ("game", "Game model"), ("strength", "Strength model"), ("items", "Item changes"),
        ("data", "Data checks")])
    ichg, iusers = read(T("item_changes.csv")), read(T("item_users.csv"))
    body = "".join([sec_pred_vs_actual(d, sdet, patch_charts), sec_luck(d), sec_breakdown(d, None), sec_champions(d), sec_calibration(d),
                    sec_bold(d, cres), sec_why(d, weights), sec_patches(cres, bres), sec_game(gres, gp, imp, effects, cp),
                    sec_strength(sres, sdet, scorr, d, gres),
                    sec_items(d, cp, ichg, iusers), sec_data(cp)])
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Model diagnostics</title><style>{CSS}</style></head><body><main>"
            f"<h1>Model diagnostics</h1><div class='sub'>Built {datetime.now():%Y-%m-%d %H:%M} "
            f"from {len(d):,} champion-patch predictions (each patch predicted from earlier "
            f"patches only)</div><nav>{nav}</nav><div class='cards'>{cards}</div>{body}</main></body></html>")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    # the same groupings as CSV, for Excel
    d2 = d.assign(**{"games bucket": games_bucket(d["games"]).astype(str),
                     "strength": strength_bucket(d["recent_winrate"]).astype(str)})
    parts = []
    for by in ["patch", "role", "class", "changed", "games bucket", "strength"]:
        t = group_table(d2, by).reset_index().rename(columns={by: "value"})
        t.insert(0, "grouped by", by)
        parts.append(t)
    pd.concat(parts).to_csv(T("diagnostics_breakdown.csv"), index=False)
    print(f"Saved {out}  (open it in your browser)")
    print(f"Saved {T('diagnostics_breakdown.csv')}")
    print(f"\nIn short: average miss {s_all['miss']:.2f} pts, of which luck explains "
          f"{s_all['luck']:.2f}; the model's own error is {s_all['own error']:.2f} pts. "
          f"{s_all['within 2 sd']:.0f}% of predictions are within 2 luck wobbles (luck alone: 95%).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build report.html with model diagnostics.")
    ap.add_argument("--dir", default="training")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--patch-charts", type=int, default=5,
                    help="predicted-vs-actual charts for the latest N patches (default 5)")
    ap.add_argument("--open", action="store_true", help="open the report in the browser")
    args = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    build(args.dir, args.models, args.out, args.patch_charts)
    if args.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
