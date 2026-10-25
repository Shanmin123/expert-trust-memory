"""Dissertation figures from the canonical run artefacts (read-only).

F1 curves.pdf      : cumulative net return, verified-global vs equal, the four
                     running examples of Chapter 5 (2x2 grid, body figure).
F1b curves_all.pdf : the same figure over all eight stocks (2x4 grid, appendix).
F2 delta.pdf       : Sharpe difference (verified global minus equal), the four
                     running examples (body figure).
F2b delta_all.pdf  : the same figure over all eight stocks (appendix).
F3 states.pdf      : SPY market-state timeline over training/embargo/test.

Net returns use the shared cost model: 10 bps per side on each exposure change.
Colours follow the Okabe-Ito colour-blind-safe palette.

Usage: PYTHONPATH=src python scripts/dissertation_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("dissertation/drafts/figures")
COST = 10.0 / 1e4

REPORTS = {
    "TSLA": "reports/final/TSLA__designD3_q_h21cert.json",
    "ADSK": "reports/final/ADSK__designD3_q_h21cert.json",
    "BIIB": "reports/final/BIIB__designD3_q_h21cert.json",
    "CVX": "reports/final/CVX__final.json",
    "DHI": "reports/final/DHI__sealed.json",
    "ECL": "reports/final/ECL__sealed.json",
    "PNC": "reports/final/PNC__sealed.json",
    "TGT": "reports/final/TGT__sealed.json",
}
VERIFIED = {"BIIB": 3, "ECL": 2, "ADSK": 1, "TSLA": 0, "CVX": 0, "DHI": 0, "PNC": 0, "TGT": 0}
# The curves figure plots the GLOBAL arm, so its panel tags count global cells only
# (ADSK's single verified cell lives in the state memory; its global series is
# identical to equal weights).
VERIFIED_GLOBAL = {"BIIB": 3, "ECL": 2, "ADSK": 0, "TSLA": 0, "CVX": 0, "DHI": 0, "PNC": 0, "TGT": 0}
# The four running examples of Chapter 5, in the order the table uses them:
# two stocks where verified trust acts, the fallback identity, the state path.
SHOWCASE = ("BIIB", "ECL", "DHI", "ADSK")
# Okabe-Ito
C_EQUAL = "#0072B2"   # blue
C_CERT = "#D55E00"    # vermilion
STATE_COLOURS = {
    # Two dimensions, two channels: hue is trend, depth is risk. All four are
    # Okabe-Ito. The previous set spent four unrelated hues on a 2x2 label, and
    # put up_stress (1,199 days) on #F0E442, which the palette validator fails at
    # lightness 0.902 and 1.29:1 against the page.
    "up_calm": "#56B4E9",     # sky blue   -- up, calm
    "up_stress": "#0072B2",   # blue       -- up, stressed
    "down_calm": "#E69F00",   # orange     -- down, calm
    "down_stress": "#D55E00", # vermilion  -- down, stressed
    "warmup": "#DDDDDD",
}


def net_curve(series):
    dates, curve, level, prev_exp = [], [], 1.0, 0.0
    for d, gross, exp in series:
        level *= 1.0 + (float(gross) - COST * abs(float(exp) - prev_exp))
        prev_exp = float(exp)
        dates.append(pd.Timestamp(d))
        curve.append(level)
    return dates, curve


def fig_curves(names, stem, nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    for ax, name in zip(axes.flat, names):
        das = json.loads(Path(REPORTS[name]).read_text(encoding="utf-8"))["daily_arm_series"]
        d_eq, c_eq = net_curve(das["equal_weight"])
        d_gl, c_gl = net_curve(das["global_memory"])
        ax.plot(d_eq, c_eq, color=C_EQUAL, lw=1.4, label="equal weights")
        ax.plot(d_gl, c_gl, color=C_CERT, lw=1.4, ls="--", label="verified global")
        n_glob = VERIFIED_GLOBAL[name]
        tag = f"  ({n_glob} ver.)" if n_glob else ""
        ax.set_title(f"{name}{tag}", fontsize=9)
        ax.axhline(1.0, color="grey", lw=0.6)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Cumulative net return, test window (10 bps per side)", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def fig_delta(stocks, stem, figsize):
    names, deltas = [], []
    for name in stocks:
        arms = json.loads(Path(REPORTS[name]).read_text(encoding="utf-8"))["arms"]
        d = arms["global_memory"]["metrics"]["sharpe"] - arms["equal_weight"]["metrics"]["sharpe"]
        names.append(name)
        deltas.append(d)
    order = sorted(range(len(names)), key=lambda i: deltas[i])
    names = [names[i] for i in order]
    deltas = [deltas[i] for i in order]
    fig, ax = plt.subplots(figsize=figsize)
    colours = [C_CERT if d > 0 else "#999999" for d in deltas]
    ax.barh(names, deltas, color=colours, height=0.62)
    for i, (n, d) in enumerate(zip(names, deltas)):
        label = f"+{d:.3f}  ({VERIFIED[n]} ver.)" if d > 0 else "0.000  (identical)"
        ax.text(max(d, 0.0) + 0.006, i, label, va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Sharpe difference: verified global minus equal weights")
    ax.set_xlim(-0.02, 0.40)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def fig_states():
    r = json.loads(Path(REPORTS["ECL"]).read_text(encoding="utf-8"))
    ml = r["market_state_labels"]
    rows = list(ml["train"])
    val = ml.get("validation")
    if isinstance(val, dict):
        val = next(iter(val.values()))
    rows += list(val or [])
    test = ml.get("arms")
    if isinstance(test, dict):
        test = test.get("equal_weight") or next(iter(test.values()))
    rows += list(test or [])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 1.95))
    start = df.loc[0, "date"]
    cur = df.loc[0, "state"]
    for i in range(1, len(df)):
        if df.loc[i, "state"] != cur:
            ax.axvspan(start, df.loc[i, "date"], color=STATE_COLOURS.get(cur, "#CCCCCC"), lw=0)
            start, cur = df.loc[i, "date"], df.loc[i, "state"]
    ax.axvspan(start, df["date"].iloc[-1], color=STATE_COLOURS.get(cur, "#CCCCCC"), lw=0)
    b1, b2 = pd.Timestamp("2023-12-31"), pd.Timestamp("2024-06-30")
    # The boundaries annotate the bands; they are not the data, so they recede.
    for boundary in (b1, b2):
        ax.axvline(boundary, color="#3D3D3D", lw=0.8, ls=(0, (2, 2)))
    for x0, x1, label in ((df["date"].iloc[0], b1, "training"),
                          (b1, b2, "embargo"),
                          (b2, df["date"].iloc[-1], "test")):
        ax.text(x0 + (x1 - x0) / 2, 1.02, label,
                transform=ax.get_xaxis_transform(), fontsize=8, ha="center")
    counts = df["state"].value_counts()
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLOURS[s])
               for s in ("up_calm", "up_stress", "down_calm", "down_stress")]
    labels = [f"{s.replace('_', '-')} ({counts.get(s, 0)}d)"
              for s in ("up_calm", "up_stress", "down_calm", "down_stress")]
    ax.legend(handles, labels, ncol=4, fontsize=8, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_yticks([])
    ax.set_xlim(df["date"].iloc[0], df["date"].iloc[-1])
    # A one-dimensional band needs no box drawn round it, and the caption already
    # says what the figure is, so the in-image title is not repeated here.
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.tick_params(axis="x", length=3, color="#B0B0B0", labelsize=8.4)
    fig.tight_layout()
    fig.savefig(OUT / "states.pdf", bbox_inches="tight")
    fig.savefig(OUT / "states.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_curves(SHOWCASE, "curves", 2, 2, (8.4, 5.0))
    fig_curves(tuple(REPORTS), "curves_all", 2, 4, (11.5, 5.2))
    fig_delta(SHOWCASE, "delta", (6.6, 2.4))
    fig_delta(tuple(REPORTS), "delta_all", (6.6, 3.4))
    fig_states()
    for f in ("curves.pdf", "curves_all.pdf", "delta.pdf", "delta_all.pdf", "states.pdf"):
        print(f"WROTE {OUT / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
