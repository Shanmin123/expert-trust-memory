"""Design, protocol and data-pipeline diagrams.

A1 fig_system.pdf   : the daily decision path and the learning path beneath it, with
                      verification standing between what is learned and what may act.
A2 fig_protocol.pdf : the train / confirm / test windows, the freeze, and the forward run.
A3 fig_data.pdf     : the IBKR collection pipeline delivered to the shared database.

Colours follow the Okabe-Ito colour-blind-safe palette used by the result figures.

Usage: PYTHONPATH=src python scripts/architecture_figures.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Rectangle)

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

CARD_LIFT = [pe.SimplePatchShadow(offset=(1.1, -1.5), shadow_rgbFace="#4A4A4A",
                               alpha=0.10), pe.Normal()]

OUT = Path("dissertation/drafts/figures")

BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#7A7A7A"
PALE = "#F2F2F2"


def box(ax, x, y, w, h, text, *, face="white", edge=BLUE, lw=1.4, fs=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.05",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35, fontweight=weight)


def arrow(ax, p0, p1, *, color=GREY, lw=1.3, style="-|>", ls="-", rad=0.0):
    if ls == "-":
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=12.5, linewidth=lw,
            color=color, linestyle=ls, zorder=1, capstyle="round",
            connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=2))
        return

    # A dashed FancyArrowPatch strokes the head outline with the same dash pattern, so
    # the head comes out with bites taken out of it. Draw the shaft and the head apart.
    # arc3 is a quadratic Bezier whose control point is the midpoint pushed sideways.
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2 - rad * dy, (y0 + y1) / 2 + rad * dx
    t = [i / 200 for i in range(201)]
    bx = [(1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u ** 2 * x1 for u in t]
    by = [(1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u ** 2 * y1 for u in t]

    # Stop the dashes where the head begins, so no dash shows through it.
    head = 0.20
    cut = len(t) - 1
    while cut > 1 and math.hypot(bx[cut] - bx[-1], by[cut] - by[-1]) < head:
        cut -= 1
    ax.plot(bx[:cut + 1], by[:cut + 1], color=color, lw=lw, linestyle=ls,
            solid_capstyle="round", dash_capstyle="round", zorder=1)
    ax.annotate("", xy=(bx[-1], by[-1]), xytext=(bx[cut], by[cut]),
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=12.5),
                zorder=1)


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- A1: system
def fig_system():
    """The two paths, vertical, labels only.

    Every explanatory sentence lives in the LaTeX caption. Boxes carry a component
    name and one muted line of parameters or maths -- what the component IS, never
    why it is there. Arrow labels name the data carried, nothing else. Verification
    is the one filled box, because it is the contribution and the single point where
    the two paths meet.
    """
    from matplotlib.colors import to_rgb

    def tint(colour, amount=0.10):
        return tuple(1 - amount + amount * c for c in to_rgb(colour))

    def ink(colour, amount=0.60):
        return tuple(amount * c for c in to_rgb(colour))

    W, H = 8.0, 8.05
    fig, ax = blank(W, H)

    def node(x, y, w, h, title, sub, *, edge=BLUE, face="white", lw=1.0,
             title_fs=9.4, sub_fs=7.4, title_color="#1A1A1A",
             sub_color="#555555", align="center", indent=0.11):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2,
            path_effects=CARD_LIFT))
        cy = y + h / 2
        tx = x + indent if align == "left" else x + w / 2
        sx = x + 0.11 if align == "left" else x + w / 2
        if sub:
            n = sub.count("\n")
            ax.text(tx, cy + (0.22 if n else 0.14), title, ha=align,
                    va="center", fontsize=title_fs, zorder=3,
                    fontweight="semibold", color=title_color)
            ax.text(sx, cy - (0.04 if n else 0.16), sub, ha=align,
                    va=("top" if n else "center"), fontsize=sub_fs, zorder=3,
                    color=sub_color, linespacing=1.35)
        else:
            ax.text(tx, cy, title, ha=align, va="center",
                    fontsize=title_fs, zorder=3, fontweight="semibold",
                    color=title_color)

    def track(s):
        """Letter-spaced caps for the zone eyebrows (hair spaces)."""
        return " ".join(s)

    # -- the decision zone ------------------------------------------------------
    # The figure had a "SYSTEM ARCHITECTURE" title here. A LaTeX figure is named by
    # its caption, so it repeated that, competed with the two zone eyebrows, and
    # cost height in the tallest figure in the thesis. Removed 2026-08-20.

    def elbow(p0, p1, *, angleA, angleB, color, lw=1.4, ls="-"):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=12.5, linewidth=lw,
            color=color, linestyle=ls, zorder=1, capstyle="round",
            connectionstyle=f"angle,angleA={angleA},angleB={angleB},rad=10",
            shrinkA=1, shrinkB=2))

    ax.add_patch(FancyBboxPatch(
        (0.30, 2.40), 7.40, 5.37, boxstyle="round,pad=0.012,rounding_size=0.10",
        facecolor=tint(BLUE, 0.035), edgecolor=tint(BLUE, 0.30), linewidth=0.9,
        zorder=0))
    ax.text(3.90, 7.53, track("DECISION PATH"), fontsize=9.4, color=ink(BLUE),
            fontweight="bold", ha="right", va="center")
    ax.text(4.10, 7.53, "every trading day", fontsize=8.0, color="#777777",
            ha="left", va="center")

    # -- observation ------------------------------------------------------------
    node(0.45, 6.55, 7.10, 0.80, "Point-in-time observation",
         "prices, news, SEC filings, FRED macro",
         face=tint(BLUE, 0.08), edge=tint(BLUE, 0.50), sub_fs=7.4)

    # -- expert committee container, market state on its own branch -------------
    ax.add_patch(FancyBboxPatch(
        (0.45, 4.86), 5.10, 1.34, boxstyle="round,pad=0.012,rounding_size=0.10",
        facecolor=tint(BLUE, 0.05), edgecolor=tint(BLUE, 0.45), linewidth=0.9,
        zorder=1))
    ax.text(0.62, 6.04, track("EXPERT COMMITTEE"), fontsize=7.0,
            color=ink(BLUE), fontweight="semibold", ha="left", va="center",
            zorder=3)
    ex_y, ex_h, ex_w = 4.98, 0.84, 1.56
    experts = [
        (0.56, "Technical", "deterministic\nmomentum, reversal, low-vol"),
        (2.21, "News", "LLM\nheadlines, attention, macro"),
        (3.86, "Fundamental", "LLM\nSEC ratios, quarterly pulse"),
    ]
    for exx, t, sub in experts:
        node(exx, ex_y, ex_w, ex_h, t, sub, face="white",
             edge=tint(BLUE, 0.60), title_fs=8.4, sub_fs=5.6, align="left",
             indent=0.30)
    node(5.95, 5.13, 1.60, 0.80, "Market state", "SPY trend × vol rank",
         face=tint(SKY, 0.14), edge=tint(SKY, 0.55), sub_fs=6.4)

    arrow(ax, (3.00, 6.55), (3.00, 6.20), color=GREY, lw=1.55)
    arrow(ax, (6.75, 6.55), (6.75, 5.93), color=GREY, lw=1.55)

    # -- weighted vote and position --------------------------------------------
    node(1.85, 3.60, 2.30, 0.84, "Weighted vote",
         r"$A_t=\sum_{i\in\mathcal{S}_t}\tilde{w}_{i,m_t}\, v_{i,t}$",
         face=tint(BLUE, 0.12), edge=tint(BLUE, 0.70), lw=1.2,
         title_fs=9.2, sub_fs=7.8)
    arrow(ax, (3.00, 4.86), (3.00, 4.44), color=GREY, lw=1.55)
    ax.text(3.14, 4.66, "votes: long / flat / abstain", fontsize=6.8,
            color="#666666", ha="left", va="center")
    elbow((6.72, 5.13), (4.15, 4.22), angleA=-90, angleB=180, color=SKY,
          lw=1.55)
    ax.text(5.55, 4.40, r"state $m_t$", fontsize=7.2, color=ink(SKY),
            ha="center", va="center")

    node(2.05, 2.55, 1.90, 0.70, "Long or cash", "executed on the next bar",
         face=tint(GREEN, 0.10), edge=tint(GREEN, 0.55), sub_fs=6.8)
    arrow(ax, (3.00, 3.60), (3.00, 3.25), color=GREY, lw=1.55)

    # -- the learning band ------------------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.30, 0.42), 7.40, 1.53, boxstyle="round,pad=0.012,rounding_size=0.10",
        facecolor=tint(VERM, 0.05), edgecolor=tint(VERM, 0.35), linewidth=0.9,
        zorder=0))
    ax.text(3.90, 1.72, track("LEARNING PATH"), fontsize=9.4, color=ink(VERM),
            fontweight="bold", ha="right", va="center")
    ax.text(4.10, 1.72, "training window only", fontsize=8.0, color="#777777",
            ha="left", va="center")

    by, bh2 = 0.55, 0.84
    band = [
        (0.58, 1.45, "Attribution", r"$g_{i,t}=v_{i,t}\, r_{t\to t+21}$"),
        (2.24, 1.75, "Expert-trust memory",
         "EW mean, var, count\nper (expert, state) record"),
        (4.20, 1.45, "Reflection", r"bounded $|a_{i,m}|\leq 0.10$"),
        (5.85, 1.60, "Verification", "block bootstrap\nHolm $\\alpha=0.10$"),
    ]
    # All four learning-path steps share one treatment. Verification used to be set
    # solid, which made it read as a different kind of object from its three siblings.
    for bx, bw, t, sub in band:
        node(bx, by, bw, bh2, t, sub,
             face="white", edge=tint(VERM, 0.65),
             lw=1.0, title_fs=8.2, sub_fs=6.2, align="left",
             title_color="#1A1A1A", sub_color="#555555")
    for i in range(3):
        arrow(ax, (band[i][0] + band[i][1], by + bh2 / 2),
              (band[i + 1][0], by + bh2 / 2), color=VERM, lw=1.3)

    # -- pictograms, one monoline glyph per data-bearing module -----------------
    def ic_bars(cx, cy, c, a):
        for dx, hh in ((-0.055, 0.07), (0.0, 0.11), (0.055, 0.15)):
            ax.plot([cx + dx, cx + dx], [cy - 0.075, cy - 0.075 + hh],
                    color=c, alpha=a, lw=1.4, solid_capstyle="round", zorder=4)

    def ic_news(cx, cy, c, a):
        ax.add_patch(Rectangle((cx - 0.09, cy - 0.07), 0.18, 0.14, fill=False,
                               edgecolor=c, alpha=a, linewidth=1.1, zorder=4))
        ax.plot([cx - 0.055, cx], [cy + 0.033, cy + 0.033], color=c, alpha=a,
                lw=0.9, zorder=4)
        for yy in (0.0, -0.035):
            ax.plot([cx - 0.055, cx + 0.055], [cy + yy, cy + yy], color=c,
                    alpha=a, lw=0.9, zorder=4)

    def ic_doc(cx, cy, c, a):
        ax.add_patch(Rectangle((cx - 0.06, cy - 0.085), 0.12, 0.17, fill=False,
                               edgecolor=c, alpha=a, linewidth=1.1, zorder=4))
        for yy in (0.04, 0.0, -0.04):
            ax.plot([cx - 0.035, cx + 0.035], [cy + yy, cy + yy], color=c,
                    alpha=a, lw=0.9, zorder=4)

    def ic_gauge(cx, cy, c, a):
        ax.add_patch(Arc((cx, cy - 0.03), 0.19, 0.16, theta1=0, theta2=180,
                         edgecolor=c, alpha=a, linewidth=1.2, zorder=4))
        ax.plot([cx, cx + 0.055], [cy - 0.03, cy + 0.028], color=c, alpha=a,
                lw=1.2, solid_capstyle="round", zorder=4)

    def ic_check(cx, cy, c, a):
        ax.add_patch(Circle((cx, cy), 0.085, fill=False, edgecolor=c, alpha=a,
                            linewidth=1.2, zorder=4))
        ax.plot([cx - 0.04, cx - 0.008, cx + 0.048], [cy - 0.002, cy - 0.038,
                cy + 0.038], color=c, alpha=a, lw=1.4, solid_capstyle="round",
                zorder=4)

    def ic_person(cx, cy, c, a):
        ax.add_patch(Circle((cx, cy + 0.035), 0.036, fill=False, edgecolor=c,
                            alpha=a, linewidth=1.2, zorder=4))
        ax.add_patch(Arc((cx, cy - 0.075), 0.15, 0.13, theta1=15, theta2=165,
                         edgecolor=c, alpha=a, linewidth=1.2, zorder=4))

    def ic_gear(cx, cy, c, a):
        ax.add_patch(Circle((cx, cy), 0.048, fill=False, edgecolor=c, alpha=a,
                            linewidth=1.2, zorder=4))
        for k in range(8):
            th = math.radians(k * 45)
            ax.plot([cx + 0.055 * math.cos(th), cx + 0.082 * math.cos(th)],
                    [cy + 0.055 * math.sin(th), cy + 0.082 * math.sin(th)],
                    color=c, alpha=a, lw=1.2, solid_capstyle="round", zorder=4)

    def ic_ledger(cx, cy, c, a):
        ax.plot([cx - 0.02, cx - 0.02], [cy - 0.075, cy + 0.075], color=c,
                alpha=a, lw=1.1, zorder=4)
        for yy, dx in ((0.045, 0.075), (0.0, 0.045), (-0.045, -0.06)):
            ax.plot([cx - 0.02, cx - 0.02 + dx], [cy + yy, cy + yy], color=c,
                    alpha=a, lw=1.5, solid_capstyle="round", zorder=4)

    def ic_db(cx, cy, c, a):
        ax.add_patch(Ellipse((cx, cy + 0.05), 0.15, 0.05, fill=False,
                             edgecolor=c, alpha=a, linewidth=1.1, zorder=4))
        for dx in (-0.075, 0.075):
            ax.plot([cx + dx, cx + dx], [cy + 0.05, cy - 0.05], color=c,
                    alpha=a, lw=1.1, zorder=4)
        ax.add_patch(Arc((cx, cy - 0.05), 0.15, 0.05, theta1=180, theta2=360,
                         edgecolor=c, alpha=a, linewidth=1.1, zorder=4))

    def ic_loop(cx, cy, c, a):
        ax.add_patch(Arc((cx, cy), 0.13, 0.13, theta1=-30, theta2=225,
                         edgecolor=c, alpha=a, linewidth=1.2, zorder=4))
        th = math.radians(-30)
        ax.plot([cx + 0.065 * math.cos(th)], [cy + 0.065 * math.sin(th)],
                marker=(3, 0, 155), markersize=3.6, color=c, alpha=a, zorder=4)

    ic_gear(0.56 + 0.155, 5.62, BLUE, 0.65)
    ic_person(2.21 + 0.155, 5.63, BLUE, 0.65)
    ic_person(3.86 + 0.155, 5.63, BLUE, 0.65)
    ic_bars(0.56 + 1.42, 5.64, BLUE, 0.65)
    ic_news(2.21 + 1.42, 5.64, BLUE, 0.65)
    ic_doc(3.86 + 1.42, 5.64, BLUE, 0.65)
    ic_gauge(7.37, 5.76, SKY, 0.90)
    ic_ledger(0.58 + 1.30, 1.21, VERM, 0.60)
    ic_db(2.24 + 1.58, 1.21, VERM, 0.60)
    ic_loop(4.20 + 1.30, 1.21, VERM, 0.60)
    ic_check(7.27, 1.21, VERM, 0.60)

    # -- the two crossings ------------------------------------------------------
    # expert views and the elapsed forward return drop into attribution
    arrow(ax, (1.00, 4.86), (1.31, by + bh2), color="#9A9A9A",
          ls=(0, (5, 3.5)), lw=1.15, rad=0.08)
    ax.text(0.70, 3.20, "expert views, next-month return", fontsize=7.0,
            color="#666666", ha="center", va="center", rotation=90)

    # verification releases the frozen weights into the vote
    elbow((6.72, by + bh2), (4.15, 3.86), angleA=90, angleB=180, color=GREEN,
          lw=1.95)
    ax.text(5.55, 3.64, "frozen weights", fontsize=8.0, color=ink(GREEN),
            ha="center", fontweight="bold")

    save(fig, "fig_system")


# -------------------------------------------------------------- A2: protocol
def fig_protocol():
    """The four evaluation windows.

    Labels only: everything that explains the design lives in the LaTeX caption, where
    it can be edited and read beside the text rather than baked into a PDF.

    Three conventions the earlier drafts broke. One date format throughout, ISO as the
    text uses, and the same shape in every box; the forward window was the odd one out
    in both. One duration unit, trading days, counted from the exchange calendar in the
    panel, which is where the 3,145 of Section 3.2 comes from; years, months and days
    mixed made the windows incomparable at a glance. And the separator belongs at the
    middle of the gap between two boxes, not at a box's right edge, where it draws over
    the rounded border and reads as misalignment.

    Widths are schematic, but ordered by duration: training > test > embargo > forward.
    """
    from matplotlib.colors import to_rgb

    def tint(colour, amount=0.10):
        """The border colour lifted most of the way to white, for a fill."""
        return tuple(1 - amount + amount * c for c in to_rgb(colour))

    def ink(colour, amount=0.65):
        """The same colour darkened for text.

        Okabe-Ito sky blue on its own tint is about 2.1:1, so the embargo label was the
        one word on the figure that had to be squinted at. Scaled to \\textwidth the
        labels print at roughly five points, which is normal text rather than large, so
        the bar is 4.5:1. At 0.65 the weakest of the four reaches 4.7 and the rest run
        6.2 to 8.2, and darkening all of them by the same factor keeps them consistent.
        """
        return tuple(amount * c for c in to_rgb(colour))

    W, H = 12.0, 2.6
    fig, ax = blank(W, H)
    y, bh = 0.75, 0.95
    segs = [
        (0.25, 4.15, "Training", "2011-07-01 to\n2023-12-31", BLUE, "3,145 days"),
        (4.70, 1.95, "Embargo", "2024-01-01 to\n2024-06-30", SKY, "124 days"),
        (6.95, 2.70, "Test", "2024-07-09 to\n2025-12-31", VERM, "373 days, 352 scored"),
        (9.95, 1.80, "Forward", "2026-08-03 to\n2026-08-14", GREEN, "10 days"),
    ]
    edges = [x + w for x, w, *_ in segs[:-1]]
    starts = [x for x, *_ in segs[1:]]
    seps = [(a + b) / 2 for a, b in zip(edges, starts)]

    for x, w, name, dates, colour, note in segs:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, bh, boxstyle="round,pad=0.012,rounding_size=0.05",
            facecolor=tint(colour), edgecolor=colour, linewidth=1.5, zorder=2))
        # The window's name carries the colour and the weight; the dates sit under it in
        # neutral grey, so the eye reads four windows first and their spans second.
        ax.text(x + w / 2, y + 0.68, name, ha="center", va="center", fontsize=9.6,
                color=ink(colour), fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + 0.29, dates, ha="center", va="center", fontsize=8,
                color="#333333", linespacing=1.35, zorder=3)
        ax.text(x + w / 2, y - 0.28, note, ha="center", fontsize=7.6,
                color=ink(colour), style="italic")

    for x in seps:
        ax.plot([x, x], [y - 0.10, y + bh + 0.12], color=GREY, lw=1.0, ls=(0, (2, 2)),
                zorder=1)

    ax.annotate("FREEZE", xy=(seps[0], y + bh + 0.14), xytext=(seps[0], y + bh + 0.52),
                fontsize=9.2, ha="center", va="bottom", color=VERM, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=VERM, lw=1.5,
                                shrinkA=2, shrinkB=0))
    save(fig, "fig_protocol")


# ------------------------------------------------------------------ A3: data
def fig_data():
    W, H = 12.0, 3.7
    fig, ax = blank(W, H)
    y, bh2 = 1.55, 1.15

    box(ax, 0.2, y, 2.0, bh2,
        "IB Gateway\n" + r"$\it{local\ session}$" + "\nmarket data + news",
        edge=GREEN, fs=8.4)
    box(ax, 2.75, y, 1.9, bh2, "Raw\n" + r"$\it{as\ received}$", edge=SKY, fs=8.4)
    box(ax, 5.2, y, 2.3, bh2,
        "Intermediate\nindicators, factors\n" + r"$\it{aligned\ per\ symbol}$",
        edge=SKY, fs=8.4)
    box(ax, 8.05, y, 2.2, bh2,
        "Final\npartitioned parquet\n" + r"$\it{Spark\text{-}readable}$", edge=BLUE, fs=8.4)
    box(ax, 10.8, y, 1.0, bh2, "Shared\nDB", edge=BLUE, fs=8.4, weight="bold")
    for a, b in ((2.2, 2.75), (4.65, 5.2), (7.5, 8.05), (10.25, 10.8)):
        arrow(ax, (a, y + bh2 / 2), (b, y + bh2 / 2))

    facts = [
        "814,022 daily bars",
        "658 S&P 500 symbols",
        "2021-07-26 to 2026-07-24",
        "14 technical indicators",
        "8 Alpha101 factors",
        "200 IBKR news headlines",
    ]
    ax.text(0.2, 1.12, "Coverage", fontsize=9, fontweight="bold", va="top", color=BLUE)
    for i, f in enumerate(facts):
        col, row = divmod(i, 3)
        ax.text(0.25 + col * 3.4, 0.78 - row * 0.30, "•  " + f, fontsize=8.3, va="top")

    ax.text(7.2, 1.12, "Contract and orchestration", fontsize=9, fontweight="bold",
            va="top", color=BLUE)
    for i, f in enumerate(["schema.md fixes the column contract",
                           "aligned row for row on (symbol, event_date)",
                           "two Airflow DAGs, plus a continuous intraday stream"]):
        ax.text(7.25, 0.78 - i * 0.30, "•  " + f, fontsize=8.3, va="top")

    ax.text(1.2, y + bh2 + 0.32,
            "local API: the integration point is the parquet output, not a remote call",
            fontsize=7.8, color=VERM, style="italic", ha="left")
    save(fig, "fig_data")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_system()
    fig_protocol()
    fig_data()
    for f in ("fig_system", "fig_protocol", "fig_data"):
        print(f"WROTE {OUT / f}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
