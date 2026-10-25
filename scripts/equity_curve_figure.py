"""The portfolio equity curve: committee, verified global arm, buy and hold.

Reads exactly what the thesis verifier reads: the twenty sealed reports resolved
by portfolio20, repriced at the next open from the panel, netted at ten basis
points per side on exposure changes; buy and hold from the baseline artifact,
whose manifest declares its cost basis. Before anything is drawn, the three
portfolio Sharpe ratios are asserted against the values tab:baselines carries,
so the figure cannot drift from the table.

    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/equity_curve_figure.py
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "dissertation" / "drafts" / "figures"
# FINSABER's own schedule, the one every row in tab:baselines is charged. The
# commission is a per-share amount, so a gross series needs the price each trade fills
# at, which under this execution is the next session's open.
PER_SHARE, MIN_COMMISSION, MAX_RATE, CASH = 0.0049, 0.99, 0.01, 100_000.0
BLUE, VERM = "#0072B2", "#D55E00"
# FinMem is a named baseline, so it carries a hue rather than a darker shade of the
# anonymous field. GREEN is the Okabe-Ito slot not already taken by BLUE or VERM.
DARK, GREEN, FIELD = "#3D3D3D", "#009E73", "#E3E3E3"

DRAW = ("ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX",
        "DDS", "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT")

# The values tab:baselines prints, used only as assertions.
EXPECT_SR = {"equal_weight": 0.770, "global_memory": 0.855, "buy_and_hold": 0.852}

# The rest of tab:baselines, drawn as the grey field behind the three focal books.
FIELD_STEMS = ("TurnOfTheMonthStrategy", "SMACrossStrategy", "WMAStrategy",
               "BollingerBandsStrategy", "ATRBandStrategy",
               "TrendFollowingStrategy", "AlphaMining_ridgepx",
               "AlphaMining_gbtpx")


def net(rows: list, opens_by_date: dict, calendar: list) -> list[float]:
    idx = {d: i for i, d in enumerate(calendar)}
    out, equity, prev = [], CASH, 0.0
    for date, r, e in rows:
        i = idx.get(date)
        nxt = calendar[i + 1] if i is not None and i + 1 < len(calendar) else None
        px = opens_by_date.get(nxt) if nxt else None
        traded = abs(e - prev)
        frac = (min(max(PER_SHARE / px, MIN_COMMISSION / equity), MAX_RATE)
                if (traded and px) else 0.0)
        out.append(r - frac * traded)
        equity *= (1.0 + r - frac * traded)
        prev = e
    return out


def sharpe(r: list[float]) -> float:
    m = sum(r) / len(r)
    v = sum((x - m) ** 2 for x in r) / (len(r) - 1)
    return m / math.sqrt(v) * math.sqrt(252) if v > 0 else 0.0


def main() -> int:
    from portfolio20 import resolve
    from reprice_next_open import adjusted_prices, reprice

    sealed, missing = resolve()
    assert not missing, missing
    opens, closes = adjusted_prices(list(DRAW))

    def baseline_maps(stem: str) -> list[dict]:
        maps = []
        for t in DRAW:
            cal_t = sorted(opens[t])
            d = json.loads((ROOT / "reports" / "baselines" /
                            f"{t}__{stem}.json").read_text(encoding="utf-8"))
            rows = d["daily_arm_series"]["equal_weight"]
            declared = d["manifest"]["returns"]
            assert declared in ("net", "gross"), declared
            vals = ([r[1] for r in rows] if declared == "net"
                    else net(rows, opens[t], cal_t))
            maps.append(dict(zip([r[0] for r in rows], vals)))
        return maps

    by_stock: dict[str, list[dict]] = {"equal_weight": [], "global_memory": []}
    calendar = None
    for t in DRAW:
        rep = json.loads(sealed[t].read_text(encoding="utf-8"))
        cal = sorted(opens[t])
        for arm in ("equal_weight", "global_memory"):
            rows = reprice(rep["daily_arm_series"][arm], opens[t], closes[t], cal)
            if t == "ADSK" and arm == "equal_weight":
                calendar = [r[0] for r in rows]
            by_stock[arm].append(dict(zip([r[0] for r in rows],
                                          net(rows, opens[t], cal))))
    by_stock["buy_and_hold"] = baseline_maps("BuyAndHoldNextOpen")
    by_stock["finmem"] = baseline_maps("FinMem")
    for stem in FIELD_STEMS:
        by_stock[f"field_{stem}"] = baseline_maps(stem)

    port = {}
    for arm, maps in by_stock.items():
        assert all(set(m) == set(calendar) for m in maps), arm
        port[arm] = [sum(m[d] for m in maps) / len(maps) for d in calendar]
        if arm in EXPECT_SR:
            got = sharpe(port[arm])
            assert abs(got - EXPECT_SR[arm]) < 5e-3, (arm, got)
            print(f"{arm:<14} SR {got:+.3f}  terminal "
                  f"{(math.prod(1 + x for x in port[arm]) - 1) * 100:+.2f}%")

    dates = [datetime.strptime(d, "%Y-%m-%d") for d in calendar]
    cum = {}
    for arm, r in port.items():
        c, w = [], 1.0
        for x in r:
            w *= 1 + x
            c.append((w - 1) * 100)
        cum[arm] = c

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    # Drawn as individual strokes, not as a filled envelope: a shaded region in a
    # cumulative-return chart reads as a confidence interval, and these are eight
    # separate strategies rather than uncertainty around the named series. They are
    # light enough to sit behind rather than compete.
    for arm in by_stock:
        if arm.startswith("field_"):
            ax.plot(dates, cum[arm], color=FIELD, lw=0.6, zorder=1)
    # A surface ring under every emphasised line, so the named series stay legible
    # where they cross each other and the grey field.
    _ring = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
    ax.plot(dates, cum["finmem"], color=GREEN, lw=1.4, ls=(0, (5, 2)), zorder=2,
            path_effects=_ring)
    style = {"buy_and_hold": (DARK, 1.5, "Buy and hold"),
             "equal_weight": (BLUE, 2.3, "Equal-weight committee"),
             "global_memory": (VERM, 1.7, "Verified global memory")}
    # The ring separates a line from marks it crosses. Buy and hold crosses the
    # field and FinMem, so it gets one. The committee and the memory coincide on
    # most days because the memory rarely overrides it, which is the first result
    # in the thesis; a halo between them would delete exactly that. Instead the
    # committee is drawn heavier and the memory thinner on top, so both show.
    for arm in ("buy_and_hold", "equal_weight", "global_memory"):
        colour, lw, label = style[arm]
        ax.plot(dates, cum[arm], color=colour, lw=lw, solid_capstyle="round",
                zorder=3, path_effects=_ring if arm == "buy_and_hold" else None)
    # Direct end labels in the right margin. Their positions used to be hand-tuned
    # offsets, which collided the moment the cost model moved FinMem up beside the
    # committee; they are spread here instead, so the figure survives the numbers.
    labels = [(cum["buy_and_hold"][-1], "Buy and hold", "#333333"),
              (cum["global_memory"][-1], "Verified global memory",
               tuple(0.72 * c for c in matplotlib.colors.to_rgb(VERM))),
              (cum["equal_weight"][-1], "Equal-weight committee",
               tuple(0.72 * c for c in matplotlib.colors.to_rgb(BLUE))),
              (cum["finmem"][-1], "FinMem",
               tuple(0.72 * c for c in matplotlib.colors.to_rgb(GREEN))),
              (min(cum[a][-1] for a in cum if a.startswith("field_")),
               "timing rules, factor models", "#A5A5A5")]
    labels.sort(key=lambda r: r[0])
    gap = 0.045 * (max(r[0] for r in labels) - min(r[0] for r in labels))
    placed = []
    for y, label, colour in labels:
        if placed and y - placed[-1][0] < gap:
            y = placed[-1][0] + gap
        placed.append((y, label, colour))
    for y, label, colour in placed:
        ax.annotate(label, (dates[-1], y), xytext=(7, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.2, clip_on=False, annotation_clip=False,
                    color=colour)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Cumulative return (%)", fontsize=9)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(labelsize=8.4)
    ax.set_xlim(dates[0], dates[-1])
    fig.subplots_adjust(left=0.07, right=0.80, top=0.97, bottom=0.12)
    fig.savefig(OUT / "equity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "equity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {OUT / 'equity.pdf'} / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
