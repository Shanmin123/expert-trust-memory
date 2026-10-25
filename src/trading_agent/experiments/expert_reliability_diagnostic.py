"""E0: LLM expert reliability diagnostic.

Before any learning claim, this diagnostic checks the two premises the memory
and reflection layer rests on, computed directly from recorded expert views:

1. Calibration by market state: do the LLM experts' stated confidences match
   their realized directional accuracy, and does that calibration shift across
   market states? A gap between mean confidence and empirical accuracy that
   changes with the state is the regime-dependent miscalibration the memory
   layer is meant to learn to correct.

2. Directional behaviour by market state: how often each expert goes long and
   how confident it is, by state -- the "hold back when rising, over-trade when
   falling" pattern documented for LLM traders.

3. Cross-expert correlation: how redundant the experts are. A learned
   weighting is only meaningful over experts that carry different information.

The diagnostic consumes attributed episodes (which already exclude safe-default
views), so parse errors and unavailable-data defaults never enter the numbers.
It performs no market simulation and issues no LLM calls; it is a pure
post-processor and can run over a frozen view corpus replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Optional, Sequence

DEFAULT_CONFIDENCE_BINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class ViewOutcome:
    """One expert judgment on one (date, ticker) with its realized outcome."""

    decision_date: str
    ticker: str
    market_state: str
    expert_id: str
    direction: str
    confidence: float
    realized_next_return: float
    direction_correct: bool

    @property
    def signed_confidence(self) -> float:
        """Directional stance in the long/flat action space: conf if long else 0."""
        return float(self.confidence) if self.direction == "long" else 0.0


def view_outcomes_from_episodes(episodes: Sequence[object]) -> list[ViewOutcome]:
    """Flatten attributed episodes into per-(date, ticker, expert) view outcomes.

    Episodes carry ``expert_errors`` records, which are built only from non-default
    views, so safe defaults are already excluded here.
    """
    out: list[ViewOutcome] = []
    for episode in episodes:
        ticker = getattr(episode, "ticker", None)
        date = getattr(episode, "decision_date", None)
        for record in getattr(episode, "expert_errors", ()) or ():
            out.append(ViewOutcome(
                decision_date=(date.isoformat() if hasattr(date, "isoformat") else str(date)),
                ticker=str(ticker) if ticker is not None else "__portfolio__",
                market_state=str(record.market_state),
                expert_id=str(record.expert_id),
                direction=str(record.direction),
                confidence=float(record.confidence),
                realized_next_return=float(record.realized_next_return),
                direction_correct=bool(record.direction_correct),
            ))
    return out


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def calibration_by_state(
    outcomes: Sequence[ViewOutcome],
    *,
    bins: Sequence[float] = DEFAULT_CONFIDENCE_BINS,
) -> dict[str, dict[str, dict]]:
    """Per (market_state, expert): mean confidence vs empirical directional accuracy.

    ``overconfidence_gap`` is mean confidence minus empirical accuracy; a positive
    gap means the expert is more confident than it is right.
    """
    edges = list(bins)
    grouped: dict[tuple[str, str], list[ViewOutcome]] = {}
    for o in outcomes:
        grouped.setdefault((o.market_state, o.expert_id), []).append(o)

    out: dict[str, dict[str, dict]] = {}
    for (state, expert_id), rows in sorted(grouped.items()):
        n = len(rows)
        mean_conf = fmean(o.confidence for o in rows)
        accuracy = sum(1 for o in rows if o.direction_correct) / n
        bin_rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Last bin is closed on the right so confidence == 1.0 is included.
            is_last = hi == edges[-1]
            if is_last:
                members = [o for o in rows if lo <= o.confidence <= hi]
            else:
                members = [o for o in rows if lo <= o.confidence < hi]
            if not members:
                continue
            bin_rows.append({
                "lo": round(lo, 6),
                "hi": round(hi, 6),
                "n": len(members),
                "mean_confidence": round(fmean(o.confidence for o in members), 6),
                "empirical_accuracy": round(
                    sum(1 for o in members if o.direction_correct) / len(members), 6
                ),
            })
        out.setdefault(state, {})[expert_id] = {
            "n": n,
            "mean_confidence": round(mean_conf, 6),
            "empirical_accuracy": round(accuracy, 6),
            "overconfidence_gap": round(mean_conf - accuracy, 6),
            "bins": bin_rows,
        }
    return out


def directional_behavior_by_state(
    outcomes: Sequence[ViewOutcome],
) -> dict[str, dict[str, dict]]:
    """Per (market_state, expert): long rate and mean confidence."""
    grouped: dict[tuple[str, str], list[ViewOutcome]] = {}
    for o in outcomes:
        grouped.setdefault((o.market_state, o.expert_id), []).append(o)

    out: dict[str, dict[str, dict]] = {}
    for (state, expert_id), rows in sorted(grouped.items()):
        n = len(rows)
        out.setdefault(state, {})[expert_id] = {
            "n": n,
            "long_rate": round(sum(1 for o in rows if o.direction == "long") / n, 6),
            "mean_confidence": round(fmean(o.confidence for o in rows), 6),
        }
    return out


def expert_signed_confidence_correlation(
    outcomes: Sequence[ViewOutcome],
) -> dict[str, object]:
    """Pairwise Pearson correlation of signed confidence, aligned by (date, ticker)."""
    by_cell: dict[tuple[str, str], dict[str, float]] = {}
    experts: set[str] = set()
    for o in outcomes:
        experts.add(o.expert_id)
        by_cell.setdefault((o.decision_date, o.ticker), {})[o.expert_id] = o.signed_confidence

    expert_ids = sorted(experts)
    pairs: dict[str, Optional[float]] = {}
    matrix: list[list[Optional[float]]] = []
    for a in expert_ids:
        row: list[Optional[float]] = []
        for b in expert_ids:
            if a == b:
                row.append(1.0)
                continue
            xs, ys = [], []
            for cell in by_cell.values():
                if a in cell and b in cell:
                    xs.append(cell[a])
                    ys.append(cell[b])
            r = _pearson(xs, ys)
            row.append(None if r is None else round(r, 6))
            if a < b:
                pairs[f"{a}|{b}"] = None if r is None else round(r, 6)
        matrix.append(row)
    return {"experts": expert_ids, "matrix": matrix, "pairs": pairs}


def buy_hold_sell_summary(episodes: Sequence[object]) -> dict[str, object]:
    """Label each day's action as BUY / HOLD / SELL from the long/flat targets.

    Mainstream single-name agents (FinMem, TradingAgents, FINSABER) emit a daily
    BUY/HOLD/SELL action; our long-only long/flat target produces exactly those
    transitions per ticker: flat->long is BUY, long->flat is SELL, no change is
    HOLD. This is a reporting view of the same decisions, computed per ticker in
    date order.
    """
    from collections import defaultdict

    by_ticker: dict[object, list] = defaultdict(list)
    for ep in episodes:
        by_ticker[getattr(ep, "ticker", None)].append(ep)

    counts = {"buy": 0, "hold": 0, "sell": 0}
    for eps in by_ticker.values():
        prev_long = False
        for ep in sorted(eps, key=lambda e: e.decision_date):
            is_long = getattr(ep, "action", "flat") == "long"
            if is_long and not prev_long:
                action = "buy"
            elif prev_long and not is_long:
                action = "sell"
            else:
                action = "hold"
            counts[action] += 1
            prev_long = is_long
    total = counts["buy"] + counts["hold"] + counts["sell"]
    return {
        **counts,
        "n": total,
        "hold_rate": round(counts["hold"] / total, 6) if total else 0.0,
    }


def expert_reliability_report(episodes: Sequence[object]) -> dict[str, object]:
    """Full E0 report from attributed episodes."""
    outcomes = view_outcomes_from_episodes(episodes)
    return {
        "n_view_outcomes": len(outcomes),
        "calibration_by_state": calibration_by_state(outcomes),
        "directional_behavior_by_state": directional_behavior_by_state(outcomes),
        "expert_correlation": expert_signed_confidence_correlation(outcomes),
    }


def view_direction_summary(records: Mapping[str, Sequence[Mapping]]) -> dict[str, dict]:
    """Direction mix per (expert, market_state) from the RAW per-view records.

    Design D3 makes "neutral" a deliberate abstention that never enters
    episodes (``signal_attribution`` and ``expert_errors`` both exclude it),
    so every episode-based summary above is neutral-blind. This summary is
    built from ``ExpertLoopResult.expert_view_records`` — captured in the run
    loop BEFORE any filtering — and is therefore the one surface where the
    neutral usage rate is visible. Safe defaults are failures, not judgments:
    they are tallied separately and excluded from the direction mix (their
    placeholder direction must not be double counted as a flat vote).

    Rates per cell:
      ``neutral_rate`` = neutral / (long + neutral + flat)   (abstention share
      among actual judgments); ``spoken_rate`` = (long + flat) / n (share of
      recorded views that entered the vote).
    """
    out: dict[str, dict] = {}
    for expert_id, rows in sorted(records.items()):
        by_state: dict[str, dict] = {}
        for row in rows:
            state = str(row.get("market_state"))
            cell = by_state.setdefault(
                state, {"long": 0, "neutral": 0, "flat": 0, "safe_default": 0, "n": 0}
            )
            cell["n"] += 1
            if row.get("safe_default"):
                cell["safe_default"] += 1
                continue
            direction = str(row.get("direction"))
            if direction in ("long", "neutral", "flat"):
                cell[direction] += 1
        total = {"long": 0, "neutral": 0, "flat": 0, "safe_default": 0, "n": 0}
        for cell in by_state.values():
            for key in total:
                total[key] += cell[key]
        for cell in [*by_state.values(), total]:
            judgments = cell["long"] + cell["neutral"] + cell["flat"]
            cell["neutral_rate"] = (
                round(cell["neutral"] / judgments, 6) if judgments else 0.0
            )
            cell["spoken_rate"] = (
                round((cell["long"] + cell["flat"]) / cell["n"], 6) if cell["n"] else 0.0
            )
        out[expert_id] = {"by_state": by_state, "all_states": total}
    return out
