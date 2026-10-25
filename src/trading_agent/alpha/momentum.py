"""Time-series momentum signal (ts_momentum).

Deterministic, transparent, point-in-time: at decision date t, the trailing
``lookback``-day return is z-scored against the distribution of the SAME rolling
return over the stock's own prior ``history_window`` days (prior-only, excluding
the current value). Positive = stronger than the stock's own recent past. This is
the time-series momentum formulation (Moskowitz-Ooi-Pedersen 2012), which answers
the long/flat timing question and stays informative for a single-name universe,
unlike a cross-sectional z-score, which degenerates to 0 with one ticker.
Emits Signal(value=z, source_id="ts_momentum").
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Sequence

from trading_agent.contracts import Observation, Signal
from trading_agent.market import LoaderPricePanel

SIGNAL_ID = "ts_momentum"


def ts_zscore(current: float, history: Sequence[float]) -> float:
    """Z-score of ``current`` against the stock's own prior-only history."""
    if not history:
        return 0.0
    sigma = pstdev(history) if len(history) > 1 else 0.0
    if sigma == 0:
        return 0.0
    return (current - mean(history)) / sigma


class MomentumAlpha:
    def __init__(
        self,
        panel: LoaderPricePanel,
        lookback: int = 10,
        history_window: int = 252,
        min_history: int = 60,
    ):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if min_history < 2 or history_window < min_history:
            raise ValueError("need history_window >= min_history >= 2")
        self._panel = panel
        self._lookback = lookback
        self._history_window = history_window
        self._min_history = min_history
        self.source_id = SIGNAL_ID

    def _rolling_returns(self, closes: Sequence[float]) -> list[float]:
        k = self._lookback
        return [
            closes[i + k] / closes[i] - 1.0
            for i in range(len(closes) - k)
            if closes[i] > 0
        ]

    def score(self, observation: Observation) -> Sequence[Signal]:
        t = observation.decision_date
        need = self._lookback + self._min_history + 1
        signals: list[Signal] = []
        for ticker in observation.tickers:
            closes = self._panel.trailing(
                ticker, t, self._lookback + self._history_window + 1
            )
            if len(closes) < need:
                continue
            rolling = self._rolling_returns(closes)
            if len(rolling) < self._min_history + 1:
                continue
            current, history = rolling[-1], rolling[:-1]
            signals.append(
                Signal(ticker=ticker, value=ts_zscore(current, history),
                       source_id=self.source_id, as_of=t)
            )
        return signals
