"""Time-series short-term reversal signal (ts_reversal).

Deterministic, point-in-time: the NEGATIVE ``lookback``-day return, skipping the
most recent ``skip`` day(s) to avoid microstructure noise, z-scored against the
stock's own prior ``history_window`` days of the same rolling quantity
(prior-only, excluding the current value). Positive = the stock is more oversold
than its own recent norm (bounce proposal). The time-series formulation stays
informative for a single-name universe, unlike a cross-sectional z-score.
Emits Signal(value=z, source_id="ts_reversal").
"""

from __future__ import annotations

from typing import Sequence

from trading_agent.alpha.momentum import ts_zscore
from trading_agent.contracts import Observation, Signal
from trading_agent.market import LoaderPricePanel

SIGNAL_ID = "ts_reversal"


class ReversalAlpha:
    def __init__(
        self,
        panel: LoaderPricePanel,
        lookback: int = 5,
        skip: int = 1,
        history_window: int = 252,
        min_history: int = 60,
    ):
        if lookback < 1 or skip < 0:
            raise ValueError("lookback>=1, skip>=0")
        if min_history < 2 or history_window < min_history:
            raise ValueError("need history_window >= min_history >= 2")
        self._panel = panel
        self._lookback = lookback
        self._skip = skip
        self._span = lookback + skip  # closes consumed by one reversal value
        self._history_window = history_window
        self._min_history = min_history
        self.source_id = SIGNAL_ID

    def _rolling_reversals(self, closes: Sequence[float]) -> list[float]:
        # Value at end index e: -(close_{e-skip} / close_{e-skip-lookback} - 1).
        out = []
        for e in range(self._span, len(closes)):
            c_old = closes[e - self._skip - self._lookback]
            if c_old <= 0:
                continue
            out.append(-(closes[e - self._skip] / c_old - 1.0))
        return out

    def score(self, observation: Observation) -> Sequence[Signal]:
        t = observation.decision_date
        need = self._span + self._min_history + 1
        signals: list[Signal] = []
        for ticker in observation.tickers:
            closes = self._panel.trailing(
                ticker, t, self._span + self._history_window + 1
            )
            if len(closes) < need:
                continue
            rolling = self._rolling_reversals(closes)
            if len(rolling) < self._min_history + 1:
                continue
            current, history = rolling[-1], rolling[:-1]
            signals.append(
                Signal(ticker=ticker, value=ts_zscore(current, history),
                       source_id=self.source_id, as_of=t)
            )
        return signals
