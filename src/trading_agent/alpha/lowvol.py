"""Time-series low-volatility defensive signal (ts_lowvol).

Deterministic, point-in-time: the NEGATIVE trailing ``window``-day realized
volatility (std of daily simple returns over data <= t), z-scored against the
stock's own prior ``history_window`` days of the same rolling quantity
(prior-only, excluding the current value). Positive = the stock is calmer than
its own recent norm (defensive). Same construction as the market-state RISK axis
but applied to the single stock; stays informative for a single-name universe.
Economic basis: the low-vol anomaly. Emits Signal(value=z, source_id="ts_lowvol").
"""

from __future__ import annotations

from statistics import pstdev
from typing import Sequence

from trading_agent.alpha.momentum import ts_zscore
from trading_agent.contracts import Observation, Signal
from trading_agent.market import LoaderPricePanel

SIGNAL_ID = "ts_lowvol"


class LowVolAlpha:
    def __init__(
        self,
        panel: LoaderPricePanel,
        window: int = 60,
        history_window: int = 252,
        min_history: int = 60,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        if min_history < 2 or history_window < min_history:
            raise ValueError("need history_window >= min_history >= 2")
        self._panel = panel
        self._window = window
        self._history_window = history_window
        self._min_history = min_history
        self.source_id = SIGNAL_ID

    def _rolling_neg_vols(self, closes: Sequence[float]) -> list[float]:
        rets = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        w = self._window
        return [-pstdev(rets[i - w:i]) for i in range(w, len(rets) + 1)]

    def score(self, observation: Observation) -> Sequence[Signal]:
        t = observation.decision_date
        need = self._window + self._min_history + 1
        signals: list[Signal] = []
        for ticker in observation.tickers:
            closes = self._panel.trailing(
                ticker, t, self._window + self._history_window + 1
            )
            if len(closes) < need or any(c <= 0 for c in closes):
                continue
            rolling = self._rolling_neg_vols(closes)
            if len(rolling) < self._min_history + 1:
                continue
            current, history = rolling[-1], rolling[:-1]
            signals.append(
                Signal(ticker=ticker, value=ts_zscore(current, history),
                       source_id=self.source_id, as_of=t)
            )
        return signals
