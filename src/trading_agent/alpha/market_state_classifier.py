"""Market-state regime classifier: 2D trend x risk, causal, point-in-time.

The design conditions memory on a single regime label. That label is
TWO-DIMENSIONAL: a trend dimension crossed with a realized-volatility (risk) dimension,
giving four states. The motivation is directional: an agent that is too conservative when
prices rise and too aggressive when prices fall fails along a trend axis that a
volatility-only label cannot see (a crisis decline and a violent rebound would otherwise
share one high-volatility state, so the trust learned there would average two opposite
behaviours).

Recipe (all strictly causal -- only data with date <= t is read):
  * market proxy = an external LEVEL series (e.g. SPY close) for single-name runs, or the
    equal-weight synthetic index of the traded universe for multi-name runs. For a single
    name the proxy MUST be external, never the traded stock itself, or the stock's own
    momentum would leak into the regime label.
  * TREND: proxy close vs its trailing ``trend_window`` (200) simple moving average. A
    switch of side is confirmed only after ``trend_confirm_days`` (5) consecutive closes on
    the new side, so the state does not oscillate near the average.  -> "up" / "down".
  * RISK: realized volatility over the trailing ``vol_window`` (21) daily LOG returns,
    zero-mean:  RV = sqrt(annualization * mean(r^2)).  The 21-day sample mean is too noisy
    to subtract, so the standard realized-vol / vol-targeting convention of not demeaning
    is used. RV is ranked within a trailing ``percentile_window`` (504) window that
    INCLUDES the current day; a rank >= ``stress_percentile`` (0.50, the median) -> "stress",
    otherwise "calm".
  * STATE = "{trend}_{risk}":  up_calm / up_stress / down_calm / down_stress.

The trend dimension follows time-series momentum (Moskowitz, Ooi & Pedersen 2012); the risk
dimension follows the volatility-regime / vol-targeting literature (Ang & Bekaert; Moreira
& Muir). VIX and HMM are diagnostic / robustness only and are NOT state sources here.

During warmup (insufficient history for the moving average, the realized vol, or its
reference window) the classifier returns ``"warmup"``. Output is always one of
MARKET_STATES or "warmup" -- never anything else.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Optional, Sequence

from trading_agent.market import LoaderPricePanel, _to_date

#: The four trend x risk states. Kept in sync with memory.expert_weights.MARKET_STATES
#: (a test asserts equality) without importing across the alpha/memory boundary.
MARKET_STATES: tuple[str, ...] = ("up_calm", "up_stress", "down_calm", "down_stress")
WARMUP: str = "warmup"

#: Human-readable display names for reports (snake_case is the canonical code/DB form).
STATE_DISPLAY: dict[str, str] = {
    "up_calm": "Up-Calm",
    "up_stress": "Up-Stress",
    "down_calm": "Down-Calm",
    "down_stress": "Down-Stress",
}


@dataclass(frozen=True)
class MarketStateConfig:
    vol_window: int = 21            # trailing daily LOG returns for realized vol
    percentile_window: int = 504    # ~2y reference window for the RV percentile rank
    annualization: int = 252        # annualisation of the daily variance
    trend_window: int = 200         # trailing simple moving average for the trend side
    trend_confirm_days: int = 5     # consecutive days on the new side to confirm a switch
    stress_percentile: float = 0.50  # RV rank >= this (median) -> "stress"
    #: "trend_risk" (default, the 2D label); "vol_quartile" (robustness: a volatility-only
    #: four-state label in the manner of Harvey 2019 that drops the trend dimension, used to
    #: test directly whether the trend axis matters); or "trend_only" (coarsening: a
    #: two-state up/down label that drops the risk dimension, used to test whether the state
    #: arm's inertness is a per-cell support problem rather than a genuine absence of
    #: state-conditional expert skill); or "risk_binary" (a two-state calm/stress label that
    #: is the coarsest partition); or the Amendment-13 Family Z modes "vol_tercile",
    #: "trend_cond_risk" and "down_merged" (balanced middle ground; see the protocol)
    #: drops the trend dimension -- the coarsest volatility partition).
    mode: str = "trend_risk"

    def __post_init__(self):
        allowed = ("trend_risk", "vol_quartile", "trend_only", "risk_binary",
                   "vol_tercile", "trend_cond_risk", "down_merged")
        if self.mode not in allowed:
            raise ValueError(
                f"unknown market-state mode {self.mode!r}; must be one of {allowed}. "
                "An unrecognised mode used to fall through silently to the 2D label."
            )


class MarketStateClassifier:
    """Causal 2D (trend x risk) market_state classifier over a market proxy."""

    semantics_version = "MarketState_v2_trend_risk"

    def __init__(
        self,
        panel: LoaderPricePanel,
        config: MarketStateConfig = MarketStateConfig(),
        proxy_returns: Optional[Mapping] = None,
        proxy_levels: Optional[Mapping] = None,
    ) -> None:
        self._panel = panel
        self._cfg = config
        # Optional external market proxy. ``proxy_levels`` (e.g. SPY close keyed by date)
        # is the faithful path: trend uses the levels directly and risk uses their log
        # returns. ``proxy_returns`` (simple returns keyed by date) is accepted for
        # backward compatibility; a scale-free synthetic level is reconstructed from them
        # for the trend (the close-vs-SMA comparison is scale-invariant). When neither is
        # set the equal-weight universe proxy is built from the panel prices.
        self._proxy_levels = dict(proxy_levels) if proxy_levels is not None else None
        self._proxy_returns = dict(proxy_returns) if proxy_returns is not None else None
        self._cache: dict = {}

    # ---- proxy level + log-return + realized-vol + trend series (each entry causal) ----
    def _series(self, universe: Sequence[str]) -> dict:
        proxy_levels = getattr(self, "_proxy_levels", None)
        proxy_returns = getattr(self, "_proxy_returns", None)
        if proxy_levels is not None:
            key: tuple = ("__proxy_levels__",)
        elif proxy_returns is not None:
            key = ("__proxy_returns__",)
        else:
            key = tuple(sorted(universe))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        dates = self._panel.dates
        n = len(dates)
        cfg = self._cfg
        level: list[Optional[float]] = [None] * n
        logret: list[Optional[float]] = [None] * n

        if proxy_levels is not None:
            prev: Optional[float] = None
            for i in range(n):
                px = proxy_levels.get(_to_date(dates[i]))
                if px is None or px <= 0:
                    level[i] = prev            # carry forward across missing days for the MA
                else:
                    level[i] = px
                    if prev is not None and prev > 0:
                        logret[i] = math.log(px / prev)
                    prev = px
        else:
            # Simple proxy returns -> synthetic (scale-free) level + log returns.
            simple: list[Optional[float]] = [None] * n
            if proxy_returns is not None:
                for i in range(n):
                    simple[i] = proxy_returns.get(_to_date(dates[i]))
            else:
                for i in range(1, n):
                    per = []
                    for tk in universe:
                        p0 = self._panel._price_on(tk, dates[i - 1])
                        p1 = self._panel._price_on(tk, dates[i])
                        if p0 and p1 and p0 > 0:
                            per.append(p1 / p0 - 1.0)
                    if per:
                        simple[i] = fmean(per)
            lvl = 1.0
            level[0] = lvl
            for i in range(1, n):
                s = simple[i]
                if s is not None and 1.0 + s > 0 and level[i - 1] is not None:
                    lvl = level[i - 1] * (1.0 + s)
                    level[i] = lvl
                    logret[i] = math.log(1.0 + s)
                else:
                    level[i] = level[i - 1]      # carry forward on a gap

        rvol = self._realized_vol_series(logret)
        trend = self._trend_series(level)

        cached = {"dates": dates, "level": level, "logret": logret, "rvol": rvol, "trend": trend}
        self._cache[key] = cached
        return cached

    def _realized_vol_series(self, logret: Sequence[Optional[float]]) -> list[Optional[float]]:
        cfg = self._cfg
        n = len(logret)
        rvol: list[Optional[float]] = [None] * n
        for i in range(n):
            lo = i - cfg.vol_window + 1
            if lo < 1:  # logret[0] is None; need vol_window real log returns at indices >= 1
                continue
            window = logret[lo:i + 1]
            if any(r is None for r in window):
                continue
            mean_sq = fmean(r * r for r in window)  # zero-mean: do NOT subtract the mean
            rvol[i] = math.sqrt(cfg.annualization * mean_sq)
        return rvol

    def _trend_series(self, level: Sequence[Optional[float]]) -> list[Optional[str]]:
        cfg = self._cfg
        n = len(level)
        raw: list[Optional[str]] = [None] * n
        for i in range(n):
            lo = i - cfg.trend_window + 1
            if lo < 0:
                continue
            window = level[lo:i + 1]
            if any(v is None for v in window) or level[i] is None:
                continue
            sma = fmean(window)
            raw[i] = "up" if level[i] > sma else "down"
        # Confirmed trend: flip the state only after ``trend_confirm_days`` consecutive
        # raw days on the new side. Causal -- confirmed[i] depends only on raw[0..i].
        confirmed: list[Optional[str]] = [None] * n
        current: Optional[str] = None
        run_side: Optional[str] = None
        run_len = 0
        for i in range(n):
            r = raw[i]
            if r is None:
                confirmed[i] = current
                continue
            if current is None:                 # initialise at the first available side
                current = run_side = r
                run_len = 1
                confirmed[i] = current
                continue
            if r == run_side:
                run_len += 1
            else:
                run_side = r
                run_len = 1
            if run_side != current and run_len >= cfg.trend_confirm_days:
                current = run_side
            confirmed[i] = current
        return confirmed

    # ---- classification ----------------------------------------------------------
    def classify_at(self, t, universe: Sequence[str]) -> str:
        """market_state at date ``t`` using only data with date <= t."""
        cfg = self._cfg
        s = self._series(universe)
        dates, rvol, trend = s["dates"], s["rvol"], s["trend"]
        idx = bisect.bisect_right(dates, _to_date(t)) - 1
        if idx < 0:
            return WARMUP
        today = rvol[idx]
        if today is None:
            return WARMUP

        # Risk reference distribution: RV over the trailing percentile_window days,
        # INCLUDING the current day (still causal -- only data with date <= t). The slice
        # [lo:idx+1] holds exactly percentile_window values ending at idx.
        lo = idx - cfg.percentile_window + 1
        if lo < 0:
            return WARMUP
        reference = rvol[lo:idx + 1]
        if len(reference) < cfg.percentile_window or any(v is None for v in reference):
            return WARMUP

        pct = sum(1 for v in reference if v <= today) / len(reference)

        if cfg.mode == "vol_tercile":
            # Family Z1 (Amendment 13): a three-bin volatility-only label. Between the
            # quartile label (rescued certification on the discovery names) and the binary
            # one (overfit), testing whether four bins were load-bearing. Balanced by
            # construction at ~33% per state; slot reuse mirrors vol_quartile.
            if pct < 1.0 / 3.0:
                return MARKET_STATES[0]
            if pct < 2.0 / 3.0:
                return MARKET_STATES[1]
            return MARKET_STATES[3]

        if cfg.mode == "vol_quartile":
            # Volatility-only robustness label: grade the RV percentile into four quartiles
            # and fill the four state slots as opaque volatility buckets, from calmest to
            # most volatile. No trend dimension, so this variant tests the trend axis
            # directly. The state names are reused as buckets so downstream is unchanged.
            if pct < 0.25:
                return MARKET_STATES[0]
            if pct < 0.50:
                return MARKET_STATES[1]
            if pct < 0.75:
                return MARKET_STATES[2]
            return MARKET_STATES[3]

        if cfg.mode == "risk_binary":
            # Risk-only TWO-state coarsening: calm / stress on the same vol percentile
            # threshold the 2D label uses, but with no trend axis. The coarsest volatility
            # partition -- the most episodes per (expert, state) cell. Reuses two state slots
            # (calm -> MARKET_STATES[0], stress -> MARKET_STATES[3]) like vol_quartile.
            return MARKET_STATES[3] if pct >= cfg.stress_percentile else MARKET_STATES[0]

        side = trend[idx]
        if side is None:
            return WARMUP

        if cfg.mode == "trend_cond_risk":
            # Family Z2 (Amendment 13): trend x risk with the risk reference restricted to
            # days on the SAME trend side, so each leg is split at its own median and no
            # cell is starved by the trend/vol correlation (down days are volatile, so the
            # unconditional median labels nearly every down day "stress" and down_calm is
            # structurally empty -- 9 of 3124 training days). The reference scans back as
            # far as data allows for up to percentile_window same-side days and needs at
            # least percentile_window // 4 of them (126 at the default 504) -- derived from
            # the existing window, not a new tuned constant.
            same_side = [rvol[j] for j in range(idx, -1, -1)
                         if trend[j] == side and rvol[j] is not None]
            same_side = same_side[:cfg.percentile_window]
            if len(same_side) < cfg.percentile_window // 4:
                return WARMUP
            pct_leg = sum(1 for v in same_side if v <= today) / len(same_side)
            risk = "stress" if pct_leg >= cfg.stress_percentile else "calm"
            return f"{side}_{risk}"

        if cfg.mode == "down_merged":
            # Family Z3 (Amendment 13): the minimal repair -- delete the structurally empty
            # down_calm cell by merging the down side into one state. Three states: up_calm,
            # up_stress, down (reusing the down_stress slot).
            if side == "down":
                return MARKET_STATES[3]
            risk = "stress" if pct >= cfg.stress_percentile else "calm"
            return f"up_{risk}"

        if cfg.mode == "trend_only":
            # Trend-only coarsening: drop the risk axis so the label has TWO states
            # (up / down) instead of four. Each (expert, state) cell then sees ~2x the
            # episodes, which directly tests whether the state arm's inertness is a
            # per-cell support problem. Reuses two of the four state slots as opaque
            # trend buckets (up -> MARKET_STATES[0], down -> MARKET_STATES[2]), mirroring
            # vol_quartile's slot reuse so downstream (scorecard, reflection) is unchanged.
            return MARKET_STATES[0] if side == "up" else MARKET_STATES[2]
        risk = "stress" if pct >= cfg.stress_percentile else "calm"
        return f"{side}_{risk}"

    def classify(self, observation) -> str:
        """market_state for an Observation (uses its decision_date and ticker universe)."""
        return self.classify_at(observation.decision_date, list(observation.tickers))

    # ---- diagnostics (for tests / attribution; never used inside a decision) ------
    def realized_vol_at(self, t, universe: Sequence[str]) -> Optional[float]:
        s = self._series(universe)
        dates, rvol = s["dates"], s["rvol"]
        idx = bisect.bisect_right(dates, _to_date(t)) - 1
        if idx < 0:
            return None
        return rvol[idx]

    def trend_at(self, t, universe: Sequence[str]) -> Optional[str]:
        s = self._series(universe)
        dates, trend = s["dates"], s["trend"]
        idx = bisect.bisect_right(dates, _to_date(t)) - 1
        if idx < 0:
            return None
        return trend[idx]
