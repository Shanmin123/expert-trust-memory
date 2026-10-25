"""Alpha158 external-feature alpha engine.

Main path: call an installed Qlib package directly for Alpha158 values. This project does not
reimplement Alpha158 formulas; it handles strategy-regime hypotheses, causal validation, and
evaluation around externally computed strategy features. A SQLite provider is kept only for
Qlib factor artifacts precomputed into a factor DB (see ``SQLiteQlibFactorProvider``) and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, datetime
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

import pandas as pd

from trading_agent.alpha.alpha158_recipes import Alpha158FeatureSpec, REGISTERED_ALPHA158_FEATURES
from trading_agent.contracts import Observation, Signal


class Alpha158FeatureProvider(Protocol):
    """Provider interface for precomputed/external Alpha158 feature values."""

    def values(
        self,
        decision_date: Date,
        tickers: Sequence[str],
        features: Sequence[Alpha158FeatureSpec],
    ) -> Mapping[str, Mapping[str, float]]:
        """Return {strategy_id: {ticker: raw_feature_value}} for one decision date."""


class QlibAlpha158FeatureProvider:
    """External Qlib-backed provider.

    Requires an installed ``qlib`` package. This class imports Qlib lazily and never imports the local
    ``infra/external/research_refs`` clone. The caller supplies a Qlib provider URI that has already
    been exported/initialized for the chosen market.
    """

    @staticmethod
    def is_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("qlib") is not None

    def __init__(self, provider_uri: str, *, market: str = "sp500", region: str = "us"):
        self.provider_uri = provider_uri
        self.market = market
        self.region = region
        self._ready = False
        self._field_by_name: dict[str, str] = {}

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            import qlib
            from qlib.config import REG_US
            from qlib.contrib.data.loader import Alpha158DL
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional external package
            raise RuntimeError(
                "Qlib is not installed in this environment. Install pyqlib and provide a Qlib provider "
                "URI, or inject a precomputed Alpha158FeatureProvider."
            ) from exc
        region = REG_US if self.region.lower() in {"us", "reg_us"} else self.region
        qlib.init(
            provider_uri=self.provider_uri,
            region=region,
            expression_cache=None,
            dataset_cache=None,
        )
        fields, names = Alpha158DL.get_feature_config()
        self._field_by_name = {name: field for field, name in zip(fields, names)}
        self._ready = True

    def values(
        self,
        decision_date: Date,
        tickers: Sequence[str],
        features: Sequence[Alpha158FeatureSpec],
    ) -> Mapping[str, Mapping[str, float]]:
        self._ensure_ready()
        from qlib.data import D  # pragma: no cover - optional external package

        wanted = [f for f in features if f.qlib_name in self._field_by_name]
        if not wanted:
            return {f.strategy_id: {} for f in features}
        fields = [self._field_by_name[f.qlib_name] for f in wanted]
        day = _iso(decision_date)
        frame = D.features(list(tickers), fields, start_time=day, end_time=day, freq="day", disk_cache=0)
        if frame is None or frame.empty:
            return {f.strategy_id: {} for f in features}
        frame = frame.copy()
        frame.columns = [f.strategy_id for f in wanted]
        reset = frame.reset_index()
        if "instrument" not in reset.columns:
            return {f.strategy_id: {} for f in features}
        out: dict[str, dict[str, float]] = {f.strategy_id: {} for f in features}
        for _, row in reset.iterrows():
            ticker = str(row["instrument"])
            for f in wanted:
                val = _finite_or_none(row.get(f.strategy_id))
                if val is not None:
                    out[f.strategy_id][ticker] = val
        return out


class SQLiteQlibFactorProvider:
    """Provider for official Qlib features precomputed into a SQLite factor DB.

    ``data_collection.qlib_provider.compute_direct_qlib_features`` writes one
    ``qlib_direct_<SYMBOL>`` table per symbol with columns like ``QLIB158_ROC20``. This
    provider reads those point-in-time (``WHERE date = ?``) so the project avoids formula
    reimplementation without requiring Qlib to be importable in the current process.
    """

    def __init__(self, factors_db: str | Path):
        self.factors_db = Path(factors_db)
        if not self.factors_db.exists():
            raise FileNotFoundError(f"Missing Qlib factor database: {self.factors_db}")

    def available_tickers(self) -> tuple[str, ...]:
        """Return symbols with precomputed Qlib factor tables in this artifact."""
        return available_tickers_from_sqlite_qlib_factors(self.factors_db)

    def values(
        self,
        decision_date: Date,
        tickers: Sequence[str],
        features: Sequence[Alpha158FeatureSpec],
    ) -> Mapping[str, Mapping[str, float]]:
        import sqlite3

        from data_collection.qlib_provider import qlib_table_name
        from data_collection.storage import quote_identifier

        day = _iso(decision_date)
        out: dict[str, dict[str, float]] = {f.strategy_id: {} for f in features}
        columns = [(f, f"QLIB158_{f.qlib_name}") for f in features]
        with sqlite3.connect(self.factors_db) as conn:
            for ticker in tickers:
                table = qlib_table_name(ticker)
                selected = [quote_identifier(col) for _, col in columns]
                sql = (
                    f"SELECT {', '.join(selected)} FROM {quote_identifier(table)} "
                    "WHERE date = ? LIMIT 1"
                )
                try:
                    row = conn.execute(sql, (day,)).fetchone()
                except sqlite3.OperationalError:
                    continue
                if row is None:
                    continue
                for (feature, _), value in zip(columns, row):
                    val = _finite_or_none(value)
                    if val is not None:
                        out[feature.strategy_id][ticker] = val
        return out

    def trailing_values(
        self,
        decision_date: Date,
        ticker: str,
        features: Sequence[Alpha158FeatureSpec],
        n: int,
    ) -> Mapping[str, list[float]]:
        """Up to ``n`` most-recent finite values per feature with date <= decision_date, ascending."""
        import sqlite3

        from data_collection.qlib_provider import qlib_table_name
        from data_collection.storage import quote_identifier

        day = _iso(decision_date)
        out: dict[str, list[float]] = {f.strategy_id: [] for f in features}
        columns = [(f, f"QLIB158_{f.qlib_name}") for f in features]
        table = qlib_table_name(ticker)
        selected = [quote_identifier(col) for _, col in columns]
        sql = (
            f"SELECT {', '.join(selected)} FROM {quote_identifier(table)} "
            "WHERE date <= ? ORDER BY date DESC LIMIT ?"
        )
        with sqlite3.connect(self.factors_db) as conn:
            try:
                rows = conn.execute(sql, (day, int(n))).fetchall()
            except sqlite3.OperationalError:
                return out
        for row in reversed(rows):  # ascending by date
            for (feature, _), value in zip(columns, row):
                val = _finite_or_none(value)
                if val is not None:
                    out[feature.strategy_id].append(val)
        return out


@dataclass
class FrameAlpha158FeatureProvider:
    """Small deterministic provider for tests or precomputed feature frames.

    The frame must contain ``date`` and ``ticker`` columns plus one column per ``strategy_id``.
    """

    frame: pd.DataFrame

    def __post_init__(self):
        required = {"date", "ticker"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"feature frame missing columns: {sorted(missing)}")
        self._frame = self.frame.copy()
        self._frame["date"] = pd.to_datetime(self._frame["date"]).dt.date
        self._by_date = {d: g for d, g in self._frame.groupby("date")}

    def values(
        self,
        decision_date: Date,
        tickers: Sequence[str],
        features: Sequence[Alpha158FeatureSpec],
    ) -> Mapping[str, Mapping[str, float]]:
        day = _to_date(decision_date)
        rows = self._by_date.get(day)
        out: dict[str, dict[str, float]] = {f.strategy_id: {} for f in features}
        if rows is None:
            return out
        wanted = set(tickers)
        for _, row in rows.iterrows():
            ticker = str(row["ticker"])
            if ticker not in wanted:
                continue
            for f in features:
                if f.strategy_id not in row:
                    continue
                val = _finite_or_none(row[f.strategy_id])
                if val is not None:
                    out[f.strategy_id][ticker] = val
        return out

    def trailing_values(
        self,
        decision_date: Date,
        ticker: str,
        features: Sequence[Alpha158FeatureSpec],
        n: int,
    ) -> Mapping[str, list[float]]:
        day = _to_date(decision_date)
        rows = self._frame[
            (self._frame["ticker"].astype(str) == str(ticker)) & (self._frame["date"] <= day)
        ].sort_values("date").tail(int(n))
        out: dict[str, list[float]] = {f.strategy_id: [] for f in features}
        for _, row in rows.iterrows():
            for f in features:
                if f.strategy_id not in row:
                    continue
                val = _finite_or_none(row[f.strategy_id])
                if val is not None:
                    out[f.strategy_id].append(val)
        return out


class Alpha158StrategyAlpha:
    """Expose external Alpha158 feature values as project ``Signal`` objects.

    Alpha158 features are ratio-style quantities (e.g. MA20 = close over its
    mean, ROC20 = close over a past close) that hover around 1.0, so raw values
    read as "positive" regardless of market direction, and a cross-sectional
    z-score degenerates to 0 for a single-name universe. Each feature is
    therefore normalised as a TIME-SERIES z-score against the same ticker's own
    prior-only history when the provider exposes ``trailing_values``
    (positive = above the stock's own recent norm). Providers without history
    fall back to raw pass-through.
    """

    def __init__(
        self,
        provider: Alpha158FeatureProvider,
        features: Sequence[Alpha158FeatureSpec] = REGISTERED_ALPHA158_FEATURES,
        history_window: int = 252,
        min_history: int = 60,
    ):
        if min_history < 2 or history_window < min_history:
            raise ValueError("need history_window >= min_history >= 2")
        self.provider = provider
        self.features = tuple(features)
        self.history_window = int(history_window)
        self.min_history = int(min_history)

    def _ts_normalised(
        self, observation: Observation, finite_by_strategy: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        from trading_agent.alpha.momentum import ts_zscore

        out: dict[str, dict[str, float]] = {f.strategy_id: {} for f in self.features}
        for ticker in observation.tickers:
            history = self.provider.trailing_values(
                observation.decision_date, ticker, self.features, self.history_window + 1
            )
            for feature in self.features:
                current = finite_by_strategy.get(feature.strategy_id, {}).get(ticker)
                if current is None:
                    continue
                series = list(history.get(feature.strategy_id, ()))
                # trailing_values includes the current date; z against prior-only.
                prior = series[:-1] if series and series[-1] == current else series
                if len(prior) < self.min_history:
                    continue
                out[feature.strategy_id][ticker] = ts_zscore(float(current), prior)
        return out

    def score_by_strategy(self, observation: Observation) -> dict[str, list[Signal]]:
        raw_by_strategy = self.provider.values(observation.decision_date, observation.tickers, self.features)
        finite_by_strategy = {
            feature.strategy_id: {
                tk: v
                for tk, v in raw_by_strategy.get(feature.strategy_id, {}).items()
                if tk in observation.tickers and _finite_or_none(v) is not None
            }
            for feature in self.features
        }
        if hasattr(self.provider, "trailing_values"):
            values_by_strategy = self._ts_normalised(observation, finite_by_strategy)
        else:
            values_by_strategy = finite_by_strategy
        return {
            feature.strategy_id: _to_signals(
                values_by_strategy.get(feature.strategy_id, {}),
                feature.strategy_id,
                observation.decision_date,
            )
            for feature in self.features
        }

    def score(self, observation: Observation) -> Sequence[Signal]:
        signals: list[Signal] = []
        for part in self.score_by_strategy(observation).values():
            signals.extend(part)
        return signals


def _to_signals(values: Mapping[str, float], strategy_id: str, as_of: Date) -> list[Signal]:
    return [
        Signal(ticker=ticker, value=float(value), source_id=strategy_id, as_of=as_of)
        for ticker, value in values.items()
    ]


def _finite_or_none(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v) or v in (float("inf"), float("-inf")):
        return None
    return v


def _to_date(value) -> Date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return pd.to_datetime(value).date()
    return value


def _iso(value) -> str:
    return _to_date(value).isoformat()


def available_tickers_from_sqlite_qlib_factors(factors_db: str | Path) -> tuple[str, ...]:
    """Return symbols represented by qlib_direct_* tables in a factor artifact."""
    import sqlite3

    prefix = "qlib_direct_"
    with sqlite3.connect(factors_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    tickers = [
        str(name)[len(prefix):]
        for (name,) in rows
        if str(name).startswith(prefix) and len(str(name)) > len(prefix)
    ]
    return tuple(sorted(tickers))


__all__ = [
    "Alpha158FeatureProvider",
    "QlibAlpha158FeatureProvider",
    "SQLiteQlibFactorProvider",
    "FrameAlpha158FeatureProvider",
    "Alpha158StrategyAlpha",
    "available_tickers_from_sqlite_qlib_factors",
]
