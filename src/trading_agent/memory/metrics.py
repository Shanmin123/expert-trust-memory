"""Validation metrics: replicas of Qlib metrics + Deflated Sharpe Ratio.

Sources (read, not imported; qlib/skfolio not installed in this env):
- IC / RankIC: Qlib `contrib/report/analysis_position/score_ic.py:18-22`: per-date IC = Pearson
  corr(label, score), RankIC = Spearman corr(label, score). ICIR = mean(IC)/std(IC).
- Deflated Sharpe Ratio: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".

No scipy dependency: the standard-normal CDF uses math.erf; the inverse-CDF uses Acklam's algorithm.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Optional, Sequence

import numpy as np

_EULER_GAMMA = 0.5772156649015329


# --- standard normal helpers (no scipy) ----------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --- correlation / IC (Qlib replica) -------------------------------------------------
def _ranks(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def pearson_corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def information_coefficient(scores: Sequence[float], labels: Sequence[float]) -> Optional[float]:
    """IC = Pearson corr(label, score) for one cross-section (Qlib score_ic)."""
    return pearson_corr(scores, labels)


def rank_information_coefficient(scores: Sequence[float], labels: Sequence[float]) -> Optional[float]:
    """RankIC = Spearman corr(label, score) = Pearson of ranks (Qlib score_ic)."""
    s = np.asarray(scores, dtype=float)
    l = np.asarray(labels, dtype=float)
    if len(s) < 2:
        return None
    return pearson_corr(_ranks(s), _ranks(l))


def icir(ic_series: Sequence[float]) -> Optional[float]:
    vals = [v for v in ic_series if v is not None]
    if len(vals) < 2:
        return None
    sd = float(np.std(vals, ddof=1))
    if sd < 1e-12:
        return None
    return float(np.mean(vals) / sd)


# --- return-series statistics --------------------------------------------------------
def sharpe(returns: Sequence[float], periods_per_year: int = 252, annualize: bool = False) -> Optional[float]:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return None
    sr = r.mean() / r.std(ddof=1)
    return float(sr * math.sqrt(periods_per_year)) if annualize else float(sr)


def max_drawdown(equity: Sequence[float]) -> float:
    peak = None
    mdd = 0.0
    for v in equity:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def utility(returns: Sequence[float], turnover: float = 0.0, drawdown: float = 0.0,
            lambda_turnover: float = 0.0, lambda_drawdown: float = 1.0) -> float:
    """Cost-adjusted utility: mean return penalised by drawdown contribution and excess turnover."""
    r = np.asarray(returns, dtype=float)
    base = float(r.mean()) if len(r) else 0.0
    return base - lambda_drawdown * drawdown - lambda_turnover * turnover


def _skew(r: np.ndarray) -> float:
    n = len(r)
    if n < 3 or r.std() == 0:
        return 0.0
    m = r.mean()
    return float(((r - m) ** 3).mean() / (r.std() ** 3))


def _kurtosis(r: np.ndarray) -> float:
    """Non-excess (Pearson) kurtosis; 3.0 for a normal distribution."""
    n = len(r)
    if n < 4 or r.std() == 0:
        return 3.0
    m = r.mean()
    return float(((r - m) ** 4).mean() / (r.std() ** 4))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """SR0: expected maximum Sharpe under multiple testing (Bailey & López de Prado, 2014)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    z1 = norm_ppf(1 - 1.0 / n_trials)
    z2 = norm_ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    trial_sharpes: Optional[Sequence[float]] = None,
    n_trials: Optional[int] = None,
    var_sr: Optional[float] = None,
) -> Optional[float]:
    """DSR: probability the observed Sharpe exceeds the multiple-testing-adjusted benchmark SR0.

    Provide ``trial_sharpes`` (the per-period Sharpe of every proposal ever evaluated) so the
    benchmark accounts for the full trial count, making "tried many, kept the best" visible.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 4 or r.std(ddof=1) < 1e-12:
        return None
    sr_hat = float(r.mean() / r.std(ddof=1))  # per-period Sharpe
    T = len(r)
    skew = _skew(r)
    kurt = _kurtosis(r)

    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        ts = np.asarray(trial_sharpes, dtype=float)
        n = len(ts)
        v = float(ts.var(ddof=1))
    elif n_trials is not None and var_sr is not None:
        n, v = n_trials, var_sr
    else:
        raise ValueError("Provide trial_sharpes, or both n_trials and var_sr.")

    sr0 = expected_max_sharpe(n, v)
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr0) * math.sqrt(T - 1) / denom
    return norm_cdf(z)
