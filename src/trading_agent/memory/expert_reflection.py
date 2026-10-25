"""Attribution-fed reflection for expert-weight proposals."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as Date
from statistics import fmean, stdev
from typing import NamedTuple, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from trading_agent.memory.expert_error import ExpertErrorRecord
from trading_agent.memory.expert_weights import (
    EXPERT_IDS,
    MARKET_STATES,
    WEIGHT_DELTA_MAX,
    ExpertWeightEntry,
    clip_weight_delta,
)

logger = logging.getLogger(__name__)

#: Fixed a priori: a |t| of 3
#: saturates the reflection bound in the t_stat step mode. Never tuned.
T_STAT_SATURATION: float = 3.0

ATTRIBUTION_COMPONENTS: tuple[str, ...] = ("market", "style", "signal", "sizing", "execution")
PARTICIPATION_FLOOR: float = 0.2

# --- Rolling verification (legacy, unit-level only; see terminal verification below) ------
# A reflection delta is written for (expert, market_state) ONLY IF that cell's counterfactual
# contribution series has a mean statistically distinguishable from zero -- i.e. the expert has
# demonstrated real directional skill in that regime -- AFTER a moving-block bootstrap
# (autocorrelation-robust, NOT an iid t-test) and Benjamini-Hochberg FDR across all tested
# cells (multiple testing over ~experts x states). This is the "independent verifier over
# self-critique" principle made deterministic; it is grid-invariant (uses ONLY the raw g
# series + these fixed a-priori params -- never tau/theta/participation/action/P&L) so the
# trained memory stays train-once/replay-grid. Opt-in; default OFF = byte-identical behaviour.
VERIFY_FDR_ALPHA: float = 0.10        # Benjamini-Hochberg false-discovery rate
VERIFY_N_BOOT: int = 2000             # moving-block bootstrap resamples
VERIFY_MIN_SUPPORT: int = 20          # a cell needs >= this many samples to be tested at all
VERIFY_SEED: int = 20260716           # fixed => reproducible / deterministic verification


def _block_bootstrap_two_sided_p(series: Sequence[float], *, n_boot: int, rng: random.Random) -> float:
    """Two-sided p-value for H0: mean(series) == 0, robust to serial correlation.

    Moving-block bootstrap: resample length-L blocks (L = max(2, round(n**(1/3))), an a-priori
    rule, not tuned) so within-block autocorrelation is preserved; the bootstrap mean
    distribution then has the right (wider) spread that an iid t-test would understate. The
    p-value is 2*min(P(boot_mean<=0), P(boot_mean>=0)) -- 0 in the tail => significant.
    """
    n = len(series)
    if n < 2:
        return 1.0
    if fmean(series) == 0.0:
        return 1.0
    block = max(2, round(n ** (1.0 / 3.0)))
    n_blocks = math.ceil(n / block)
    max_start = max(0, n - block)
    below = above = 0
    for _ in range(n_boot):
        vals: list[float] = []
        for _b in range(n_blocks):
            start = rng.randint(0, max_start)
            vals.extend(series[start:start + block])
        boot_mean = fmean(vals[:n])
        if boot_mean <= 0.0:
            below += 1
        if boot_mean >= 0.0:
            above += 1
    return max(0.0, min(1.0, 2.0 * min(below, above) / n_boot))


def verify_reliable_cells(
    by_state: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    min_support: int = VERIFY_MIN_SUPPORT,
    fdr_alpha: float = VERIFY_FDR_ALPHA,
    n_boot: int = VERIFY_N_BOOT,
    seed: int = VERIFY_SEED,
) -> tuple[set[tuple[str, str]], dict]:
    """Which (market_state, expert_id) cells carry statistically-verified directional skill.

    Tests every cell with >= min_support samples (block-bootstrap p that mean g != 0), then
    applies Benjamini-Hochberg FDR across all tested cells at ``fdr_alpha``. Returns the set of
    verified cells plus an info dict (cells_tested, cells_verified, verification_coverage).
    Deterministic given ``seed``. Grid-invariant: raw g series + fixed a-priori params only.
    """
    rng = random.Random(seed)
    pvals: dict[tuple[str, str], float] = {}
    for state in sorted(by_state):
        for expert_id, series in sorted(by_state[state].items()):
            if len(series) >= min_support:
                pvals[(state, expert_id)] = _block_bootstrap_two_sided_p(
                    list(series), n_boot=n_boot, rng=rng
                )
    m = len(pvals)
    verified: set[tuple[str, str]] = set()
    if m:
        ordered = sorted(pvals.items(), key=lambda kv: kv[1])
        thresh_k = 0
        for k, (_cell, p) in enumerate(ordered, start=1):
            if p <= (k / m) * fdr_alpha:
                thresh_k = k
        verified = {cell for cell, _p in ordered[:thresh_k]}
    info = {
        "cells_tested": m,
        "cells_verified": len(verified),
        "verification_coverage": (len(verified) / m) if m else 0.0,
    }
    return verified, info


# --- TERMINAL verification (protocol repair, research_review_2026-07-16) -------------------
# Replaces the rolling per-cycle check as the system mechanism. Runs ONCE, at freeze time,
# over ALL training episodes, and clears every reflection delta whose (state, expert) cell is
# not statistically verified. Backend-agnostic BY CONSTRUCTION: it cleans whatever the maker
# (LLM or deterministic) wrote into memory.
#
# Estimand: the RELATIVE, grid-invariant marginal proxy
#     m_{e,t} = r_{t+1}/N_t * (v_{e,t} - vbar_{-e,t}) = (g_{e,t} - gbar_{-e,t}) / N_t
# over AVAILABLE experts (safe-default views never enter signal_attribution) -- proportional
# to the change in the equal-weight committee margin when e replaces the average of the
# others. Matches the delta's relative semantics (H0: no edge OVER THE COMMITTEE), unlike the
# absolute H0 E[g]=0.
# Inference: centred+STUDENTIZED moving-block bootstrap on SYNCHRONIZED calendar dates (blocks
# are drawn over the global date index jointly, preserving cross-cell dependence),
# p = (b+1)/(B+1), then HOLM (FWER, valid under arbitrary dependence) across tested cells --
# not BH, whose PRDS assumption is not established for correlated cells.
TERMINAL_VERIFY_ALPHA: float = 0.10   # Holm FWER level, fixed a priori
TERMINAL_VERIFY_N_BOOT: int = 2000
TERMINAL_VERIFY_MIN_SUPPORT: int = 20
TERMINAL_VERIFY_SEED: int = 20260716


def marginal_series(
    episodes: Sequence["AttributedEpisode"], *, pool_states: bool = False,
) -> dict[tuple[str, str], list[tuple[Date, float]]]:
    """Per (state, expert): date-ordered (decision_date, m) marginal-proxy observations.

    ``pool_states=True`` keys every cell under GLOBAL (for the global-scope store).
    Episodes with fewer than 2 available experts carry no relative signal and are skipped.
    """
    out: dict[tuple[str, str], list[tuple[Date, float]]] = defaultdict(list)
    for ep in episodes:
        contribs = {e: float(g) for e, g in ep.signal_attribution.items() if e in EXPERT_IDS}
        n = len(contribs)
        if n < 2:
            continue
        total = sum(contribs.values())
        state = "global" if pool_states else ep.market_state
        for e, g in contribs.items():
            gbar_others = (total - g) / (n - 1)
            out[(state, e)].append((ep.decision_date, (g - gbar_others) / n))
    for series in out.values():
        series.sort(key=lambda pair: pair[0])
    return out


def _holm(pvals: dict[tuple[str, str], float], alpha: float) -> set[tuple[str, str]]:
    """Holm step-down: reject smallest p while p_(k) <= alpha/(m-k+1)."""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected: set[tuple[str, str]] = set()
    for k, (cell, p) in enumerate(ordered, start=1):
        if p <= alpha / (m - k + 1):
            rejected.add(cell)
        else:
            break
    return rejected


def _benjamini_hochberg(pvals: dict[tuple[str, str], float], alpha: float) -> set[tuple[str, str]]:
    """Benjamini-Hochberg step-up: reject the k smallest p for the LARGEST k with
    p_(k) <= (k/m)*alpha. Controls the false discovery rate rather than the familywise
    error rate, so it is less conservative than Holm -- the right choice when several
    true cells are expected and a bounded fraction of false ones is tolerable."""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    k_max = 0
    for k, (_cell, p) in enumerate(ordered, start=1):
        if p <= (k / m) * alpha:
            k_max = k
    return {cell for cell, _ in ordered[:k_max]}


def _kfold_confirms(
    dmap: "dict[Date, list[float]]", full_mean: float, *,
    k: int, min_agree: int, min_fold_n: int = 3,
) -> tuple[bool, list[int]]:
    """Temporal k-fold confirmation: split the cell's own episode dates into ``k``
    contiguous folds; a fold confirms when its mean shares the sign of the full-train
    mean and holds at least ``min_fold_n`` samples. The cell passes with >= ``min_agree``
    confirming folds. Generalises the split-half persistence check (k=2). Returns
    (passed, per-fold sign list: +1 agree / -1 disagree / 0 insufficient)."""
    dates = sorted(dmap)
    if k < 2 or not dates:
        return False, []
    fold_size = max(1, len(dates) // k)
    signs: list[int] = []
    agree = 0
    for i in range(k):
        lo = i * fold_size
        hi = (i + 1) * fold_size if i < k - 1 else len(dates)
        vals = [m for d in dates[lo:hi] for m in dmap[d]]
        if len(vals) < min_fold_n:
            signs.append(0)
            continue
        same = (fmean(vals) > 0.0) == (full_mean > 0.0)
        signs.append(1 if same else -1)
        if same:
            agree += 1
    return agree >= min_agree, signs


def _persists_across_halves(
    dmap: "dict[Date, list[float]]", split_date: "Date", min_half: int = 5
) -> bool:
    """Temporal-persistence check: the cell's mean marginal contribution keeps the same
    sign in the first and second half of the training window (split at ``split_date``).
    Real skill persists; an in-sample fluke that flips between halves is rejected. Halves
    with fewer than ``min_half`` samples are treated as non-persistent (insufficient
    evidence to confirm)."""
    first = [m for d, ms in dmap.items() if d <= split_date for m in ms]
    second = [m for d, ms in dmap.items() if d > split_date for m in ms]
    if len(first) < min_half or len(second) < min_half:
        return False
    return (fmean(first) > 0.0) == (fmean(second) > 0.0)


def terminal_verify(
    episodes: Sequence["AttributedEpisode"],
    *,
    pool_states: bool = False,
    min_support: int = TERMINAL_VERIFY_MIN_SUPPORT,
    alpha: float = TERMINAL_VERIFY_ALPHA,
    n_boot: int = TERMINAL_VERIFY_N_BOOT,
    seed: int = TERMINAL_VERIFY_SEED,
    block_len: Optional[int] = None,
    correction: str = "holm",
    require_persistence: bool = False,
    min_effect: float = 0.0,
    oos_series: Optional[Mapping[tuple[str, str], Sequence[tuple["Date", float]]]] = None,
    min_oos: int = 5,
    kfold: int = 0,
    kfold_min_agree: Optional[int] = None,
) -> tuple[set[tuple[str, str]], dict]:
    """ONE terminal verification pass at freeze time.

    Returns (verified cells, report). Report carries per-cell n / unique dates / mean / se /
    t / p / p-adjusted-rank info / verified flag / rough 80%-power MDE, plus coverage counts
    (treatment receipt). Deterministic given ``seed``; uses ONLY counterfactual records +
    fixed a-priori parameters (grid-invariant).
    """
    series = marginal_series(episodes, pool_states=pool_states)
    dates = sorted({d for pairs in series.values() for d, _ in pairs})
    t_len = len(dates)
    cells = {cell: pairs for cell, pairs in series.items() if len(pairs) >= min_support}
    report: dict = {
        "alpha": alpha, "n_boot": n_boot, "min_support": min_support,
        "correction": correction,
        "estimand": "marginal_proxy_m",
        "cells_planned": len(series), "cells_tested": len(cells),
        "per_cell": {},
    }
    if not cells or t_len < 4:
        report.update({"cells_verified": 0, "verification_coverage": 0.0})
        return set(), report

    by_date: dict[tuple[str, str], dict[Date, list[float]]] = {}
    stats: dict[tuple[str, str], tuple[float, float, float, int]] = {}
    for cell, pairs in cells.items():
        dmap: dict[Date, list[float]] = defaultdict(list)
        for d, m in pairs:
            dmap[d].append(m)
        by_date[cell] = dmap
        vals = [m for _, m in pairs]
        n = len(vals)
        mean = fmean(vals)
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        stats[cell] = (mean, se, (mean / se) if se > 0 else 0.0, n)

    block = block_len if block_len is not None else max(2, round(t_len ** (1.0 / 3.0)))
    n_blocks = math.ceil(t_len / block)
    max_start = max(0, t_len - block)
    rng = random.Random(seed)
    exceed = {cell: 0 for cell in cells}
    for _ in range(n_boot):
        resampled: list[Date] = []
        for _b in range(n_blocks):
            start = rng.randint(0, max_start)
            resampled.extend(dates[start:start + block])
        resampled = resampled[:t_len]
        for cell in cells:
            dmap = by_date[cell]
            vals = [m for d in resampled for m in dmap.get(d, ())]
            nb = len(vals)
            if nb < 2:
                exceed[cell] += 1  # uninformative resample counts against significance
                continue
            mean_b = fmean(vals)
            var_b = sum((x - mean_b) ** 2 for x in vals) / (nb - 1)
            se_b = math.sqrt(var_b / nb)
            mean0, _se0, t0, _n = stats[cell]
            if se_b <= 0.0:
                exceed[cell] += 1
                continue
            t_b = (mean_b - mean0) / se_b   # centred + studentized
            if abs(t_b) >= abs(t0):
                exceed[cell] += 1

    pvals = {cell: (exceed[cell] + 1) / (n_boot + 1) for cell in cells}
    if correction not in ("holm", "fdr"):
        raise ValueError(f"unknown correction {correction!r}; must be 'holm' or 'fdr'")
    verified = _benjamini_hochberg(pvals, alpha) if correction == "fdr" else _holm(pvals, alpha)
    # Out-of-sample-generalisation filters (opt-in): a cell that is significant in-sample
    # can still fail out-of-sample (the observed ADSK false positive). These only REMOVE
    # cells from the verified set, never add.
    if min_effect > 0.0:
        verified = {c for c in verified if abs(stats[c][0]) >= min_effect}
    if require_persistence:
        split_date = dates[len(dates) // 2]
        verified = {c for c in verified if _persists_across_halves(by_date[c], split_date)}
    kfold_receipt: dict[str, list[int]] = {}
    if kfold and kfold >= 2:
        # Full-train temporal-fold confirmation (the principled replacement for thin
        # held-out-window confirmation): certification requires fold-stable sign.
        min_agree = kfold_min_agree if kfold_min_agree is not None else kfold - 1
        confirmed_k: set[tuple[str, str]] = set()
        for c in verified:
            ok, signs = _kfold_confirms(
                by_date[c], stats[c][0], k=kfold, min_agree=min_agree
            )
            kfold_receipt["|".join(c)] = signs
            if ok:
                confirmed_k.add(c)
        verified = confirmed_k
    if oos_series is not None:
        # True out-of-sample confirmation: a verified cell must keep the same-signed mean
        # marginal contribution on a HELD-OUT window (the validation split), not just be
        # significant in-sample. A cell the held-out window cannot confirm (no data or an
        # opposite sign) is dropped -- the direct guard against an in-sample false positive.
        confirmed: set[tuple[str, str]] = set()
        for c in verified:
            oos_vals = [m for _, m in oos_series.get(c, ())]
            if len(oos_vals) >= min_oos and (fmean(oos_vals) > 0.0) == (stats[c][0] > 0.0):
                confirmed.add(c)
        verified = confirmed
    report["require_persistence"] = require_persistence
    report["min_effect"] = min_effect
    report["oos_confirmed"] = oos_series is not None
    report["kfold"] = int(kfold)
    if kfold_receipt:
        report["kfold_signs"] = kfold_receipt
    # Holm step-down ADJUSTED p-values with ranks, reported alongside the raw
    # bootstrap p so the artifact is self-contained (round-2 review receipt).
    m_tests = len(cells)
    holm_adjusted: dict[tuple[str, str], tuple[float, int]] = {}
    running = 0.0
    for rank, (cell, p_raw) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m_tests - rank) * p_raw))
        holm_adjusted[cell] = (running, rank + 1)
    z80 = 0.8416
    z_alpha = 2.807 if m_tests <= 1 else _z_two_sided(alpha / m_tests)
    for cell, (mean, se, t0, n) in stats.items():
        report["per_cell"]["|".join(cell)] = {
            "n": n, "unique_dates": len(by_date[cell]),
            "mean_m": mean, "se": se, "t": t0, "p": pvals[cell],
            "p_holm": holm_adjusted[cell][0],
            "holm_rank": holm_adjusted[cell][1],
            "verified": cell in verified,
            "mde_80pct": (z_alpha + z80) * se,
        }
    report["cells_verified"] = len(verified)
    report["verification_coverage"] = len(verified) / m_tests if m_tests else 0.0
    # Receipt denominators + omitted cells (support below min_support), so the
    # writeup can state eligible-vs-planned coverage without re-deriving it.
    report["cells_omitted_below_support"] = sorted(
        "|".join(cell) for cell in series if cell not in cells
    )
    report["verified_over_planned"] = (
        len(verified) / len(series) if series else 0.0
    )
    # Field semantics (round-2 review): se/t are IID sample quantities; the
    # INFERENTIAL object is the centred+studentized moving-block bootstrap p;
    # mde_80pct is a Bonferroni-normal PLANNING approximation, not an MBB/Holm
    # power calculation. Recorded in the artifact to prevent misreading.
    report["se_note"] = (
        "se and t are ordinary IID sample quantities; inference uses the "
        "centred+studentized moving-block bootstrap p (p_holm = Holm step-down "
        "adjusted); mde_80pct is a Bonferroni-normal planning approximation"
    )
    report["block_len"] = block
    return verified, report


def _z_two_sided(p: float) -> float:
    """Two-sided normal quantile z_{1-p/2} via Acklam-style rational approximation."""
    p = min(max(p / 2.0, 1e-12), 0.5)
    # Beasley-Springer-Moro style approximation for the lower-tail quantile of |z|.
    t = math.sqrt(-2.0 * math.log(p))
    z = t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)
    return z


def apply_terminal_gate(store, verified: set[tuple[str, str]]) -> int:
    """Zero every reflection delta whose cell is NOT in ``verified``. Returns #cleared.

    Store must still be writable (call BEFORE freeze). Cells the maker never touched are
    untouched; verified cells keep their trained delta (the 'final verified set').

    DELIBERATELY SIGN-BLIND (settled 2026-07-18, user decision, principle-first): the
    verification certifies that a cell carries reliable relative skill in EITHER
    direction (two-sided test over the FULL undecayed training window), while the delta
    is a bounded RECENCY tilt (last reflection window, SET semantics) and the scorecard
    S is EW-decayed — three time-weightings by design. Forcing sign agreement would
    impose a stationarity assumption the EW design rejects. The tension is made
    auditable by ``delta_sign_receipt`` below, and cross-name replication adjudicates
    whether the recency tilt is systematic or single-name luck.
    """
    cleared = 0
    for entry in store.learned():
        cell = (entry.market_state, entry.expert_id)
        if cell not in verified and entry.weight_delta != 0.0:
            store.set_reflection_delta(entry.market_state, entry.expert_id, 0.0)
            cleared += 1
    return cleared


def delta_sign_receipt(store, report: Mapping) -> dict[str, dict]:
    """Per-cell receipt of the maker's pre-clearing deltas vs the verifier's direction.

    Compute BEFORE ``apply_terminal_gate`` (it reads the store's live deltas). For every
    nonzero delta: the delta, whether its cell verified, the verified full-train mean m,
    and whether the two directions agree. ``sign_agreement=False`` on a verified cell is
    the documented time-horizon tension (recency tilt vs undecayed full-train skill),
    not an error — the writeup discloses it and cross-name evidence adjudicates it.
    """
    per_cell = report.get("per_cell", {}) if isinstance(report, Mapping) else {}
    out: dict[str, dict] = {}
    for entry in store.learned():
        cell_key = f"{entry.market_state}|{entry.expert_id}"
        v = per_cell.get(cell_key, {})
        m = v.get("mean_m")
        agrees = None
        if m is not None and entry.weight_delta != 0.0:
            agrees = (float(m) > 0.0) == (entry.weight_delta > 0.0)
        out[cell_key] = {
            "delta": round(float(entry.weight_delta), 6),
            "verified": bool(v.get("verified", False)),
            "verified_mean_m": m,
            "sign_agreement": agrees,
        }
    return out


def _default_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


@dataclass(frozen=True)
class AttributedEpisode:
    """One decision date outcome with decomposed PnL fields."""

    decision_date: Date
    market_state: str
    attribution: Mapping[str, float]
    signal_attribution: Mapping[str, float]
    expert_errors: tuple[ExpertErrorRecord, ...] = ()
    action: str = "long"
    ticker: str | None = None

    def __post_init__(self) -> None:
        if self.market_state not in MARKET_STATES:
            raise ValueError(f"unknown market_state {self.market_state!r}")
        missing = [c for c in ATTRIBUTION_COMPONENTS if c not in self.attribution]
        if missing:
            raise ValueError(f"missing attribution components: {missing}")


@dataclass
class ReflectionTrialLog:
    """Running tally of reflection proposals."""

    trials: int = 0
    proposed: int = 0
    records: list[dict] = field(default_factory=list)

    def record(self, entry: ExpertWeightEntry, reason: str) -> None:
        self.trials += 1
        self.proposed += 1
        self.records.append({
            "proposal_index": self.trials,
            "proposal_id": entry.entry_id,
            "expert_id": entry.expert_id,
            "market_state": entry.market_state,
            "weight_delta": entry.weight_delta,
            "reason": reason,
        })
        logger.info(
            "reflection proposal #%d: %s @ %s delta=%+.3f (%s)",
            self.trials, entry.expert_id, entry.market_state, entry.weight_delta, reason,
        )


@runtime_checkable
class ExpertReflectionBackend(Protocol):
    def reflect(
        self,
        episodes: Sequence[AttributedEpisode],
        created_at: Date,
        participation_rate: float,
    ) -> Sequence[ExpertWeightEntry]: ...


class ReflectionInput(NamedTuple):
    """The ONLY episode fields any reflection maker may see (grid invariance).

    Everything else an AttributedEpisode carries -- the system's action, the
    weights used, participation, portfolio P&L -- depends on tau/theta and is
    stripped HERE, structurally, so a maker cannot read it even by accident.
    """

    market_state: str
    expert_id: str
    contribution: float


def strip_to_reflection_inputs(
    episodes: Sequence[AttributedEpisode],
) -> list[ReflectionInput]:
    out: list[ReflectionInput] = []
    for ep in episodes:
        for expert_id, contribution in ep.signal_attribution.items():
            if expert_id in EXPERT_IDS:
                out.append(ReflectionInput(ep.market_state, expert_id, float(contribution)))
    return out


def _reliability_by_state(
    episodes: Sequence[AttributedEpisode],
) -> dict[str, dict[str, list[float]]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in strip_to_reflection_inputs(episodes):
        acc[record.market_state][record.expert_id].append(record.contribution)
    return acc


def _compact_lessons_by_state(
    episodes: Sequence[AttributedEpisode],
) -> dict[str, list[str]]:
    by_state = _reliability_by_state(episodes)
    lessons: dict[str, list[str]] = {}
    for market_state in sorted(by_state):
        means = {eid: fmean(vals) for eid, vals in by_state[market_state].items() if vals}
        if not means:
            continue
        ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_mean = ranked[0]
        if len(ranked) > 1:
            runner_id, runner_mean = ranked[1]
            lesson = (
                f"{market_state}: {best_id} led attributed signal contribution "
                f"({best_mean:.4f}) versus {runner_id} ({runner_mean:.4f}); "
                "treat as a compact lesson for similar market_state windows."
            )
        else:
            lesson = (
                f"{market_state}: {best_id} has the only attributed signal contribution "
                f"sample ({best_mean:.4f}); treat as low-support context."
            )
        lessons[market_state] = [lesson]
    return lessons


class DeterministicExpertReflection:
    """Transparent reflection over the attributed contribution scorecard (g = v*r).

    ``step_mode='fixed'``: ``delta = clip((mean_e - state_mean) * step_scale)``.
    On daily-return scales the deviations are ~1e-3, so the deltas are ~1e-2 and
    functionally a no-op next to the +/-0.10 bound (the TSLA sweep measured zero
    decision flips). ``step_mode='scale_free'``: rescale by the largest deviation
    in the state, ``delta_e = BOUND * dev_e / max|dev|``, so the best/worst
    experts get near-bound deltas regardless of the return scale -- commensurate
    with what the LLM reflection actually emits.
    """

    model_version = "deterministic-expert-reflection-v1"

    def __init__(self, min_support: int = 3, step_scale: float = 2.0,
                 step_mode: str = "scale_free", verify_gate: bool = False,
                 verify_min_support: int = VERIFY_MIN_SUPPORT):
        if step_mode not in ("fixed", "scale_free", "t_stat"):
            raise ValueError(f"unknown step_mode {step_mode!r}")
        self.min_support = min_support
        self.step_scale = step_scale
        self.step_mode = step_mode
        # verify_gate=True: only write a delta for a cell whose counterfactual skill is
        # statistically verified (block-bootstrap + BH-FDR); the verification-on/off ablation.
        # Default OFF -> behaviour is byte-identical to before. Populated on each reflect().
        self.verify_gate = bool(verify_gate)
        self._verify_min_support = int(verify_min_support)
        self.verified_cells: set[tuple[str, str]] = set()
        self.verification_coverage: Optional[float] = None
        self.log = ReflectionTrialLog()

    def reflect(
        self,
        episodes: Sequence[AttributedEpisode],
        created_at: Date,
        participation_rate: float,
    ) -> list[ExpertWeightEntry]:
        # Reads ONLY the counterfactual contribution scorecard (g = v*r), never the
        # confidence-based expert-error summaries: the whole system uses direction, not the
        # language model's self-reported confidence (Chapter 3). For each state, experts are
        # scored by their mean contribution and nudged toward or away from the state mean.
        # participation_rate is not read, which keeps the trained memory grid-invariant.
        del participation_rate
        by_state = _reliability_by_state(episodes)
        if self.verify_gate:
            self.verified_cells, _info = verify_reliable_cells(
                by_state, min_support=self._verify_min_support
            )
            self.verification_coverage = _info["verification_coverage"]
            logger.info(
                "verification: %d/%d cells verified (coverage=%.3f) @ %s; verified=%s",
                _info["cells_verified"], _info["cells_tested"],
                _info["verification_coverage"], created_at.isoformat(),
                sorted(self.verified_cells),
            )
        out: list[ExpertWeightEntry] = []
        for market_state in sorted(by_state):
            per_expert = by_state[market_state]
            means = {
                eid: fmean(vals)
                for eid, vals in per_expert.items()
                if len(vals) >= self.min_support
            }
            if len(means) < 2:
                continue
            state_mean = fmean(means.values())
            max_dev = max(abs(mean - state_mean) for mean in means.values())
            for expert_id, mean in sorted(means.items()):
                if self.step_mode == "t_stat":
                    # The step
                    # carries the cell's own uncertainty. t = (mean - state
                    # mean) / (s / sqrt(n)); |t| = T_STAT_SATURATION saturates
                    # the +-WEIGHT_DELTA_MAX bound. s == 0 or n < 2 -> no step.
                    vals = per_expert[expert_id]
                    n = len(vals)
                    s = stdev(vals) if n >= 2 else 0.0
                    if s <= 0.0:
                        continue
                    t = (mean - state_mean) / (s / math.sqrt(n))
                    raw_delta = WEIGHT_DELTA_MAX * t / T_STAT_SATURATION
                elif self.step_mode == "scale_free":
                    if max_dev <= 0.0:
                        continue
                    raw_delta = WEIGHT_DELTA_MAX * (mean - state_mean) / max_dev
                else:
                    raw_delta = (mean - state_mean) * self.step_scale
                delta = clip_weight_delta(raw_delta, expert_id=expert_id, market_state=market_state)
                if delta == 0.0:
                    continue
                if self.verify_gate and (market_state, expert_id) not in self.verified_cells:
                    # maker proposed a tilt, but the independent verifier found the cell's
                    # counterfactual skill indistinguishable from noise -> no rule written.
                    continue
                entry = ExpertWeightEntry(
                    entry_id=f"det::{expert_id}::{market_state}::{created_at.isoformat()}",
                    expert_id=expert_id,
                    market_state=market_state,
                    weight_delta=delta,
                    status="learned",
                    created_at=created_at,
                    learned_at=created_at,
                    rationale=(
                        f"expert '{expert_id}' contribution mean {mean:.4f} vs state mean "
                        f"{state_mean:.4f} in {market_state}. Audit-only."
                    ),
                )
                self.log.record(entry, reason="contribution_vs_state_mean")
                out.append(entry)
        return out


class LLMExpertReflection:
    """Injectable-LLM reflection with strict JSON and deterministic fallback."""

    model_version = "llm-expert-reflection-v1"

    def __init__(
        self,
        backend: Callable[[str, str], str],
        fallback: Optional[ExpertReflectionBackend] = None,
        min_support: int = 3,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        sleeper: Callable[[float], None] | None = None,
    ):
        self._backend = backend
        self._fallback = fallback or DeterministicExpertReflection(min_support=min_support)
        self.log = ReflectionTrialLog()
        self.fallback_count = 0
        # Retry the backend on TRANSIENT failures (rate limit / timeout / a one-off
        # malformed response) with exponential backoff BEFORE falling back to the
        # deterministic rule. A rate-limit blip during a concurrent job otherwise
        # silently degrades reflection to the rule and corrupts the comparison.
        self._max_retries = int(max_retries)
        self._backoff_seconds = float(backoff_seconds)
        self._sleeper = sleeper or _default_sleep
        self.retry_count = 0

    def _system_prompt(self, participation_rate: float) -> str:
        return (
            "You reflect over an attributed scorecard to propose weight adjustments over "
            "three experts (technical, news, fundamental). Judge an expert ONLY by its "
            "per-market_state contribution statistics (mean contribution, hit rate, count), "
            "never raw portfolio PnL and never a self-reported confidence. For each "
            "market_state, shift toward experts with a higher mean contribution and away "
            "from those with a lower one. Every weight_delta MUST be within [-0.10, 0.10]. "
            'Return JSON: {"proposals":[{"expert_id":...,"market_state":...,'
            '"weight_delta":...,"rationale":...}]}. Rationale is audit-only.'
        )

    def _user_prompt(self, episodes: Sequence[AttributedEpisode]) -> str:
        by_state = _reliability_by_state(episodes)
        # Scorecard statistics per (state, expert): mean attributed contribution, hit
        # rate (fraction of episodes with positive contribution), and support count. No
        # confidence-based error summary -- confidence is not part of the mechanism.
        scorecard = {
            state: {
                eid: {
                    "mean_contribution": round(fmean(vals), 6),
                    "hit_rate": round(sum(1 for v in vals if v > 0) / len(vals), 4),
                    "count": len(vals),
                }
                for eid, vals in experts.items() if vals
            }
            for state, experts in by_state.items()
        }
        return json.dumps(
            {
                "scorecard_by_state": scorecard,
                "recent_lessons_by_state": _compact_lessons_by_state(episodes),
            },
            sort_keys=True,
        )

    def _parse(
        self, raw: str, created_at: Date, participation_rate: float
    ) -> list[ExpertWeightEntry]:
        data = json.loads(raw.strip(), strict=False)
        proposals = data["proposals"]
        if not isinstance(proposals, list):
            raise ValueError("'proposals' must be a list")
        out: list[ExpertWeightEntry] = []
        for i, proposal in enumerate(proposals):
            expert_id = str(proposal["expert_id"])
            market_state = str(proposal["market_state"])
            entry = ExpertWeightEntry(
                entry_id=f"llm::{expert_id}::{market_state}::{created_at.isoformat()}::{i}",
                expert_id=expert_id,
                market_state=market_state,
                weight_delta=float(proposal["weight_delta"]),
                status="learned",
                created_at=created_at,
                learned_at=created_at,
                rationale=str(proposal.get("rationale", ""))[:200],
            )
            self.log.record(entry, reason="llm_proposal")
            out.append(entry)
        return out

    def reflect(
        self,
        episodes: Sequence[AttributedEpisode],
        created_at: Date,
        participation_rate: float,
    ) -> list[ExpertWeightEntry]:
        system = self._system_prompt(participation_rate)
        user = self._user_prompt(episodes)
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._backend(system, user)
                return self._parse(raw, created_at, participation_rate)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    self.retry_count += 1
                    logger.info(
                        "LLM reflection attempt %d failed (%r); retrying after backoff",
                        attempt + 1, exc,
                    )
                    self._sleeper(self._backoff_seconds * (2 ** attempt))
        self.fallback_count += 1
        logger.warning(
            "LLM reflection failed after %d attempts (%r); falling back to deterministic",
            self._max_retries + 1, last_exc,
        )
        return list(self._fallback.reflect(episodes, created_at, participation_rate))
