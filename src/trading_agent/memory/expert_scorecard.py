"""Scorecard-based expert-trust memory (Fable learning layer).

For each ``(expert_id, market_state)`` the store keeps an exponentially weighted mean and
variance of the expert's counterfactual net contribution (the learning evidence built by
``runner._attribute``), a hit rate and sample count, and a bounded reflection logit
adjustment. Retrieval turns the per-state mean-variance scores into weights through a
temperature-scaled softmax projected onto the same floored simplex the aggregation uses:

    S_e      = ew_mean_e - lambda * ew_var_e                 (per-state mean-variance score)
    z_e      = S_e / tau + reflection_delta_e                (logit; delta = 0 for numeric-only)
    w        = floored_simplex( softmax_e(z_e) )             (floor 0.05, sums to 1)

The scorecard is the statistical baseline, accumulated mechanically from attributed
outcomes; the bounded reflection delta is the only channel through which language-model
reflection can move the weights. The numeric constants below are fixed a priori from the
online-learning / mean-variance literature (the exponential-weights forecaster of
Cesa-Bianchi & Lugosi, 2006) and are NOT tuned on the validation window; only whether the
learned trust is used, whether it is conditioned on the market state, and whether the
reflection delta is applied, are selected. This keeps the multiple-testing budget small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Optional

from trading_agent.aggregation.soft_weighting import EXPERT_IDS
from trading_agent.aggregation.weights import WEIGHT_FLOOR, project_to_floored_simplex

MARKET_STATES: tuple[str, ...] = ("up_calm", "up_stress", "down_calm", "down_stress")
GLOBAL_MARKET_STATE: str = "global"
_SCOPES: tuple[str, ...] = ("state", "global")

# --- learning-layer constants. lambda and the decay are fixed a priori; tau is the centre
# of the validation grid {0.0005, 0.001, 0.002} (daily-return scale). See dissertation Ch3.
SCORECARD_HALFLIFE: int = 63          # active-episode half-life of the EW decay
VARIANCE_PENALTY: float = 1.0         # S = ew_mean - VARIANCE_PENALTY * ew_var
SOFTMAX_TEMPERATURE: float = 0.001    # raw-retrieval grid centre (the --raw-memory ablation)
DEFAULT_TEMPERATURE: float = 0.5      # standardized retrieval, the selected tau (system default)
REFLECTION_DELTA_BOUND: float = 0.10  # |reflection_delta| clipped to this (logit units)
# Per-state episodes before the scorecard tilts off equal weights. The separate,
# smaller reporting threshold that marks a state "underpowered" (default 5) lives in the
# evaluation layer, not here: 5-19 episodes are reported descriptively but still weighted
# equally, and only >= 20 episodes actually move the weights.
DEFAULT_MIN_SUPPORT: int = 20

#: EW smoothing derived from the half-life: mean_{t} = mean_{t-1} + alpha*(x - mean_{t-1}).
DECAY_ALPHA: float = 1.0 - 0.5 ** (1.0 / SCORECARD_HALFLIFE)


def equal_weights() -> dict[str, float]:
    """The equal-weight fallback: 0.25 per expert."""
    return {eid: 1.0 / len(EXPERT_IDS) for eid in EXPERT_IDS}


def clip_delta(value: float) -> float:
    return max(-REFLECTION_DELTA_BOUND, min(REFLECTION_DELTA_BOUND, float(value)))


@dataclass(frozen=True)
class LearnedDelta:
    """One reflection delta, exposed for the holdout's weight-adjustment audit."""

    market_state: str
    expert_id: str
    weight_delta: float
    rationale: str = ""
    status: str = "learned"


@dataclass
class Scorecard:
    """Exponentially weighted mean/variance of one expert's contribution in one state."""

    ew_mean: float = 0.0
    ew_var: float = 0.0
    count: int = 0
    hits: int = 0
    reflection_delta: float = 0.0

    def observe(self, x: float, alpha: float = DECAY_ALPHA) -> None:
        """Fold one counterfactual contribution into the EW mean/variance."""
        x = float(x)
        self.count += 1
        if x > 0.0:
            self.hits += 1
        if self.count == 1:
            self.ew_mean = x
            self.ew_var = 0.0
            return
        diff = x - self.ew_mean
        self.ew_mean += alpha * diff
        # Exponentially weighted variance (Finch, 2009): stays non-negative by construction.
        self.ew_var = (1.0 - alpha) * (self.ew_var + alpha * diff * diff)

    @property
    def score(self) -> float:
        return self.ew_mean - VARIANCE_PENALTY * self.ew_var

    @property
    def hit_rate(self) -> float:
        return self.hits / self.count if self.count else 0.0


class ExpertScorecardStore:
    """Per-(expert, state) scorecard memory with a floored-softmax retrieval.

    ``scope='state'`` conditions the weights on the market state (the market_state_memory
    arm); ``scope='global'`` pools all observations under a single key (the global_memory
    arm). ``use_reflection=False`` ignores the reflection deltas (the numeric-only ablation).
    """

    def __init__(
        self,
        *,
        scope: str = "state",
        use_reflection: bool = True,
        min_support: int = DEFAULT_MIN_SUPPORT,
        temperature: float = DEFAULT_TEMPERATURE,
        bound_reflection: bool = True,
        certified_only: bool = False,
        standardize: bool = True,
        shrink_to_global: bool = False,
        half_life: float = float(SCORECARD_HALFLIFE),
        variance_penalty: float = VARIANCE_PENALTY,
        score_mode: str = "ewvar",
        shrinkage: str = "none",
        overlap: int = 1,
    ) -> None:
        if scope not in _SCOPES:
            raise ValueError(f"unknown scope {scope!r}; must be one of {_SCOPES}")
        if score_mode not in ("ewvar", "hit_logodds"):
            raise ValueError(f"unknown score_mode {score_mode!r}; must be 'ewvar' or 'hit_logodds'")
        if shrinkage not in ("none", "empirical_bayes"):
            raise ValueError(
                f"unknown shrinkage {shrinkage!r}; must be 'none' or 'empirical_bayes'"
            )
        if shrinkage == "empirical_bayes" and shrink_to_global:
            raise ValueError(
                "empirical_bayes replaces --shrink-to-global; they are two versions of the "
                "same idea and composing them shrinks twice"
            )
        # shrinkage="empirical_bayes" (Family S, prereg Amendment 11): instead of a binary
        # certify/reject decision, every state cell is pulled toward the global card by its
        # own reliability, B = tau^2 / (tau^2 + se^2). A well-supported consistent cell keeps
        # its estimate (B -> 1); a thin noisy one falls back to global (B -> 0); nothing is
        # discarded and nothing is trusted blindly. James-Stein (1961) / Efron-Morris (1975),
        # the form Jorion (1986) and Vasicek (1973) use on portfolio inputs and betas.
        self._shrinkage = shrinkage
        # Contributions g = v*r built at attribution horizon h are h-day OVERLAPPING sums:
        # consecutive observations share h-1 days, so the variance of their mean is inflated
        # by ~h relative to the independent case (the same dependence that forces
        # terminal_verify onto a block bootstrap). ``overlap`` carries h into the shrinkage
        # standard error; it is threaded from the fixed a-priori attribution horizon, never
        # tuned. It only affects _shrinkage_factor, so the default of 1 changes nothing.
        if int(overlap) < 1:
            raise ValueError("overlap must be >= 1")
        self._overlap = int(overlap)
        if score_mode == "hit_logodds" and shrink_to_global and scope == "state":
            raise ValueError("hit_logodds retrieval does not compose with shrink_to_global")
        # score_mode="hit_logodds" (Family E1, prereg Amendment 3): retrieval scores each
        # cell by the Laplace-shrunk log-odds of its hit rate, logit((hits+1)/(count+2)),
        # instead of ew_mean - lambda*ew_var. The logit is already dimensionless, so the
        # pooled-std standardization step is skipped; tau and the reflection delta apply
        # on the same logit scale. Moments and learning are unchanged (grid invariance).
        self._score_mode = score_mode
        self._scope = scope
        self._use_reflection = bool(use_reflection)
        # certified_only=True (2026-07-19, "deviate from equal only where the
        # terminal verification certified reliable skill"): retrieval zeroes the
        # z of every cell OUTSIDE the certified set, so states with nothing
        # certified fall back to equal weights. The certified set is written
        # once, next to the terminal delta clearing; during training it is
        # empty, so a certified-only arm trades equal-weight until freeze.
        self._certified_only = bool(certified_only)
        self._certified: set[tuple[str, str]] = set()
        self._min_support = int(min_support)
        self._tau = float(temperature)
        # bound_reflection=False is the E3 unbounded-memory ablation: the reflection delta
        # is written without the +/-0.10 logit clip. The main configuration always clips.
        self._bound_reflection = bool(bound_reflection)
        # standardize=True divides the score by the state's pooled EW std before the
        # temperature, so z is dimensionless: tau becomes a pure number (grid ~ 0.5/1/2),
        # comparable across names, and the +/-0.10 reflection delta is commensurate with z.
        # A pure retrieval transform -- the moments are unchanged (grid invariance holds).
        self._standardize = bool(standardize)
        # shrink_to_global=True (state scope only) replaces the equal-weight fallback with
        # partial pooling: per-expert moments blend state and global cards with
        # lambda = n_state / (n_state + min_support), so an under-supported state shrinks
        # toward the pooled (global) evidence instead of discarding it.
        self._shrink_to_global = bool(shrink_to_global) and scope == "state"
        self._cards: dict[tuple[str, str], Scorecard] = {}
        self._frozen = False
        self._frozen_at = None
        if not (half_life > 0):
            raise ValueError("half_life must be positive")
        self._half_life = float(half_life)
        self._alpha = 1.0 - 0.5 ** (1.0 / self._half_life)
        self._variance_penalty = float(variance_penalty)

    # ---- keys -------------------------------------------------------------------
    def _key_state(self, market_state: str) -> str:
        return market_state if self._scope == "state" else GLOBAL_MARKET_STATE

    def _card(self, state: str, expert_id: str) -> Scorecard:
        key = (state, expert_id)
        sc = self._cards.get(key)
        if sc is None:
            sc = Scorecard()
            self._cards[key] = sc
        return sc

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _ensure_writable(self) -> None:
        if self._frozen:
            raise RuntimeError("ExpertScorecardStore is frozen")

    # ---- accumulation -----------------------------------------------------------
    def observe(self, market_state: str, contributions: Mapping[str, float]) -> None:
        """Fold one episode's per-expert counterfactual contributions into the scorecard.

        SIGN CONVENTION: each contribution is ``ep.signal_attribution[expert]`` from
        ``runner._attribute`` -- the expert's own view executed at unit conviction, net of
        its own cost, NOT demeaned. Larger is better (a long view in a rising market is
        positive; a long view in a falling market is negative), which is what makes the
        score ``mean - lambda*var`` and the softmax tilt toward the more reliable expert.
        """
        self._ensure_writable()
        if market_state not in MARKET_STATES:
            return
        state = self._key_state(market_state)
        for expert_id, x in contributions.items():
            if expert_id in EXPERT_IDS:
                self._card(state, expert_id).observe(x, alpha=self._alpha)
                if self._shrink_to_global or self._shrinkage == "empirical_bayes":
                    # Parallel pooled aggregate: the shrinkage target for both the crude
                    # count-based blend and the empirical-Bayes one.
                    self._card(GLOBAL_MARKET_STATE, expert_id).observe(x, alpha=self._alpha)

    def set_reflection_delta(self, market_state: str, expert_id: str, delta: float) -> None:
        self._ensure_writable()
        if expert_id not in EXPERT_IDS:
            raise ValueError(f"unknown expert_id {expert_id!r}")
        state = self._key_state(market_state)
        value = clip_delta(delta) if self._bound_reflection else float(delta)
        self._card(state, expert_id).reflection_delta = value

    # ---- drop-in shim so the store stands in for the runner/holdout weight store -------
    def add_learned_adjustment(self, entry) -> str:
        """Apply a reflection proposal (carrying expert_id, market_state, weight_delta as a
        logit adjustment) to the scorecard's reflection delta."""
        self.set_reflection_delta(entry.market_state, entry.expert_id, entry.weight_delta)
        return getattr(entry, "entry_id", "")

    def freeze(self, as_of=None) -> None:
        self._frozen = True
        self._frozen_at = as_of

    @property
    def frozen_at(self):
        return self._frozen_at

    def count_learned(self) -> int:
        return self.reflection_delta_count()

    def learned(self) -> list["LearnedDelta"]:
        """Reflection deltas as audit entries (one per expert-state with a nonzero delta)."""
        return [
            LearnedDelta(market_state=state, expert_id=expert_id,
                         weight_delta=sc.reflection_delta,
                         rationale=f"scorecard mean={sc.ew_mean:.6f} n={sc.count}")
            for (state, expert_id), sc in sorted(self._cards.items())
            if sc.reflection_delta != 0.0
        ]

    # ---- inspection -------------------------------------------------------------
    def support(self, market_state: str) -> int:
        state = self._key_state(market_state)
        return max(
            (self._cards[(state, e)].count for e in EXPERT_IDS if (state, e) in self._cards),
            default=0,
        )

    def learned_state_count(self) -> int:
        """Number of states whose support clears min_support (tilts off equal weights)."""
        states = {state for state, _ in self._cards}
        return sum(
            1 for state in states
            if max((self._cards[(state, e)].count for e in EXPERT_IDS if (state, e) in self._cards),
                   default=0) >= self._min_support
        )

    def reflection_delta_count(self) -> int:
        return sum(1 for sc in self._cards.values() if sc.reflection_delta != 0.0)

    # ---- retrieval --------------------------------------------------------------
    def set_certified_cells(self, cells) -> None:
        """Record the terminally-verified cells (call next to the delta clearing)."""
        self._ensure_writable()
        self._certified = {(str(s), str(e)) for s, e in cells}

    def _ew_mean_variance(self, card: "Scorecard", sigma2: Optional[float] = None) -> float:
        """Sampling variance of this card's exponentially weighted mean.

        At steady state the EW mean's effective sample size is (2-alpha)/alpha -- 182 at the
        a-priori half-life of 63 -- but a card observed fewer times than that has only
        ``count`` observations' worth of information, so n_eff = min(count, cap). The v1
        estimator used the steady-state form alone, which divided a 7-observation cell's
        variance by 182 and made the LEAST-evidenced cell look the MOST precise (the TSLA
        down_calm pathology, median B 0.962). ``sigma2`` overrides the card's own ew_var for
        under-supported cells, whose accumulated-from-zero variance is not a valid dispersion
        estimate.
        """
        cap = (2.0 - self._alpha) / self._alpha
        n_eff = max(1.0, min(float(card.count), cap))
        base = float(card.ew_var) if sigma2 is None else float(sigma2)
        return base * float(self._overlap) / n_eff

    def _shrinkage_factor(self, state: str, expert_id: str) -> Optional[float]:
        """Empirical-Bayes B for one cell, or None when shrinkage cannot apply.

        B = tau^2 / (tau^2 + se^2), with tau^2 the between-state dispersion of THIS expert's
        SUPPORTED state cells (count >= min_support) around the global card, by method of
        moments. Under-supported cells are excluded from tau^2 -- a 7-observation mean is not
        evidence about between-state structure -- and priced with the pooled variance of the
        supported peers, so the thinner the cell, the harder it is pulled toward global.
        B -> 1 keeps the cell's own estimate, B -> 0 falls back to the global card.
        """
        if self._shrinkage != "empirical_bayes" or state == GLOBAL_MARKET_STATE:
            return None
        gc = self._cards.get((GLOBAL_MARKET_STATE, expert_id))
        sc = self._cards.get((state, expert_id))
        if gc is None or gc.count <= 0 or sc is None or sc.count <= 0:
            return None
        supported = [
            c for (s, e), c in self._cards.items()
            if e == expert_id and s != GLOBAL_MARKET_STATE and c.count >= self._min_support
        ]
        if len(supported) < 2:
            return None
        between = fmean((c.ew_mean - gc.ew_mean) ** 2 for c in supported)
        noise = fmean(self._ew_mean_variance(c) for c in supported)
        tau2 = max(0.0, between - noise)
        if tau2 <= 0.0:
            return 0.0                      # no real between-state signal: use global for all
        own_supported = sc.count >= self._min_support
        sigma2 = None if own_supported else fmean(c.ew_var for c in supported)
        se2 = self._ew_mean_variance(sc, sigma2=sigma2)
        if se2 <= 0.0:
            # A supported constant cell genuinely has no sampling noise; an under-supported
            # one with a degenerate pooled variance has no usable evidence at all.
            return 1.0 if own_supported else 0.0
        return tau2 / (tau2 + se2)

    def _effective_moments(self, state: str, expert_id: str) -> tuple[float, float, float]:
        """(mean, var, delta) for retrieval, applying shrinkage when enabled."""
        if self._certified_only and (state, expert_id) not in self._certified:
            return 0.0, 0.0, 0.0
        sc = self._cards.get((state, expert_id))
        mean = sc.ew_mean if sc is not None else 0.0
        var = sc.ew_var if sc is not None else 0.0
        delta = sc.reflection_delta if (sc is not None and self._use_reflection) else 0.0
        if self._shrink_to_global and state != GLOBAL_MARKET_STATE:
            gc = self._cards.get((GLOBAL_MARKET_STATE, expert_id))
            g_mean = gc.ew_mean if gc is not None else 0.0
            g_var = gc.ew_var if gc is not None else 0.0
            n_state = sc.count if sc is not None else 0
            lam = n_state / (n_state + self._min_support)
            mean = lam * mean + (1.0 - lam) * g_mean
            var = lam * var + (1.0 - lam) * g_var
        b = self._shrinkage_factor(state, expert_id)
        if b is not None:
            gc = self._cards[(GLOBAL_MARKET_STATE, expert_id)]
            mean = gc.ew_mean + b * (mean - gc.ew_mean)
            var = gc.ew_var + b * (var - gc.ew_var)
        return mean, var, delta

    def _effective_hit_logit(self, state: str, expert_id: str) -> tuple[float, float]:
        """(logit, delta) for hit-rate retrieval, respecting certified-only retrieval.

        An absent or uncertified cell contributes logit 0 (the Laplace prior's even
        odds), which after the softmax is exactly the equal-weight fallback.
        """
        if self._certified_only and (state, expert_id) not in self._certified:
            return 0.0, 0.0
        sc = self._cards.get((state, expert_id))
        if sc is None or sc.count <= 0:
            return 0.0, 0.0
        p = (sc.hits + 1.0) / (sc.count + 2.0)
        delta = sc.reflection_delta if self._use_reflection else 0.0
        return math.log(p / (1.0 - p)), delta

    def get_weights(self, market_state: str) -> dict[str, float]:
        """Floored, normalised four-weight dict for the market state.

        Without shrinkage: equal weights when the (scoped) state has fewer than
        ``min_support`` episodes. With shrinkage: equal weights only while the
        POOLED (global) support is below ``min_support``; after that an
        under-supported state shrinks toward the pooled evidence instead of
        discarding it.
        """
        state = self._key_state(market_state)
        if self._shrink_to_global:
            pooled = max(
                (self._cards[(GLOBAL_MARKET_STATE, e)].count
                 for e in EXPERT_IDS if (GLOBAL_MARKET_STATE, e) in self._cards),
                default=0,
            )
            if pooled < self._min_support:
                return equal_weights()
        elif self.support(market_state) < self._min_support:
            return equal_weights()

        if self._score_mode == "hit_logodds":
            zs = {}
            for e in EXPERT_IDS:
                logit, delta = self._effective_hit_logit(state, e)
                zs[e] = logit / self._tau + delta
            z_max = max(zs.values())
            exps = {e: math.exp(zs[e] - z_max) for e in EXPERT_IDS}
            total = sum(exps.values())
            return project_to_floored_simplex({e: exps[e] / total for e in EXPERT_IDS}, WEIGHT_FLOOR)

        moments = {e: self._effective_moments(state, e) for e in EXPERT_IDS}
        scale = 1.0
        if self._standardize:
            # Pool over every cell that CONTRIBUTES moments, zero variance
            # included: a fully supported constant expert is real evidence
            # about the state's dispersion, and dropping it inflated the
            # pooled scale. Under shrinkage an unseen state borrows the
            # global cell, so the global count keeps it in the pool.
            def _in_pool(e: str) -> bool:
                sc = self._cards.get((state, e))
                if sc is not None and sc.count > 0:
                    return True
                if self._shrink_to_global or self._shrinkage == "empirical_bayes":
                    gc = self._cards.get((GLOBAL_MARKET_STATE, e))
                    return gc is not None and gc.count > 0
                return False

            variances = [moments[e][1] for e in EXPERT_IDS if _in_pool(e)]
            pooled_std = math.sqrt(fmean(variances)) if variances else 0.0
            if pooled_std > 0.0:
                scale = pooled_std
        zs: dict[str, float] = {}
        for e in EXPERT_IDS:
            mean, var, delta = moments[e]
            score = mean - self._variance_penalty * var
            zs[e] = (score / scale) / self._tau + delta
        z_max = max(zs.values())
        exps = {e: math.exp(zs[e] - z_max) for e in EXPERT_IDS}
        total = sum(exps.values())
        raw = {e: exps[e] / total for e in EXPERT_IDS}
        return project_to_floored_simplex(raw, WEIGHT_FLOOR)

    # ---- persistence ------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "type": "expert_scorecard",
            "scope": self._scope,
            "use_reflection": self._use_reflection,
            "certified_only": self._certified_only,
            "certified_cells": sorted(list(c) for c in self._certified),
            "min_support": self._min_support,
            "temperature": self._tau,
            "score_mode": self._score_mode,
            "shrinkage": self._shrinkage,
            "overlap": self._overlap,
            "standardize": self._standardize,
            "shrink_to_global": self._shrink_to_global,
            "half_life": self._half_life,
            "variance_penalty": self._variance_penalty,
            "frozen": self._frozen,
            "cards": [
                {
                    "market_state": state,
                    "expert_id": expert_id,
                    "ew_mean": sc.ew_mean,
                    "ew_var": sc.ew_var,
                    "count": sc.count,
                    "hits": sc.hits,
                    "reflection_delta": sc.reflection_delta,
                }
                for (state, expert_id), sc in sorted(self._cards.items())
            ],
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping, *, temperature: Optional[float] = None) -> "ExpertScorecardStore":
        """Rebuild a store from a snapshot. ``temperature`` overrides the stored tau, which
        is how the validation grid re-reads one frozen scorecard at each grid temperature
        without retraining (the scorecard moments are invariant to tau)."""
        store = cls(
            scope=str(payload.get("scope", "state")),
            use_reflection=bool(payload.get("use_reflection", True)),
            certified_only=bool(payload.get("certified_only", False)),
            min_support=int(payload.get("min_support", DEFAULT_MIN_SUPPORT)),
            temperature=float(temperature) if temperature is not None
            else float(payload.get("temperature", SOFTMAX_TEMPERATURE)),
            score_mode=str(payload.get("score_mode", "ewvar")),
            shrinkage=str(payload.get("shrinkage", "none")),
            overlap=int(payload.get("overlap", 1)),
            standardize=bool(payload.get("standardize", False)),
            shrink_to_global=bool(payload.get("shrink_to_global", False)),
            half_life=float(payload.get("half_life", SCORECARD_HALFLIFE)),
            variance_penalty=float(payload.get("variance_penalty", VARIANCE_PENALTY)),
        )
        for row in payload.get("cards", []):
            sc = Scorecard(
                ew_mean=float(row["ew_mean"]),
                ew_var=float(row["ew_var"]),
                count=int(row["count"]),
                hits=int(row.get("hits", 0)),
                reflection_delta=float(row.get("reflection_delta", 0.0)),
            )
            store._cards[(str(row["market_state"]), str(row["expert_id"]))] = sc
        store._certified = {
            (str(s), str(e)) for s, e in payload.get("certified_cells", [])
        }
        if bool(payload.get("frozen", False)):
            store._frozen = True
        return store


def snapshot_payload(store: ExpertScorecardStore, *, status: str = "frozen") -> dict:
    """Auditable snapshot of a (frozen) scorecard store for the run manifest."""
    payload = store.snapshot()
    payload["status"] = status
    payload["frozen_at"] = store.frozen_at.isoformat() if store.frozen_at is not None else None
    return payload


def snapshot_entries(payload: Mapping) -> list[LearnedDelta]:
    """Reflection deltas from a snapshot payload (audit / paper-mode application)."""
    return [
        LearnedDelta(
            market_state=str(row["market_state"]),
            expert_id=str(row["expert_id"]),
            weight_delta=float(row.get("reflection_delta", 0.0)),
        )
        for row in payload.get("cards", [])
        if float(row.get("reflection_delta", 0.0)) != 0.0
    ]
