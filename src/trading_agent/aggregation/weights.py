"""Projection onto the floored probability simplex.

The single source of truth for "valid expert weights" in this system.
Aggregation and memory retrieval both call ``project_to_floored_simplex`` instead of
re-implementing floor logic independently.

A floored simplex is the set ``{ w : w_i >= floor for all i, sum_i w_i == 1 }``.
Projecting onto it by a single clamp-then-renormalise pass does not work: renorm
can push a clamped weight back below the floor. We instead iterate clamp/renorm
to convergence, which is the standard water-filling projection: weights that fall
below the floor are pinned at the floor, and the remaining budget is distributed
proportionally over the still-free weights until every weight is >= floor.
"""

from __future__ import annotations

import math

from typing import Mapping

WEIGHT_FLOOR: float = 0.05
_TOL: float = 1e-12


def project_to_floored_simplex(
    weights: Mapping[str, float],
    floor: float = WEIGHT_FLOOR,
) -> dict[str, float]:
    """Project ``weights`` onto the floored simplex {w_i >= floor, sum == 1}.

    Iterative clamp-renorm (water-filling) until convergence:
      1. distribute the remaining budget over the free weights proportional to
         their (non-negative) input value;
      2. any free weight whose allocation falls below ``floor`` is pinned at
         ``floor`` and removed from the free set;
      3. repeat until no free weight violates the floor.

    Guarantees on the returned dict (same keys as input):
      * every value >= floor (up to floating-point tolerance)
      * values sum to exactly 1.0 (up to floating-point tolerance)

    Negative inputs are treated as 0. If every input is 0 (or non-positive) the
    budget is split equally. Raises ``ValueError`` if ``n * floor > 1`` (the floor
    is then infeasible) or if ``floor`` is negative.
    """
    keys = list(weights)
    n = len(keys)
    if n == 0:
        return {}
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}")
    if n * floor > 1.0 + _TOL:
        raise ValueError(
            f"infeasible floor: {n} weights cannot all be >= {floor} and sum to 1"
        )

    vals: dict[str, float] = {}
    for k in keys:
        v = float(weights[k])
        # NaN propagates through max() and +inf overflows free_sum to inf,
        # both of which broke the projection; neither carries information.
        vals[k] = v if math.isfinite(v) and v > 0.0 else 0.0
    peak = max(vals.values(), default=0.0)
    if peak > 0.0:
        # Ratios are all the projection uses; rescaling by the peak keeps
        # free_sum finite even for inputs near the float ceiling.
        vals = {k: v / peak for k, v in vals.items()}
    fixed: set[str] = set()

    while True:
        free = [k for k in keys if k not in fixed]
        if not free:
            # Reached only when n * floor == 1 and all weights were pinned.
            return {k: floor for k in keys}

        remaining = 1.0 - floor * len(fixed)
        free_sum = sum(vals[k] for k in free)
        if free_sum <= 0.0:
            # No information among the free weights: split the remainder equally.
            # Feasibility (n*floor <= 1) guarantees this is >= floor, so it never
            # violates and the loop terminates here.
            alloc = {k: remaining / len(free) for k in free}
        else:
            alloc = {k: remaining * vals[k] / free_sum for k in free}

        violating = [k for k in free if alloc[k] < floor - _TOL]
        if not violating:
            result = {k: floor for k in fixed}
            result.update(alloc)
            return result
        fixed.update(violating)
