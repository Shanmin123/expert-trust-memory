"""Aggregation layer: combines the ExpertViews with floored, normalised weights.

``aggregate`` is the runner entry point. ``project_to_floored_simplex`` is the
shared floor/normalisation primitive used by aggregation and learned-memory
retrieval. ``normalise_weights`` is the EXPERT_IDS-aware wrapper around it.
"""

from trading_agent.aggregation.soft_weighting import (
    EXPERT_IDS,
    AggregationResult,
    aggregate,
    normalise_weights,
)
from trading_agent.aggregation.weights import WEIGHT_FLOOR, project_to_floored_simplex

__all__ = [
    "EXPERT_IDS",
    "WEIGHT_FLOOR",
    "AggregationResult",
    "aggregate",
    "normalise_weights",
    "project_to_floored_simplex",
]
