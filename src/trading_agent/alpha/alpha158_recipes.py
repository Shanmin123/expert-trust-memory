"""Registered Alpha158 feature metadata (external-library first; no formula reimplementation).

The project may call an installed Qlib package for Alpha158 features. This registry only names the
feature subset we want to test and records provenance; it does not reimplement Qlib expressions and
it does not import Qlib or any local research clone at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

FeatureFamily = Literal["momentum", "trend", "volatility", "range", "volume"]

DEFINITION_REF = "qlib.contrib.data.loader.Alpha158DL.get_feature_config"


@dataclass(frozen=True)
class Alpha158FeatureSpec:
    """One external Alpha158 feature treated as a strategy raw material."""

    strategy_id: str
    qlib_name: str
    family: FeatureFamily
    definition_ref: str = DEFINITION_REF
    economic_prior_notes: str = ""

    def __post_init__(self):
        if not self.strategy_id.startswith("alpha158_"):
            raise ValueError("strategy_id must use the stable alpha158_ prefix")
        if not self.qlib_name:
            raise ValueError("qlib_name must be non-empty")


REGISTERED_ALPHA158_FEATURES: tuple[Alpha158FeatureSpec, ...] = (
    Alpha158FeatureSpec("alpha158_roc5", "ROC5", "momentum",
                        economic_prior_notes="Rate-of-change feature; raw material for regime dispatch."),
    Alpha158FeatureSpec("alpha158_roc20", "ROC20", "momentum",
                        economic_prior_notes="Medium-horizon rate-of-change feature; may be regime dependent."),
    Alpha158FeatureSpec("alpha158_ma5", "MA5", "trend",
                        economic_prior_notes="Short moving-average displacement; trend-state raw material."),
    Alpha158FeatureSpec("alpha158_ma20", "MA20", "trend",
                        economic_prior_notes="Medium moving-average displacement; trend-state raw material."),
    Alpha158FeatureSpec("alpha158_std20", "STD20", "volatility",
                        economic_prior_notes="Realized-volatility feature; may help risk-off dispatch."),
    Alpha158FeatureSpec("alpha158_std60", "STD60", "volatility",
                        economic_prior_notes="Longer realized-volatility feature; defensive/risk-off raw material."),
    Alpha158FeatureSpec("alpha158_max20", "MAX20", "range",
                        economic_prior_notes="Recent high relative to current close; breakout/range raw material."),
    Alpha158FeatureSpec("alpha158_min20", "MIN20", "range",
                        economic_prior_notes="Recent low relative to current close; range/mean-reversion raw material."),
    Alpha158FeatureSpec("alpha158_rsv20", "RSV20", "range",
                        economic_prior_notes="Position in recent high-low range; sideways-regime raw material."),
    Alpha158FeatureSpec("alpha158_vma20", "VMA20", "volume",
                        economic_prior_notes="Volume moving-average displacement; attention/liquidity raw material."),
    Alpha158FeatureSpec("alpha158_vstd20", "VSTD20", "volume",
                        economic_prior_notes="Volume variability; attention/liquidity raw material."),
)


def features_by_strategy_id(
    features: Sequence[Alpha158FeatureSpec] = REGISTERED_ALPHA158_FEATURES,
) -> dict[str, Alpha158FeatureSpec]:
    return {f.strategy_id: f for f in features}


def features_by_qlib_name(
    features: Sequence[Alpha158FeatureSpec] = REGISTERED_ALPHA158_FEATURES,
) -> dict[str, Alpha158FeatureSpec]:
    return {f.qlib_name: f for f in features}


__all__ = [
    "Alpha158FeatureSpec",
    "REGISTERED_ALPHA158_FEATURES",
    "features_by_strategy_id",
    "features_by_qlib_name",
    "DEFINITION_REF",
]
