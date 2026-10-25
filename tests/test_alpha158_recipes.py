from __future__ import annotations

from pathlib import Path

from trading_agent.alpha.alpha158_recipes import (
    DEFINITION_REF,
    REGISTERED_ALPHA158_FEATURES,
    features_by_qlib_name,
    features_by_strategy_id,
)


def test_alpha158_registry_is_metadata_only_and_stable():
    assert "qlib.contrib.data.loader.Alpha158DL" in DEFINITION_REF
    ids = [f.strategy_id for f in REGISTERED_ALPHA158_FEATURES]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("alpha158_") for i in ids)
    assert {"momentum", "trend", "volatility", "range", "volume"}.issubset(
        {f.family for f in REGISTERED_ALPHA158_FEATURES}
    )
    assert features_by_strategy_id()["alpha158_roc20"].qlib_name == "ROC20"
    assert features_by_qlib_name()["MA20"].strategy_id == "alpha158_ma20"


def test_alpha158_registry_does_not_import_research_clone_or_qlib_runtime():
    src = (Path(__file__).parent.parent / "src/trading_agent/alpha/alpha158_recipes.py").read_text(encoding="utf-8")
    assert "research_refs" not in src
    assert "import qlib" not in src
