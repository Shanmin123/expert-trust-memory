"""CLI/reporting tests for the expert-weight holdout script."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from trading_agent.experiments import expert_weight_holdout_launcher


def _load_script_module():
    return expert_weight_holdout_launcher


def test_explicit_tickers_llm_reflection_and_numeric_only_are_wired(
    tmp_path, regime_dataset, mem_backtest_config, monkeypatch
):
    script = _load_script_module()

    monkeypatch.setattr(script, "load_panel", lambda *args, **kwargs: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))
    monkeypatch.setattr(script, "_protocol_universe_has_price_data", lambda *args, **kwargs: True)

    def fake_build_sources(**kwargs):
        from trading_agent.expert_context import ExpertSignalSources

        return ExpertSignalSources()

    monkeypatch.setattr(script, "build_expert_signal_sources", fake_build_sources)

    def liquidity_selection_must_not_run(*args, **kwargs):
        raise AssertionError("--tickers must bypass the liquidity-ranked selection")

    monkeypatch.setattr(script, "select_fixed_universe", liquidity_selection_must_not_run)

    class StubChatBackend:
        """Stands in for OpenAIChatBackend; returns an empty proposal set."""

        def __init__(self, *, model, temperature=0.0, model_revision=None):
            self.model = model
            self.model_name = model
            self.model_revision = model_revision or model
            self.temperature = temperature

        def call(self, system_prompt, user_prompt):
            return json.dumps({"proposals": []})

    monkeypatch.setattr(script, "OpenAIChatBackend", StubChatBackend)

    output = tmp_path / "explicit_out"
    rc = script.main([
        "--data-root", "unused",
        "--start", mem_backtest_config.date_from.isoformat(),
        "--validation-boundary", "2022-04-08",
        "--boundary", "2022-05-06",
        "--end", mem_backtest_config.date_to.isoformat(),
        "--tickers", "t1,T2,T3,T4",
        "--embargo-days", "2",
        "--reflect-every", "10",
        "--reflection-window", "7",
        "--reflection-min-support", "1",
        "--reflection-backend", "llm",
        "--numeric-only-memory",
        "--decision-thresholds", "0.3,0.4,0.5",
        "--vol-window", "5",
        "--percentile-window", "20",
        "--trend-window", "10",
        "--trend-confirm-days", "2",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    assert manifest["tickers"] == ["T1", "T2", "T3", "T4"]   # upper-cased, order kept
    assert manifest["universe_label"].startswith("explicit fixed universe")
    assert manifest["reflection_backend"] == "llm"
    assert manifest["reflection_model"] == manifest["expert_model"]  # default follows expert model
    assert manifest["reflection_fallback_count"] == 0
    assert manifest["use_reflection"] is False
    assert manifest["exit_threshold"] == 0.0   # HOLD semantics on by default


def test_exit_threshold_none_disables_hold_semantics():
    script = _load_script_module()
    base = ["--start", "2020-01-01", "--boundary", "2021-01-01", "--end", "2022-01-01"]
    parser = script.build_parser()
    assert parser.parse_args(base).exit_threshold == 0.0
    assert parser.parse_args(base + ["--exit-threshold", "none"]).exit_threshold is None
    assert parser.parse_args(base + ["--exit-threshold", "-0.25"]).exit_threshold == -0.25


def test_script_writes_json_and_markdown_reports(tmp_path, regime_dataset, mem_backtest_config, monkeypatch):
    script = _load_script_module()

    class Selection:
        tickers = ["T1", "T2", "T3", "T4"]
        label = "synthetic"

    monkeypatch.setattr(script, "select_fixed_universe", lambda *args, **kwargs: Selection())
    monkeypatch.setattr(script, "load_panel", lambda *args, **kwargs: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))
    captured_sources = {}

    def fake_build_sources(**kwargs):
        from trading_agent.expert_context import ExpertSignalSources

        captured_sources.update(kwargs)
        return ExpertSignalSources()

    monkeypatch.setattr(script, "build_expert_signal_sources", fake_build_sources)

    output = tmp_path / "expert_weight_holdout"
    memory_output = tmp_path / "expert_weight_memory.json"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps({
            "reserved_test_periods": [
                {"name": "reserved", "start": "2030-01-01", "end": "2030-12-31"}
            ]
        }),
        encoding="utf-8",
    )
    rc = script.main([
        "--data-root", "unused",
        "--alpha158-factors-db", "alpha158.sqlite",
        "--start", mem_backtest_config.date_from.isoformat(),
        # Training spans into the mean-reversion segment so a flat expert reliably scores
        # positive there (the fixture's momentum-vs-reversal learning intent); an all-trend
        # training window under the 2D trend x risk state yields no learnable pattern.
        "--validation-boundary", "2022-04-08",
        "--boundary", "2022-05-06",
        "--end", mem_backtest_config.date_to.isoformat(),
        "--names", "4",
        "--embargo-days", "2",
        "--reflect-every", "10",
        "--reflection-window", "7",
        "--reflection-min-support", "1",
        "--decision-thresholds", "0.3,0.4,0.5",
        # heuristic: the experts read the (warmed-up) technical/fundamental context and
        # genuinely disagree. The static backend cannot feed this fixture's learning
        # assertions any more: its factor/news views are data-unavailable safe defaults
        # here, and those are excluded from learning evidence by design.
        "--expert-backend", "heuristic",
        "--run-label", "REHEARSAL",
        "--commission-per-share", "0.02",
        "--min-commission", "1.25",
        "--max-commission-rate", "0.03",
        "--slippage-perc", "0.001",
        "--liquidity-cap-pct", "0.15",
        "--vol-window", "5",
        "--percentile-window", "20",
        "--trend-window", "10",
        "--trend-confirm-days", "2",
        "--output", str(output),
        "--memory-output", str(memory_output),
        "--protocol", str(protocol_path),
    ])

    assert rc == 0
    assert captured_sources["data_root"] == "unused"
    assert captured_sources["alpha158_factors_db"] == "alpha158.sqlite"
    assert captured_sources["include_news"] is True
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")

    assert payload["mode"] == "holdout"
    assert payload["run_label"] == "REHEARSAL"
    assert payload["system"] == "four_expert_reflection"
    assert payload["freeze_ok"] is True
    assert payload["trial_count"] > 0
    assert payload["manifest"]["seed"] == 11
    assert payload["manifest"]["market_state"] == {
        "vol_window": 5,
        "percentile_window": 20,
        "trend_window": 10,
        "trend_confirm_days": 2,
    }
    assert payload["manifest"]["validation_boundary"] == "2022-04-08"
    assert payload["split_protocol"]["train_end"] == "2022-04-08"
    assert payload["split_protocol"]["validation_end"] == "2022-05-06"
    assert set(payload["validation_arms"]) == {"equal_weight", "global_memory", "market_state_memory"}
    assert set(payload["validation_arm_comparison_vs_equal_weight"]) == {
        "global_memory",
        "market_state_memory",
    }
    for comparison in payload["validation_arm_comparison_vs_equal_weight"].values():
        assert set(comparison) >= {
            "total_return_delta",
            "sharpe_delta",
            "max_drawdown_delta",
            "turnover_delta",
            "participation_rate_delta",
        }
    assert "Split Protocol" in markdown
    assert "Validation Arms" in markdown
    assert "Validation Comparison vs Equal Weight" in markdown
    assert payload["manifest"]["reflection_window"] == 7
    assert payload["manifest"]["reflection_min_support"] == 1
    assert payload["manifest"]["decision_thresholds"] == [0.3, 0.4, 0.5]
    assert payload["manifest"]["run_label"] == "REHEARSAL"
    assert payload["manifest"]["reserved_test_periods"]
    assert payload["selected_decision_threshold"] in [0.3, 0.4, 0.5]
    assert payload["selected_decision_threshold"] == payload["split_protocol"]["selected_decision_threshold"]
    assert payload["validation_threshold_sweep"][f"{payload['selected_decision_threshold']:.4f}"]["selected"] is True
    assert "expert_confidence_diagnostics" in payload
    assert payload["expert_confidence_diagnostics"]["train"]["technical"]["n"] > 0
    assert "Decision Threshold Selection" in markdown
    assert "Expert Confidence Diagnostics" in markdown
    assert payload["manifest"]["strict_news"] is False
    assert payload["manifest"]["expert_backend"] == "heuristic"
    assert payload["manifest"]["expert_model"] == "gpt-4.1-mini"
    assert payload["manifest"]["expert_model_revision"] == "heuristic-deterministic"
    assert payload["manifest"]["expert_transport"] == "heuristic"
    assert payload["manifest"]["expert_temperature"] == 0.0
    assert payload["manifest"]["codex_sdk_available"] in (True, False)
    assert payload["manifest"]["configuration_trial_count"] == payload["configuration_trial_count"]
    # trial_count = |tau grid| x |theta grid|; standardized memory is the default,
    # so the tau grid defaults to the three-point {0.5, 1, 2}.
    assert payload["configuration_trial_count"] == (
        len(payload["manifest"]["decision_thresholds"]) * len(payload["manifest"]["temperature_grid"])
    )
    assert payload["trial_count"] == payload["configuration_trial_count"]
    assert payload["candidate_audit_count"] == (
        payload["reflection_trial_audit"]["proposal_count"]
        + payload["reflection_trial_audit"]["global_proposal_count"]
    )
    assert payload["manifest"]["embargo_warning"]["minimum_non_diagnostic_embargo_days"] == 5
    assert "WARNING: embargo_days below main-protocol minimum" in markdown
    assert "expert_model_revision: heuristic-deterministic" in markdown
    # With the bps cost model active, the ignored commission parameters are
    # zeroed so the manifest states what is actually charged.
    assert payload["manifest"]["execution"] == {
        "initial_cash": 100000.0,
        "commission_per_share": 0.0,
        "min_commission": 0.0,
        "max_commission_rate": 0.03,
        "slippage_perc": 0.001,
        "execution_cost_bps_per_side": 10.0,
        "liquidity_cap_pct": 0.15,
    }
    assert payload["manifest"]["cost_protocol"] == {
        "model": "bps_per_side_on_traded_notional",
        "bps_replaces_per_share_commission": True,
        "headline_bps_per_side": 10.0,
        "sensitivity_bps_per_side": 25.0,
        "arm_identical": True,
    }
    assert payload["manifest"]["preload"]["calendar_days"] == 900
    assert payload["cost_sensitivity"]["cost_bps_per_side"] == 25.0
    assert payload["train"]["n_applied"] == payload["train"]["n_proposals"]
    assert "reflection_trial_audit" in payload
    assert payload["reflection_trial_audit"]["round_count"] == payload["train"]["n_reflection_rounds"]
    assert payload["reflection_trial_audit"]["applied_count"] == payload["train"]["n_applied"]
    assert isinstance(payload["reflection_trial_audit"]["rounds"], list)
    assert set(payload["research_claims"]) >= {
        "reflection_contribution",
        "regime_conditioning_contribution",
        "best_expert_ablation",
        "frozen_test_protocol",
    }
    assert payload["research_claims"]["reflection_contribution"]["comparison"] == (
        "market_state_memory_vs_equal_weight"
    )
    assert payload["research_claims"]["regime_conditioning_contribution"]["comparison"] == (
        "market_state_memory_vs_global_memory"
    )
    assert payload["research_claims"]["frozen_test_protocol"] == {
        "freeze_ok": True,
        "allow_reflection": False,
        "memory_status": "frozen",
    }
    assert payload["system_completion"]["research_system_complete"] is True
    assert payload["system_completion"]["external_paper_validation_required"] is False
    assert payload["system_completion"]["ready_for_defense_sections"] == [
        "data_observation",
        "market_state_classifier",
        "four_experts",
        "expert_error_analysis",
        "reflection_memory",
        "train_freeze_validation_test",
        "three_arm_evaluation",
        "per_state_ablation",
    ]
    assert payload["system_completion"]["checks"]["freeze_ok"] is True
    assert payload["system_completion"]["checks"]["test_reflection_disabled"] is True
    assert payload["system_completion"]["checks"]["memory_snapshot_frozen"] is True
    assert payload["system_completion"]["checks"]["all_required_arms_present"] is True
    assert payload["system_completion"]["checks"]["expert_error_reported"] is True
    assert payload["system_completion"]["checks"]["per_state_ablation_reported"] is True
    assert "expert_error_by_state" in payload
    assert "weight_adjustments_by_state" in payload
    assert "expert_adjustment_audit" in payload
    assert payload["expert_adjustment_audit"]
    assert set(payload["expert_adjustment_audit"][0]) >= {
        "market_state",
        "expert_id",
        "weight_delta",
        "train_accuracy",
        "state_signal_component_delta",
    }
    assert set(payload["expert_ablation_by_state"]) == {"technical", "news", "fundamental"}
    assert "Expert Ablation By State" in markdown
    assert "market_state_coverage" in payload
    assert "market_state_labels" in payload
    assert payload["market_state_labels"]["arms"]["market_state_memory"]
    assert "per_state_arm_metrics" in payload
    assert "per_state_arm_comparison_vs_equal_weight" in payload
    assert "underpowered_cells" in payload
    assert "train" in payload["market_state_coverage"]
    assert set(payload["market_state_coverage"]["arms"]) == {"equal_weight", "global_memory", "market_state_memory", "market_state_memory_online"}
    assert "market_state_memory" in payload["per_state_arm_metrics"]
    for comparison in payload["per_state_arm_comparison_vs_equal_weight"].values():
        for metrics in comparison.values():
            assert set(metrics) >= {
                "mean_market_return_delta",
                "mean_signal_component_delta",
                "participation_rate_delta",
                "underpowered",
                "exclude_from_headline_comparison",
            }
    assert "memory_snapshot" in payload
    assert "arm_memory_snapshots" in payload
    assert payload["memory_snapshot"]["status"] == "frozen"
    assert payload["memory_snapshot"]["status"] == "frozen"
    assert isinstance(payload["memory_snapshot"]["cards"], list)
    memory_payload = json.loads(memory_output.read_text(encoding="utf-8"))
    assert memory_payload == payload["memory_snapshot"]
    assert set(payload["arm_comparison_vs_equal_weight"]) == {
        "global_memory", "market_state_memory", "market_state_memory_online",
    }
    for comparison in payload["arm_comparison_vs_equal_weight"].values():
        assert set(comparison) >= {
            "total_return_delta",
            "sharpe_delta",
            "max_drawdown_delta",
            "turnover_delta",
            "participation_rate_delta",
        }
    assert set(payload["arms"]) == {"equal_weight", "global_memory", "market_state_memory", "market_state_memory_online"}
    for arm in payload["arms"].values():
        assert set(arm["metrics"]) >= {
            "total_return",
            "sharpe",
            "max_drawdown",
            "turnover",
            "participation_rate",
            "trial_count",
            "learned_adjustment_count",
            "sharpe_trial_count",
            "deflated_sharpe_ratio",
        }
        assert arm["metrics"]["sharpe_trial_count"] == payload["trial_count"]
        assert arm["metrics"]["sharpe_trial_count"] == payload["configuration_trial_count"]
    assert "# REHEARSAL Expert Weight Holdout" in markdown
    assert "commission_per_share: 0.0" in markdown
    assert "slippage_perc: 0.001" in markdown
    assert "headline_cost_bps_per_side: 10.0" in markdown
    assert "Cost Sensitivity Arms" in markdown
    assert "applied adjustments" in markdown
    assert "Research Claims" in markdown
    assert "System Completion" in markdown
    assert "market_state_memory_vs_equal_weight" in markdown
    assert "market_state_memory_vs_global_memory" in markdown
    assert "Expert Error Summary" in markdown
    assert "Expert Adjustment Audit" in markdown
    assert "Market State Coverage" in markdown
    assert "Per-State Arm Metrics" in markdown
    assert "Per-State Comparison vs Equal Weight" in markdown
    assert "Underpowered Cells" in markdown
    assert "Comparison vs Equal Weight" in markdown
    assert "market_state_memory" in markdown
    assert "turnover" in markdown
    assert "validation" in payload["per_name_episode_counts"]
    assert "validation" in payload["episode_count_before_after"]


def test_script_records_then_replays_a_frozen_view_corpus(
    tmp_path, regime_dataset, mem_backtest_config, monkeypatch
):
    script = _load_script_module()

    class Selection:
        tickers = ["T1", "T2", "T3", "T4"]
        label = "synthetic"

    monkeypatch.setattr(script, "select_fixed_universe", lambda *a, **k: Selection())
    monkeypatch.setattr(script, "load_panel", lambda *a, **k: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))

    from trading_agent.expert_context import ExpertSignalSources
    monkeypatch.setattr(script, "build_expert_signal_sources", lambda **k: ExpertSignalSources())

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"reserved_test_periods": []}), encoding="utf-8")
    corpus_path = tmp_path / "views.jsonl"

    def run(mode, out):
        return script.main([
            "--data-root", "unused",
            "--start", mem_backtest_config.date_from.isoformat(),
            "--validation-boundary", "2022-03-01",
            "--boundary", "2022-04-01",
            "--end", mem_backtest_config.date_to.isoformat(),
            "--names", "4",
            "--embargo-days", "2",
            "--reflect-every", "10",
            "--reflection-min-support", "1",
            "--decision-thresholds", "0.3,0.5",
            "--vol-window", "5",
            "--percentile-window", "20",
            "--trend-window", "10",
            "--trend-confirm-days", "2",
            "--sensitivity-cost-bps-per-side", "10.0",  # skip the sensitivity re-run
            "--view-corpus", str(corpus_path),
            "--view-corpus-mode", mode,
            "--output", str(out),
            "--protocol", str(protocol_path),
        ])

    # Record with the (deterministic static) backend, then replay off the file.
    assert run("readthrough", tmp_path / "rec") == 0
    assert corpus_path.exists()
    rec_payload = json.loads((tmp_path / "rec.json").read_text(encoding="utf-8"))
    # The main configuration is the not-demeaned counterfactual (raw), recorded in manifest.
    assert rec_payload["manifest"]["attribution_mode"] == "raw"
    assert rec_payload["manifest"]["bounded_memory"] is True

    corpus_meta = rec_payload["manifest"]["view_corpus"]
    assert corpus_meta["mode"] == "readthrough"
    assert corpus_meta["entries"] > 0
    assert corpus_meta["records_written"] > 0

    # E0 reliability diagnostic is emitted for train and test windows.
    reliability = rec_payload["expert_reliability"]
    assert reliability["train"]["n_view_outcomes"] > 0
    assert "calibration_by_state" in reliability["train"]
    assert "expert_correlation" in reliability["test"]

    # D3 neutral-usage instrumentation: raw direction mix per expert x state.
    directions = rec_payload["view_direction_summary"]
    for window in ("train", "test"):
        assert directions[window], window
        for expert_id, summary in directions[window].items():
            total = summary["all_states"]
            assert total["n"] > 0, expert_id
            assert (
                total["long"] + total["neutral"] + total["flat"] + total["safe_default"]
                <= total["n"]
            )
            assert 0.0 <= total["neutral_rate"] <= 1.0
            assert 0.0 <= total["spoken_rate"] <= 1.0

    assert run("replay", tmp_path / "rep") == 0
    rep_payload = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    assert rep_payload["manifest"]["view_corpus"]["mode"] == "replay"
    # Replay wrote nothing new and served everything from the corpus.
    assert rep_payload["manifest"]["view_corpus"]["records_written"] == 0
    assert rep_payload["manifest"]["view_corpus"]["hits"] > 0
    # Same frozen views -> same headline arms.
    assert rep_payload["arms"]["market_state_memory"]["metrics"]["total_return"] == pytest.approx(
        rec_payload["arms"]["market_state_memory"]["metrics"]["total_return"]
    )


def test_config_zeroes_ignored_commission_params_when_bps_model_active():
    script = _load_script_module()
    base_argv = [
        "--data-root", "unused",
        "--start", "2022-01-03",
        "--boundary", "2022-04-01",
        "--end", "2022-07-08",
    ]

    args = script.build_parser().parse_args(base_argv)
    args.preload_start = script._preload_start(args.start, args.preload_calendar_days)
    cfg = script._config(args, ("T1",))
    # Default headline cost is 10 bps, which replaces per-share commission in the
    # adapter, so the ignored commission parameters must be zeroed in the config.
    assert cfg.execution_cost_bps_per_side == 10.0
    assert cfg.commission_per_share == 0.0
    assert cfg.min_commission == 0.0

    args_no_bps = script.build_parser().parse_args(
        base_argv + ["--headline-cost-bps-per-side", "0.0"]
    )
    args_no_bps.preload_start = script._preload_start(
        args_no_bps.start, args_no_bps.preload_calendar_days
    )
    cfg_no_bps = script._config(args_no_bps, ("T1",))
    assert cfg_no_bps.execution_cost_bps_per_side == 0.0
    assert cfg_no_bps.commission_per_share == pytest.approx(0.0049)
    assert cfg_no_bps.min_commission == pytest.approx(0.99)


def test_script_rejects_invalid_date_window_before_data_resolution(monkeypatch):
    script = _load_script_module()

    def fail_if_resolved(root=None):
        raise AssertionError("invalid dates should be rejected before data root resolution")

    monkeypatch.setattr(script, "resolve_data_root", fail_if_resolved)

    with pytest.raises(ValueError, match="start.*boundary.*end"):
        script.main([
            "--data-root", "unused",
            "--start", "2022-04-01",
            "--boundary", "2022-04-01",
            "--end", "2022-07-08",
        ])


def test_script_rejects_invalid_validation_boundary_before_data_resolution(monkeypatch):
    script = _load_script_module()

    def fail_if_resolved(root=None):
        raise AssertionError("invalid validation dates should be rejected before data root resolution")

    monkeypatch.setattr(script, "resolve_data_root", fail_if_resolved)

    with pytest.raises(ValueError, match="validation-boundary"):
        script.main([
            "--data-root", "unused",
            "--start", "2022-01-03",
            "--validation-boundary", "2022-04-01",
            "--boundary", "2022-04-01",
            "--end", "2022-07-08",
        ])


def test_script_rejects_non_positive_parameter_before_data_resolution(monkeypatch):
    script = _load_script_module()

    def fail_if_resolved(root=None):
        raise AssertionError("invalid parameter should be rejected before data root resolution")

    monkeypatch.setattr(script, "resolve_data_root", fail_if_resolved)

    with pytest.raises(ValueError, match="--reflection-window.*positive"):
        script.main([
            "--data-root", "unused",
            "--start", "2022-01-03",
            "--boundary", "2022-04-01",
            "--end", "2022-07-08",
            "--reflection-window", "0",
        ])


def test_script_rejects_missing_expert_identity_before_trading():
    from argparse import Namespace

    script = _load_script_module()

    with pytest.raises(ValueError, match="--expert-backend"):
        script._require_expert_model_identity(Namespace(
            expert_backend="",
            expert_model="gpt-4.1-mini",
            expert_model_revision=None,
        ))


def test_codex_sdk_identity_defaults_to_codex_model():
    from argparse import Namespace

    script = _load_script_module()
    args = Namespace(
        expert_backend="codex_sdk",
        expert_model=None,
        expert_model_revision=None,
    )

    script._require_expert_model_identity(args)

    assert args.expert_model == "gpt-5.4-mini"
    assert args.expert_model_revision == "gpt-5.4-mini"


def test_protocol_universe_takes_precedence_over_auto_selection(tmp_path, monkeypatch):
    script = _load_script_module()

    protocol = {
        "data_scope": {"universe": ["AAPL", "AMD", "MSFT"]},
        "reserved_test_periods": [],
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    class Selection:
        tickers = ["WRONG"]
        label = "auto"

    monkeypatch.setattr(script, "_protocol_universe_has_price_data", lambda *args, **kwargs: True)
    monkeypatch.setattr(script, "select_fixed_universe", lambda *args, **kwargs: Selection())

    selection = script._resolve_universe_selection(
        "unused",
        start=script._parse_date("2010-01-04"),
        end=script._parse_date("2016-12-30"),
        names=10,
        protocol=script._read_protocol(protocol_path),
    )

    assert selection.tickers == ("AAPL", "AMD", "MSFT")
    assert selection.label == "protocol fixed universe: 3 names"


def test_protocol_universe_falls_back_when_not_available(monkeypatch):
    script = _load_script_module()

    class Selection:
        tickers = ["T1", "T2"]
        label = "auto"

    monkeypatch.setattr(script, "_protocol_universe_has_price_data", lambda *args, **kwargs: False)
    monkeypatch.setattr(script, "select_fixed_universe", lambda *args, **kwargs: Selection())

    selection = script._resolve_universe_selection(
        "unused",
        start=script._parse_date("2022-01-03"),
        end=script._parse_date("2022-04-08"),
        names=2,
        protocol={"data_scope": {"universe": ["AAPL", "AMD"]}},
    )

    assert selection.tickers == ["T1", "T2"]
    assert selection.label == "auto"


def test_compare_market_state_labels_to_timeline_csv_exact_match(tmp_path):
    script = _load_script_module()
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "date,state\n"
        "2016-01-11,up_stress\n"
        "2016-01-12,down_calm\n",
        encoding="utf-8",
    )

    result = script.compare_market_state_labels_to_timeline_csv(
        [(script._parse_date("2016-01-11"), "up_stress"), (script._parse_date("2016-01-12"), "down_calm")],
        csv_path,
        start=script._parse_date("2016-01-11"),
        end=script._parse_date("2016-01-12"),
    )

    assert result == {"matched": True, "n_checked": 2, "mismatches": []}


def test_compare_market_state_labels_to_timeline_csv_rejects_mismatch(tmp_path):
    script = _load_script_module()
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "date,state\n"
        "2016-01-11,up_stress\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="market-state label mismatch"):
        script.compare_market_state_labels_to_timeline_csv(
            [(script._parse_date("2016-01-11"), "down_stress")],
            csv_path,
            start=script._parse_date("2016-01-11"),
            end=script._parse_date("2016-01-11"),
        )


def test_state_power_marks_absent_states_underpowered():
    script = _load_script_module()
    loop_result = SimpleNamespace(
        episodes=[
            SimpleNamespace(market_state="up_stress", action="long"),
            SimpleNamespace(market_state="up_stress", action="flat"),
            SimpleNamespace(market_state="down_calm", action="long"),
        ]
    )

    power = script._state_power(loop_result, min_support=2)

    assert power["down_stress"]["episode_count"] == 0
    assert power["down_stress"]["absent"] is True
    assert power["down_stress"]["underpowered"] is True
    assert power["down_stress"]["exclude_from_headline_comparison"] is True


def test_negative_decision_thresholds_allowed_for_participation_bias():
    # A = sum w*v in [-1, 1], so a negative entry threshold is a valid
    # participation-biased rule (long unless strongly bearish). Passed with '='
    # so argparse does not read the leading '-' as a flag.
    from trading_agent.experiments.expert_weight_holdout_launcher import _parse_thresholds
    from trading_agent.experiments.expert_weight_holdout import _normalise_threshold_grid
    import pytest

    assert _parse_thresholds("-0.5,-0.25,0.0") == (-0.5, -0.25, 0.0)
    assert _normalise_threshold_grid((-0.5, -0.25, 0.0)) == (-0.5, -0.25, 0.0)
    with pytest.raises(Exception):
        _parse_thresholds("-1.5")          # below -1 rejected


def test_runner_accepts_negative_decision_threshold():
    # runner.run_expert_loop is the third and final threshold guard (after the
    # launcher parser and the holdout grid). It must also admit the negative
    # participation-biased range so a full replay does not crash mid-run.
    import inspect
    from trading_agent import runner
    src = inspect.getsource(runner.run_expert_loop)
    assert "-1.0 <= float(decision_threshold) <= 1.0" in src
    assert "must be in [0, 1]" not in src


def test_single_name_run_without_market_proxy_is_refused(
    tmp_path, regime_dataset, mem_backtest_config, monkeypatch
):
    # Market-state leakage rule, ENFORCED 2026-07-18: a single-name run's
    # equal-weight universe IS the traded stock, so omitting --market-proxy
    # would silently derive the market_state label from the stock itself.
    # Hard error, not a manifest footnote.
    script = _load_script_module()
    monkeypatch.setattr(script, "load_panel", lambda *a, **k: regime_dataset)
    monkeypatch.setattr(script, "resolve_data_root", lambda root=None: Path(root or "unused"))
    monkeypatch.setattr(script, "_protocol_universe_has_price_data", lambda *a, **k: True)

    def fake_build_sources(**kwargs):
        from trading_agent.expert_context import ExpertSignalSources

        return ExpertSignalSources()

    monkeypatch.setattr(script, "build_expert_signal_sources", fake_build_sources)

    with pytest.raises(SystemExit, match="market-proxy is REQUIRED"):
        script.main([
            "--data-root", "unused",
            "--start", mem_backtest_config.date_from.isoformat(),
            "--validation-boundary", "2022-04-08",
            "--boundary", "2022-05-06",
            "--end", mem_backtest_config.date_to.isoformat(),
            "--tickers", "T1",
            "--embargo-days", "2",
            "--vol-window", "5",
            "--percentile-window", "20",
            "--trend-window", "10",
            "--trend-confirm-days", "2",
            "--output", str(tmp_path / "refused"),
        ])
