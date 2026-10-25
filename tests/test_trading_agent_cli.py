"""CLI contract tests for the three-mode trading-agent launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


class _FakePaperIB:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


def _write_tiny_finsaber_root(root: Path, *, n_days: int = 70) -> tuple[date, date]:
    price_dir = root / "price_daily" / "year=2022"
    price_dir.mkdir(parents=True)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for i, ts in enumerate(dates):
        for j, symbol in enumerate(("T1", "T2", "T3", "T4")):
            close = 100.0 + i * (1.0 + j * 0.1) + j * 5.0
            rows.append({
                "date": ts.date(),
                "symbol": symbol,
                "cik": str(1000 + j),
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.7,
                "close": close,
                "adjusted_close": close,
                "volume": 1_000_000 + j * 100_000,
            })
    pd.DataFrame(rows).to_parquet(price_dir / "part-000.parquet", index=False)
    return dates[0].date(), dates[-1].date()


def _assert_launcher_report_schema(payload: dict, *, mode: str, launcher: str) -> None:
    assert payload["mode"] == mode
    assert payload["system"] == "four_expert_reflection"
    assert payload["schema_version"] == "trading-agent-report/v1"
    assert payload["manifest"]["mode"] == mode
    assert payload["manifest"]["system"] == "four_expert_reflection"
    assert payload["manifest"]["schema_version"] == "trading-agent-report/v1"
    assert payload["manifest"]["launcher"] == launcher


def test_cli_parser_exposes_three_modes():
    from trading_agent.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()

    assert "backtest" in help_text
    assert "holdout" in help_text
    assert "paper" in help_text
    assert "backtest: run historical committee backtests" in help_text
    assert "holdout: train memory, freeze it, and test it on a later window" in help_text
    assert "paper: connect to IB Gateway/TWS paper trading" in help_text
    assert "Example:" in help_text
    assert "trading-agent holdout" in help_text


def test_package_module_entrypoint_shows_cli_help():
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [sys.executable, "-m", "trading_agent", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    assert "{backtest,holdout,paper}" in completed.stdout


def test_package_module_entrypoint_shows_version():
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [sys.executable, "-m", "trading_agent", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "trading-agent 0.1.0"


def test_script_entrypoint_shows_cli_help_without_install():
    script = Path(__file__).resolve().parents[1] / "scripts" / "trading_agent_cli.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert completed.returncode == 0
    assert "{backtest,holdout,paper}" in completed.stdout


def test_installed_console_script_shows_cli_help():
    if shutil.which("trading-agent") is None:
        pytest.skip("trading-agent console script is not installed")

    completed = subprocess.run(
        ["trading-agent", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "{backtest,holdout,paper}" in completed.stdout
    assert completed.stderr == ""


def test_installed_console_script_shows_version():
    if shutil.which("trading-agent") is None:
        pytest.skip("trading-agent console script is not installed")

    completed = subprocess.run(
        ["trading-agent", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "trading-agent 0.1.0"
    assert completed.stderr == ""


def test_installed_console_script_holdout_writes_auditable_reports(tmp_path):
    if shutil.which("trading-agent") is None:
        pytest.skip("trading-agent console script is not installed")

    data_root = tmp_path / "finsaber"
    _write_tiny_finsaber_root(data_root)
    output = tmp_path / "holdout_report"
    memory_output = tmp_path / "holdout_memory.json"

    completed = subprocess.run(
        [
            "trading-agent", "holdout",
            "--data-root", str(data_root),
            "--start", "2022-01-03",
            "--validation-boundary", "2022-01-31",
            "--boundary", "2022-02-28",
            "--end", "2022-04-08",
            "--names", "4",
            "--embargo-days", "1",
            "--reflect-every", "5",
            "--reflection-window", "10",
            "--reflection-min-support", "1",
            "--vol-window", "5",
            "--percentile-window", "10",
            "--output", str(output),
            "--memory-output", str(memory_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    memory_payload = json.loads(memory_output.read_text(encoding="utf-8"))
    _assert_launcher_report_schema(payload, mode="holdout", launcher="trading-agent holdout")
    assert memory_payload == payload["memory_snapshot"]
    assert set(payload["arms"]) == {"equal_weight", "global_memory", "market_state_memory", "market_state_memory_online"}


def test_package_module_holdout_entrypoint_writes_auditable_reports(tmp_path):
    data_root = tmp_path / "finsaber"
    _write_tiny_finsaber_root(data_root)
    output = tmp_path / "holdout_report"
    memory_output = tmp_path / "holdout_memory.json"
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [
            sys.executable, "-m", "trading_agent", "holdout",
            "--data-root", str(data_root),
            "--start", "2022-01-03",
            "--validation-boundary", "2022-01-31",
            "--boundary", "2022-02-28",
            "--end", "2022-04-08",
            "--names", "4",
            "--embargo-days", "1",
            "--reflect-every", "5",
            "--reflection-window", "10",
            "--reflection-min-support", "1",
            "--vol-window", "5",
            "--percentile-window", "10",
            "--output", str(output),
            "--memory-output", str(memory_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    memory_payload = json.loads(memory_output.read_text(encoding="utf-8"))
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    _assert_launcher_report_schema(payload, mode="holdout", launcher="trading-agent holdout")
    assert memory_payload == payload["memory_snapshot"]
    assert payload["split_protocol"]["train_end"] == "2022-01-31"
    assert set(payload["arms"]) == {"equal_weight", "global_memory", "market_state_memory", "market_state_memory_online"}
    assert set(payload["validation_arms"]) == {"equal_weight", "global_memory", "market_state_memory"}
    assert "WROTE" in completed.stdout
    assert "trading-agent holdout" in markdown
    assert "Split Protocol" in markdown
    assert "Validation Arms" in markdown


def test_package_module_backtest_entrypoint_writes_launcher_manifest(tmp_path):
    data_root = tmp_path / "finsaber"
    _write_tiny_finsaber_root(data_root)
    output = tmp_path / "backtest.json"
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [
            sys.executable, "-m", "trading_agent", "backtest",
            "--data-root", str(data_root),
            "--start", "2022-01-03",
            "--end", "2022-03-18",
            "--names", "4",
            "--vol-window", "5",
            "--percentile-window", "10",
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    _assert_launcher_report_schema(payload, mode="backtest", launcher="trading-agent backtest")
    assert payload["manifest"]["tickers"] == ["T1", "T2", "T3", "T4"]
    assert "WROTE" in completed.stdout


def test_package_module_paper_dry_run_entrypoint_writes_launcher_manifest(tmp_path):
    output = tmp_path / "paper.json"
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [
            sys.executable, "-m", "trading_agent", "paper",
            "--symbols", "AAPL,MSFT",
            "--order", "AAPL:buy:2",
            "--order", "MSFT:sell:1",
            "--dry-run",
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    _assert_launcher_report_schema(payload, mode="paper", launcher="trading-agent paper")
    assert payload["manifest"]["run_intent"] == "dry_run"
    assert payload["order_preview_count"] == 2
    assert "WROTE" in completed.stdout


def test_package_module_holdout_memory_output_feeds_paper_dry_run(tmp_path):
    data_root = tmp_path / "finsaber"
    _write_tiny_finsaber_root(data_root)
    holdout_output = tmp_path / "holdout_report"
    memory_output = tmp_path / "holdout_memory.json"
    paper_output = tmp_path / "paper_from_memory.json"
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    holdout = subprocess.run(
        [
            sys.executable, "-m", "trading_agent", "holdout",
            "--data-root", str(data_root),
            "--start", "2022-01-03",
            "--validation-boundary", "2022-01-31",
            "--boundary", "2022-02-28",
            "--end", "2022-04-08",
            "--names", "4",
            "--embargo-days", "1",
            "--reflect-every", "5",
            "--reflection-window", "10",
            "--reflection-min-support", "1",
            "--vol-window", "5",
            "--percentile-window", "10",
            "--output", str(holdout_output),
            "--memory-output", str(memory_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )
    assert holdout.returncode == 0, holdout.stderr

    paper = subprocess.run(
        [
            sys.executable, "-m", "trading_agent", "paper",
            "--symbols", "AAPL,MSFT",
            "--order", "AAPL:buy:2",
            "--dry-run",
            "--memory-snapshot", str(memory_output),
            "--output", str(paper_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )
    assert paper.returncode == 0, paper.stderr

    memory_payload = json.loads(memory_output.read_text(encoding="utf-8"))
    paper_payload = json.loads(paper_output.read_text(encoding="utf-8"))
    _assert_launcher_report_schema(paper_payload, mode="paper", launcher="trading-agent paper")
    assert paper_payload["memory_snapshot"]["path"] == str(memory_output)
    assert paper_payload["memory_snapshot"]["status"] == "frozen"
    assert paper_payload["memory_snapshot"]["status"] == "frozen"
    assert paper_payload["memory_snapshot"]["entries_loaded"] == len(memory_payload["cards"])
    assert paper_payload["order_preview_count"] == 1


def test_pyproject_declares_trading_agent_console_script():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "[project.scripts]" in text
    assert 'trading-agent = "trading_agent.cli:main"' in text


def test_pyproject_declares_explicit_build_backend():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "[build-system]" in text
    assert '"setuptools>=68"' in text
    assert 'build-backend = "setuptools.build_meta"' in text


def test_pyproject_declares_documented_install_extras():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "dependencies = [" in text
    for dependency in ('"pandas>=2.2,<3"', '"numpy>=1.23,<2"', '"requests>=2.31"', '"pyarrow>=18.0,<26"'):
        assert dependency in text
    assert "[project.optional-dependencies]" in text
    assert "dev = [" in text
    assert '"pytest"' in text
    assert "research = [" in text
    assert "paper = [" in text
    assert '"ib_async>=2.1,<3"' in text
    assert "llm = [" in text
    assert '"openai>=1.0"' in text


def test_paper_help_describes_execute_orders_as_live_paper_submission():
    from trading_agent.cli import build_parser

    parser = build_parser()
    paper_parser = parser._subparsers._group_actions[0].choices["paper"]
    help_text = paper_parser.format_help()
    normalized_help = " ".join(help_text.split())

    assert "Reserved safety switch" not in help_text
    assert "Submit market orders to the connected IB paper account" in normalized_help
    assert "--allow-non-paper-port" in help_text
    assert "--connection-smoke" in help_text
    assert "4002" in help_text
    assert "7497" in help_text


def test_research_modes_expose_expert_backend_selection():
    from trading_agent.cli import build_parser

    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices

    for mode in ("backtest", "holdout"):
        help_text = choices[mode].format_help()
        assert "--expert-backend" in help_text
        assert "--expert-model" in help_text
        assert "--expert-model-revision" in help_text
        assert "static" in help_text
        assert "heuristic" in help_text
        assert "openai" in help_text
        assert "codex_sdk" in help_text


def test_cli_reuses_paper_adapter_port_constant():
    from argparse import Namespace

    from trading_agent import cli
    from trading_agent.paper_trading import COMMON_IB_PAPER_PORTS

    assert not hasattr(cli, "IB_PAPER_PORTS")
    assert cli._uses_non_paper_port(Namespace(port=COMMON_IB_PAPER_PORTS[0])) is False
    assert cli._uses_non_paper_port(Namespace(port=4001)) is True


def test_backtest_mode_writes_json_report(tmp_path, demo_dataset, monkeypatch):
    from trading_agent import cli

    class Selection:
        tickers = ["AAA", "BBB"]
        label = "demo"

    monkeypatch.setattr(cli, "select_fixed_universe", lambda *args, **kwargs: Selection())
    monkeypatch.setattr(cli, "load_panel", lambda *args, **kwargs: demo_dataset)
    monkeypatch.setattr(cli, "resolve_data_root", lambda root=None: "demo-root")

    output = tmp_path / "backtest.json"
    rc = cli.main([
        "backtest",
        "--data-root", "demo-root",
        "--start", "2024-01-02",
        "--end", "2024-01-08",
        "--names", "2",
        "--vol-window", "2",
        "--percentile-window", "3",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "backtest"
    assert payload["system"] == "four_expert_reflection"
    assert payload["universe"]["tickers"] == ["AAA", "BBB"]
    assert payload["n_dates"] == 5
    assert "participation_rate" in payload
    assert "n_applied" in payload
    assert "market_state_coverage" in payload
    assert "expert_error_by_state" in payload
    assert "per_state_metrics" in payload
    assert isinstance(payload["market_state_coverage"], dict)
    assert isinstance(payload["expert_error_by_state"], dict)
    assert isinstance(payload["per_state_metrics"], dict)
    for state_metrics in payload["per_state_metrics"].values():
        assert set(state_metrics) >= {
            "episode_count",
            "mean_market_return",
            "mean_signal_component",
            "participation_rate",
        }
    assert payload["reflection_trial_audit"]["round_count"] == payload["n_reflection_rounds"]
    assert payload["reflection_trial_audit"]["applied_count"] == payload["n_applied"]
    assert isinstance(payload["reflection_trial_audit"]["rounds"], list)
    assert payload["research_claims"] == {
        "run_role": "exploratory_backtest",
        "evidence_scope": "in_sample_only",
        "final_evidence_mode": "holdout",
        "reflection_learning": {
            "allow_reflection": False,
            "n_reflection_rounds": payload["n_reflection_rounds"],
            "n_learned": payload["n_learned"],
        },
    }
    manifest = payload["manifest"]
    assert manifest["data_root"] == "demo-root"
    assert manifest["universe_label"] == "demo"
    assert manifest["tickers"] == ["AAA", "BBB"]
    assert manifest["start"] == "2024-01-02"
    assert manifest["end"] == "2024-01-08"
    assert manifest["run_id"] == "cli-backtest"
    assert manifest["seed"] == 11
    assert manifest["reflect_every"] == 21
    assert manifest["reflection_window"] == 63
    assert manifest["reflection_min_support"] == 5
    assert manifest["market_state"] == {
        "vol_window": 2,
        "percentile_window": 3,
    }
    # The bps model replaces per-share commission in the adapter, so the
    # manifest records the effective (zeroed) commission parameters.
    assert manifest["execution"] == {
        "initial_cash": 100_000.0,
        "commission_per_share": 0.0,
        "min_commission": 0.0,
        "max_commission_rate": 0.01,
        "slippage_perc": 0.0005,
        "execution_cost_bps_per_side": 10.0,
        "liquidity_cap_pct": 0.0,
    }
    assert manifest["cost_protocol"] == {
        "model": "bps_per_side_on_traded_notional",
        "bps_replaces_per_share_commission": True,
        "headline_bps_per_side": 10.0,
        "sensitivity_bps_per_side": 25.0,
        "arm_identical": True,
    }
    assert manifest["preload"]["calendar_days"] == 900
    assert manifest["expert_backend"] == "static"
    assert manifest["alpha158_factors_db"] is None
    assert manifest["qlib_provider_uri"] is None
    assert manifest["news_enabled"] is True
    assert manifest["strict_news"] is False
    assert manifest["allow_reflection"] is False
    assert manifest["expert_backend"] == "static"
    assert manifest["expert_model"] == "gpt-4.1-mini"
    assert manifest["expert_model_revision"] == "static-deterministic"


def test_backtest_mode_rejects_invalid_date_window_before_loading(monkeypatch):
    from trading_agent import cli

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("invalid backtest dates should be rejected before data loading")

    monkeypatch.setattr(cli, "select_fixed_universe", fail_if_loaded)

    with pytest.raises(ValueError, match="start.*before.*end"):
        cli.main([
            "backtest",
            "--data-root", "demo-root",
            "--start", "2024-01-08",
            "--end", "2024-01-02",
        ])


def test_backtest_mode_rejects_non_positive_experiment_parameter_before_loading(monkeypatch):
    from trading_agent import cli

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("invalid backtest parameter should be rejected before data loading")

    monkeypatch.setattr(cli, "select_fixed_universe", fail_if_loaded)

    with pytest.raises(ValueError, match="--names.*positive"):
        cli.main([
            "backtest",
            "--data-root", "demo-root",
            "--start", "2024-01-02",
            "--end", "2024-01-08",
            "--names", "0",
        ])


def test_holdout_mode_delegates_to_existing_driver(tmp_path, monkeypatch):
    from trading_agent import cli

    captured = {}

    def fake_holdout_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli, "run_holdout_script", fake_holdout_main)
    output = tmp_path / "holdout"
    memory_output = tmp_path / "memory.json"

    rc = cli.main([
        "holdout",
        "--data-root", "demo-root",
        "--start", "2022-01-03",
        "--validation-boundary", "2022-03-01",
        "--boundary", "2022-04-01",
        "--end", "2022-07-08",
        "--commission-per-share", "0.02",
        "--min-commission", "1.25",
        "--max-commission-rate", "0.03",
        "--slippage-perc", "0.001",
        "--liquidity-cap-pct", "0.15",
        "--output", str(output),
        "--memory-output", str(memory_output),
    ])

    assert rc == 0
    assert captured["argv"][:2] == ["--data-root", "demo-root"]
    assert "--validation-boundary" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--validation-boundary") + 1] == "2022-03-01"
    assert "--boundary" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--commission-per-share") + 1] == "0.02"
    assert captured["argv"][captured["argv"].index("--min-commission") + 1] == "1.25"
    assert captured["argv"][captured["argv"].index("--max-commission-rate") + 1] == "0.03"
    assert captured["argv"][captured["argv"].index("--slippage-perc") + 1] == "0.001"
    assert captured["argv"][captured["argv"].index("--liquidity-cap-pct") + 1] == "0.15"
    assert "--expert-model-revision" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--expert-model-revision") + 1] == "static-deterministic"
    assert str(output) in captured["argv"]
    assert "--memory-output" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--memory-output") + 1] == str(memory_output)


def test_holdout_mode_rejects_invalid_date_window_before_driver(monkeypatch):
    from trading_agent import cli

    def fail_if_called(argv):
        raise AssertionError("invalid holdout dates should be rejected before driver delegation")

    monkeypatch.setattr(cli, "run_holdout_script", fail_if_called)

    with pytest.raises(ValueError, match="start.*boundary.*end"):
        cli.main([
            "holdout",
            "--data-root", "demo-root",
            "--start", "2022-04-01",
            "--boundary", "2022-04-01",
            "--end", "2022-07-08",
        ])


def test_holdout_mode_rejects_non_positive_experiment_parameter_before_driver(monkeypatch):
    from trading_agent import cli

    def fail_if_called(argv):
        raise AssertionError("invalid holdout parameter should be rejected before driver delegation")

    monkeypatch.setattr(cli, "run_holdout_script", fail_if_called)

    with pytest.raises(ValueError, match="--reflection-window.*positive"):
        cli.main([
            "holdout",
            "--data-root", "demo-root",
            "--start", "2022-01-03",
            "--boundary", "2022-04-01",
            "--end", "2022-07-08",
            "--reflection-window", "0",
        ])


def test_paper_mode_dry_run_writes_manifest(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL,MSFT",
        "--host", "127.0.0.1",
        "--port", "4002",
        "--client-id", "17",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "paper"
    assert payload["dry_run"] is True
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 4002
    assert payload["client_id"] == 17
    assert payload["symbols"] == ["AAPL", "MSFT"]
    assert payload["execute_orders"] is False
    assert payload["gateway_connected"] is False
    assert payload["orders_submitted"] == 0
    assert payload["order_previews"] == []
    assert payload["order_source"] == "none"
    assert payload["manifest"] == {
        "mode": "paper",
        "system": "four_expert_reflection",
        "schema_version": "trading-agent-report/v1",
        "launcher": "trading-agent paper",
        "host": "127.0.0.1",
        "port": 4002,
        "client_id": 17,
        "symbols": ["AAPL", "MSFT"],
        "dry_run": True,
        "execute_orders": False,
        "memory_required": False,
        "gateway_connected": False,
        "connection_smoke": False,
        "run_intent": "dry_run",
        "common_paper_port": True,
        "allowed_paper_ports": [4002, 7497],
        "non_paper_port_override": False,
        "order_preview_count": 0,
            "orders_submitted": 0,
            "orders_pending_submission": 0,
            "submission_coverage": 0.0,
            "ib_connector_source": "data_collection.ib_connector",
            "external_validation_steps": [
                "connection_smoke",
                "memory_dry_run",
                "paper_order_submission",
            ],
            "ib_paper_validation": {
                "requires_external_gateway": True,
                "status": "dry_run_only",
                "connection_verified": False,
                "actual_order_submission": False,
            },
        }
    assert "no orders submitted" in payload["safety_note"]


def test_paper_mode_rejects_empty_symbol_list(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_empty_symbols.json"
    with pytest.raises(ValueError, match="at least one symbol"):
        cli.main([
            "paper",
            "--symbols", ",,,",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_rejects_dry_run_execute_orders_conflict(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_conflict.json"
    with pytest.raises(ValueError, match="--dry-run.*--execute-orders"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--order", "AAPL:buy:1",
            "--dry-run",
            "--execute-orders",
            "--output", str(output),
        ])


def test_paper_mode_execute_orders_requires_order_source_before_connect(tmp_path, monkeypatch):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("execute without order source should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_execute_no_source.json"
    with pytest.raises(ValueError, match="--execute-orders.*--order.*--auto-decide"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--execute-orders",
            "--output", str(output),
        ])


def test_paper_mode_rejects_non_positive_client_id_before_connect(tmp_path, monkeypatch):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("invalid paper client id should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_bad_client_id.json"
    with pytest.raises(ValueError, match="--client-id.*positive"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--order", "AAPL:buy:1",
            "--client-id", "0",
            "--execute-orders",
            "--output", str(output),
        ])


def test_paper_mode_rejects_empty_host_before_connect(tmp_path, monkeypatch):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("invalid paper host should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_bad_host.json"
    with pytest.raises(ValueError, match="--host.*non-empty"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--host", " ",
            "--output", str(output),
        ])


@pytest.mark.parametrize("bad_port", ["0", "70000"])
def test_paper_mode_rejects_invalid_port_before_connect(
    bad_port,
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("invalid paper port should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / f"paper_bad_port_{bad_port}.json"
    with pytest.raises(ValueError, match="--port.*1.*65535"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--port", bad_port,
            "--output", str(output),
        ])


def test_paper_mode_execute_orders_uses_submission_helper(tmp_path, monkeypatch):
    from trading_agent import cli

    class FakeIB:
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    fake_ib = FakeIB()

    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)

    def fail_if_bars_requested(*args, **kwargs):
        raise AssertionError("manual connected paper orders should not request daily bars")

    monkeypatch.setattr(cli, "request_daily_bars", fail_if_bars_requested)

    def fake_submit(ib, previews):
        assert ib is fake_ib
        assert previews[0]["ib_symbol"] == "AAPL"
        return [{"symbol": "AAPL", "ib_symbol": "AAPL", "action": "BUY", "quantity": 2, "order_id": 99, "status": "Submitted"}]

    monkeypatch.setattr(cli, "submit_order_previews", fake_submit)

    output = tmp_path / "paper_execute.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--order", "AAPL:buy:2",
        "--execute-orders",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dry_run"] is False
    assert payload["execute_orders"] is True
    assert payload["gateway_connected"] is True
    assert payload["orders_submitted"] == 1
    assert payload["order_source"] == "manual"
    assert payload["submission_reports"][0]["order_id"] == 99
    assert fake_ib.disconnected is True


def test_paper_mode_connected_smoke_checks_account_without_submitting_orders(
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    class FakeIB:
        disconnected = False

        def accountSummary(self):
            return [object(), object()]

        def disconnect(self):
            self.disconnected = True

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)

    def fail_if_bars_requested(*args, **kwargs):
        raise AssertionError("connected paper smoke should not request daily bars")

    def fail_if_submitted(*args, **kwargs):
        raise AssertionError("connected paper smoke should not submit orders")

    monkeypatch.setattr(cli, "request_daily_bars", fail_if_bars_requested)
    monkeypatch.setattr(cli, "submit_order_previews", fail_if_submitted)

    output = tmp_path / "paper_connected_smoke.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--output", str(output),
    ])

    assert rc == 0
    assert fake_ib.disconnected is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dry_run"] is False
    assert payload["execute_orders"] is False
    assert payload["gateway_connected"] is True
    assert payload["orders_submitted"] == 0
    assert payload["submission_reports"] == []
    assert payload["order_source"] == "none"
    assert payload["gateway_smoke"] == {
        "account_summary_checked": True,
        "account_summary_rows": 2,
    }
    assert payload["connection_smoke"] is False
    assert "no orders submitted" in payload["safety_note"]


def test_paper_mode_explicit_connection_smoke_is_account_only(tmp_path, monkeypatch):
    from trading_agent import cli

    class FakeIB:
        disconnected = False

        def accountSummary(self):
            return [object(), object(), object()]

        def disconnect(self):
            self.disconnected = True

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)

    def fail_if_bars_requested(*args, **kwargs):
        raise AssertionError("explicit connection smoke should not request daily bars")

    def fail_if_submitted(*args, **kwargs):
        raise AssertionError("explicit connection smoke should not submit orders")

    monkeypatch.setattr(cli, "request_daily_bars", fail_if_bars_requested)
    monkeypatch.setattr(cli, "submit_order_previews", fail_if_submitted)

    output = tmp_path / "paper_explicit_smoke.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--connection-smoke",
        "--output", str(output),
    ])

    assert rc == 0
    assert fake_ib.disconnected is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["connection_smoke"] is True
    assert payload["smoke_status"] == "passed"
    assert payload["manifest"]["connection_smoke"] is True
    assert payload["manifest"]["run_intent"] == "connection_smoke"
    assert payload["gateway_connected"] is True
    assert payload["orders_submitted"] == 0
    assert payload["order_previews"] == []
    assert payload["order_source"] == "none"
    assert payload["gateway_smoke"] == {
        "account_summary_checked": True,
        "account_summary_rows": 3,
    }


def test_paper_mode_connection_smoke_does_not_require_symbols(tmp_path, monkeypatch, capsys):
    from trading_agent import cli

    class FakeIB:
        disconnected = False

        def accountSummary(self):
            return [object()]

        def disconnect(self):
            self.disconnected = True

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)

    def fail_if_bars_requested(*args, **kwargs):
        raise AssertionError("connection smoke without symbols should not request daily bars")

    monkeypatch.setattr(cli, "request_daily_bars", fail_if_bars_requested)

    output = tmp_path / "paper_smoke_no_symbols.json"
    rc = cli.main([
        "paper",
        "--connection-smoke",
        "--output", str(output),
    ])

    assert rc == 0
    assert fake_ib.disconnected is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["symbols"] == []
    assert payload["connection_smoke"] is True
    assert payload["smoke_status"] == "passed"
    assert payload["manifest"]["connection_smoke"] is True
    assert payload["manifest"]["run_intent"] == "connection_smoke"
    assert payload["gateway_smoke"]["account_summary_rows"] == 1
    stdout = capsys.readouterr().out
    assert "CONNECTION SMOKE PASSED" in stdout
    assert "account_summary_rows=1" in stdout


def test_paper_mode_connection_smoke_connection_failure_is_actionable(tmp_path, monkeypatch):
    from trading_agent import cli

    def fail_connect(settings):
        raise OSError("connection refused")

    monkeypatch.setattr(cli, "connect_ib", fail_connect)

    output = tmp_path / "paper_smoke_failure.json"
    with pytest.raises(RuntimeError, match="IB Gateway paper connection smoke failed.*127.0.0.1:4002"):
        cli.main([
            "paper",
            "--connection-smoke",
            "--output", str(output),
        ])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "paper"
    assert payload["connection_smoke"] is True
    assert payload["smoke_status"] == "failed"
    assert payload["gateway_connected"] is False
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 4002
    assert payload["manifest"] == {
        "mode": "paper",
        "system": "four_expert_reflection",
        "schema_version": "trading-agent-report/v1",
        "launcher": "trading-agent paper",
        "host": "127.0.0.1",
        "port": 4002,
        "client_id": 1,
        "symbols": [],
        "dry_run": False,
        "execute_orders": False,
        "memory_required": False,
        "gateway_connected": False,
        "connection_smoke": True,
        "run_intent": "connection_smoke",
        "common_paper_port": True,
        "allowed_paper_ports": [4002, 7497],
        "non_paper_port_override": False,
        "order_preview_count": 0,
            "orders_submitted": 0,
            "orders_pending_submission": 0,
            "submission_coverage": 0.0,
            "ib_connector_source": "data_collection.ib_connector",
            "external_validation_steps": [
                "connection_smoke",
                "memory_dry_run",
                "paper_order_submission",
            ],
            "ib_paper_validation": {
                "requires_external_gateway": True,
                "status": "connection_smoke_failed",
                "connection_verified": False,
                "actual_order_submission": False,
            },
        }
    assert "connection refused" in payload["error"]


def test_paper_mode_requires_symbols_outside_connection_smoke(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_no_symbols.json"
    with pytest.raises(ValueError, match="at least one symbol"):
        cli.main([
            "paper",
            "--dry-run",
            "--output", str(output),
        ])


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--dry-run"],
        ["--execute-orders"],
        ["--auto-decide"],
        ["--order", "AAPL:buy:1"],
    ],
)
def test_paper_mode_connection_smoke_rejects_non_smoke_options(
    bad_args,
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("invalid connection smoke args should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_bad_explicit_smoke.json"
    with pytest.raises(ValueError, match="--connection-smoke"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--connection-smoke",
            "--output", str(output),
            *bad_args,
        ])


def test_paper_mode_execute_orders_rejects_unresolved_all_before_connect(
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("unresolved ALL quantity should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_all_execute.json"
    with pytest.raises(ValueError, match="ALL quantity"):
        cli.main([
            "paper",
            "--symbols", "MSFT",
            "--order", "MSFT:sell:ALL",
            "--execute-orders",
            "--output", str(output),
        ])


def test_paper_mode_execute_orders_rejects_common_live_ports(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_live_port.json"
    with pytest.raises(ValueError, match="non-paper IB port"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--order", "AAPL:buy:2",
            "--port", "4001",
            "--execute-orders",
            "--output", str(output),
        ])


def test_paper_mode_execute_orders_can_explicitly_allow_non_paper_port(
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    class FakeIB:
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(
        cli,
        "request_daily_bars",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manual connected paper orders should not request daily bars")
        ),
    )

    def fake_submit(ib, previews):
        assert ib is fake_ib
        return [{
            "symbol": "AAPL",
            "ib_symbol": "AAPL",
            "action": "BUY",
            "quantity": 2,
            "order_id": 101,
            "status": "Submitted",
        }]

    monkeypatch.setattr(cli, "submit_order_previews", fake_submit)

    output = tmp_path / "paper_live_port_allowed.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--order", "AAPL:buy:2",
        "--port", "4001",
        "--allow-non-paper-port",
        "--execute-orders",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execute_orders"] is True
    assert payload["port"] == 4001
    assert payload["non_paper_port_override"] is True
    assert payload["manifest"]["common_paper_port"] is False
    assert payload["manifest"]["non_paper_port_override"] is True
    assert payload["manifest"]["run_intent"] == "paper_order_submission"
    assert payload["submission_reports"][0]["order_id"] == 101
    assert fake_ib.disconnected is True


def test_paper_auto_decide_rejects_non_positive_cash_before_connect(tmp_path, monkeypatch):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("invalid paper cash should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_bad_cash.json"
    with pytest.raises(ValueError, match="paper cash"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--auto-decide",
            "--paper-cash", "0",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_auto_decide_rejects_non_positive_classifier_window_before_connect(
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    def fail_if_connected(settings):
        raise AssertionError("invalid classifier windows should be rejected before IB connect")

    monkeypatch.setattr(cli, "connect_ib", fail_if_connected)

    output = tmp_path / "paper_bad_window.json"
    with pytest.raises(ValueError, match="window"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--auto-decide",
            "--paper-vol-window", "0",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_can_use_ib_account_cash_for_auto_decide(tmp_path, monkeypatch):
    from trading_agent import cli

    class AccountRow:
        def __init__(self, tag, value, currency="USD"):
            self.tag = tag
            self.value = value
            self.currency = currency

    class FakeIB:
        def accountSummary(self):
            return [AccountRow("AvailableFunds", "2000", "USD")]

        def disconnect(self):
            pass

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    output = tmp_path / "paper_account_cash.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-market-state", "up_stress",
        "--use-account-cash",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cash_basis"]["source"] == "ib_account_available_funds"
    assert payload["cash_basis"]["amount"] == 2000.0
    assert payload["order_previews"][0]["quantity"] == 20


def test_paper_mode_dry_run_can_preview_with_ib_account_cash(tmp_path, monkeypatch):
    from trading_agent import cli

    class AccountRow:
        def __init__(self, tag, value, currency="USD"):
            self.tag = tag
            self.value = value
            self.currency = currency

    class FakeIB:
        def __init__(self):
            self.disconnected = False

        def accountSummary(self):
            return [AccountRow("AvailableFunds", "1500", "USD")]

        def disconnect(self):
            self.disconnected = True

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    output = tmp_path / "paper_dry_account_cash.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-market-state", "up_stress",
        "--use-account-cash",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["gateway_connected"] is True
    assert payload["cash_basis"]["source"] == "ib_account_available_funds"
    assert payload["cash_basis"]["amount"] == 1500.0
    assert payload["order_previews"][0]["quantity"] == 15
    assert fake_ib.disconnected is True


def test_paper_mode_caps_auto_decide_order_value(tmp_path, monkeypatch):
    from trading_agent import cli

    class AccountRow:
        def __init__(self, tag, value, currency="USD"):
            self.tag = tag
            self.value = value
            self.currency = currency

    class FakeIB:
        def accountSummary(self):
            return [AccountRow("AvailableFunds", "100000", "USD")]

        def disconnect(self):
            pass

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = FakeIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    output = tmp_path / "paper_capped.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-market-state", "up_stress",
        "--use-account-cash",
        "--max-order-value", "1000",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["risk_limits"]["max_order_value"] == 1000.0
    assert payload["order_previews"][0]["quantity"] == 10


def test_paper_mode_auto_decide_generates_order_preview_from_latest_bars(tmp_path, monkeypatch):
    from trading_agent import cli

    class FakeFrame:
        empty = False

        def __init__(self, close):
            self._close = close

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return self
            if key == "close":
                return self
            raise KeyError(key)

        @property
        def iloc(self):
            return self

        def __getitem_iloc__(self, index):
            return "2026-06-25"

        def __getattr__(self, name):
            if name == "iloc":
                return self
            raise AttributeError(name)

        def __iter__(self):
            return iter([])

        def __repr__(self):
            return f"FakeFrame({self._close})"

        def __call__(self, index):
            return self._close

    class FakeSeries:
        def __init__(self, value):
            self.iloc = [value]

    class BetterFrame:
        empty = False

        def __init__(self, close):
            self.close = close

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return FakeSeries("2026-06-25")
            if key == "close":
                return FakeSeries(self.close)
            raise KeyError(key)

    class FakeIB:
        def __init__(self):
            self.disconnected = False

        def disconnect(self):
            self.disconnected = True

    fake_ib = FakeIB()
    requested_with = []
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)

    def fake_request_daily_bars(ib, symbol, **kwargs):
        requested_with.append(ib)
        return BetterFrame(100.0 if symbol == "AAPL" else 50.0)

    monkeypatch.setattr(cli, "request_daily_bars", fake_request_daily_bars)

    output = tmp_path / "paper_auto.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL,MSFT",
        "--auto-decide",
        "--paper-market-state", "up_stress",
        "--paper-cash", "1000",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    assert requested_with == [fake_ib, fake_ib]
    assert fake_ib.disconnected is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    # No --committee-read here, so the views did come from the static backend and the
    # manifest says so. With one, the label names the file instead; the hardcoded
    # "four_expert_static" it used to carry described neither.
    assert payload["decision_source"] == "three_expert_static"
    assert payload["order_source"] == "auto_decide"
    assert payload["orders_submitted"] == 0
    assert payload["order_previews"] == [
        {
            "action": "BUY",
            "ib_symbol": "AAPL",
            "order_type": "MKT",
            "quantity": 5,
            "symbol": "AAPL",
            "time_in_force": "DAY",
            "transmit": True,
        },
        {
            "action": "BUY",
            "ib_symbol": "MSFT",
            "order_type": "MKT",
            "quantity": 10,
            "symbol": "MSFT",
            "time_in_force": "DAY",
            "transmit": True,
        },
    ]


def _sc_card(state, expert, *, count=25, ew_mean=0.0, ew_var=0.0, delta=0.0):
    """One scorecard card. count >= 20 clears the tilt threshold (per-name episodes)."""
    return {
        "market_state": state, "expert_id": expert,
        "ew_mean": ew_mean, "ew_var": ew_var, "count": count, "hits": count,
        "reflection_delta": delta,
    }


def _scorecard_snapshot_payload(cards, *, scope="state", status="frozen"):
    """A frozen scorecard snapshot payload (Fable learning-layer format)."""
    payload = {
        "type": "expert_scorecard", "scope": scope, "use_reflection": True,
        "min_support": 20, "temperature": 0.001, "frozen": True,
        "cards": list(cards), "status": status, "frozen_at": None,
    }
    return payload


def test_paper_mode_loads_memory_snapshot_metadata(tmp_path):
    from trading_agent import cli

    snapshot = tmp_path / "memory.json"
    snapshot_payload = _scorecard_snapshot_payload(
        [_sc_card("up_stress", "technical", ew_mean=0.001, delta=0.05)]
    )
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    output = tmp_path / "paper_snapshot.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--memory-snapshot", str(snapshot),
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["memory_snapshot"]["path"] == str(snapshot)
    assert payload["memory_snapshot"]["entries_loaded"] == 1        # one scorecard
    assert payload["memory_snapshot"]["learned_adjustments_loaded"] == 1   # one nonzero delta
    assert payload["memory_snapshot"]["status"] == "frozen"
    assert payload["memory_snapshot"]["status"] == "frozen"
    assert payload["memory_snapshot"]["frozen_at"] is None
    assert payload["memory_snapshot"]["adjustments"] == [
        {
            "expert_id": "technical",
            "market_state": "up_stress",
            "status": "learned",
            "weight_delta": 0.05,
        }
    ]


def test_paper_mode_accepts_full_holdout_report_as_memory_snapshot(tmp_path):
    from trading_agent import cli

    report = tmp_path / "holdout.json"
    report.write_text(json.dumps({
        "memory_snapshot": _scorecard_snapshot_payload(
            [_sc_card("up_stress", "technical", ew_mean=0.001, delta=0.05)]
        )
    }), encoding="utf-8")

    output = tmp_path / "paper_report_snapshot.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--memory-snapshot", str(report),
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["memory_snapshot"]["entries_loaded"] == 1


def test_paper_auto_decide_ignores_undersupported_memory_for_application(tmp_path, monkeypatch):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    # A state below the scorecard tilt threshold (count < 20) is not applied: the weights
    # fall back to equal, exactly as a deprecated entry did under the old delta memory.
    snapshot = tmp_path / "undersupported_memory.json"
    snapshot.write_text(json.dumps(_scorecard_snapshot_payload(
        [_sc_card("up_stress", "technical", count=5, ew_mean=0.001)]
    )), encoding="utf-8")

    output = tmp_path / "paper_undersupported_ignored.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--memory-snapshot", str(snapshot),
        "--paper-market-state", "up_stress",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["memory_snapshot"]["entries_loaded"] == 1
    assert payload["memory_snapshot"]["learned_adjustments_loaded"] == 0
    assert "memory_applied" not in payload
    assert payload["aggregation"]["weights"]["technical"] == pytest.approx(1.0 / 3.0)


def test_paper_mode_rejects_unfrozen_memory_snapshot(tmp_path):
    from trading_agent import cli

    snapshot = tmp_path / "memory_training.json"
    snapshot.write_text(json.dumps({
        "status": "training",
        "entries": [
            {
                "entry_id": "m1",
                "expert_id": "technical",
                "market_state": "up_stress",
                "weight_delta": 0.05,
                "status": "learned",
            }
        ],
    }), encoding="utf-8")

    output = tmp_path / "paper_unfrozen.json"
    with pytest.raises(ValueError, match="frozen"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--memory-snapshot", str(snapshot),
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_require_memory_rejects_missing_snapshot(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_require_memory.json"
    with pytest.raises(ValueError, match="--require-memory"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--auto-decide",
            "--require-memory",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_require_memory_requires_auto_decision(tmp_path):
    from trading_agent import cli

    output = tmp_path / "paper_require_memory_manual.json"
    with pytest.raises(ValueError, match="--require-memory.*--auto-decide"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--order", "AAPL:buy:1",
            "--require-memory",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_require_memory_rejects_empty_snapshot(tmp_path):
    from trading_agent import cli

    snapshot = tmp_path / "empty_memory.json"
    snapshot_payload = _scorecard_snapshot_payload([])
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    output = tmp_path / "paper_empty_memory.json"
    with pytest.raises(ValueError, match="at least one learned"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--auto-decide",
            "--memory-snapshot", str(snapshot),
            "--require-memory",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_require_memory_rejects_undersupported_only_snapshot(tmp_path):
    from trading_agent import cli

    # No state clears the tilt threshold (count < 20) -> no learned market state.
    snapshot = tmp_path / "undersupported_memory.json"
    snapshot_payload = _scorecard_snapshot_payload(
        [_sc_card("up_stress", "technical", count=5, ew_mean=0.001)]
    )
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    output = tmp_path / "paper_undersupported_memory.json"
    with pytest.raises(ValueError, match="at least one learned"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--auto-decide",
            "--memory-snapshot", str(snapshot),
            "--require-memory",
            "--dry-run",
            "--output", str(output),
        ])


def test_paper_mode_require_memory_rejects_snapshot_without_current_state_adjustment(
    tmp_path,
    monkeypatch,
):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    # The snapshot has learned memory for down_stress, but the classified state is up_stress.
    snapshot = tmp_path / "other_state_memory.json"
    snapshot_payload = _scorecard_snapshot_payload(
        [_sc_card("down_stress", "technical", ew_mean=0.001)]
    )
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    output = tmp_path / "paper_other_state_memory.json"
    with pytest.raises(ValueError, match="learned memory"):
        cli.main([
            "paper",
            "--symbols", "AAPL",
            "--auto-decide",
            "--paper-cash", "1000",
            "--memory-snapshot", str(snapshot),
            "--require-memory",
            "--paper-market-state", "up_stress",
            "--dry-run",
            "--output", str(output),
        ])
    assert fake_ib.disconnected is True


def test_paper_auto_decide_applies_loaded_memory_weights(tmp_path, monkeypatch):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    snapshot = tmp_path / "memory.json"
    snapshot_payload = _scorecard_snapshot_payload(
        [_sc_card("up_stress", "technical", ew_mean=0.001)]   # positive score -> tilts up
    )
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    output = tmp_path / "paper_memory_applied.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-cash", "1000",
        "--memory-snapshot", str(snapshot),
        "--require-memory",
        "--paper-market-state", "up_stress",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["market_state"] == "up_stress"
    assert payload["market_state_source"] == "manual_cli"
    assert payload["memory_required"] is True
    assert payload["memory_applied"]["market_state"] == "up_stress"
    assert payload["memory_applied"]["weights"]["technical"] > 0.25
    assert payload["memory_applied"]["state_support"] == 25
    assert fake_ib.disconnected is True


def test_paper_auto_decide_classifies_market_state_by_default(tmp_path, monkeypatch):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    class FakeClassifier:
        def __init__(self, panel, config, proxy_levels=None):
            self.panel = panel
            self.config = config
            self.proxy_levels = proxy_levels

        def classify(self, observation):
            assert observation.tickers == ["AAPL"]
            # An external market proxy (SPY by default) must be supplied so the state
            # is not conditioned on the traded name's own bars.
            assert self.proxy_levels
            return "down_calm"

    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())
    monkeypatch.setattr(cli, "MarketStateClassifier", FakeClassifier)

    output = tmp_path / "paper_classified_state.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-cash", "1000",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["market_state"] == "down_calm"
    assert payload["market_state_source"] == "classifier_proxy"
    assert fake_ib.disconnected is True


def test_paper_auto_decide_requests_enough_history_for_default_classifier(tmp_path, monkeypatch):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    durations = []
    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)

    def fake_request_daily_bars(ib, *args, **kwargs):
        durations.append(kwargs["duration"])
        return Frame()

    monkeypatch.setattr(cli, "request_daily_bars", fake_request_daily_bars)

    output = tmp_path / "paper_default_history.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    # Enough history to cover the 200-day trend window (not just the RV windows), and
    # one extra request for the SPY market proxy. Both use the same duration.
    assert durations == ["305 D", "305 D"]


def test_paper_auto_decide_manual_market_state_overrides_classifier(tmp_path, monkeypatch):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    class FakeClassifier:
        def __init__(self, panel, config):
            pass

        def classify(self, observation):
            return "down_stress"

    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())
    monkeypatch.setattr(cli, "MarketStateClassifier", FakeClassifier)

    output = tmp_path / "paper_manual_state.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-cash", "1000",
        "--paper-market-state", "up_stress",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["market_state"] == "up_stress"
    assert payload["market_state_source"] == "manual_cli"
    assert fake_ib.disconnected is True


def test_paper_auto_decide_deploys_full_cash_when_long(tmp_path, monkeypatch):
    from trading_agent import cli

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())

    snapshot = tmp_path / "memory_negative.json"
    snapshot.write_text(json.dumps(_scorecard_snapshot_payload(
        [_sc_card("up_stress", "technical", count=25, delta=-0.10)]   # down-weight technical
    )), encoding="utf-8")

    output = tmp_path / "paper_memory_scaled.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-cash", "1000",
        "--memory-snapshot", str(snapshot),
        "--paper-market-state", "up_stress",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    # Under v12 the risk layer does no discretionary sizing: a long decision deploys the full
    # paper cash (1000 / 100 = 10 shares). With all experts long, the +/-1 vote score is 1.0
    # regardless of the memory weights, so the memory changes the weights but not the size.
    assert payload["memory_applied"]["direction"] == "long"
    assert payload["memory_applied"]["aggregation_score"] == pytest.approx(1.0)
    assert payload["order_previews"][0]["quantity"] == 10
    assert fake_ib.disconnected is True


def test_paper_auto_decide_calls_experts_and_records_views(tmp_path, monkeypatch):
    from trading_agent import cli
    from trading_agent.contracts import ExpertView

    class Series:
        def __init__(self, value):
            self.iloc = [value]

    class Frame:
        empty = False

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "date":
                return Series("2026-06-25")
            if key == "close":
                return Series(100.0)
            raise KeyError(key)

    class FakeExpert:
        def __init__(self, expert_id, confidence):
            self.expert_id = expert_id
            self.confidence = confidence
            self.calls = 0

        def view(self, observation, market_state, context=None):
            self.calls += 1
            assert observation.tickers == ["AAPL"]
            assert market_state == "up_stress"
            return ExpertView(
                expert_id=self.expert_id,
                direction="long",
                confidence=self.confidence,
                reasoning=f"{self.expert_id} online",
            )

    experts = [
        FakeExpert("technical", 0.9),
        FakeExpert("factor", 0.6),
        FakeExpert("news", 0.5),
        FakeExpert("fundamental", 0.4),
    ]
    fake_ib = _FakePaperIB()
    monkeypatch.setattr(cli, "connect_ib", lambda settings: fake_ib)
    monkeypatch.setattr(cli, "request_daily_bars", lambda ib, *args, **kwargs: Frame())
    monkeypatch.setattr(cli, "build_static_experts", lambda: experts)

    output = tmp_path / "paper_views.json"
    rc = cli.main([
        "paper",
        "--symbols", "AAPL",
        "--auto-decide",
        "--paper-cash", "1000",
        "--paper-market-state", "up_stress",
        "--dry-run",
        "--output", str(output),
    ])

    assert rc == 0
    assert [expert.calls for expert in experts] == [1, 1, 1, 1]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["expert_views"]["technical"]["confidence"] == 0.9
    assert payload["aggregation"]["direction"] == "long"
    assert payload["aggregation"]["score"] > 0.0
    assert fake_ib.disconnected is True


def test_paper_memory_application_uses_frozen_theta_not_aggregation_default():
    # 2 long vs 1 flat -> equal-weight score +1/3, inside (0, 0.5). The paper
    # decision path must use the FROZEN system threshold (theta=0, strict) ->
    # long; the aggregation signature's 0.5 default would have said flat (the
    # pre-2026-07-18 drift this test pins down). A tie (score exactly 0) stays
    # flat: stateless paper invocations are strict entry re-votes.
    from trading_agent.cli import _memory_application
    from trading_agent.contracts import ExpertView

    def v(eid, direction):
        return ExpertView(expert_id=eid, direction=direction, confidence=0.8,
                          reasoning="t", evidence=("e",), risk_flags=(),
                          data_quality="high")

    views = {
        "technical": {"AAA": v("technical", "long")},
        "news": {"AAA": v("news", "long")},
        "fundamental": {"AAA": v("fundamental", "flat")},
    }
    _, payload, fraction = _memory_application("up_calm", None, views, ("AAA",))
    assert payload["by_ticker"]["AAA"]["direction"] == "long"
    assert payload["by_ticker"]["AAA"]["score"] == pytest.approx(1.0 / 3.0)
    assert fraction == 1.0

    tie = {
        "technical": {"AAA": v("technical", "long")},
        "news": {"AAA": v("news", "neutral")},
        "fundamental": {"AAA": v("fundamental", "flat")},
    }
    _, tie_payload, _ = _memory_application("up_calm", None, tie, ("AAA",))
    assert tie_payload["by_ticker"]["AAA"]["score"] == pytest.approx(0.0)
    assert tie_payload["by_ticker"]["AAA"]["direction"] == "flat"


def test_paper_boundary_score_maintains_a_held_position():
    # The frozen action rule maintains a position at a boundary score, because zero net
    # spoken edge is evidence for nothing. The paper bridge aggregated without the
    # account's position until 2026-08-05, which made every session a fresh entry vote
    # and would have closed O that day on a score of exactly 0.000 against 78 shares.
    from trading_agent.cli import _memory_application
    from trading_agent.contracts import ExpertView

    def v(eid, direction):
        return ExpertView(expert_id=eid, direction=direction, confidence=0.8,
                          reasoning="t", evidence=("e",), risk_flags=(),
                          data_quality="high")

    tie = {
        "technical": {"AAA": v("technical", "long")},
        "news": {"AAA": v("news", "neutral")},
        "fundamental": {"AAA": v("fundamental", "flat")},
    }
    _, held_payload, _ = _memory_application("up_calm", None, tie, ("AAA",),
                                             held={"AAA": 78.0})
    assert held_payload["by_ticker"]["AAA"]["score"] == pytest.approx(0.0)
    assert held_payload["by_ticker"]["AAA"]["direction"] == "long"

    # A read account holding nothing is not the same as an unread one: the stock stays
    # flat, and a strictly positive score still enters.
    _, flat_payload, _ = _memory_application("up_calm", None, tie, ("AAA",), held={})
    assert flat_payload["by_ticker"]["AAA"]["direction"] == "flat"

    enter = {
        "technical": {"AAA": v("technical", "long")},
        "news": {"AAA": v("news", "long")},
        "fundamental": {"AAA": v("fundamental", "flat")},
    }
    _, enter_payload, _ = _memory_application("up_calm", None, enter, ("AAA",), held={})
    assert enter_payload["by_ticker"]["AAA"]["direction"] == "long"

    # A held stock the committee turns against is still exited: maintain is a boundary
    # rule, not a ratchet.
    against = {
        "technical": {"AAA": v("technical", "flat")},
        "news": {"AAA": v("news", "flat")},
        "fundamental": {"AAA": v("fundamental", "long")},
    }
    _, exit_payload, _ = _memory_application("up_calm", None, against, ("AAA",),
                                             held={"AAA": 78.0})
    assert exit_payload["by_ticker"]["AAA"]["score"] < 0
    assert exit_payload["by_ticker"]["AAA"]["direction"] == "flat"
