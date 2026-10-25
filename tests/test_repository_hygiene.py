"""Repository hygiene checks for generated local artifacts."""

from __future__ import annotations

import subprocess
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generated_local_artifacts_are_git_ignored():
    generated_paths = [
        "build",
        "data",
        "reports/trading_agent_backtest.json",
        "src/trading_agent.egg-info",
        ".agents",
        ".claude",
        ".gemini",
        ".idea",
        ".qoder",
        "skills-lock.json",
    ]

    completed = subprocess.run(
        ["git", "check-ignore", *generated_paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    ignored = set(completed.stdout.splitlines())
    assert ignored == set(generated_paths)


def test_package_dependencies_avoid_numpy_two_abi_breakage():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"numpy>=1.23,<2"' in pyproject
    assert '"numpy>=1.26"' not in pyproject


def test_cli_holdout_launcher_is_package_native():
    cli_source = (ROOT / "src" / "trading_agent" / "cli.py").read_text(encoding="utf-8")

    assert "scripts.run_expert_weight_holdout" not in cli_source
    assert '"scripts" / "run_expert_weight_holdout.py"' not in cli_source


def test_cli_top_level_import_does_not_require_data_backends():
    cli_source = (ROOT / "src" / "trading_agent" / "cli.py").read_text(encoding="utf-8")
    module = ast.parse(cli_source)

    forbidden_top_level_imports = [
        "trading_agent.data.finsaber_panel",
        "trading_agent.expert_context",
        "trading_agent.runner",
    ]
    top_level_imports = {
        node.module
        for node in module.body
        if isinstance(node, ast.ImportFrom)
    }
    assert not (top_level_imports & set(forbidden_top_level_imports))
