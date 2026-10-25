"""Bootstrap path discovery for installed CLI runs."""

from __future__ import annotations

from pathlib import Path


def test_finsaber_root_discovery_includes_working_tree_external(monkeypatch, tmp_path):
    from trading_agent import _bootstrap

    project = tmp_path / "project"
    finsaber = project / "external" / "FINSABER"
    finsaber.mkdir(parents=True)
    monkeypatch.chdir(project)

    roots = _bootstrap._possible_finsaber_roots(package_file=Path("site/trading_agent/_bootstrap.py"))

    assert finsaber in roots
