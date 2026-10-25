"""Tests for data_collection.config.load_env_file (the lightweight .env loader).

Secrets (OPENAI_API_KEY etc.) live in a gitignored .env that the experiment launcher
loads at startup; these guard the parsing, especially the Windows BOM case where a
UTF-8-BOM editor would otherwise fuse the BOM into the first key name.
"""
from __future__ import annotations

from data_collection.config import load_env_file

_KEY = "TA_TEST_DOTENV_KEY"


def _write(path, text, encoding):
    path.write_text(text, encoding=encoding)
    return path


def test_load_env_file_reads_plain_utf8(tmp_path, monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)
    env = _write(tmp_path / ".env", f"{_KEY}=sk-plain-123\n", "utf-8")
    load_env_file(env)
    import os
    assert os.environ.get(_KEY) == "sk-plain-123"


def test_load_env_file_strips_leading_bom(tmp_path, monkeypatch):
    # Notepad / PowerShell Set-Content -Encoding utf8 prepend a BOM. The first key
    # must still parse as OPENAI_API_KEY, not ﻿OPENAI_API_KEY.
    monkeypatch.delenv(_KEY, raising=False)
    env = _write(tmp_path / ".env", f"{_KEY}=sk-bom-123\n", "utf-8-sig")
    load_env_file(env)
    import os
    assert os.environ.get(_KEY) == "sk-bom-123"


def test_existing_env_var_wins_over_dotenv(tmp_path, monkeypatch):
    # setdefault semantics: an already-exported shell var overrides the file.
    monkeypatch.setenv(_KEY, "sk-from-shell")
    env = _write(tmp_path / ".env", f"{_KEY}=sk-from-file\n", "utf-8")
    load_env_file(env)
    import os
    assert os.environ.get(_KEY) == "sk-from-shell"


def test_missing_file_is_a_noop(tmp_path):
    load_env_file(tmp_path / "does_not_exist.env")  # must not raise
