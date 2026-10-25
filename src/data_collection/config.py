from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    # utf-8-sig strips a leading BOM: Windows editors (Notepad, PowerShell
    # Set-Content -Encoding utf8) prepend one, which would otherwise fuse into the
    # first key name (﻿OPENAI_API_KEY) and silently break key loading.
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    ib_host: str
    ib_port: int
    ib_client_id: int
    data_dir: Path
    reports_dir: Path
    rapidapi_key: str | None
    rapidapi_host: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        if not reports_dir.is_absolute():
            reports_dir = PROJECT_ROOT / reports_dir

        rapidapi_key = os.getenv("RAPIDAPI_KEY") or None
        return cls(
            ib_host=os.getenv("IB_HOST", "127.0.0.1"),
            ib_port=int(os.getenv("IB_PORT", "4002")),
            ib_client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            data_dir=data_dir,
            reports_dir=reports_dir,
            rapidapi_key=rapidapi_key,
            rapidapi_host=os.getenv(
                "RAPIDAPI_HOST", "seeking-alpha-finance.p.rapidapi.com"
            ),
        )

    @property
    def prices_db(self) -> Path:
        return self.data_dir / "sp500_prices.db"

    @property
    def indicators_db(self) -> Path:
        return self.data_dir / "sp500_indicators.db"

    @property
    def alphas_db(self) -> Path:
        return self.data_dir / "sp500_alphas.db"

    @property
    def qlib_direct_factors_db(self) -> Path:
        return self.data_dir / "sp500_qlib_direct_factors.db"

    @property
    def qlib_provider_dir(self) -> Path:
        return self.data_dir / "qlib_provider" / "us_sp500"

    @property
    def finsaber_dir(self) -> Path:
        return self.data_dir / "finsaber"

    @property
    def seeking_alpha_dir(self) -> Path:
        return self.data_dir / "seeking_alpha"

    @property
    def universe_snapshots_dir(self) -> Path:
        return self.data_dir / "universe_snapshots"

    @property
    def sp500_universe_snapshot(self) -> Path:
        return self.data_dir / "sp500_universe_current_snapshot.csv"


def ensure_project_dirs(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    for path in (
        settings.data_dir,
        settings.reports_dir,
        settings.finsaber_dir,
        settings.seeking_alpha_dir,
        settings.universe_snapshots_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
