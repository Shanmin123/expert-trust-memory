from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
