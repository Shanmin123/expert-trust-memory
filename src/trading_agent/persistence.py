"""Append-only, replayable run log (JSONL).

Records one line per event with an idempotency key (run_id, decision_id, kind): a duplicate
key is skipped, so re-emitting a decision does not double-write. Payloads are deterministic
(decision-time timestamps only — never wall-clock), so a replay produces an identical log.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date as Date, datetime
from pathlib import Path
from typing import Any


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, Date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class RunLogger:
    """Idempotent JSONL logger keyed on (run_id, decision_id, kind)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[tuple] = set()
        if self.path.exists():
            for record in self.records():
                self._seen.add(
                    (record.get("run_id"), record.get("decision_id"), record.get("kind"))
                )

    def log(self, kind: str, run_id: str, decision_id: str, payload: dict) -> bool:
        key = (run_id, decision_id, kind)
        if key in self._seen:
            return False
        record = {"kind": kind, "run_id": run_id, "decision_id": decision_id}
        record.update(_jsonable(payload))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._seen.add(key)
        return True

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
