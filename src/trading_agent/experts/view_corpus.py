"""Frozen expert-view corpus: record and replay LLM expert calls.

The LLM experts are the project's distinguishing feature, but a live language
model is non-deterministic (temperature zero does not guarantee bit-identical
output across backend versions), costs money per call, and cannot be replayed
for ablations. This module records every expert call once and lets all later
learning-layer experiments -- reflection, attribution-vs-profit, bounded-vs-
unbounded, the three arms -- run as a deterministic replay over the *same* set
of LLM judgments. The corpus is the experiment's raw data, analogous to an
offline reinforcement-learning dataset.

A record is keyed by ``(expert_id, system_prompt, user_prompt)``. The user
prompt already encodes the decision date and the tickers, and each expert has
its own system prompt, so the key uniquely identifies a request; the expert id
is included explicitly so two experts cannot collide on an identical prompt.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

RECORD = "record"
REPLAY = "replay"
REPLAY_BY_DATE = "replay_by_date"
READTHROUGH = "readthrough"
_MODES = frozenset({RECORD, REPLAY, REPLAY_BY_DATE, READTHROUGH})


def _cache_key(expert_id: str, user_prompt: str) -> tuple[str, str]:
    """What a record is about: which expert was asked, and what it was asked.

    Both halves are stored on the record, so a lookup is an identity and not an
    agreement between two digests.
    """
    return (expert_id, user_prompt)


def _extract_date_and_tickers(user_prompt: str) -> tuple[Optional[str], list[str]]:
    try:
        payload = json.loads(user_prompt)
    except (TypeError, ValueError):
        return None, []
    if not isinstance(payload, dict):
        return None, []
    date = payload.get("date")
    tickers = payload.get("tickers")
    return (
        str(date) if date is not None else None,
        [str(t) for t in tickers] if isinstance(tickers, list) else [],
    )


@dataclass(frozen=True)
class CorpusRecord:
    expert_id: str
    date: Optional[str]
    tickers: tuple[str, ...]
    response: str
    user_prompt: str

    @property
    def key(self) -> tuple[str, str]:
        return _cache_key(self.expert_id, self.user_prompt)

    def to_json(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "date": self.date,
            "tickers": list(self.tickers),
            "response": self.response,
            "user_prompt": self.user_prompt,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CorpusRecord":
        # Corpora recorded before the key became readable also carry "key" and
        # "prompt_sha". Both are ignored, so those files load unchanged.
        if "user_prompt" not in data:
            raise KeyError("user_prompt")
        return cls(
            expert_id=str(data.get("expert_id", "")),
            date=(str(data["date"]) if data.get("date") is not None else None),
            tickers=tuple(str(t) for t in data.get("tickers", [])),
            response=str(data.get("response", "")),
            user_prompt=str(data.get("user_prompt", "")),
        )


class ViewCorpus:
    """A JSONL-backed store of expert view responses, shared across experts.

    Thread-safe: ``run_expert_loop`` may call the experts in parallel, and
    they share one corpus. The file is append-only; loading reads the last entry
    for any repeated key so a re-recorded corpus stays self-consistent.
    """

    def __init__(self, path: str | Path, mode: str = READTHROUGH) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown corpus mode {mode!r}; must be one of {sorted(_MODES)}")
        self.path = Path(path)
        self.mode = mode
        self._store: dict[tuple[str, str], CorpusRecord] = {}
        # Secondary index for prompt-drift-tolerant replay: the exact (expert_id,
        # date, tickers) request, independent of the prompt string. A frozen view
        # is the expert's decision for that name/date; downstream experiments that
        # change only the market state, attribution horizon, or scorecard scope do
        # not change the views, so serving them by identity reproduces the frozen
        # decisions even when serialisation drift moves the prompt hash.
        self._by_date: dict[tuple[str, Optional[str], tuple[str, ...]], CorpusRecord] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.records_written = 0
        if mode in (REPLAY, REPLAY_BY_DATE, READTHROUGH):
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            if self.mode == REPLAY:
                raise FileNotFoundError(
                    f"replay corpus not found: {self.path}. Record it first with mode='record'."
                )
            return
        self.corrupt_lines = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = CorpusRecord.from_json(json.loads(line))
                except (ValueError, KeyError):
                    # A torn line (e.g. from an interrupted or concurrent
                    # writer) must not brick the whole corpus; the affected
                    # request is simply re-recorded on the next readthrough.
                    self.corrupt_lines += 1
                    continue
                self._store[record.key] = record
                # Last-wins by identity, mirroring the append-only store's own
                # last-wins for a repeated hash key: when a name/date/expert view
                # was recorded more than once (e.g. across recording passes), the
                # most recent frozen response is served.
                self._by_date[(record.expert_id, record.date, record.tickers)] = record
        if self.corrupt_lines:
            import logging
            logging.getLogger(__name__).warning(
                "view corpus %s: skipped %d corrupt line(s)", self.path, self.corrupt_lines
            )

    def lookup(self, expert_id: str, user_prompt: str) -> Optional[str]:
        """The stored response to this exact prompt, or None.

        A hit means the stored prompt is this prompt, so there is nothing further
        to check: the corpus cannot serve one prompt's answer under another's key.
        """
        record = self._store.get(_cache_key(expert_id, user_prompt))
        return record.response if record is not None else None

    def lookup_by_date(
        self, expert_id: str, date: Optional[str], tickers: Sequence[str]
    ) -> Optional[str]:
        """Return the frozen response for an exact (expert, date, tickers) request.

        Prompt-drift-tolerant fallback used only in ``replay_by_date`` mode. When a
        view was recorded more than once, the most recent response wins (mirroring
        the store's last-wins for a repeated hash key).

        A MULTI-ticker request that has no exact record is assembled from the
        per-ticker records for the same (expert, date): each recorded response is a
        list of per-ticker view objects, so concatenating them reproduces what one
        batched call over those tickers would have returned. This lets several
        single-name corpora be replayed as one pooled universe at zero cost.

        Ragged coverage is normal in a multi-name universe (names start late, have
        data gaps), so a name with no frozen view for that date is simply omitted --
        it then takes the expert's data-unavailable safe default and abstains, which
        is the same thing that would have happened live. Only a date where NO name
        has a view is a miss. Single-ticker lookups are unaffected.
        """
        names = tuple(str(t) for t in tickers)
        record = self._by_date.get((expert_id, date, names))
        if record is not None:
            return record.response
        if len(names) <= 1:
            return None
        merged: list = []
        for ticker in names:
            part_record = self._by_date.get((expert_id, date, (ticker,)))
            if part_record is None:
                continue            # this name has no frozen view for the date -> abstains
            try:
                part = json.loads(part_record.response)
            except (TypeError, ValueError):
                continue
            if isinstance(part, list):
                merged.extend(part)
        return json.dumps(merged, ensure_ascii=False) if merged else None

    def note_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def note_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def append(self, record: CorpusRecord) -> None:
        with self._lock:
            self._store[record.key] = record
            self.records_written += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "mode": self.mode,
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "records_written": self.records_written,
        }


class CachedExpertBackend:
    """Wrap an expert backend so its calls are recorded to / replayed from a corpus.

    - ``record``: always call the inner backend, store the response, return it.
    - ``replay``: return the stored response; a missing key is an error, because
      it means the frozen corpus does not cover the request being replayed.
    - ``readthrough``: return the stored response if present, otherwise call the
      inner backend and store it (incremental corpus building across runs).
    """

    def __init__(self, inner: Any, corpus: ViewCorpus, expert_id: str, mode: Optional[str] = None) -> None:
        self._inner = inner
        self._corpus = corpus
        self.expert_id = expert_id
        self.mode = mode or corpus.mode
        if self.mode not in _MODES:
            raise ValueError(f"unknown corpus mode {self.mode!r}")
        # Proxy model identity so the run manifest can record what produced the corpus.
        self.model = getattr(inner, "model", None)
        self.model_revision = getattr(inner, "model_revision", None)

    def call(self, system_prompt: str, user_prompt: str) -> str:
        if self.mode in (REPLAY, REPLAY_BY_DATE, READTHROUGH):
            cached = self._corpus.lookup(self.expert_id, user_prompt)
            if cached is not None:
                self._corpus.note_hit()
                return cached
            if self.mode == REPLAY_BY_DATE:
                # Exact prompt hash missed (serialisation drift): fall back to the
                # frozen view for this exact (expert, date, tickers) request.
                date, tickers = _extract_date_and_tickers(user_prompt)
                cached = self._corpus.lookup_by_date(self.expert_id, date, tickers)
                if cached is not None:
                    self._corpus.note_hit()
                    return cached
            if self.mode in (REPLAY, REPLAY_BY_DATE):
                date, _tk = _extract_date_and_tickers(user_prompt)
                raise KeyError(
                    f"replay corpus miss for expert={self.expert_id} date={date}; "
                    "the frozen corpus does not cover this request."
                )
        self._corpus.note_miss()
        response = self._inner.call(system_prompt, user_prompt)
        date, tickers = _extract_date_and_tickers(user_prompt)
        self._corpus.append(CorpusRecord(
            expert_id=self.expert_id,
            date=date,
            tickers=tuple(tickers),
            response=response,
            user_prompt=user_prompt,
        ))
        return response

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


def attach_corpus(experts: Sequence[Any], corpus: ViewCorpus, *, mode: Optional[str] = None) -> Sequence[Any]:
    """Wrap each expert's backend with a CachedExpertBackend sharing one corpus.

    Mutates the experts in place (replacing ``_backend``) and returns them.
    """
    for expert in experts:
        inner = expert._backend
        expert._backend = CachedExpertBackend(inner, corpus, expert.expert_id, mode)
    return experts
