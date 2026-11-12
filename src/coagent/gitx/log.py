"""The session log: one JSONL record per turn (DESIGN §3.4, FR-TOL-8).

Written per turn (not per session) because a preemption must not lose the tail.
Every record is **redacted at write time** — redaction after the fact is a
promise, redaction before the write is a property (PRD FR-PRV-3).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..safety.redact import Redactor

__all__ = ["SessionLog", "LogRecord"]


@dataclass(slots=True)
class LogRecord:
    step: int
    phase: str
    tool: str
    ok: bool
    tokens_in: int = 0
    tokens_out: int = 0
    ms: int = 0
    commit: str = ""
    args_hash: str = ""
    error: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "t": round(time.time(), 3),
            "step": self.step,
            "phase": self.phase,
            "tool": self.tool,
            "ok": self.ok,
        }
        if self.tokens_in or self.tokens_out:
            data["tokens"] = {"in": self.tokens_in, "out": self.tokens_out}
        if self.ms:
            data["ms"] = self.ms
        if self.commit:
            data["commit"] = self.commit
        if self.args_hash:
            data["args_hash"] = self.args_hash
        if self.error:
            data["error"] = self.error
        if self.note:
            data["note"] = self.note
        return data


class SessionLog:
    """Append-only JSONL log for one session."""

    def __init__(self, repo_root: str | Path, session_id: str, *, redactor: Redactor | None = None) -> None:
        self.path = Path(repo_root) / ".coagent" / "sessions" / f"{session_id}.jsonl"
        self.redactor = redactor or Redactor()

    def append(self, record: LogRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Redact the serialised record: every string field, including nested ones.
        safe = self.redactor.redact_obj(record.to_dict())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def tail(self, count: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-count:]
