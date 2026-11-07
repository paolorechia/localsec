"""A record of what the agents did.

Security work has to be auditable: what was asked, what was changed, what
was found, and who approved it. Every entry is one JSON object on one line,
so the log can be read, tailed, and shipped while it is written.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["AuditLog"]


class AuditLog:
    """An append-only log of what the agents did."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the log at `path`."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: object) -> dict[str, Any]:
        """Write one entry, and return it."""
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        """Return every entry written so far."""
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries
