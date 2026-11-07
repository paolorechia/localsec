"""Tests for the audit log."""

from __future__ import annotations

import json
from pathlib import Path

from localsec.runtime.audit import AuditLog


def test_a_record_is_written_as_one_json_object(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    entry = log.record("tool_call", tool="write_file", path="a.py")

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry
    assert entry["event"] == "tool_call"
    assert entry["tool"] == "write_file"
    assert entry["ts"].startswith("20")


def test_entries_come_back_in_order(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    log.record("first")
    log.record("second")

    assert [entry["event"] for entry in log.entries()] == ["first", "second"]


def test_the_log_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "audit.jsonl"

    AuditLog(path).record("before")
    AuditLog(path).record("after")

    assert [entry["event"] for entry in AuditLog(path).entries()] == ["before", "after"]


def test_a_value_that_is_not_json_is_kept_as_text(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")

    log.record("tool_call", result=Path("/tmp/x"))

    assert log.entries()[0]["result"] == "/tmp/x"


def test_an_empty_log_has_no_entries(tmp_path: Path) -> None:
    assert AuditLog(tmp_path / "audit.jsonl").entries() == []
