"""Tests for the approval gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from localsec.runtime.approvals import (
    ApprovalDeniedError,
    ApprovalGate,
    ApprovalRequest,
    Decision,
    console_approver,
)
from localsec.runtime.audit import AuditLog


def _request(tool: str = "write_file") -> ApprovalRequest:
    return ApprovalRequest(
        tool=tool,
        action="write a file",
        detail="path='a.py'",
        arguments={"path": "a.py"},
    )


def test_a_gate_without_an_approver_denies_everything() -> None:
    gate = ApprovalGate()

    with pytest.raises(ApprovalDeniedError):
        gate.check(_request())


def test_an_approver_can_allow_an_action() -> None:
    gate = ApprovalGate(lambda _request: Decision.ALLOW)

    gate.check(_request())


def test_an_approver_can_deny_an_action() -> None:
    gate = ApprovalGate(lambda _request: Decision.DENY)

    with pytest.raises(ApprovalDeniedError, match="declined"):
        gate.check(_request())


def test_allowing_a_tool_always_remembers_it() -> None:
    calls: list[ApprovalRequest] = []

    def approver(request: ApprovalRequest) -> Decision:
        calls.append(request)
        return Decision.ALLOW_ALWAYS

    gate = ApprovalGate(approver)

    gate.check(_request("write_file"))
    gate.check(_request("write_file"))
    gate.check(_request("run_tests"))

    # The second write is not asked about, the run is.
    assert [request.tool for request in calls] == ["write_file", "run_tests"]
    assert gate.always_allowed == frozenset({"write_file", "run_tests"})


def test_the_console_approver_declines_when_nobody_can_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_stdin(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)

    assert console_approver(_request()) is Decision.DENY


def test_the_console_approver_allows_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    assert console_approver(_request()) is Decision.ALLOW


def test_the_console_approver_allows_always(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "a")

    assert console_approver(_request()) is Decision.ALLOW_ALWAYS


def test_the_console_approver_declines_anything_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "maybe")

    assert console_approver(_request()) is Decision.DENY


def test_every_decision_is_audited(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    gate = ApprovalGate(lambda _request: Decision.ALLOW, audit=audit)

    gate.check(_request())

    entry = audit.entries()[0]
    assert entry["event"] == "approval"
    assert entry["tool"] == "write_file"
    assert entry["granted"] is True


def test_a_denial_is_audited_too(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    gate = ApprovalGate(lambda _request: Decision.DENY, audit=audit)

    with pytest.raises(ApprovalDeniedError):
        gate.check(_request())

    assert audit.entries()[0]["granted"] is False
