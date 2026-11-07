"""Tests for a patching run."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from helpers import FINDINGS, NOT_VERIFIED, PLAN, VERIFIED, prompt_text, scripted_output
from localsec.agents.patching.orchestrator import PatchingRun
from localsec.agents.patching.workspace import RepoDeps, Workspace
from localsec.runtime.approvals import ApprovalGate, ApprovalRequest, Decision
from localsec.runtime.audit import AuditLog
from localsec.runtime.sandbox import DockerSandbox


@pytest.fixture
def deps(tmp_path: Path) -> RepoDeps:
    """A workspace with one file in it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def load(p):\n    return open(p)\n")
    return RepoDeps(workspace=Workspace(root))


def test_a_run_with_a_closed_gate_changes_nothing(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.DENY)
    run = PatchingRun(scripted_output(FINDINGS, PLAN, VERIFIED), deps, gate=gate)

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.declined == ["app.py"]
    assert outcome.applied == []
    assert deps.workspace.read("app.py") == "def load(p):\n    return open(p)\n"


def test_a_run_whose_changes_were_all_declined_did_not_pass(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.DENY)
    run = PatchingRun(scripted_output(FINDINGS, PLAN, VERIFIED), deps, gate=gate)

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.verification is not None
    assert outcome.verification.passed
    assert not outcome.passed


def test_a_run_with_nothing_to_fix_passes(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    empty = {"findings": [], "notes": "The code is not here."}
    run = PatchingRun(scripted_output(empty, PLAN, VERIFIED), deps, gate=gate)

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.passed


def test_a_run_applies_the_change_when_the_gate_allows(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    run = PatchingRun(scripted_output(FINDINGS, PLAN, VERIFIED), deps, gate=gate)

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.applied == ["app.py"]
    assert outcome.declined == []
    assert "encoding='utf-8'" in deps.workspace.read("app.py")
    assert outcome.passed


def test_the_gate_is_asked_before_every_change(deps: RepoDeps) -> None:
    asked: list[str] = []

    def approver(request: ApprovalRequest) -> Decision:
        asked.append(request.tool)
        return Decision.ALLOW

    gate = ApprovalGate(approver)
    run = PatchingRun(scripted_output(FINDINGS, PLAN, VERIFIED), deps, gate=gate)

    asyncio.run(run.run("The path is not checked."))

    assert asked == ["edit_file", "run_tests"]


def test_a_change_that_does_not_apply_is_recorded(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    plan = {
        "summary": "Fix it.",
        "edits": [
            {
                "path": "app.py",
                "old": "    return closed(p)",
                "new": "    return open(p)",
                "rationale": "The wrong call.",
            }
        ],
        "test_command": "pytest -q",
    }
    run = PatchingRun(scripted_output(FINDINGS, plan, VERIFIED), deps, gate=gate)

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.failed == ["app.py"]
    assert outcome.applied == []
    assert deps.workspace.read("app.py") == "def load(p):\n    return open(p)\n"


def test_a_run_that_does_not_verify_tries_again(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    run = PatchingRun(
        scripted_output(FINDINGS, PLAN, NOT_VERIFIED, PLAN, VERIFIED),
        deps,
        gate=gate,
        max_rounds=2,
    )

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.rounds == 2
    assert outcome.passed


def test_a_run_that_never_verifies_gives_up_after_its_rounds(deps: RepoDeps) -> None:
    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    run = PatchingRun(
        scripted_output(FINDINGS, PLAN, NOT_VERIFIED, PLAN, NOT_VERIFIED),
        deps,
        gate=gate,
        max_rounds=2,
    )

    outcome = asyncio.run(run.run("The path is not checked."))

    assert outcome.rounds == 2
    assert not outcome.passed
    assert outcome.verification is not None
    assert outcome.verification.remaining_issues == ["the encoding is still missing"]


def test_the_configured_test_command_wins_over_the_agent(deps: RepoDeps, tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\necho "$@"\n', encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    deps.sandbox = DockerSandbox(docker=str(docker))
    deps.test_command = "make test"
    seen: list[str] = []
    payloads: list[dict[str, Any]] = [FINDINGS, PLAN, VERIFIED]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = payloads[len(seen)]
        seen.append(prompt_text(messages))
        return ModelResponse(parts=[ToolCallPart("final_result", payload)])

    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    run = PatchingRun(FunctionModel(respond), deps, gate=gate)

    asyncio.run(run.run("The path is not checked."))

    assert "make test" in seen[2]
    assert "pytest -q" not in seen[2]


def test_the_run_is_audited(deps: RepoDeps, tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    run = PatchingRun(scripted_output(FINDINGS, PLAN, VERIFIED), deps, gate=gate, audit=audit)

    asyncio.run(run.run("The path is not checked."))

    events = [entry["event"] for entry in audit.entries()]
    assert events == [
        "triage_started",
        "triage_completed",
        "plan_created",
        "round_applied",
        "tests_completed",
        "verification_completed",
    ]


def test_the_test_output_reaches_the_verification(deps: RepoDeps, tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\necho '2 passed'\n", encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    deps.sandbox = DockerSandbox(docker=str(docker))
    seen: list[str] = []
    payloads: list[dict[str, Any]] = [FINDINGS, PLAN, VERIFIED]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = payloads[len(seen)]
        seen.append(prompt_text(messages))
        return ModelResponse(parts=[ToolCallPart("final_result", payload)])

    gate = ApprovalGate(lambda _request: Decision.ALLOW)
    run = PatchingRun(FunctionModel(respond), deps, gate=gate)

    asyncio.run(run.run("The path is not checked."))

    assert "2 passed" in seen[2]
