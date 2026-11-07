"""Tests for the patching agents.

The model is stood in for by a script, so the tests are about what the agents
do with what they are given rather than about what a model would say.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from helpers import prompt_text
from localsec.agents.patching.agents import PatchAgent, TriageAgent, VerifyAgent
from localsec.agents.patching.types import Findings, PatchPlan, Verification
from localsec.agents.patching.workspace import RepoDeps, Workspace


def scripted(*steps: tuple[str, str, dict[str, Any]]) -> FunctionModel:
    """Return a model that follows a script of tool calls and answers.

    A step is `("tool", name, arguments)` or `("output", "", arguments)`, and
    an output step fills in the agent's output tool.
    """
    remaining = list(steps)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        kind, name, arguments = remaining.pop(0)
        if kind == "output":
            return ModelResponse(parts=[ToolCallPart("final_result", arguments)])
        return ModelResponse(parts=[ToolCallPart(name, arguments)])

    return FunctionModel(respond)


@pytest.fixture
def deps(tmp_path: Path) -> RepoDeps:
    """A workspace with one file in it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def load(p):\n    return open(p)\n")
    return RepoDeps(workspace=Workspace(root))


def test_the_triage_agent_reports_the_findings(deps: RepoDeps) -> None:
    model = scripted(
        (
            "tool",
            "read_file",
            {"path": "app.py"},
        ),
        (
            "output",
            "",
            {
                "findings": [
                    {
                        "title": "Unvalidated open",
                        "severity": "Medium",
                        "file": "app.py",
                        "line": 2,
                        "cve": None,
                        "description": "The path is not checked.",
                    }
                ],
                "notes": "",
            },
        ),
    )
    agent = TriageAgent(model, deps)

    findings = asyncio.run(agent.run("A path is opened without checking it."))

    assert isinstance(findings, Findings)
    assert findings.findings[0].title == "Unvalidated open"
    assert findings.findings[0].line == 2


def test_the_triage_agent_can_report_nothing(deps: RepoDeps) -> None:
    model = scripted(("output", "", {"findings": [], "notes": "The code is not here."}))
    agent = TriageAgent(model, deps)

    findings = asyncio.run(agent.run("A path is opened without checking it."))

    assert findings.findings == []
    assert findings.notes == "The code is not here."


def test_the_patch_agent_writes_the_plan(deps: RepoDeps) -> None:
    model = scripted(
        (
            "output",
            "",
            {
                "summary": "Check the path before opening it.",
                "edits": [
                    {
                        "path": "app.py",
                        "old": "    return open(p)",
                        "new": "    return open(p, encoding='utf-8')",
                        "rationale": "The encoding is always required.",
                    }
                ],
                "test_command": "pytest -q",
            },
        ),
    )
    agent = PatchAgent(model, deps)
    findings = Findings(findings=[])

    plan = asyncio.run(agent.plan(findings, "The encoding is missing."))

    assert isinstance(plan, PatchPlan)
    assert plan.edits[0].path == "app.py"
    assert plan.test_command == "pytest -q"


def test_the_patch_agent_is_told_what_went_wrong(deps: RepoDeps) -> None:
    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(prompt_text(messages))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_result",
                    {"summary": "Try again.", "edits": [], "test_command": "pytest -q"},
                )
            ]
        )

    agent = PatchAgent(FunctionModel(respond), deps)

    asyncio.run(
        agent.plan(
            Findings(findings=[]),
            "The encoding is missing.",
            "The tests failed: the encoding is still missing.",
        )
    )

    assert "The tests failed" in seen[0]


def test_the_verify_agent_judges_the_fix(deps: RepoDeps) -> None:
    model = scripted(
        (
            "output",
            "",
            {
                "passed": True,
                "summary": "The tests pass and the encoding is now required.",
                "command": "pytest -q",
                "remaining_issues": [],
            },
        ),
    )
    agent = VerifyAgent(model, deps)
    plan = PatchPlan(summary="Fix the encoding.")

    verification = asyncio.run(agent.check(plan, "exit code: 0", 1))

    assert isinstance(verification, Verification)
    assert verification.passed
    assert verification.remaining_issues == []
