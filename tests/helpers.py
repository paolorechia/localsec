"""Helpers the tests share."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

FINDINGS = {
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
}
PLAN = {
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
}
VERIFIED = {
    "passed": True,
    "summary": "The tests pass.",
    "command": "pytest -q",
    "remaining_issues": [],
}
NOT_VERIFIED = {
    "passed": False,
    "summary": "The tests still fail.",
    "command": "pytest -q",
    "remaining_issues": ["the encoding is still missing"],
}


def prompt_text(messages: list[ModelMessage]) -> str:
    """Return the text of the user prompt in the last request."""
    for part in messages[-1].parts:
        if isinstance(part, UserPromptPart):
            return str(part.content)
    return ""


def scripted_output(*payloads: dict[str, Any]) -> FunctionModel:
    """Return a model that answers each agent in turn with one payload."""
    remaining = list(payloads)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = remaining.pop(0)
        return ModelResponse(parts=[ToolCallPart("final_result", payload)])

    return FunctionModel(respond)
