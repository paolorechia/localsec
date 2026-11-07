"""Tests for the command line."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from helpers import FINDINGS, PLAN, VERIFIED, prompt_text
from localsec import cli


class _FakeSession:
    """A model session that does not talk to the server."""

    def __init__(self, model: FunctionModel) -> None:
        self.model = model

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tree with one file in it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def load(p):\n    return open(p)\n")
    return root


def test_the_patch_command_applies_the_fix(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str] = []
    payloads: list[dict[str, Any]] = [FINDINGS, PLAN, VERIFIED]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = payloads[len(seen)]
        seen.append(prompt_text(messages))
        return ModelResponse(parts=[ToolCallPart("final_result", payload)])

    monkeypatch.setattr(cli, "ModelSession", lambda: _FakeSession(FunctionModel(respond)))

    code = cli.main(
        [
            "patch",
            "The path is not checked.",
            "--workspace",
            str(workspace),
            "--yes",
            "--no-tests",
        ]
    )

    assert code == 0
    assert "encoding='utf-8'" in (workspace / "app.py").read_text()
    assert "applied:  app.py" in capsys.readouterr().out


def test_the_patch_command_reports_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    payloads: list[dict[str, Any]] = [FINDINGS, PLAN, VERIFIED]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = payloads.pop(0)
        return ModelResponse(parts=[ToolCallPart("final_result", payload)])

    monkeypatch.setattr(cli, "ModelSession", lambda: _FakeSession(FunctionModel(respond)))

    code = cli.main(
        [
            "patch",
            "The path is not checked.",
            "--workspace",
            str(workspace),
            "--no-tests",
        ]
    )

    # The gate declines when nobody is there to answer it.
    assert code == 1
    assert (workspace / "app.py").read_text() == "def load(p):\n    return open(p)\n"
