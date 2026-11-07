"""Tests for the tool registry."""

from __future__ import annotations

from pathlib import Path

from localsec.runtime.approvals import ApprovalGate, ApprovalPolicy, ApprovalRequest, Decision
from localsec.runtime.tools import ToolRegistry, tool


@tool(policy=ApprovalPolicy.READ_ONLY)
def read_file(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool(policy=ApprovalPolicy.MUTATING, action="write a file")
def write_file(path: str, content: str) -> str:
    """Write a file."""
    return f"wrote {len(content)} bytes to {path}"


def test_a_tool_records_its_policy() -> None:
    registry = ToolRegistry()
    registry.extend([read_file, write_file])

    specs = {spec.name: spec for spec in registry.specs()}
    assert specs["read_file"].policy is ApprovalPolicy.READ_ONLY
    assert specs["write_file"].policy is ApprovalPolicy.MUTATING
    assert specs["write_file"].action == "write a file"


def test_a_tool_without_a_policy_is_read_only() -> None:
    def plain() -> str:
        """A plain function."""
        return "ok"

    registry = ToolRegistry()
    registry.add(plain)

    assert registry.specs()[0].policy is ApprovalPolicy.READ_ONLY


def test_a_read_only_tool_runs_without_asking() -> None:
    registry = ToolRegistry(ApprovalGate(lambda _request: Decision.DENY))
    registry.extend([read_file])

    (func,) = registry.functions()

    assert func(path="a.py") == "contents of a.py"


def test_a_mutating_tool_asks_the_gate() -> None:
    asked: list[str] = []

    def approver(request: object) -> Decision:
        asked.append(getattr(request, "tool", "?"))
        return Decision.ALLOW

    registry = ToolRegistry(ApprovalGate(approver))
    registry.extend([write_file])

    (func,) = registry.functions()
    result = func(path="a.py", content="x")

    assert asked == ["write_file"]
    assert result == "wrote 1 bytes to a.py"


def test_a_declined_tool_tells_the_model_rather_than_failing() -> None:
    registry = ToolRegistry(ApprovalGate(lambda _request: Decision.DENY))
    registry.extend([write_file])

    (func,) = registry.functions()
    result = func(path="a.py", content="x")

    assert isinstance(result, str)
    assert result.startswith("DECLINED:")
    assert "write a file" in result


def test_the_gated_tool_keeps_the_original_signature() -> None:
    registry = ToolRegistry(ApprovalGate(lambda _request: Decision.ALLOW))
    registry.extend([write_file])

    (func,) = registry.functions()

    assert func.__name__ == "write_file"
    assert func.__doc__ == "Write a file."


def test_the_arguments_reach_the_operator() -> None:
    seen: list[ApprovalRequest] = []

    def approver(request: ApprovalRequest) -> Decision:
        seen.append(request)
        return Decision.ALLOW

    registry = ToolRegistry(ApprovalGate(approver))
    registry.extend([write_file])

    (func,) = registry.functions()
    func(path="a.py", content="x")

    request = seen[0]
    assert request.detail == "path='a.py', content='x'"
    assert request.arguments == {"path": "a.py", "content": "x"}


def test_a_registry_can_be_used_without_a_gate(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.extend([write_file])

    (func,) = registry.functions()

    assert func(path="a.py", content="x") == "wrote 1 bytes to a.py"
