"""Agent tools, and the policy that says which ones change the world.

A tool is a plain function. The `@tool` decorator records whether it is
read-only or mutating, and the registry hands the agent a wrapped version
that asks the approval gate before a mutating tool runs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from localsec.runtime.approvals import (
    ApprovalDeniedError,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalRequest,
)

__all__ = ["ToolRegistry", "ToolSpec", "tool"]

P = ParamSpec("P")
R = TypeVar("R")

_POLICY_ATTRIBUTE = "localsec_policy"
_ACTION_ATTRIBUTE = "localsec_action"


def tool(
    *,
    policy: ApprovalPolicy,
    action: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as an agent tool, with the policy it runs under.

    A `READ_ONLY` tool only observes. A `MUTATING` tool changes the world,
    and the approval gate is asked before it runs.

    Args:
        policy: whether the tool changes anything.
        action: how to describe it to the operator, e.g. "write a file".
    """

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        setattr(func, _POLICY_ATTRIBUTE, policy)
        setattr(func, _ACTION_ATTRIBUTE, action or func.__name__.replace("_", " "))
        return func

    return decorate


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool, and what it does."""

    name: str
    policy: ApprovalPolicy
    action: str
    func: Callable[..., Any]


def _describe(arguments: dict[str, Any]) -> str:
    """Render a tool's arguments for the operator."""
    if not arguments:
        return ""
    rendered = ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    return rendered if len(rendered) <= 400 else rendered[:397] + "..."


class ToolRegistry:
    """The tools an agent may use."""

    def __init__(self, gate: ApprovalGate | None = None) -> None:
        """Open a registry. Tools are gated by `gate`, when one is given."""
        self._tools: dict[str, ToolSpec] = {}
        self._gate = gate

    def add(self, func: Callable[..., Any], *, name: str | None = None) -> ToolSpec:
        """Add a tool, and return what was recorded for it."""
        tool_name = name or func.__name__
        policy = getattr(func, _POLICY_ATTRIBUTE, ApprovalPolicy.READ_ONLY)
        action = getattr(func, _ACTION_ATTRIBUTE, tool_name.replace("_", " "))
        spec = ToolSpec(name=tool_name, policy=policy, action=action, func=func)
        self._tools[tool_name] = spec
        return spec

    def extend(self, funcs: list[Callable[..., Any]]) -> None:
        """Add several tools."""
        for func in funcs:
            self.add(func)

    def specs(self) -> list[ToolSpec]:
        """Return the tools recorded."""
        return list(self._tools.values())

    def functions(self) -> list[Callable[..., Any]]:
        """Return the tools to hand to an agent, gated where needed."""
        return [self._gated(spec) for spec in self._tools.values()]

    def function(self, name: str) -> Callable[..., Any]:
        """Return one tool, gated where needed.

        This is how a run calls a tool itself, rather than leaving it to the
        model: the gate is asked in exactly the same way.
        """
        return self._gated(self._tools[name])

    def _gated(self, spec: ToolSpec) -> Callable[..., Any]:
        """Return the tool, asking the gate first when it changes anything."""
        gate = self._gate
        if gate is None or spec.policy is ApprovalPolicy.READ_ONLY:
            return spec.func

        @functools.wraps(spec.func)
        def wrapper(*args: object, **kwargs: object) -> object:
            request = ApprovalRequest(
                tool=spec.name,
                action=spec.action,
                detail=_describe(kwargs),
                arguments=kwargs,
            )
            try:
                gate.check(request)
            except ApprovalDeniedError as denial:
                # The model is told the action was declined, so it can try
                # something else rather than failing the whole run.
                return f"DECLINED: {denial}"
            return spec.func(*args, **kwargs)

        return wrapper
