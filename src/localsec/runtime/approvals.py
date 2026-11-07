"""The gate that state-changing actions pass through.

Every tool is either read-only or mutating. A read-only tool observes; a
mutating one changes the world -- it writes a file, runs a command, or
sends a request that alters something. Mutating tools ask the gate first,
and a gate with no approver denies: an unattended agent cannot change
anything by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from localsec.runtime.audit import AuditLog

__all__ = [
    "ApprovalDeniedError",
    "ApprovalGate",
    "ApprovalPolicy",
    "ApprovalRequest",
    "Decision",
    "console_approver",
]


class ApprovalPolicy(StrEnum):
    """What a tool does to the world."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"


class Decision(StrEnum):
    """What the operator decided."""

    ALLOW = "allow"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A request to do something that changes the world."""

    tool: str
    action: str
    detail: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


class ApprovalDeniedError(Exception):
    """The operator declined an action."""


Approver = Callable[[ApprovalRequest], Decision]


def console_approver(request: ApprovalRequest) -> Decision:
    """Ask the operator on the terminal.

    Reads one word: `y` allows the action, `a` allows it for the rest of
    the session, `n` declines, and anything else declines as well.
    """
    print(f"\nThe agent wants to {request.action}:")
    print(f"  {request.detail}")
    for name, value in request.arguments.items():
        print(f"  {name}: {value}")
    answer = input("Allow? [y] once  [a] always  [n] no: ").strip().lower()
    if answer == "a":
        return Decision.ALLOW_ALWAYS
    if answer == "y":
        return Decision.ALLOW
    return Decision.DENY


class ApprovalGate:
    """Decides whether a state-changing action may happen."""

    def __init__(
        self,
        approver: Approver | None = None,
        *,
        audit: AuditLog | None = None,
    ) -> None:
        """Open the gate. Without an approver, everything is declined."""
        self._approver = approver
        self._audit = audit
        self._always: set[str] = set()

    @property
    def always_allowed(self) -> frozenset[str]:
        """Return the tools that have been allowed for the whole session."""
        return frozenset(self._always)

    def check(self, request: ApprovalRequest) -> None:
        """Approve the action, or raise `ApprovalDeniedError`."""
        granted = request.tool in self._always
        if not granted:
            decision = self._approver(request) if self._approver else Decision.DENY
            granted = decision in (Decision.ALLOW, Decision.ALLOW_ALWAYS)
            if decision is Decision.ALLOW_ALWAYS:
                self._always.add(request.tool)
        if self._audit:
            self._audit.record(
                "approval",
                tool=request.tool,
                action=request.action,
                detail=request.detail,
                granted=granted,
            )
        if not granted:
            raise ApprovalDeniedError(f"the operator declined {request.action}: {request.detail}")
