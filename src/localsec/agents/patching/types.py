"""What the patching agents produce.

Each of these is a structured result the model fills in. The local server
constrains the model to a grammar built from the schema, so the fields below
are what actually comes back -- not a shape the code has to hope for and
scrape out of prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Edit", "Finding", "Findings", "PatchPlan", "Verification"]


@dataclass
class Finding:
    """A vulnerability, and where it lives."""

    title: str
    severity: str
    file: str
    line: int | None = None
    cve: str | None = None
    description: str = ""


@dataclass
class Findings:
    """The findings of one triage.

    A container rather than a bare list, because the model fills in one
    structured object and the grammar is simpler that way.
    """

    findings: list[Finding] = field(default_factory=list)
    notes: str = ""


@dataclass
class Edit:
    """One change to one file."""

    path: str
    old: str
    new: str
    rationale: str = ""


@dataclass
class PatchPlan:
    """The changes that fix the findings."""

    summary: str = ""
    edits: list[Edit] = field(default_factory=list)
    test_command: str = ""


@dataclass
class Verification:
    """Whether the fix holds."""

    passed: bool
    summary: str
    command: str = ""
    remaining_issues: list[str] = field(default_factory=list)
