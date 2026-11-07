"""Tests for the workspace the patching agents work in."""

from __future__ import annotations

from pathlib import Path

import pytest

from localsec.agents.patching.workspace import (
    RepoDeps,
    Workspace,
    WorkspaceError,
    mutating_tools,
    read_only_tools,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A workspace with a couple of files in it."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("import os\n\n\ndef load(p):\n    return open(p)\n")
    (root / "README.md").write_text("# a project\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: main\n")
    return Workspace(root)


def test_a_file_can_be_read(workspace: Workspace) -> None:
    assert "def load(p):" in workspace.read("src/app.py")


def test_a_file_that_is_not_there_is_an_error(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="does not exist"):
        workspace.read("src/nope.py")


def test_a_path_that_leaves_the_workspace_is_refused(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.read("../../etc/passwd")


def test_an_absolute_path_is_refused(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="absolute"):
        workspace.read("/etc/passwd")


def test_a_symlink_out_of_the_workspace_is_refused(workspace: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("hunter2\n")
    (workspace.root / "link.txt").symlink_to(secret)

    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.read("link.txt")


def test_the_listing_leaves_the_hidden_directories_out(workspace: Workspace) -> None:
    listed = workspace.listing()

    assert listed == ["README.md", "src/app.py"]


def test_a_pattern_narrows_the_listing(workspace: Workspace) -> None:
    assert workspace.listing("*.md") == ["README.md"]


def test_the_search_finds_the_line(workspace: Workspace) -> None:
    hits = workspace.search("open(p)")

    assert hits == ["src/app.py:5: return open(p)"]


def test_the_search_that_finds_nothing_is_empty(workspace: Workspace) -> None:
    assert workspace.search("nothing here") == []


def test_a_file_can_be_written(workspace: Workspace) -> None:
    workspace.write("src/new.py", "x = 1\n")

    assert workspace.read("src/new.py") == "x = 1\n"


def test_a_write_cannot_leave_the_workspace(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.write("../escaped.py", "x = 1\n")


def test_an_edit_replaces_the_text(workspace: Workspace) -> None:
    workspace.edit("src/app.py", "return open(p)", "return open(p, encoding='utf-8')")

    assert "encoding='utf-8'" in workspace.read("src/app.py")


def test_an_edit_that_matches_nothing_is_an_error(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="does not contain"):
        workspace.edit("src/app.py", "return closed(p)", "return open(p)")


def test_an_edit_that_matches_twice_is_refused(workspace: Workspace) -> None:
    workspace.write("twice.py", "x = 1\nx = 1\n")

    with pytest.raises(WorkspaceError, match="2 times"):
        workspace.edit("twice.py", "x = 1", "x = 2")


def test_the_read_only_tools_do_not_change_anything(workspace: Workspace) -> None:
    deps = RepoDeps(workspace=workspace)
    list_files, read_file, search_code = read_only_tools(deps)

    assert "src/app.py" in list_files()
    assert "def load(p):" in read_file(path="src/app.py")
    assert search_code(query="open(p)").startswith("src/app.py:5:")


def test_a_read_of_a_missing_file_reports_the_error(workspace: Workspace) -> None:
    deps = RepoDeps(workspace=workspace)
    _, read_file, _ = read_only_tools(deps)

    assert read_file(path="nope.py").startswith("ERROR:")


def test_the_mutating_tools_report_errors_rather_than_raising(workspace: Workspace) -> None:
    deps = RepoDeps(workspace=workspace)
    _, edit_file, _ = mutating_tools(deps)

    result = edit_file(path="src/app.py", old="nothing", new="something")

    assert result.startswith("ERROR:")


def test_the_tests_cannot_run_without_a_sandbox(workspace: Workspace) -> None:
    deps = RepoDeps(workspace=workspace)
    _, _, run_tests = mutating_tools(deps)

    assert run_tests().startswith("ERROR:")
