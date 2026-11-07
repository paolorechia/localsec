"""Tests for the container sandbox."""

from __future__ import annotations

import stat
from pathlib import Path

from localsec.runtime.sandbox import DockerSandbox


def _fake_docker(
    directory: Path,
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> Path:
    """Write a script that stands in for the docker command.

    It records the arguments it was given and copies the workspace it was
    pointed at, so a test can look at what the container would have seen.
    """
    args_file = directory / "args.txt"
    captured = directory / "captured"
    script = directory / "docker"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" >> "{args_file}"\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        '    *:/work) src="${arg%:/work}";;\n'
        "  esac\n"
        "done\n"
        f'[ -n "$src" ] && cp -R "$src" "{captured}"\n'
        f"printf '%s' '{stdout}'\n"
        f"printf '%s' '{stderr}' >&2\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_the_command_runs_the_script_in_the_workspace() -> None:
    sandbox = DockerSandbox(image="python:3.12-slim")

    args = sandbox.command("pytest -q", workspace="/tmp/work")

    assert args[:5] == ["docker", "run", "--rm", "-v", "/tmp/work:/work"]
    assert args[5:7] == ["-w", "/work"]
    assert args[-4:] == ["python:3.12-slim", "sh", "-c", "pytest -q"]


def test_the_network_can_be_cut_off() -> None:
    sandbox = DockerSandbox(network="none")

    args = sandbox.command("pytest -q", workspace="/tmp/work")

    assert "--network" in args
    assert args[args.index("--network") + 1] == "none"


def test_a_run_copies_the_workspace_and_runs_the_command(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path, exit_code=0, stdout="2 passed\n")
    workspace = tmp_path / "repo"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "HEAD").write_text("ref: main\n", encoding="utf-8")

    sandbox = DockerSandbox(docker=str(docker))
    result = sandbox.run("pytest -q", workspace=workspace)

    assert result.ok
    assert result.exit_code == 0
    assert result.stdout == "2 passed\n"
    assert result.duration >= 0

    recorded = (tmp_path / "args.txt").read_text(encoding="utf-8").splitlines()
    assert recorded[-4:] == ["python:3.12-slim", "sh", "-c", "pytest -q"]


def test_a_failing_command_keeps_its_output(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path, exit_code=1, stdout="out", stderr="boom")
    workspace = tmp_path / "repo"
    workspace.mkdir()

    sandbox = DockerSandbox(docker=str(docker))
    result = sandbox.run("pytest -q", workspace=workspace)

    assert not result.ok
    assert result.exit_code == 1
    assert result.stdout == "out"
    assert result.stderr == "boom"


def test_the_workspace_is_copied_without_the_git_directory(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path)
    workspace = tmp_path / "repo"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "HEAD").write_text("ref: main\n", encoding="utf-8")
    (workspace / "keep.py").write_text("x = 1\n", encoding="utf-8")

    sandbox = DockerSandbox(docker=str(docker))
    sandbox.run("ls", workspace=workspace)

    captured = tmp_path / "captured"
    assert (captured / "keep.py").exists()
    assert not (captured / ".git").exists()


def test_the_runtime_is_reported_as_missing_when_it_is_not_there() -> None:
    sandbox = DockerSandbox(docker="definitely-not-docker")

    assert sandbox.available() is False


def test_the_runtime_is_reported_as_present_when_it_is(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path)

    assert DockerSandbox(docker=str(docker)).available() is True


def test_a_command_that_times_out_is_reported(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path)
    docker.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    workspace = tmp_path / "repo"
    workspace.mkdir()

    sandbox = DockerSandbox(docker=str(docker), timeout=0.2)
    result = sandbox.run("sleep 5", workspace=workspace)

    assert result.exit_code == 124
    assert "timed out" in result.stderr
