from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATHS = (
    Path("examples/workflows/daily-digest.yml"),
    Path("examples/workflows/signal-collection.yml"),
)
CLONE_URL = '"https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"'


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, message: str) -> None:
    _run("git", "add", "-A", cwd=repo)
    _run(
        "git",
        "-c",
        "user.name=Workflow Test",
        "-c",
        "user.email=workflow-test@example.invalid",
        "commit",
        "-m",
        message,
        cwd=repo,
    )


def _create_remote(tmp_path: Path, *, with_state_branch: bool) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    _run("git", "init", "-b", "main", cwd=source)
    (source / "src").mkdir()
    (source / "tests").mkdir()
    (source / "src" / "application.py").write_text("SOURCE = True\n")
    (source / "tests" / "test_application.py").write_text("def test_source(): pass\n")
    _commit(source, "source branch")

    if with_state_branch:
        _run("git", "switch", "--orphan", "news-state", cwd=source)
        (source / "state").mkdir()
        (source / "state" / "signals.jsonl").write_text('{"signal": 1}\n')
        _commit(source, "private state")
        _run("git", "switch", "main", cwd=source)

    remote = tmp_path / "remote.git"
    _run("git", "clone", "--bare", str(source), str(remote), cwd=tmp_path)
    if with_state_branch:
        _run("git", "push", str(remote), "news-state", cwd=source)
    return source, remote


def _workflow_step(path: Path, name: str) -> str:
    workflow = yaml.safe_load(path.read_text())
    job = next(iter(workflow["jobs"].values()))
    return next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == name
    )


def _execute_checkout(
    path: Path,
    *,
    source: Path,
    remote: Path,
    runner_temp: Path,
    break_remote_probe: bool = False,
) -> subprocess.CompletedProcess[str]:
    script = _workflow_step(
        path, "Initialize or checkout private news state"
    ).replace(CLONE_URL, shlex.quote(str(remote)))
    if break_remote_probe:
        probe = "git ls-remote --exit-code --heads origin news-state"
        broken_probe = (
            'git ls-remote --exit-code --heads "$RUNNER_TEMP/missing.git" news-state'
        )
        script = script.replace(probe, broken_probe)

    environment = {
        **os.environ,
        "GH_TOKEN": "test-token",
        "GITHUB_REPOSITORY": "test/example",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "RUNNER_TEMP": str(runner_temp),
    }
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _tracked_files(repo: Path) -> list[str]:
    result = _run("git", "ls-files", cwd=repo)
    return result.stdout.splitlines()


def _remote_files(remote: Path) -> list[str]:
    result = _run(
        "git",
        "--git-dir",
        str(remote),
        "ls-tree",
        "-r",
        "--name-only",
        "news-state",
        cwd=remote.parent,
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS)
def test_new_state_branch_is_initialized_outside_source_without_source_files(
    workflow_path: Path, tmp_path: Path
) -> None:
    source, remote = _create_remote(tmp_path, with_state_branch=False)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()

    result = _execute_checkout(
        workflow_path,
        source=source,
        remote=remote,
        runner_temp=runner_temp,
    )

    assert result.returncode == 0, result.stderr
    state_checkout = runner_temp / "news-state"
    assert _tracked_files(state_checkout) == [".gitkeep"]
    assert _remote_files(remote) == [".gitkeep"]
    assert not (state_checkout / "src").exists()
    assert not (state_checkout / "tests").exists()
    assert not (source / "news-state").exists()


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS)
def test_existing_state_branch_is_checked_out_without_source_files(
    workflow_path: Path, tmp_path: Path
) -> None:
    source, remote = _create_remote(tmp_path, with_state_branch=True)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()

    result = _execute_checkout(
        workflow_path,
        source=source,
        remote=remote,
        runner_temp=runner_temp,
    )

    assert result.returncode == 0, result.stderr
    state_checkout = runner_temp / "news-state"
    assert _tracked_files(state_checkout) == ["state/signals.jsonl"]
    assert _remote_files(remote) == ["state/signals.jsonl"]
    assert not (state_checkout / "src").exists()
    assert not (state_checkout / "tests").exists()
    assert not (source / "news-state").exists()


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS)
def test_remote_probe_failure_does_not_initialize_a_state_branch(
    workflow_path: Path, tmp_path: Path
) -> None:
    source, remote = _create_remote(tmp_path, with_state_branch=False)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()

    result = _execute_checkout(
        workflow_path,
        source=source,
        remote=remote,
        runner_temp=runner_temp,
        break_remote_probe=True,
    )

    assert result.returncode != 0
    assert not (runner_temp / "news-state" / ".gitkeep").exists()


@pytest.mark.parametrize(
    ("workflow_path", "persist_step"),
    (
        (WORKFLOW_PATHS[0], "Persist private news state"),
        (WORKFLOW_PATHS[1], "Persist private signal state"),
    ),
)
def test_new_state_branch_supports_later_plain_push(
    workflow_path: Path, persist_step: str, tmp_path: Path
) -> None:
    source, remote = _create_remote(tmp_path, with_state_branch=False)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    checkout = _execute_checkout(
        workflow_path,
        source=source,
        remote=remote,
        runner_temp=runner_temp,
    )
    assert checkout.returncode == 0, checkout.stderr
    (runner_temp / "news-state" / "state.jsonl").write_text('{"state": 1}\n')

    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            _workflow_step(workflow_path, persist_step),
        ],
        cwd=source,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "RUNNER_TEMP": str(runner_temp),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert _remote_files(remote) == [".gitkeep", "state.jsonl"]
