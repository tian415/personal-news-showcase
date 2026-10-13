from pathlib import Path

import yaml

WORKFLOW_DIRECTORY = Path("examples/workflows")
STATE_WORKFLOWS = (
    WORKFLOW_DIRECTORY / "daily-digest.yml",
    WORKFLOW_DIRECTORY / "signal-collection.yml",
)


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_every_state_workflow_rejects_public_repositories_before_state_access() -> None:
    for path in STATE_WORKFLOWS:
        job = next(iter(_workflow(path)["jobs"].values()))
        steps = job["steps"]
        guard_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Require private repository for news state"
        )
        state_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Initialize or checkout private news state"
        )
        guard = steps[guard_index]

        assert guard_index < state_index
        assert guard["if"] == "${{ github.event.repository.private != true }}"
        assert "private running repository or a local workspace" in guard["run"].lower()
        assert "exit 1" in guard["run"]


def test_daily_digest_repository_and_base_url_have_owner_derived_defaults() -> None:
    workflow = _workflow(WORKFLOW_DIRECTORY / "daily-digest.yml")
    env = workflow["jobs"]["build"]["env"]

    assert env["DIGEST_REPOSITORY"] == (
        "${{ vars.DIGEST_REPOSITORY || format('{0}/personal-news-digest', github.repository_owner) }}"
    )
    assert env["DIGEST_BASE_URL"] == (
        "${{ vars.DIGEST_BASE_URL || format('https://{0}.github.io/personal-news-digest/', github.repository_owner) }}"
    )

    checkout = next(
        step
        for step in workflow["jobs"]["build"]["steps"]
        if step.get("name") == "Checkout public digest repository"
    )
    assert checkout["with"]["repository"] == "${{ env.DIGEST_REPOSITORY }}"


def test_daily_digest_url_comes_from_one_shell_environment_value() -> None:
    serialized = (WORKFLOW_DIRECTORY / "daily-digest.yml").read_text()
    workflow = _workflow(WORKFLOW_DIRECTORY / "daily-digest.yml")
    steps = workflow["jobs"]["build"]["steps"]
    build = next(step for step in steps if step.get("name") == "Build public digest")
    email = next(step for step in steps if step.get("name") == "Send Gmail digest")
    expected_arg = '--digest-url "$DIGEST_BASE_URL"'

    assert build["run"].count(expected_arg) == 1
    assert email["run"].count(expected_arg) == 1
    assert serialized.count(expected_arg) == 2
    assert "${{ vars.DIGEST_BASE_URL" not in build["run"]
    assert "${{ vars.DIGEST_BASE_URL" not in email["run"]
