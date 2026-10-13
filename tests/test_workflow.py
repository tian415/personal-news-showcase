import shlex
from pathlib import Path

import yaml

WORKFLOW_PATH = Path("examples/workflows/daily-digest.yml")


def _steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return workflow["jobs"]["build"]["steps"]


def test_workflow_audits_publishes_then_sends() -> None:
    steps = _steps()
    names = [step.get("name", step.get("uses", "")) for step in steps]

    assert names.index("Build public digest") < names.index("Audit public digest")
    assert names.index("Audit public digest") < names.index(
        "Publish public digest"
    )
    assert names.index("Checkout public digest repository") < names.index(
        "Publish public digest"
    )
    assert names.index("Publish public digest") < names.index("Send Gmail digest")


def test_workflow_recovers_confirmed_deliveries_before_building_and_persists_before_publish():
    steps = _steps()
    names = [step.get("name") for step in steps]
    recovery = names.index("Recover confirmed web deliveries")
    assert names.index("Initialize or checkout private news state") < recovery
    assert names.index("Checkout public digest repository") < recovery
    assert recovery < names.index("Build public digest")
    assert names.index("Build public digest") < names.index("Persist private news state")
    assert names.index("Persist private news state") < names.index("Publish public digest")
    command = steps[recovery]["run"]
    assert "news-assistant reconcile-deliveries" in command
    assert '--state-dir "$RUNNER_TEMP/news-state"' in command
    assert "--published digest-site" in command


def test_workflow_preserves_and_audits_history_before_publish() -> None:
    steps = _steps()
    names = [step.get("name") for step in steps]
    restore_index = names.index("Preserve and audit published history")
    assert names.index("Checkout public digest repository") < restore_index
    assert restore_index < names.index("Publish public digest")
    assert "news-assistant preserve-history" in steps[restore_index]["run"]


def test_workflow_targets_public_repository_and_builds_once() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    env = workflow["jobs"]["build"]["env"]

    assert env["DIGEST_REPOSITORY"] == (
        "${{ vars.DIGEST_REPOSITORY || format('{0}/personal-news-digest', github.repository_owner) }}"
    )
    assert "{}/personal-news-digest".format("example-user") == "example-user/personal-news-digest"
    assert env["DIGEST_BASE_URL"] == (
        "${{ vars.DIGEST_BASE_URL || format('https://{0}.github.io/personal-news-digest/', github.repository_owner) }}"
    )
    assert "https://{}.github.io/personal-news-digest/".format("example-user") == (
        "https://example-user.github.io/personal-news-digest/"
    )
    serialized = WORKFLOW_PATH.read_text()
    assert "secrets.DIGEST_REPO_TOKEN" in serialized
    assert "actions/configure-pages" not in serialized
    assert "actions/upload-pages-artifact" not in serialized
    assert "actions/deploy-pages" not in serialized
    assert serialized.count("news-assistant build") == 1
    assert "news-assistant send-email" in serialized
    assert 'repository: ${{ env.DIGEST_REPOSITORY }}' in serialized
    assert '--digest-url "$DIGEST_BASE_URL"' in serialized


def test_workflow_builds_private_email_artifact_and_does_not_publish_it() -> None:
    serialized = WORKFLOW_PATH.read_text()
    assert "--email-output work/digest-email.html" in serialized
    assert "--input work/digest-email.html" in serialized
    assert "rsync -a --delete --exclude .git/ public/ digest-site/" in serialized
    assert serialized.count("news-assistant build") == 1


def test_workflow_keeps_daily_schedule_and_serializes_private_shadow_state() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())

    assert workflow[True]["schedule"] == [{"cron": "17 23 * * *"}]
    assert workflow["permissions"] == {"contents": "write"}
    assert workflow["concurrency"] == {"group": "news-state", "cancel-in-progress": False}
    assert "Initialize or checkout private news state" in [step.get("name") for step in _steps()]
    assert "Persist private news state" in [step.get("name") for step in _steps()]
    assert "--selection-engine layered_shadow_v1" in WORKFLOW_PATH.read_text()
    assert '--state-dir "$RUNNER_TEMP/news-state"' in WORKFLOW_PATH.read_text()
    assert "environment" not in workflow["jobs"]["build"]


def test_workflow_validates_source_before_checking_out_private_state() -> None:
    steps = _steps()
    names = [step.get("name", step.get("uses", "")) for step in steps]

    state_index = names.index("Initialize or checkout private news state")
    assert names.index("astral-sh/setup-uv@v6") < state_index
    runs = [step.get("run") for step in steps]
    assert runs.index("uv sync --locked") < state_index
    assert runs.index("uv run pytest -v") < state_index
    assert runs.index("uv run ruff check .") < state_index


def test_reruns_use_distinct_delivery_ids_shared_by_build_and_mark() -> None:
    commands = {step.get("name"): step.get("run", "") for step in _steps()}
    ids = []
    for attempt in (1, 2):
        parsed_ids = []
        for name in ("Build public digest", "Mark web delivery in private state"):
            command = commands[name].replace("${{ github.run_id }}", "1234").replace(
                "${{ github.run_attempt }}", str(attempt)
            )
            args = shlex.split(command)
            parsed_ids.append(args[args.index("--run-id") + 1])
        assert parsed_ids[0] == parsed_ids[1]
        ids.append(parsed_ids[0])
    assert ids == ["1234-1", "1234-2"]


def test_manual_dry_run_skips_all_external_writes_and_public_checkout_dependents() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert workflow[True]["workflow_dispatch"]["inputs"]["dry_run"]["type"] == "boolean"
    guarded = {
        "Initialize or checkout private news state", "Checkout public digest repository",
        "Recover confirmed web deliveries", "Persist private news state",
        "Preserve and audit published history", "Publish public digest",
        "Mark web delivery in private state", "Persist delivery state", "Send Gmail digest",
    }
    steps = {step.get("name"): step for step in _steps()}
    for name in guarded:
        assert steps[name]["if"] == "${{ inputs.dry_run != true }}", name
