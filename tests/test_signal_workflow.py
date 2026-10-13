from pathlib import Path

import yaml


def test_signal_workflow_only_collects_and_persists_private_state() -> None:
    path = Path("examples/workflows/signal-collection.yml")
    workflow = yaml.safe_load(path.read_text())
    serialized = path.read_text()

    assert workflow[True]["schedule"] == [{"cron": "17 */6 * * *"}]
    assert "news-assistant collect-signals" in serialized
    assert "send-email" not in serialized
    assert "personal-news-digest" not in serialized
    assert "news-assistant build" not in serialized
    assert '--state-dir "$RUNNER_TEMP/news-state"' in serialized
    assert 'cd "$RUNNER_TEMP/news-state"' in serialized
