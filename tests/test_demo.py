from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime

import pytest

from news_assistant.demo import run_demo
from news_assistant.feedback import public_event_ref
from news_assistant.models import DeliveryRecord
from news_assistant.publication import audit_public_directory
from news_assistant.state import StateStore

NOW = datetime(2026, 9, 28, 8, tzinfo=UTC)


def _forbid_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import smtplib

    import httpx

    from news_assistant.deepseek import DeepSeekClient

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline demo attempted an external call")

    monkeypatch.setattr(httpx.Client, "request", blocked)
    monkeypatch.setattr(smtplib, "SMTP_SSL", blocked)
    monkeypatch.setattr(DeepSeekClient, "enrich", blocked)


def test_run_demo_builds_audited_offline_feedback_loop(tmp_path, monkeypatch) -> None:
    _forbid_external_calls(monkeypatch)
    root = tmp_path / "demo"

    report = run_demo(root, now=NOW)

    assert report["schema_version"] == "offline-demo-v1"
    assert report["labels"] == {
        "content": "mock",
        "feedback": "simulated",
        "publication": "simulated_local_copy",
    }
    assert report["counts"] == {
        "input_articles": 10,
        "round_1_selected": 10,
        "round_2_selected": 10,
        "feedback_records": 2,
        "tracked_events": 1,
        "confirmed_simulated_publications": 1,
    }
    assert report["legacy_selection_unchanged"] is True
    assert report["stages"]["simulated_publication"] == "hash_confirmed"
    assert report["stages"]["feedback_import"] == "idempotent"
    assert report["stages"]["email_preview"] == "rendered_not_sent"
    assert report["stages"]["layered_evaluation"] == "not_ready"
    assert report["evaluation"]["release_readiness"]["status"] == "not_ready"
    assert report["feedback_effect"]["public_selection"] == "legacy_unchanged"
    assert report["feedback_effect"]["feedback_records"] == 2
    assert report["feedback_effect"]["tracked_event_ids"]
    assert report["feedback_effect_change"] == {
        "round_1_feedback_records": 0,
        "round_2_feedback_records": 2,
        "effective_preferences_changed": True,
        "tracked_events_delta": 1,
    }
    assert report["layer_coverage"]["empty_layers"]
    assert report["artifacts"]["visibility"] == "private_demo_root"
    probe = report["feedback_scoring_probe"]
    assert probe["candidate_id"] == "private-feedback-probe"
    assert probe["actual_relevance_after"] > probe["control_relevance_before"]
    assert probe["proof_scope"] == "feedback_entered_scoring_not_recommendation_quality"

    expected = {
        "public/index.html",
        "public-next/index.html",
        "published/index.html",
        "private-state/feedback-state.json",
        "email.html",
        "simulation-feedback.json",
        "report.json",
    }
    assert all((root / relative).is_file() for relative in expected)
    assert json.loads((root / "report.json").read_text()) == report
    assert "演示数据" in (root / "public/index.html").read_text()
    assert "example.com" in (root / "public/index.html").read_text()
    private_title = "合成演示：私有候选发布反馈验证工具"
    assert private_title not in (root / "public/index.html").read_text()
    assert private_title not in (root / "public-next/index.html").read_text()
    topics = Counter(
        re.findall(r'data-topic="([a-z]+)"', (root / "public/index.html").read_text())
    )
    assert sum(topics[topic] for topic in ("ai", "tech", "science")) == 5
    assert sum(topics[topic] for topic in ("business", "markets")) == 3
    assert topics["exploration"] == 2
    assert audit_public_directory(root / "public")
    assert audit_public_directory(root / "public-next")
    assert audit_public_directory(root / "published")
    second_run = StateStore(root / "private-state")._find_run_directory("demo-round-2")
    layered = json.loads((second_run / "layered-shadow.json").read_text())
    assert any(
        probe["candidate_id"] in item["event"]["candidate_ids"]
        for item in layered["items"]
    )


def test_demo_feedback_is_bound_to_confirmed_first_round_and_replay_is_stable(
    tmp_path,
) -> None:
    root = tmp_path / "demo"
    run_demo(root, now=NOW)
    store = StateStore(root / "private-state")

    deliveries = [
        DeliveryRecord.model_validate(value)
        for path in (root / "private-state/deliveries").glob("*/*/deliveries.jsonl")
        for value in store._read_jsonl(path)
    ]
    assert deliveries
    assert {delivery.run_id for delivery in deliveries} == {"demo-round-1"}
    assert all(delivery.event is not None for delivery in deliveries)
    assert all(delivery.event.event_id == delivery.event_id for delivery in deliveries)
    assert all(delivery.event.latest_update_id == delivery.update_id for delivery in deliveries)

    first_html = (root / "public/index.html").read_text()
    refs = set(re.findall(r'data-event-ref="([a-f0-9]{24})"', first_html))
    assert refs
    confirmed_refs = {
        public_event_ref(
            delivery.event_id,
            run_id=delivery.run_id,
            update_id=delivery.update_id,
        )
        for delivery in deliveries
    }
    payload = json.loads((root / "simulation-feedback.json").read_text())
    assert {entry["event_ref"] for entry in payload["events"]} <= refs == confirmed_refs
    state_path = root / "private-state/feedback-state.json"
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()

    from news_assistant.feedback_store import import_feedback_file

    import_feedback_file(root / "simulation-feedback.json", root / "private-state", now=NOW)
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == before


def test_run_demo_rejects_nonempty_output_without_overwriting(tmp_path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(ValueError, match="empty|non-empty|非空"):
        run_demo(root, now=NOW)

    assert sentinel.read_text() == "keep"
    assert list(root.iterdir()) == [sentinel]


def test_run_demo_requires_timezone_aware_now(tmp_path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        run_demo(tmp_path / "demo", now=NOW.replace(tzinfo=None))
