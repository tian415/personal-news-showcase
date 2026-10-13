import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from news_assistant.models import (
    CandidateArticle,
    CoverageDomain,
    DecisionRecord,
    DeliveryRecord,
    Event,
    ProcessingStage,
    ReasonCode,
    RunRecord,
    SignalSnapshot,
    SourceHealthRecord,
    Topic,
)
from news_assistant.state import StateStore


def _candidate_at(candidate_id: str, timestamp: datetime) -> CandidateArticle:
    return CandidateArticle(
        id=candidate_id,
        candidate_id=candidate_id,
        title=candidate_id,
        url=f"https://example.com/{candidate_id}",
        source="Example",
        source_id="example",
        language="en",
        published_at=timestamp,
        topic=Topic.AI,
    )


def test_state_store_writes_auditable_private_jsonl_and_loads_recent_event(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    candidate = CandidateArticle(
        id="article-1",
        candidate_id="candidate-1",
        title="Official announcement",
        url="https://example.com/announcement",
        source="Example",
        source_id="example",
        language="en",
        published_at=now,
        topic=Topic.AI,
    )
    event = Event(
        event_id="evt-1",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title=candidate.title,
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=[candidate.candidate_id],
    )
    run = RunRecord(run_id="run-1", started_at=now, selection_engine="shadow_event_v1")
    decision = DecisionRecord(
        run_id=run.run_id,
        event_id=event.event_id,
        stage=ProcessingStage.SELECTED,
        selected=True,
        primary_reason=ReasonCode.MAJOR_GUARANTEE,
        occurred_at=now,
    )

    store = StateStore(tmp_path / "state")
    store.write_run(run, [candidate], [decision], [event])
    assert store.load_recent_events(now, days=90) == []
    store.mark_candidates_delivered(run.run_id, [candidate.candidate_id], now)

    assert [item.event_id for item in store.load_recent_events(now, days=90)] == ["evt-1"]
    assert (tmp_path / "state" / "runs" / "2026" / "08" / "13" / "run-1" / "candidates.jsonl").exists()


def test_write_run_saves_event_snapshot_and_assigns_run_id_to_decisions(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    run = RunRecord(run_id="run-snapshot", started_at=now, selection_engine="shadow_event_v1")
    event = Event(
        event_id="evt-snapshot",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Snapshot",
        first_event_time=now,
        latest_event_time=now,
    )
    decision = DecisionRecord(
        event_id=event.event_id,
        stage=ProcessingStage.SELECTED,
        selected=True,
        primary_reason=ReasonCode.MAJOR_GUARANTEE,
        occurred_at=now,
    )
    store = StateStore(tmp_path / "state")

    store.write_run(run, [], [decision], [event])

    run_dir = store.root / "runs" / "2026" / "08" / "13" / run.run_id
    saved_decision = json.loads((run_dir / "decisions.jsonl").read_text())
    saved_event = json.loads((run_dir / "events.jsonl").read_text())
    assert saved_decision["run_id"] == run.run_id
    assert saved_event["event_id"] == event.event_id


def test_delivery_uses_only_the_requested_run_snapshot(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    first_run = RunRecord(run_id="run-first", started_at=now, selection_engine="shadow_event_v1")
    second_run = RunRecord(
        run_id="run-second",
        started_at=now + timedelta(minutes=1),
        selection_engine="shadow_event_v1",
    )
    first_event = Event(
        event_id="evt-first",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="First run",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["shared-candidate"],
    )
    second_event = first_event.model_copy(
        update={"event_id": "evt-second", "title": "Second run"}
    )
    candidate = _candidate_at("shared-candidate", now)
    store.write_run(first_run, [candidate], [], [first_event])
    store.write_run(second_run, [candidate], [], [second_event])

    store.mark_candidates_delivered(first_run.run_id, ["shared-candidate"], now)

    assert [event.event_id for event in store.load_recent_events(now)] == ["evt-first"]


def test_load_recent_events_returns_only_the_exact_delivered_event_version(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    published_run = RunRecord(
        run_id="run-published", started_at=now, selection_engine="shadow_event_v1"
    )
    unpublished_run = RunRecord(
        run_id="run-unpublished",
        started_at=now + timedelta(hours=1),
        selection_engine="shadow_event_v1",
    )
    published = Event(
        event_id="evt-versioned",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Published version",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["published-candidate"],
        latest_update_id="update-published",
    )
    unpublished = published.model_copy(
        update={
            "title": "Unpublished newer version",
            "latest_event_time": now + timedelta(hours=1),
            "latest_update_id": "update-unpublished",
        }
    )
    candidate = _candidate_at("published-candidate", now)
    store.write_run(published_run, [candidate], [], [published])
    store.mark_candidates_delivered(
        published_run.run_id, ["published-candidate"], now + timedelta(minutes=1)
    )
    store.write_run(unpublished_run, [candidate], [], [unpublished])

    assert store.load_recent_events(now + timedelta(hours=2)) == [published]


def test_delivered_run_rejects_changed_rewrite_without_mutating_snapshot(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-immutable", started_at=now, selection_engine="shadow_event_v1")
    published = Event(
        event_id="evt-immutable",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Published",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["candidate-immutable"],
    )
    candidate = _candidate_at("candidate-immutable", now)
    store.write_run(run, [candidate], [], [published])
    store.mark_candidates_delivered(run.run_id, ["candidate-immutable"], now)
    event_path = store._run_directory(run) / "events.jsonl"
    original_contents = event_path.read_bytes()

    with pytest.raises(ValueError, match="delivered run"):
        store.write_run(
            run,
            [candidate],
            [],
            [published.model_copy(update={"title": "Unpublished replacement"})],
        )

    assert event_path.read_bytes() == original_contents
    assert store.load_recent_events(now) == [published]


def test_delivered_run_allows_identical_idempotent_rewrite(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-retry", started_at=now, selection_engine="shadow_event_v1")
    event = Event(
        event_id="evt-retry",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Retry",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["candidate-retry"],
    )
    candidate = _candidate_at("candidate-retry", now)
    store.write_run(run, [candidate], [], [event])
    store.mark_candidates_delivered(run.run_id, ["candidate-retry"], now)

    store.write_run(run, [candidate], [], [event])

    global_events = store.root / "events" / "2026" / "08" / "13" / "events.jsonl"
    assert len(store._read_jsonl(global_events)) == 1


def test_load_recent_events_keeps_newer_version_when_older_delivery_is_appended_later(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    old_run = RunRecord(run_id="run-old", started_at=now, selection_engine="shadow_event_v1")
    new_run = RunRecord(
        run_id="run-new",
        started_at=now + timedelta(hours=1),
        selection_engine="shadow_event_v1",
    )
    old = Event(
        event_id="evt-reordered",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Old version",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["candidate-old"],
        latest_update_id="update-old",
    )
    new = old.model_copy(
        update={
            "title": "New version",
            "latest_event_time": now + timedelta(hours=1),
            "candidate_ids": ["candidate-new"],
            "latest_update_id": "update-new",
        }
    )
    store.write_run(old_run, [_candidate_at("candidate-old", now)], [], [old])
    store.write_run(new_run, [_candidate_at("candidate-new", now)], [], [new])
    store.mark_candidates_delivered(new_run.run_id, ["candidate-new"], now + timedelta(hours=1))
    store.mark_candidates_delivered(old_run.run_id, ["candidate-old"], now + timedelta(hours=2))

    assert store.load_recent_events(now + timedelta(hours=3)) == [new]


def test_mark_candidates_delivered_is_idempotent(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-idempotent", started_at=now, selection_engine="shadow_event_v1")
    event = Event(
        event_id="evt-idempotent",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Idempotent",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["candidate-idempotent"],
    )
    store.write_run(run, [_candidate_at("candidate-idempotent", now)], [], [event])

    store.mark_candidates_delivered(run.run_id, ["candidate-idempotent"], now)
    store.mark_candidates_delivered(run.run_id, ["candidate-idempotent"], now)

    target = store.root / "deliveries" / "2026" / "08" / "deliveries.jsonl"
    assert len(store._read_jsonl(target)) == 1


def test_mark_candidates_delivered_rejects_legacy_run_without_event_snapshot(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-legacy", started_at=now, selection_engine="shadow_event_v1")
    run_dir = store._run_directory(run)
    store._write_json(run_dir / "run.json", run.model_dump(mode="json"))

    with pytest.raises(ValueError, match="event snapshot"):
        store.mark_candidates_delivered(run.run_id, ["candidate-legacy"], now)


def test_mark_candidates_delivered_rejects_candidates_outside_run(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-manifest", started_at=now, selection_engine="shadow_event_v1")
    event = Event(
        event_id="evt-manifest",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Manifest",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["known-candidate"],
    )
    store.write_run(run, [_candidate_at("known-candidate", now)], [], [event])

    with pytest.raises(ValueError, match="unknown candidate"):
        store.mark_candidates_delivered(
            run.run_id, ["known-candidate", "unknown-candidate"], now
        )

    assert not (store.root / "deliveries").exists()


def test_mark_rejects_event_snapshot_referencing_candidate_outside_run(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-inconsistent", started_at=now, selection_engine="shadow_event_v1")
    candidate = CandidateArticle(
        id="known-candidate",
        candidate_id="known-candidate",
        title="Known",
        url="https://example.com/known",
        source="Example",
        source_id="example",
        language="en",
        published_at=now,
        topic=Topic.AI,
    )
    event = Event(
        event_id="evt-inconsistent",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Inconsistent",
        first_event_time=now,
        latest_event_time=now,
        candidate_ids=["missing-from-run"],
    )
    store.write_run(run, [candidate], [], [event])

    with pytest.raises(ValueError, match="event snapshot references unknown candidate"):
        store.mark_candidates_delivered(run.run_id, [candidate.candidate_id], now)


def test_load_recent_events_ignores_unproven_or_ineligible_deliveries(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    run = RunRecord(run_id="run-filter", started_at=now, selection_engine="shadow_event_v1")
    events = [
        Event(
            event_id=event_id,
            coverage_domain=CoverageDomain.AI_TECHNOLOGY,
            event_type="release",
            title=event_id,
            first_event_time=now,
            latest_event_time=now,
        )
        for event_id in ("legacy", "future", "future-event", "email", "failed")
    ]
    events[2] = events[2].model_copy(update={"latest_event_time": now + timedelta(minutes=1)})
    store.write_run(run, [], [], events)
    target = store.root / "deliveries" / "2026" / "08" / "deliveries.jsonl"
    store._write_jsonl(
        target,
        [
            {
                "event_id": "legacy",
                "channel": "web",
                "status": "delivered",
                "delivered_at": now.isoformat(),
            },
            DeliveryRecord(
                run_id=run.run_id,
                event_id="future",
                channel="web",
                status="delivered",
                delivered_at=now + timedelta(minutes=1),
            ).model_dump(mode="json"),
            DeliveryRecord(
                run_id=run.run_id,
                event_id="future-event",
                channel="web",
                status="delivered",
                delivered_at=now,
            ).model_dump(mode="json"),
            DeliveryRecord(
                run_id=run.run_id,
                event_id="email",
                channel="email",
                status="delivered",
                delivered_at=now,
            ).model_dump(mode="json"),
            DeliveryRecord(
                run_id=run.run_id,
                event_id="failed",
                channel="web",
                status="failed",
                delivered_at=now,
            ).model_dump(mode="json"),
        ],
    )

    assert store.load_recent_events(now) == []


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", r"nested\run", ".", ".."])
def test_run_id_rejects_path_escape(run_id) -> None:
    with pytest.raises(ValidationError):
        RunRecord(
            run_id=run_id,
            started_at=datetime(2026, 8, 13, 8, tzinfo=UTC),
            selection_engine="shadow_event_v1",
        )


def test_only_legacy_published_candidates_mark_events_delivered(tmp_path) -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    run = RunRecord(run_id="run-2", started_at=now, selection_engine="shadow_event_v1")
    candidates = [
        CandidateArticle(id="published", candidate_id="published", title="Published", url="https://example.com/published", source="Example", source_id="example", language="en", published_at=now, topic=Topic.AI),
        CandidateArticle(id="shadow-only", candidate_id="shadow-only", title="Shadow only", url="https://example.com/shadow", source="Example", source_id="example", language="en", published_at=now, topic=Topic.AI),
    ]
    events = [
        Event(event_id="evt-published", coverage_domain=CoverageDomain.AI_TECHNOLOGY, event_type="release", title="Published", first_event_time=now, latest_event_time=now, candidate_ids=["published"]),
        Event(event_id="evt-shadow", coverage_domain=CoverageDomain.AI_TECHNOLOGY, event_type="release", title="Shadow only", first_event_time=now, latest_event_time=now, candidate_ids=["shadow-only"]),
    ]
    store = StateStore(tmp_path / "state")
    store.write_run(run, candidates, [], events)

    store.mark_candidates_delivered(run.run_id, ["published"], now)

    assert [event.event_id for event in store.load_recent_events(now)] == ["evt-published"]


def test_state_store_persists_and_reads_private_signal_snapshots(tmp_path) -> None:
    captured = datetime(2026, 8, 17, 0, tzinfo=UTC)
    later = datetime(2026, 8, 17, 6, tzinfo=UTC)
    store = StateStore(tmp_path / "state")
    first = SignalSnapshot(
        source_id="v2ex_latest",
        external_id="42",
        captured_at=captured,
        rank=3,
        reply_count=12,
    )
    second = SignalSnapshot(
        source_id="v2ex_latest",
        external_id="42",
        captured_at=later,
        rank=1,
        reply_count=20,
    )
    health = SourceHealthRecord(
        source_id="v2ex_latest", captured_at=captured, status="ok", candidate_count=1
    )

    store.write_signal_collection(captured, [first], [health])
    store.write_signal_collection(later, [second], [])

    assert store.load_recent_snapshots("v2ex_latest", "42", later, hours=24) == [first, second]
    assert not (tmp_path / "public").exists()
