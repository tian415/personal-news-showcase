from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from news_assistant.delivery_recovery import (
    confirm_local_reading,
    reconcile_publications,
    record_publication,
)
from news_assistant.feedback import public_event_ref
from news_assistant.feedback_store import import_feedback_file
from news_assistant.models import (
    CandidateArticle,
    CoverageDomain,
    DeliveryRecord,
    Event,
    RunRecord,
    Topic,
)
from news_assistant.state import StateStore

NOW = datetime(2026, 9, 28, 8, tzinfo=UTC)


def _write_run(
    store: StateStore,
    run_id: str = "local-run",
    *,
    started_at: datetime = NOW,
    event_time: datetime | None = None,
    selection_engine: str = "reviewed_local_v1",
) -> tuple[RunRecord, CandidateArticle, Event]:
    candidate = CandidateArticle(
        id=f"candidate-{run_id}",
        candidate_id=f"candidate-{run_id}",
        title="Reviewed local article",
        url=f"https://example.com/{run_id}",
        source="Example",
        source_id="example",
        language="en",
        published_at=NOW,
        topic=Topic.AI,
    )
    event = Event(
        event_id=f"event-{run_id}",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="review",
        title=candidate.title,
        first_event_time=event_time or started_at,
        latest_event_time=event_time or started_at,
        candidate_ids=[candidate.candidate_id],
        primary_candidate_id=candidate.candidate_id,
        latest_update_id=f"update-{run_id}",
    )
    run = RunRecord(
        run_id=run_id,
        started_at=started_at,
        selection_engine=selection_engine,
    )
    store.write_run(run, [candidate], [], [event])
    return run, candidate, event


def _record_local_page(
    store: StateStore,
    root: Path,
    *,
    run_id: str = "local-run",
    started_at: datetime = NOW,
    event_time: datetime | None = None,
) -> tuple[RunRecord, CandidateArticle, Event, Path]:
    run, candidate, event = _write_run(
        store,
        run_id,
        started_at=started_at,
        event_time=event_time,
    )
    page = root / f"{started_at.date().isoformat()}.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_bytes(f"reviewed page for {run_id}".encode())
    record_publication(store, run, started_at.date(), [candidate.candidate_id], page)
    return run, candidate, event, page


def _feedback(path: Path, event_ref: str, *, updated_at: datetime, vote: str | None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "news-feedback-v2",
                "events": [
                    {
                        "event_ref": event_ref,
                        "vote": vote,
                        "reason": None,
                        "tracking": vote is not None,
                        "updated_at": updated_at.isoformat(),
                    }
                ],
            }
        )
    )


def test_local_feedback_requires_reading_and_stays_separate_from_web_state(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, _, event, page = _record_local_page(store, tmp_path / "preview")
    ref = public_event_ref(
        event.event_id,
        run_id=run.run_id,
        update_id=event.latest_update_id,
    )
    feedback_path = tmp_path / "feedback.json"
    _feedback(feedback_path, ref, updated_at=NOW, vote="helpful")

    with pytest.raises(ValueError, match="unknown event_ref"):
        import_feedback_file(feedback_path, store.root, now=NOW, channel="local")
    assert not (store.root / "local-feedback-state.json").exists()

    assert confirm_local_reading(store, run.run_id, page, now=NOW) is True
    assert store.load_recent_events(NOW) == []
    assert store.load_recent_events(NOW, channel="local") == [event]
    with pytest.raises(ValueError, match="unknown event_ref"):
        import_feedback_file(feedback_path, store.root, now=NOW)
    assert not (store.root / "feedback-state.json").exists()

    state = import_feedback_file(feedback_path, store.root, now=NOW, channel="local")
    assert state.preferences.topic_weights == {Topic.AI: 1.1}
    assert (store.root / "local-feedback-state.json").exists()

    _feedback(feedback_path, ref, updated_at=NOW + timedelta(minutes=1), vote=None)
    revoked = import_feedback_file(
        feedback_path,
        store.root,
        now=NOW + timedelta(minutes=1),
        channel="local",
    )
    assert revoked.preferences.topic_weights == {}
    assert revoked.tracked_event_ids == []
    before = (store.root / "local-feedback-state.json").read_bytes()
    import_feedback_file(
        feedback_path,
        store.root,
        now=NOW + timedelta(minutes=1),
        channel="local",
    )
    assert (store.root / "local-feedback-state.json").read_bytes() == before


def test_delivery_channels_are_validated_and_independently_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    local_run, local_candidate, _ = _write_run(store)
    web_run, web_candidate, _ = _write_run(
        store,
        "web-run",
        selection_engine="shadow_event_v1",
    )

    with pytest.raises(ValueError, match="reviewed_local_v1"):
        store.mark_candidates_delivered(
            local_run.run_id,
            [local_candidate.candidate_id],
            NOW,
        )
    with pytest.raises(ValueError, match="local"):
        store.mark_candidates_delivered(
            web_run.run_id,
            [web_candidate.candidate_id],
            NOW,
            channel="local",
        )
    with pytest.raises(ValueError, match="channel"):
        store.mark_candidates_delivered(
            web_run.run_id,
            [web_candidate.candidate_id],
            NOW,
            channel="email",
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        store.mark_candidates_delivered(
            web_run.run_id,
            [web_candidate.candidate_id],
            NOW.replace(tzinfo=None),
        )

    store.mark_candidates_delivered(web_run.run_id, [web_candidate.candidate_id], NOW)

    # A historical record from another channel must not suppress a valid local mark.
    cross_store = StateStore(tmp_path / "cross-state")
    cross_run, cross_candidate, cross_event = _write_run(cross_store, "cross-run")
    cross_target = cross_store.root / "deliveries" / "2026" / "09" / "deliveries.jsonl"
    cross_store._write_jsonl(
        cross_target,
        [
            DeliveryRecord(
                run_id=cross_run.run_id,
                event_id=cross_event.event_id,
                update_id=cross_event.latest_update_id,
                event=cross_event,
                candidate_ids=[cross_candidate.candidate_id],
                channel="web",
                status="delivered",
                delivered_at=NOW,
            ).model_dump(mode="json")
        ],
    )
    cross_store.mark_candidates_delivered(
        cross_run.run_id,
        [cross_candidate.candidate_id],
        NOW,
        channel="local",
    )
    records = [
        row
        for path in (cross_store.root / "deliveries").glob("*/*/deliveries.jsonl")
        for row in cross_store._read_jsonl(path)
    ]
    assert {row["channel"] for row in records} == {"web", "local"}
    assert len(records) == 2


@pytest.mark.parametrize("future", ["run", "event"])
def test_mark_rejects_future_run_or_event(tmp_path: Path, future: str) -> None:
    store = StateStore(tmp_path / "state")
    future_time = NOW + timedelta(minutes=1)
    run, candidate, _ = _write_run(
        store,
        started_at=future_time if future == "run" else NOW,
        event_time=future_time if future == "event" else NOW,
    )

    with pytest.raises(ValueError, match="future"):
        store.mark_candidates_delivered(
            run.run_id,
            [candidate.candidate_id],
            NOW,
            channel="local",
        )
    assert not (store.root / "deliveries").exists()


@pytest.mark.parametrize("naive", ["run", "event"])
def test_mark_rejects_naive_run_or_event_time(tmp_path: Path, naive: str) -> None:
    store = StateStore(tmp_path / "state")
    naive_time = NOW.replace(tzinfo=None)
    run, candidate, _ = _write_run(
        store,
        started_at=naive_time if naive == "run" else NOW,
        event_time=naive_time if naive == "event" else NOW,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        store.mark_candidates_delivered(
            run.run_id,
            [candidate.candidate_id],
            NOW,
            channel="local",
        )


def test_confirm_local_reading_checks_exact_page_binding_and_is_cross_month_idempotent(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state")
    run, _, event, page = _record_local_page(store, tmp_path / "preview")

    wrong_name = page.with_name("2026-09-27.html")
    wrong_name.write_bytes(page.read_bytes())
    with pytest.raises(ValueError, match="date|basename|name"):
        confirm_local_reading(store, run.run_id, wrong_name, now=NOW)
    page.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|match"):
        confirm_local_reading(store, run.run_id, page, now=NOW)
    page.write_bytes(b"reviewed page for local-run")

    assert confirm_local_reading(store, run.run_id, page, now=NOW) is True
    assert confirm_local_reading(
        store,
        run.run_id,
        page,
        now=NOW + timedelta(days=35),
    ) is False
    assert store.load_recent_events(NOW + timedelta(days=35), channel="local") == [event]
    records = [
        row
        for path in (store.root / "deliveries").glob("*/*/deliveries.jsonl")
        for row in store._read_jsonl(path)
        if row["channel"] == "local"
    ]
    assert len(records) == 1
    assert len(list((store.root / "deliveries").glob("*/*/deliveries.jsonl"))) == 1


def test_confirm_local_reading_rejects_future_run_and_symlink(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, _, _, page = _record_local_page(
        store,
        tmp_path / "preview",
        started_at=NOW + timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="future"):
        confirm_local_reading(store, run.run_id, page, now=NOW)

    other_store = StateStore(tmp_path / "other-state")
    other_run, _, _, other_page = _record_local_page(other_store, tmp_path / "other-preview")
    target = tmp_path / "target.html"
    other_page.replace(target)
    other_page.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        confirm_local_reading(other_store, other_run.run_id, other_page, now=NOW)


def test_reconcile_publications_never_turns_local_review_into_web_delivery(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    _, _, _, page = _record_local_page(store, tmp_path / "preview")
    published = tmp_path / "published"
    published.mkdir()
    (published / page.name).write_bytes(page.read_bytes())

    assert reconcile_publications(store, published, now=NOW) == []
    assert store.load_recent_events(NOW) == []
