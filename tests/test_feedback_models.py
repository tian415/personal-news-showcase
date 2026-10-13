from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from pydantic import ValidationError

from news_assistant.feedback import (
    FeedbackEntry,
    FeedbackPayload,
    FeedbackState,
    FeedbackTarget,
    merge_feedback,
    public_event_ref,
)
from news_assistant.models import Preferences, Topic

NOW = datetime(2026, 9, 12, 8, tzinfo=UTC)


def target(event_id: str, topic: Topic = Topic.AI, source: str = "source") -> FeedbackTarget:
    return FeedbackTarget(event_id=event_id, topic=topic, source=source)


def entry(
    event_id: str,
    *,
    vote: str | None = "helpful",
    reason: str | None = None,
    tracking: bool = False,
    minutes: int = 0,
) -> FeedbackEntry:
    return FeedbackEntry(
        event_ref=public_event_ref(event_id),
        vote=vote,
        reason=reason,
        tracking=tracking,
        updated_at=NOW + timedelta(minutes=minutes),
    )


def test_public_event_ref_is_a_stable_24_hex_alias() -> None:
    expected = sha256(b"news-event:event-1").hexdigest()[:24]

    assert public_event_ref("event-1") == expected
    assert public_event_ref("event-1") == public_event_ref("event-1")


def test_feedback_models_forbid_extra_fields_and_bool_coercion() -> None:
    with pytest.raises(ValidationError):
        FeedbackTarget(event_id="event-1", topic=Topic.AI, source="source", extra="no")
    with pytest.raises(ValidationError):
        FeedbackEntry(
            event_ref="ref",
            vote="helpful",
            reason=None,
            tracking=1,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError):
        FeedbackEntry(
            event_ref="ref",
            vote="helpful",
            reason=None,
            tracking=False,
            updated_at=1,
        )
    with pytest.raises(ValidationError):
        FeedbackEntry(
            event_ref="ref",
            vote="helpful",
            reason="irrelevant",
            tracking=False,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError):
        FeedbackEntry(
            event_ref="ref",
            vote="reduce",
            reason=None,
            tracking=False,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError):
        FeedbackEntry(
            event_ref="ref",
            vote=None,
            reason="duplicate",
            tracking=False,
            updated_at=NOW,
        )


def test_feedback_payload_is_versioned_and_limited_to_10000_events() -> None:
    payload = FeedbackPayload(schema_version="news-feedback-v2", events=[])

    assert payload.schema_version == "news-feedback-v2"
    with pytest.raises(ValidationError):
        FeedbackPayload(schema_version="old", events=[])
    with pytest.raises(ValidationError):
        FeedbackPayload(schema_version="news-feedback-v2", events=[], extra="no")
    with pytest.raises(ValidationError):
        FeedbackPayload(schema_version="news-feedback-v2", events=[entry("event-1")] * 10001)


def test_merge_rejects_unknown_ref_without_changing_previous_state() -> None:
    known = {public_event_ref("event-1"): target("event-1")}
    previous = merge_feedback(
        {"schema_version": "news-feedback-v2", "events": [entry("event-1")]},
        known,
        now=NOW,
    )
    unknown = FeedbackEntry(
        event_ref=public_event_ref("unknown"),
        vote="helpful",
        reason=None,
        tracking=False,
        updated_at=NOW + timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="unknown event_ref"):
        merge_feedback(
            {"schema_version": "news-feedback-v2", "events": [unknown]},
            known,
            previous=previous,
            now=NOW,
        )
    assert previous.events[public_event_ref("event-1")].vote == "helpful"


def test_merge_is_idempotent_and_latest_timestamp_wins_regardless_of_order() -> None:
    known = {
        public_event_ref("event-1"): target("event-1"),
        public_event_ref("event-2"): target("event-2", Topic.BUSINESS, "wire"),
    }
    older = entry("event-1", vote="helpful", minutes=0)
    newer = entry("event-1", vote="reduce", reason="irrelevant", minutes=1)
    second = entry("event-2", vote="helpful", tracking=True)
    payload = {"schema_version": "news-feedback-v2", "events": [newer, second, older]}

    first = merge_feedback(payload, known, now=NOW + timedelta(minutes=1))
    replay = merge_feedback(
        {"schema_version": "news-feedback-v2", "events": [older, second, newer]},
        known,
        previous=first,
        now=NOW + timedelta(minutes=1),
    )

    assert replay == first
    assert first.events[public_event_ref("event-1")].vote == "reduce"
    assert first.preferences.topic_weights[Topic.AI] == 0.9
    assert first.tracked_event_ids == ["event-2"]


def test_same_timestamp_conflict_is_rejected() -> None:
    known = {public_event_ref("event-1"): target("event-1")}
    first = entry("event-1", vote="helpful")
    conflict = entry("event-1", vote=None)

    with pytest.raises(ValueError, match="same updated_at"):
        merge_feedback(
            {"schema_version": "news-feedback-v2", "events": [first, conflict]},
            known,
            now=NOW,
        )


def test_tombstone_revokes_vote_and_tracking_but_is_retained() -> None:
    ref = public_event_ref("event-1")
    known = {ref: target("event-1")}
    initial = merge_feedback(
        {"schema_version": "news-feedback-v2", "events": [entry("event-1", tracking=True)]},
        known,
        now=NOW,
    )
    tombstone = entry("event-1", vote=None, tracking=False, minutes=1)

    final = merge_feedback(
        {"schema_version": "news-feedback-v2", "events": [tombstone]},
        known,
        previous=initial,
        now=NOW + timedelta(minutes=1),
    )

    assert final.events[ref] == tombstone
    assert final.preferences == Preferences()
    assert final.tracked_event_ids == []


def test_reduce_reasons_update_only_irrelevant_and_count_other_diagnostics() -> None:
    known = {
        public_event_ref("irrelevant"): target("irrelevant"),
        public_event_ref("known"): target("known"),
        public_event_ref("shallow"): target("shallow"),
    }
    entries = [
        entry("irrelevant", vote="reduce", reason="irrelevant"),
        entry("known", vote="reduce", reason="already_known", minutes=1),
        entry("shallow", vote="reduce", reason="shallow", minutes=2),
    ]

    result = merge_feedback(
        {"schema_version": "news-feedback-v2", "events": entries},
        known,
        now=NOW + timedelta(minutes=2),
    )

    assert result.preferences.topic_weights[Topic.AI] == 0.9
    assert result.diagnostics == {"already_known": 1, "shallow": 1}


def test_helpful_and_irrelevant_reduce_cancel_for_topic_and_source() -> None:
    known = {
        public_event_ref("helpful"): target("helpful", Topic.TECH, "wire"),
        public_event_ref("reduce"): target("reduce", Topic.TECH, "wire"),
    }
    result = merge_feedback(
        {
            "schema_version": "news-feedback-v2",
            "events": [
                entry("helpful"),
                entry("reduce", vote="reduce", reason="irrelevant", minutes=1),
            ],
        },
        known,
        now=NOW + timedelta(minutes=1),
    )

    assert result.preferences.topic_weights[Topic.TECH] == 1.0
    assert result.preferences.source_weights["wire"] == 1.0


def test_future_and_naive_timestamps_are_rejected() -> None:
    known = {public_event_ref("event-1"): target("event-1")}
    future = entry("event-1", minutes=6)
    with pytest.raises(ValueError, match="future"):
        merge_feedback(
            {"schema_version": "news-feedback-v2", "events": [future]},
            known,
            now=NOW,
        )
    with pytest.raises(ValidationError, match="timezone"):
        FeedbackEntry(
            event_ref=public_event_ref("event-1"),
            vote="helpful",
            reason=None,
            tracking=False,
            updated_at=datetime.fromisoformat("2026-09-12T08:00:00"),
        )


def test_previous_unknown_ref_requires_target_mapping() -> None:
    old_ref = public_event_ref("old")
    previous = FeedbackState(
        events={old_ref: entry("old")},
        preferences=Preferences(),
        tracked_event_ids=[],
        diagnostics={},
    )

    with pytest.raises(ValueError, match="previous event_ref"):
        merge_feedback(
            {"schema_version": "news-feedback-v2", "events": []},
            {},
            previous=previous,
            now=NOW,
        )
