from datetime import UTC, datetime

from news_assistant.models import (
    DecisionRecord,
    DeliveryRecord,
    ProcessingStage,
    ReasonCode,
    ScoreCard,
    SelectionPolicy,
)


def test_score_card_keeps_five_dimensions_and_calculates_versioned_rank() -> None:
    scores = ScoreCard(
        freshness=50,
        importance=80,
        relevance=70,
        exploration=60,
        evidence_quality=90,
    )

    assert scores.rank_score(SelectionPolicy()) == 71.5
    assert scores.importance == 80


def test_decision_record_keeps_stage_reason_and_replay_metadata() -> None:
    record = DecisionRecord(
        event_id="evt-1",
        stage=ProcessingStage.SCORED,
        selected=False,
        primary_reason=ReasonCode.LOW_EVIDENCE,
        occurred_at=datetime(2026, 8, 13, tzinfo=UTC),
    )

    assert record.primary_reason is ReasonCode.LOW_EVIDENCE
    assert record.ruleset_version == "selection-v1"
    assert record.model_version == "none"


def test_legacy_delivery_record_remains_readable_without_proving_a_run_version() -> None:
    record = DeliveryRecord.model_validate(
        {
            "event_id": "evt-legacy",
            "channel": "web",
            "status": "delivered",
            "delivered_at": "2026-08-13T08:00:00Z",
        }
    )

    assert record.run_id is None
    assert record.event is None
