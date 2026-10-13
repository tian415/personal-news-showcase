from datetime import UTC, datetime

from news_assistant.evidence import EvidenceAssessment, is_major
from news_assistant.models import CoverageDomain, Event, ScoreCard, SelectionPolicy


def test_anonymous_signal_cannot_pass_major_evidence_gate() -> None:
    event = Event(
        event_id="evt-1",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title="Rumored launch",
        first_event_time=datetime(2026, 8, 13, tzinfo=UTC),
        latest_event_time=datetime(2026, 8, 13, tzinfo=UTC),
    )
    scores = ScoreCard(
        freshness=90, importance=95, relevance=100, exploration=0, evidence_quality=20
    )

    assert is_major(event, scores, EvidenceAssessment(quality=20), SelectionPolicy()) is False
