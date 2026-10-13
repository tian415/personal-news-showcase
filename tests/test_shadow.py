from datetime import UTC, datetime

import pytest

from news_assistant.models import (
    CandidateArticle,
    CoverageDomain,
    EvidenceRole,
    Preferences,
    SourceTier,
    Topic,
)
from news_assistant.shadow import run_shadow_selection


def test_major_events_can_exceed_soft_limit_in_shadow_selection() -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    candidates = [
        CandidateArticle(
            id=str(index),
            candidate_id=str(index),
            title=f"OpenAI launches product {index}",
            url=f"https://official.example.com/{index}",
            source="Official",
            source_id=f"official-{index}",
            source_tier=SourceTier.T0,
            coverage_domain=CoverageDomain.AI_TECHNOLOGY,
            language="en",
            published_at=now,
            topic=Topic.AI,
            quality_score=0.9,
            importance_score=0.95,
        )
        for index in range(11)
    ]

    result = run_shadow_selection(candidates, [], Preferences(), now=now)

    assert len(result.selected_events) == 11
    assert all(event.major for event in result.selected_events)


def test_community_signal_cannot_become_major_when_legacy_tier_is_misconfigured() -> None:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    candidate = CandidateArticle(
        id="community-1",
        candidate_id="community-1",
        title="Community launches product",
        url="https://community.example.com/1",
        source="Community",
        source_id="community",
        source_tier=SourceTier.T0,
        evidence_role=EvidenceRole.COMMUNITY_SIGNAL,
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        language="en",
        published_at=now,
        topic=Topic.AI,
        quality_score=0.9,
        importance_score=0.95,
    )

    result = run_shadow_selection([candidate], [], Preferences(), now=now)

    assert result.events[0].major is False
    assert result.events[0].scores.evidence_quality == 20


def test_source_feedback_changes_relevance_without_promoting_weak_evidence() -> None:
    now = datetime(2026, 9, 28, 8, tzinfo=UTC)
    candidate = CandidateArticle(
        id="community-source-feedback", candidate_id="community-source-feedback",
        title="Community discusses a new language model",
        url="https://community.example.com/model", source="Community",
        source_id="community", source_tier=SourceTier.T3,
        evidence_role=EvidenceRole.COMMUNITY_SIGNAL,
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        language="en", published_at=now, topic=Topic.AI,
        quality_score=0.9, importance_score=0.95,
    )
    baseline = run_shadow_selection([candidate], [], Preferences(), now=now)
    preferred = run_shadow_selection(
        [candidate], [], Preferences(source_weights={"Community": 1.1}), now=now,
    )
    assert baseline.events[0].scores.relevance == 50
    assert preferred.events[0].scores.relevance == pytest.approx(55)
    assert preferred.events[0].scores.evidence_quality == 20
    assert preferred.events[0].major is False
    assert preferred.selected_events == []
