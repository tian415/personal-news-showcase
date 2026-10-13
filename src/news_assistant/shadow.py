from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, Field

from .events import assign_events
from .evidence import EvidenceAssessment, is_major
from .models import (
    CandidateArticle,
    DecisionRecord,
    Event,
    EvidenceRole,
    Preferences,
    ProcessingStage,
    ReasonCode,
    ScoreCard,
    SelectionPolicy,
)


class ShadowSelectionResult(BaseModel):
    events: list[Event] = Field(default_factory=list)
    selected_events: list[Event] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)


def run_shadow_selection(
    candidates: Iterable[CandidateArticle],
    historical_events: Iterable[Event],
    preferences: Preferences,
    *,
    now: datetime,
    policy: SelectionPolicy | None = None,
    limit: int = 10,
) -> ShadowSelectionResult:
    active_policy = policy or SelectionPolicy()
    candidate_list = list(candidates)
    assignment = assign_events(candidate_list, historical_events)
    by_id = {candidate.candidate_id: candidate for candidate in candidate_list}
    scored_events: list[Event] = []
    decisions = list(assignment.decisions)
    for event in assignment.events:
        members = [by_id[item] for item in event.candidate_ids]
        score = _scores(event, members, preferences, now)
        evidence = _evidence(members)
        major = is_major(event, score, evidence, active_policy)
        scored = event.model_copy(update={"scores": score, "major": major})
        scored_events.append(scored)

    eligible_events = [event for event in scored_events if event.scores and event.scores.evidence_quality >= 70]
    major_events = [event for event in eligible_events if event.major]
    non_major = sorted(
        (event for event in eligible_events if not event.major),
        key=lambda event: event.scores.rank_score(active_policy) if event.scores else 0,
        reverse=True,
    )
    selected = [*major_events, *non_major[: max(0, limit - len(major_events))]]
    selected_ids = {event.event_id for event in selected}
    for event in scored_events:
        selected_event = event.event_id in selected_ids
        decisions.append(
            DecisionRecord(
                event_id=event.event_id,
                stage=ProcessingStage.SELECTED if selected_event else ProcessingStage.SCORED,
                selected=selected_event,
                primary_reason=(ReasonCode.MAJOR_GUARANTEE if event.major else
                                ReasonCode.LOW_EVIDENCE if event not in eligible_events else
                                ReasonCode.LOST_TO_HIGHER_VALUE),
                scores=event.scores,
            )
        )
    return ShadowSelectionResult(events=scored_events, selected_events=selected, decisions=decisions)


def _scores(
    event: Event, candidates: list[CandidateArticle], preferences: Preferences, now: datetime
) -> ScoreCard:
    primary = candidates[0]
    age_hours = max(0, (now - event.latest_event_time).total_seconds() / 3600)
    freshness = max(0, 100 - age_hours * 4)
    relevance = (
        50 * preferences.topic_weights.get(primary.topic, 1.0)
        * preferences.source_weights.get(primary.source, 1.0)
    )
    return ScoreCard(
        freshness=freshness,
        importance=primary.importance_score * 100,
        relevance=min(100, relevance),
        exploration=70 if primary.topic.value == "exploration" else 30,
        evidence_quality=_evidence(candidates).quality,
    )


def _evidence(candidates: list[CandidateArticle]) -> EvidenceAssessment:
    roles = {candidate.evidence_role for candidate in candidates}
    independent_reporting_count = len(
        {
            candidate.publisher_group or candidate.source_id
            for candidate in candidates
            if candidate.evidence_role is EvidenceRole.INDEPENDENT_REPORTING
        }
    )
    if EvidenceRole.PRIMARY_FACT in roles:
        return EvidenceAssessment(quality=90, has_primary_fact=True)
    if independent_reporting_count >= 2:
        return EvidenceAssessment(
            quality=80, independent_reporting_count=independent_reporting_count
        )
    if EvidenceRole.INDEPENDENT_REPORTING in roles:
        return EvidenceAssessment(
            quality=65, independent_reporting_count=independent_reporting_count
        )
    if EvidenceRole.EXPERT_ANALYSIS in roles:
        return EvidenceAssessment(quality=55)
    return EvidenceAssessment(quality=20)
