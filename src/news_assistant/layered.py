"""Conservative, explainable three-layer selection for private shadow evaluation."""

from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from .models import CandidateArticle, Event, EvidenceRole, Preferences, SignalSnapshot
from .shadow import run_shadow_selection
from .trends import TrendAssessment, assess_trend


class ContentLayer(StrEnum):
    DECISION_BRIEF = "decision_brief"
    SIGNAL_RADAR = "signal_radar"
    EXPLORATION = "exploration"


class LayeredItem(BaseModel):
    event: Event
    source_ids: list[str] = Field(default_factory=list)
    layer: ContentLayer
    reasons: list[str]
    fact_status: str
    independent_platforms: int = 0
    trends: dict[str, TrendAssessment] = Field(default_factory=dict)
    expansion_connection: str | None = None
    manipulation_risk: str = "not_assessed"


class LayeredSelectionResult(BaseModel):
    schema_version: str = "layered-shadow-v1"
    items: list[LayeredItem] = Field(default_factory=list)
    excluded: dict[str, list[str]] = Field(default_factory=dict)

    def comparison(self, legacy_candidate_ids: Iterable[str]) -> dict:
        legacy = set(legacy_candidate_ids)
        selected = {identifier for item in self.items for identifier in item.event.candidate_ids}
        return {
            **self.model_dump(mode="json"),
            "layer_counts": {
                layer.value: sum(item.layer is layer for item in self.items)
                for layer in ContentLayer
            },
            "legacy_only": sorted(legacy - selected),
            "layered_only": sorted(selected - legacy),
            "overlap": sorted(legacy & selected),
            "limitations": [
                "Private shadow output; public selection remains legacy.",
                "Trend counters do not establish factual truth or rule out manipulation.",
                "Conservative title grouping is not complete multilingual event resolution.",
            ],
        }


def run_layered_selection(
    candidates: Iterable[CandidateArticle],
    historical_events: Iterable[Event],
    preferences: Preferences,
    *,
    snapshots: Iterable[SignalSnapshot],
    now: datetime,
    tracked_event_ids: Iterable[str] = (),
) -> LayeredSelectionResult:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    supplied_candidates = list(candidates)
    candidate_ids = [item.candidate_id for item in supplied_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate_id values are not allowed")
    role_order = {
        EvidenceRole.PRIMARY_FACT: 3,
        EvidenceRole.INDEPENDENT_REPORTING: 2,
        EvidenceRole.EXPERT_ANALYSIS: 1,
        EvidenceRole.COMMUNITY_SIGNAL: 0,
    }
    candidate_list = sorted(
        (
            item
            for item in supplied_candidates
            if now - timedelta(hours=36) <= item.published_at <= now
        ),
        key=lambda item: (
            role_order[item.evidence_role],
            item.quality_score,
            item.importance_score,
            item.candidate_id,
        ),
        reverse=True,
    )
    by_id = {item.candidate_id: item for item in candidate_list}
    baseline = run_shadow_selection(candidate_list, historical_events, preferences, now=now)
    snapshot_list = list(snapshots)
    result = LayeredSelectionResult()
    tracked = set(tracked_event_ids)
    pools: dict[ContentLayer, list[LayeredItem]] = {layer: [] for layer in ContentLayer}
    for event in baseline.events:
        members = [by_id[identifier] for identifier in event.candidate_ids]
        score = event.scores
        if score is None or max(item.quality_score for item in members) < 0.6:
            result.excluded[event.event_id] = ["BELOW_QUALITY_THRESHOLD"]
            continue
        trends: dict[str, TrendAssessment] = {}
        for member in members:
            if member.evidence_role is not EvidenceRole.COMMUNITY_SIGNAL:
                continue
            external_id = member.signal_external_id or member.candidate_id
            trend = assess_trend(
                [
                    row
                    for row in snapshot_list
                    if row.source_id == member.source_id and row.external_id == external_id
                ],
                now=now,
            )
            platform = member.publisher_group or member.source_id
            # Two feeds of the same publisher count only once, with the best observed series.
            previous = trends.get(platform)
            if previous is None or (trend.attention_velocity or 0) > (
                previous.attention_velocity or 0
            ):
                trends[platform] = trend
        rising = sum((trend.attention_velocity or 0) > 0 for trend in trends.values())
        connection = next(
            (
                item.expansion_connection.strip()
                for item in members
                if item.expansion_connection and item.expansion_connection.strip()
            ),
            None,
        )
        tracked_update = (event.event_id in tracked and bool(event.update_history)
                          and event.update_history[-1].material)
        if event.major or (score.evidence_quality >= 70 and (score.importance >= 60 or tracked_update)):
            layer = ContentLayer.DECISION_BRIEF
            reasons = ["MAJOR_GUARANTEE" if event.major else "EVIDENCED_DECISION_VALUE"]
            status = "evidence_supported"
            if tracked_update:
                reasons.append("TRACKED_MATERIAL_UPDATE")
        elif rising >= 2:
            layer = ContentLayer.SIGNAL_RADAR
            reasons = ["MULTI_PLATFORM_GROWTH", "FACT_NOT_CONFIRMED", "MANIPULATION_NOT_ASSESSED"]
            status = "unconfirmed_signal"
        elif score.evidence_quality >= 70 and connection:
            layer = ContentLayer.EXPLORATION
            reasons = ["EVIDENCED_CROSS_DOMAIN_CONNECTION"]
            status = "evidence_supported"
        else:
            result.excluded[event.event_id] = [
                "INSUFFICIENT_TREND_EVIDENCE" if trends else "NO_LAYER_THRESHOLD_MET"
            ]
            continue
        pools[layer].append(
            LayeredItem(
                event=event,
                source_ids=sorted({member.source_id for member in members}),
                layer=layer,
                reasons=reasons,
                fact_status=status,
                independent_platforms=rising,
                trends=trends,
                expansion_connection=connection,
            )
        )
    limits = {
        ContentLayer.DECISION_BRIEF: 5,
        ContentLayer.SIGNAL_RADAR: 5,
        ContentLayer.EXPLORATION: 3,
    }
    for layer, entries in pools.items():
        entries.sort(
            key=lambda entry: (
                entry.event.major,
                entry.event.scores.importance,
                entry.event.scores.relevance,
                entry.event.latest_event_time,
                entry.event.event_id,
            ),
            reverse=True,
        )
        for index, entry in enumerate(entries):
            if entry.event.major or index < limits[layer]:
                result.items.append(entry)
            else:
                result.excluded[entry.event.event_id] = ["LAYER_SOFT_LIMIT"]
    for decision in baseline.decisions:
        if decision.candidate_id and decision.primary_reason.value == "EVENT_DUPLICATE_NO_DELTA":
            result.excluded[f"candidate:{decision.candidate_id}"] = [decision.primary_reason.value]
    return result
