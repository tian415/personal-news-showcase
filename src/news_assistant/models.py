from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class Topic(StrEnum):
    AI = "ai"
    TECH = "tech"
    SCIENCE = "science"
    BUSINESS = "business"
    MARKETS = "markets"
    EXPLORATION = "exploration"


class CoverageDomain(StrEnum):
    AI_TECHNOLOGY = "ai_technology"
    BUSINESS_MACRO = "business_macro"
    SCIENCE_HEALTH = "science_health"
    POLICY_PUBLIC_AFFAIRS = "policy_public_affairs"
    EXPLORATION = "exploration"


class SourceTier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class EvidenceRole(StrEnum):
    PRIMARY_FACT = "PRIMARY_FACT"
    INDEPENDENT_REPORTING = "INDEPENDENT_REPORTING"
    EXPERT_ANALYSIS = "EXPERT_ANALYSIS"
    COMMUNITY_SIGNAL = "COMMUNITY_SIGNAL"


class SourceKind(StrEnum):
    OFFICIAL = "OFFICIAL"
    MEDIA = "MEDIA"
    CREATOR = "CREATOR"
    FORUM = "FORUM"
    AGGREGATOR = "AGGREGATOR"
    REPOSITORY = "REPOSITORY"


class ProcessingStage(StrEnum):
    DISCOVERED = "DISCOVERED"
    NORMALIZED = "NORMALIZED"
    ARTICLE_DEDUPED = "ARTICLE_DEDUPED"
    EVENT_ASSIGNED = "EVENT_ASSIGNED"
    EVIDENCE_CHECKED = "EVIDENCE_CHECKED"
    SCORED = "SCORED"
    SELECTED = "SELECTED"
    SUMMARIZED = "SUMMARIZED"
    DELIVERED = "DELIVERED"


class ReasonCode(StrEnum):
    FETCH_FAILED = "FETCH_FAILED"
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
    EXACT_ARTICLE_DUPLICATE = "EXACT_ARTICLE_DUPLICATE"
    EVENT_DUPLICATE_NO_DELTA = "EVENT_DUPLICATE_NO_DELTA"
    LOW_EVIDENCE = "LOW_EVIDENCE"
    BELOW_SCORE_THRESHOLD = "BELOW_SCORE_THRESHOLD"
    LOST_TO_HIGHER_VALUE = "LOST_TO_HIGHER_VALUE"
    TOPIC_CAP = "TOPIC_CAP"
    SOURCE_CAP = "SOURCE_CAP"
    MODEL_UNCERTAIN = "MODEL_UNCERTAIN"
    COST_BUDGET_FALLBACK = "COST_BUDGET_FALLBACK"
    MAJOR_GUARANTEE = "MAJOR_GUARANTEE"
    MATERIAL_UPDATE_INCLUDED = "MATERIAL_UPDATE_INCLUDED"


class SelectionPolicy(BaseModel):
    version: str = "selection-v1"
    importance_weight: float = Field(default=0.35, ge=0, le=1)
    relevance_weight: float = Field(default=0.30, ge=0, le=1)
    freshness_weight: float = Field(default=0.15, ge=0, le=1)
    exploration_weight: float = Field(default=0.10, ge=0, le=1)
    evidence_quality_weight: float = Field(default=0.10, ge=0, le=1)


class ScoreCard(BaseModel):
    freshness: float = Field(ge=0, le=100)
    importance: float = Field(ge=0, le=100)
    relevance: float = Field(ge=0, le=100)
    exploration: float = Field(ge=0, le=100)
    evidence_quality: float = Field(ge=0, le=100)

    def rank_score(self, policy: SelectionPolicy) -> float:
        return (
            self.importance * policy.importance_weight
            + self.relevance * policy.relevance_weight
            + self.freshness * policy.freshness_weight
            + self.exploration * policy.exploration_weight
            + self.evidence_quality * policy.evidence_quality_weight
        )


class Article(BaseModel):
    id: str
    title: str
    url: HttpUrl
    source: str
    language: str = Field(pattern=r"^(zh|en)$")
    published_at: datetime
    topic: Topic
    summary: str = ""
    quality_score: float = Field(default=0.7, ge=0, le=1)
    importance_score: float = Field(default=0.5, ge=0, le=1)

    @field_validator("published_at")
    @classmethod
    def publication_time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware")
        return value


class CandidateArticle(Article):
    candidate_id: str = ""
    source_id: str = ""
    source_tier: SourceTier = SourceTier.T2
    evidence_role: EvidenceRole | None = None
    publisher_group: str = ""
    coverage_domain: CoverageDomain = CoverageDomain.AI_TECHNOLOGY
    canonical_url: HttpUrl | None = None
    content_fingerprint: str = ""
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_time: datetime | None = None
    adapter_version: str = "rss-v1"
    discovery_query: str | None = None
    signal_external_id: str | None = None
    expansion_connection: str | None = None

    @model_validator(mode="after")
    def apply_legacy_evidence_role(self) -> CandidateArticle:
        if self.evidence_role is None:
            self.evidence_role = {
                SourceTier.T0: EvidenceRole.PRIMARY_FACT,
                SourceTier.T1: EvidenceRole.INDEPENDENT_REPORTING,
                SourceTier.T2: EvidenceRole.EXPERT_ANALYSIS,
                SourceTier.T3: EvidenceRole.COMMUNITY_SIGNAL,
            }[self.source_tier]
        return self


class SignalSnapshot(BaseModel):
    source_id: str
    external_id: str
    captured_at: datetime
    rank: int = Field(ge=1)
    reply_count: int = Field(default=0, ge=0)

    @field_validator("captured_at")
    @classmethod
    def capture_time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value


class SourceHealthRecord(BaseModel):
    source_id: str
    captured_at: datetime
    status: str = Field(pattern=r"^(ok|error)$")
    candidate_count: int = Field(ge=0)
    error_type: str | None = None

    @field_validator("captured_at")
    @classmethod
    def health_time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value


class EventUpdate(BaseModel):
    update_id: str = Field(default_factory=lambda: f"upd_{uuid4()}")
    event_id: str
    previous_update_id: str | None = None
    delta_fields: list[str] = Field(default_factory=list)
    material: bool = False
    material_update_reason: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4()}")
    coverage_domain: CoverageDomain
    entities: list[str] = Field(default_factory=list)
    event_type: str
    title: str
    first_event_time: datetime
    latest_event_time: datetime
    state: str = "discovered"
    editorial_content_fingerprints: list[str] = Field(default_factory=list)
    editorial_source_urls: list[str] = Field(default_factory=list)
    editorial_development: str | None = None
    editorial_evidence_sha256: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    primary_candidate_id: str | None = None
    independent_source_count: int = Field(default=0, ge=0)
    scores: ScoreCard | None = None
    major: bool = False
    latest_update_id: str | None = None
    update_history: list[EventUpdate] = Field(default_factory=list)
    identity_terms: list[str] = Field(default_factory=list)
    config_version: str = "selection-v1"
    model_confidence: float | None = Field(default=None, ge=0, le=1)


class DecisionRecord(BaseModel):
    run_id: str = ""
    candidate_id: str | None = None
    event_id: str | None = None
    stage: ProcessingStage
    selected: bool
    primary_reason: ReasonCode
    secondary_reasons: list[ReasonCode] = Field(default_factory=list)
    scores: ScoreCard | None = None
    final_rank: int | None = Field(default=None, ge=1)
    ruleset_version: str = "selection-v1"
    model_version: str = "none"
    prompt_version: str | None = None
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    model_cache_hit: bool = False
    estimated_cost: float = Field(default=0, ge=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DeliveryRecord(BaseModel):
    run_id: str | None = None
    event_id: str
    update_id: str | None = None
    event: Event | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    channel: str
    status: str = Field(pattern=r"^(pending|delivered|failed)$")
    delivered_at: datetime | None = None

    @field_validator("run_id")
    @classmethod
    def optional_run_id_must_be_path_safe(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_run_id(value)
        return value

    @model_validator(mode="after")
    def embedded_event_must_match_delivery_version(self) -> DeliveryRecord:
        if self.event is not None and (
            self.event.event_id != self.event_id
            or self.event.latest_update_id != self.update_id
        ):
            raise ValueError("embedded event must match delivery version")
        return self


class RunRecord(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    selection_engine: str
    ruleset_version: str = "selection-v1"
    model_version: str = "none"
    source_candidate_counts: dict[str, int] = Field(default_factory=dict)
    stage_counts: dict[ProcessingStage, int] = Field(default_factory=dict)
    reason_counts: dict[ReasonCode, int] = Field(default_factory=dict)
    model_calls: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0, ge=0)
    fallback_reasons: list[ReasonCode] = Field(default_factory=list)

    @field_validator("run_id")
    @classmethod
    def run_id_must_be_path_safe(cls, value: str) -> str:
        return _validate_run_id(value)


def _validate_run_id(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("run_id must be a non-empty path segment")
    return value


class NewsValueDisplay(BaseModel):
    why_today: str
    impact: str
    event_time: datetime
    date_label: Literal["事件／进展日期", "本次披露日期", "实质更新日期"] = "事件／进展日期"
    attention_note: str
    attention_url: HttpUrl | None = None
    value_basis_url: HttpUrl | None = None
    value_basis_source: str | None = None


class EnrichedArticle(Article):
    news_value: NewsValueDisplay | None = None
    event_ref: str | None = None
    chinese_title: str
    chinese_summary: str
    key_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    beginner_context: str | None = None
    claim_type: str = Field(default="fact", pattern=r"^(fact|opinion|speculation)$")
    degraded: bool = False


class BackgroundReading(BaseModel):
    """Public projection only; private review material never enters a digest."""

    id: str
    title: str
    url: HttpUrl
    source: str
    published_on: date
    kind: Literal["background", "analysis", "opinion"]
    why_read: str
    limitation: str
    language: Literal["zh", "en"]


class Digest(BaseModel):
    demo_mode: bool = False
    local_preview: bool = False
    coverage_notes: list[str] = Field(default_factory=list)
    digest_date: date
    articles: list[EnrichedArticle]
    background_readings: list[BackgroundReading] = Field(default_factory=list, max_length=2)
    public_url: HttpUrl | None = None


class Preferences(BaseModel):
    topic_weights: dict[Topic, float] = Field(default_factory=dict)
    source_weights: dict[str, float] = Field(default_factory=dict)
    positive_tags: dict[str, int] = Field(default_factory=dict)
    negative_tags: dict[str, int] = Field(default_factory=dict)

    @field_validator("topic_weights", "source_weights", mode="before")
    @classmethod
    def clamp_weights(cls, values: dict | None) -> dict:
        return {key: min(2.0, max(0.5, float(value))) for key, value in (values or {}).items()}
