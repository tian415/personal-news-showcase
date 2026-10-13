"""Content-bound editorial selection for explicit local previews, never production.

Scores are reviewer judgments, not automatically verified truth or a quality metric.
The exact input and quotation bindings prevent stale/unattributed judgments from
silently turning into recommendations. Pending and unverified material stays out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .article_content import CapturedContent, captured_contents
from .content_quality import promotional_reason, roundup_reason
from .coverage_evaluation import NonEmpty, _AuditSnapshot, _aware, _fingerprint
from .events import assign_events
from .feeds import canonicalize_url
from .models import (
    Article,
    CandidateArticle,
    CoverageDomain,
    Event,
    EvidenceRole,
    Preferences,
    Topic,
)
from .news_value import (
    assign_value_events,
    evaluate_value,
    has_attention,
    rank_value,
    validate_value_review,
)
from .selection import QUALITY_FLOOR, select_digest
from .source_text import readable_source_text


class ArticleReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: NonEmpty
    article_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "accepted", "rejected"] = "pending"
    evidence_excerpt: NonEmpty
    quotation_basis: Literal["captured-v1", "readable-v1"] = "captured-v1"
    quality_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    importance_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    fact_status: Literal["reported", "opinion", "unverified"] = "unverified"
    story_shape: Literal["unclassified", "single_event", "roundup"] = "unclassified"
    origin_publisher_group: NonEmpty | None = None
    topic: Topic
    coverage_domain: CoverageDomain
    rationale: str = ""
    major_reason: str | None = None
    chinese_title: str | None = Field(default=None, min_length=1, max_length=160)
    chinese_summary: str | None = Field(default=None, min_length=1, max_length=800)

    @model_validator(mode="after")
    def require_judgment(self) -> ArticleReview:
        if bool(self.chinese_title) != bool(self.chinese_summary):
            raise ValueError("review copy requires both title and summary")
        if self.chinese_title is not None and (
                not self.chinese_title.strip() or not self.chinese_summary.strip()):
            raise ValueError("review copy requires meaningful title and summary")
        if self.status != "pending" and not self.rationale.strip():
            raise ValueError("a review decision requires a rationale")
        if self.status == "accepted" and (
            self.quality_score is None or self.importance_score is None
        ):
            raise ValueError("accepted articles need explicit content scores")
        if self.major_reason is not None and not self.major_reason.strip():
            raise ValueError("major_reason must be meaningful or null")
        return self


class EditorialReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["editorial-review-v1"]
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    items: list[ArticleReview]

    @field_validator("reviewed_at")
    @classmethod
    def aware_review_time(cls, value):
        return _aware(value) if value is not None else None

    @model_validator(mode="after")
    def attributed_decisions(self) -> EditorialReview:
        if any(item.status != "pending" for item in self.items) and (
            not self.reviewed_by or not self.reviewed_by.strip() or self.reviewed_at is None
        ):
            raise ValueError("decisions require a reviewer and review time")
        if len({item.candidate_id for item in self.items}) != len(self.items):
            raise ValueError("duplicate candidate reviews")
        return self


def article_fingerprint(article: Article | dict, content: CapturedContent | None = None) -> str:
    value = Article.model_validate(article).model_dump(mode="json")
    if content is not None:
        value["article_content"] = content.model_dump(mode="json")
    return _fingerprint(value)


def _unique_articles(articles: list[Article]) -> list[Article]:
    unique: dict[str, Article] = {}
    for article in articles:
        if article.id in unique and unique[article.id] != article:
            raise ValueError("conflicting repeated article content")
        unique[article.id] = article
    return list(unique.values())


def prepare_review(audit: dict) -> dict:
    snapshot = _AuditSnapshot.model_validate(audit)
    contents = captured_contents(audit, snapshot)
    domains = {source["id"]: CoverageDomain(source["coverage_domains"][0]) for source in audit["sources"]}
    return EditorialReview(
        schema_version="editorial-review-v1", snapshot_sha256=_fingerprint(audit),
        items=[ArticleReview(
            candidate_id=f"{source.id}:{article.id}",
            article_sha256=article_fingerprint(
                article, contents.get(f"{source.id}:{article.id}")), evidence_excerpt=article.title,
            topic=article.topic, coverage_domain=domains[source.id],
            story_shape="roundup" if roundup_reason(article) else "unclassified",
        ) for source in snapshot.sources for article in _unique_articles(source.candidates)],
    ).model_dump(mode="json")


@dataclass
class ReviewedSelection:
    selected: list[CandidateArticle] = field(default_factory=list)
    candidates: list[CandidateArticle] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    value_details: dict[str, dict] = field(default_factory=dict)


def select_reviewed(
    audit: dict, review: dict, preferences: Preferences, *, now: datetime,
    historical_events: list[Event] = (), include_shadow: bool = False,
    tracked_event_ids: list[str] = (), limit: int | None = None,
    value_review: dict | None = None,
) -> ReviewedSelection:
    now = _aware(now)
    limit = (15 if value_review is not None else 10) if limit is None else limit
    if limit < 0:
        raise ValueError("limit must be non-negative")
    snapshot = _AuditSnapshot.model_validate(audit)
    contents = captured_contents(audit, snapshot)
    assessed = EditorialReview.model_validate(review)
    if snapshot.captured_at > now:
        raise ValueError("capture is in the future")
    if assessed.snapshot_sha256 != _fingerprint(audit):
        raise ValueError("review is bound to a different candidate snapshot")
    if assessed.reviewed_at and not snapshot.captured_at <= assessed.reviewed_at <= now:
        raise ValueError("review time must fall between capture and selection")
    if any(item.retrieved_at > (assessed.reviewed_at or now) for item in contents.values()):
        raise ValueError("body capture must precede review and selection")
    entries = {item.candidate_id: item for item in assessed.items}
    expected = {f"{s.id}:{a.id}" for s in snapshot.sources for a in s.candidates}
    if entries.keys() != expected:
        raise ValueError("review must cover every captured candidate exactly once")
    raw_sources = {source["id"]: source for source in audit["sources"]}
    values = (validate_value_review(value_review, audit, review, now)
              if value_review is not None else None)
    material_hashes = {}
    result = ReviewedSelection()
    for source in snapshot.sources:
        metadata = raw_sources[source.id]
        role = EvidenceRole(metadata["evidence_role"])
        for article in _unique_articles(source.candidates):
            identifier = f"{source.id}:{article.id}"
            item = entries[identifier]
            content = contents.get(identifier)
            if item.article_sha256 != article_fingerprint(article, content):
                raise ValueError("review article fingerprint mismatch")
            transform = readable_source_text if item.quotation_basis == "readable-v1" else str
            captured_text = f"{transform(article.title)}\n{transform(article.summary)}"
            if content is not None and content.status == "extracted":
                captured_text += f"\n{content.text}"
            if item.evidence_excerpt not in captured_text:
                raise ValueError("review quotation is absent from captured content")
            value_reason = None
            if values is not None:
                if identifier not in values:
                    value_reason = "pending_news_value"
                else:
                    # Value annotations always quote readable text, independently of old review receipts.
                    value_text = f"{readable_source_text(article.title)}\n{readable_source_text(article.summary)}"
                    if content is not None and content.status == "extracted":
                        value_text += f"\n{content.text}"
                    material_hashes[identifier] = _fingerprint(value_text)
                    value_reason, result.value_details[identifier] = evaluate_value(
                        values[identifier], value_text, now)
            reason = None
            if not source.production_enabled and not include_shadow:
                reason = "shadow_source_not_enabled_for_preview"
            elif not now - timedelta(hours=36) <= article.published_at <= now:
                reason = "outside_window"
            elif item.status == "pending":
                reason = "pending_review"
            elif item.status == "rejected":
                reason = "editorial_rejection"
            elif item.story_shape == "roundup" or roundup_reason(article):
                reason = "roundup_requires_event_split"
            elif item.fact_status == "unverified":
                reason = "unverified_claim"
            elif item.quality_score < QUALITY_FLOOR:
                reason = "below_quality_floor"
            elif promotional_reason(article):
                reason = "promotional_content"
            elif role == EvidenceRole.COMMUNITY_SIGNAL:
                reason = "community_signal_not_fact_evidence"
            reason = reason or value_reason
            if reason:
                result.excluded[identifier] = reason
                continue
            candidate = CandidateArticle(
                **{**article.model_dump(), "id": identifier, "topic": item.topic,
                   "quality_score": item.quality_score, "importance_score": item.importance_score},
                candidate_id=identifier, source_id=source.id,
                publisher_group=item.origin_publisher_group or source.publisher_group,
                evidence_role=role, coverage_domain=item.coverage_domain, canonical_url=article.url,
                discovered_at=snapshot.captured_at,
            )
            result.candidates.append(candidate)
            result.reasons[identifier] = item.rationale
    # More reliable representatives are first, independently of preference weights.
    result.candidates.sort(key=lambda c: (
        c.evidence_role == EvidenceRole.PRIMARY_FACT, c.quality_score,
        c.importance_score, c.published_at, c.candidate_id,
    ), reverse=True)
    # A hosting publisher does not establish the origin of a syndicated report.
    # Keep unknown origins eligible on their own merit, without corroboration credit.
    supported_reporting_ids = {
        c.id for c in result.candidates
        if entries[c.id].fact_status == "reported"
        and (c.evidence_role == EvidenceRole.PRIMARY_FACT
             or (c.evidence_role == EvidenceRole.INDEPENDENT_REPORTING
                 and entries[c.id].origin_publisher_group is not None))
    }
    if values is not None:
        assigned_events, excluded = assign_value_events(
            result.candidates, values, historical_events,
            supported_reporting_ids,
            material_hashes)
        result.excluded.update(excluded)
    else:
        assignment = assign_events(result.candidates, historical_events)
        assigned_events = assignment.events
        for decision in assignment.decisions:
            if decision.primary_reason.value == "EVENT_DUPLICATE_NO_DELTA":
                result.excluded[decision.candidate_id] = "event_duplicate_no_delta"
    by_id = {item.id: item for item in result.candidates}
    major: list[CandidateArticle] = []
    tracked: list[CandidateArticle] = []
    ordinary: list[CandidateArticle] = []
    for event in assigned_events:
        members = [by_id[identifier] for identifier in event.candidate_ids]
        representative = members[0]
        reported = [c for c in members if entries[c.id].fact_status == "reported"]
        reporting = [c for c in reported if c.evidence_role == EvidenceRole.INDEPENDENT_REPORTING
                     and c.id in supported_reporting_ids]
        # Copies of one URL or one publisher do not establish corroboration.
        corroborated = (len({c.publisher_group for c in reporting}) >= 2
                        and len({canonicalize_url(str(c.url)) for c in reporting}) >= 2)
        strong_evidence = any(c.evidence_role == EvidenceRole.PRIMARY_FACT for c in reported) or corroborated
        if values is not None:
            count = len({c.publisher_group for c in reporting}) if corroborated else 0
            for member in members:
                result.value_details[member.id]['independent_reporting_count'] = count
            valuable = [c for c in members if result.value_details[c.id]['importance'] >= 0.55
                        or has_attention(result.value_details[c.id])]
            if not valuable:
                for member in members:
                    result.excluded[member.id] = "low_news_value"
                continue
            # Keep evidence preference, but never choose an ineligible representative.
            members = valuable
            reported = [c for c in members if entries[c.id].fact_status == "reported"]
            reporting = [c for c in reported if c.evidence_role == EvidenceRole.INDEPENDENT_REPORTING
                         and c.id in supported_reporting_ids]
            representative = members[0]
        protected = strong_evidence and any(
            entries[c.id].major_reason and c.importance_score >= 0.85 and c.quality_score >= 0.6
            and (values is None or result.value_details[c.id]["importance"] >= 0.85)
            for c in reported
        )
        if protected:
            supporting = [c for c in reported if c.evidence_role == EvidenceRole.PRIMARY_FACT]
            representative = (supporting or reporting)[0]
        elif reported:
            representative = reported[0]
        if values is not None:
            strongest = max(members, key=lambda c: result.value_details[c.id]['importance'])
            strongest_detail = dict(result.value_details[strongest.id])
            strongest_detail['value_basis_candidate_id'] = strongest.id
            strongest_detail['value_basis_url'] = str(strongest.url)
            strongest_detail['value_basis_source'] = strongest.source
            result.value_details[representative.id] = strongest_detail
        unique_evidence = {canonicalize_url(str(c.url)): c for c in members
                           if c.id in supported_reporting_ids}
        event = event.model_copy(update={
            "major": bool(protected), "primary_candidate_id": representative.id,
            "independent_source_count": len({c.publisher_group for c in unique_evidence.values()}),
        })
        result.events.append(event)
        is_tracked_update = (event.event_id in tracked_event_ids and event.update_history
                             and event.update_history[-1].material and strong_evidence)
        (major if protected else tracked if is_tracked_update else ordinary).append(representative)
    # Major events are protected even when they exceed the normal ten-item budget.
    # Ordinary local items keep 5/3/2 slots; missing topics stay visibly empty.
    prioritized = [*major, *tracked[:max(0, limit - len(major))]]
    if values is not None:
        major = rank_value(major, result.value_details, preferences)
        tracked = rank_value(tracked, result.value_details, preferences)
        prioritized = [*major, *tracked[:max(0, limit - len(major))]]
        ranked = rank_value(ordinary, result.value_details, preferences)
    else:
        ranked = select_digest(ordinary, preferences, limit=limit, now=now, fill_shortfalls=False)
    result.selected = [*prioritized, *ranked[:max(0, limit - len(prioritized))]]
    selected_ids = {item.id for item in result.selected}
    represented_ids = {item.id for item in [*prioritized, *ordinary]}
    for candidate in result.candidates:
        if candidate.id not in selected_ids and candidate.id not in result.excluded:
            result.excluded[candidate.id] = (
                "lost_to_higher_value" if candidate.id in represented_ids else "same_event_alternative"
            )
    return result
