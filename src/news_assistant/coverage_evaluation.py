"""Offline reference-URL coverage replay; never claims fact or delivery acceptance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

from .feeds import canonicalize_url
from .models import Article, CoverageDomain, Preferences
from .selection import select_digest

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("reference times must be timezone-aware")
    return value.astimezone(UTC)


class ReferenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: NonEmpty
    title: NonEmpty
    coverage_domain: CoverageDomain
    available_at: datetime
    major: bool = Field(strict=True)
    verified: bool = Field(strict=True)
    verification_note: NonEmpty
    evidence_urls: list[HttpUrl] = Field(min_length=1)

    _aware_time = field_validator("available_at")(_aware)


class CoverageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["coverage-reference-v1"]
    synthetic: bool = Field(strict=True)
    purpose: Literal["calibration", "evaluation"]
    reviewed_by: NonEmpty
    reviewed_at: datetime
    reference_origin: NonEmpty
    events: list[ReferenceEvent]

    _aware_review = field_validator("reviewed_at")(_aware)

    @model_validator(mode="after")
    def unambiguous_references(self) -> CoverageReference:
        seen_ids: set[str] = set()
        seen_urls: dict[str, str] = {}
        for event in self.events:
            if event.event_id in seen_ids:
                raise ValueError("duplicate reference event_id")
            seen_ids.add(event.event_id)
            for url in event.evidence_urls:
                canonical = canonicalize_url(str(url))
                if canonical in seen_urls and seen_urls[canonical] != event.event_id:
                    raise ValueError("one URL cannot identify multiple reference events")
                seen_urls[canonical] = event.event_id
        return self


class _SourceSnapshot(BaseModel):
    id: NonEmpty
    name: NonEmpty
    publisher_group: NonEmpty
    production_enabled: bool = Field(strict=True)
    status: Literal["ok", "no_recent", "error"]
    parsed_count: int = Field(ge=0, strict=True)
    fresh_count: int = Field(ge=0, strict=True)
    stale_count: int = Field(ge=0, strict=True)
    future_count: int = Field(ge=0, strict=True)
    candidates: list[Article]

    @model_validator(mode="after")
    def complete_capture(self) -> _SourceSnapshot:
        if self.fresh_count != len(self.candidates):
            raise ValueError("candidate snapshot count mismatch")
        if self.parsed_count != self.fresh_count + self.stale_count + self.future_count:
            raise ValueError("source snapshot counts are inconsistent")
        if self.status == "error" and self.parsed_count:
            raise ValueError("failed source cannot contain parsed candidates")
        if (self.status == "ok") != bool(self.fresh_count):
            raise ValueError("source status and fresh count disagree")
        if any(article.source != self.name for article in self.candidates):
            raise ValueError("candidate source does not match captured source")
        return self


class _AuditSnapshot(BaseModel):
    schema_version: Literal["source-audit-v1"]
    candidate_snapshot_available: bool = Field(strict=True)
    captured_at: datetime
    max_age_hours: int = Field(gt=0, strict=True)
    sources: list[_SourceSnapshot]

    _aware_capture = field_validator("captured_at")(_aware)

    @model_validator(mode="after")
    def validate_capture(self) -> _AuditSnapshot:
        if not self.candidate_snapshot_available:
            raise ValueError("audit must have candidate capture enabled")
        cutoff = self.captured_at - timedelta(hours=self.max_age_hours)
        source_ids: set[str] = set()
        source_names: set[str] = set()
        article_urls: dict[str, str] = {}
        for source in self.sources:
            if source.id in source_ids or source.name in source_names:
                raise ValueError("duplicate captured source identity")
            source_ids.add(source.id)
            source_names.add(source.name)
            for article in source.candidates:
                if not cutoff <= article.published_at <= self.captured_at:
                    raise ValueError("candidate is outside the captured window")
                url = canonicalize_url(str(article.url))
                if article.id in article_urls and article_urls[article.id] != url:
                    raise ValueError("conflicting article identity")
                article_urls[article.id] = url
        return self


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _metrics(events: list[dict]) -> dict:
    major = [event for event in events if event["major"]]

    def ratio(key: str) -> float | None:
        return sum(bool(event[key]) for event in major) / len(major) if major else None

    return {
        "reference_event_count": len(events),
        "major_reference_count": len(major),
        "production_discovered_count": sum(bool(e["production_source_ids"]) for e in events),
        "all_sources_discovered_count": sum(bool(e["all_source_ids"]) for e in events),
        "legacy_selected_count": sum(e["legacy_selected"] for e in events),
        "shadow_only_count": sum(e["status"] == "shadow_only" for e in events),
        "production_major_discovery_recall": ratio("production_source_ids"),
        "all_sources_major_discovery_recall": ratio("all_source_ids"),
        "legacy_major_selection_recall": ratio("legacy_selected"),
    }


def evaluate_coverage(audit: dict, reference: dict, preferences: Preferences) -> dict:
    """Replay legacy on captured production inputs; match reviewed URL aliases only.

    A reference's verified flag is a reviewer's declaration, not verification by this tool.
    Unknown URL aliases need human review; title similarity never becomes a scored hit.
    """
    snapshot = _AuditSnapshot.model_validate(audit)
    refs = CoverageReference.model_validate(reference)
    cutoff = snapshot.captured_at - timedelta(hours=snapshot.max_age_hours)
    selected = select_digest(
        [article for source in snapshot.sources if source.production_enabled
         for article in source.candidates], preferences, now=snapshot.captured_at,
    )
    selected_urls = {canonicalize_url(str(article.url)) for article in selected}
    by_url: dict[str, list[_SourceSnapshot]] = {}
    for source in snapshot.sources:
        for article in source.candidates:
            by_url.setdefault(canonicalize_url(str(article.url)), []).append(source)
    events = []
    excluded = []
    for event in refs.events:
        if not event.verified or not cutoff <= event.available_at <= snapshot.captured_at:
            excluded.append({"event_id": event.event_id, "reason": (
                "unverified" if not event.verified else "outside_capture_window"
            )})
            continue
        urls = {canonicalize_url(str(url)) for url in event.evidence_urls}
        matches = {s.id: s for url in urls for s in by_url.get(url, [])}
        production_ids = sorted(s.id for s in matches.values() if s.production_enabled)
        legacy_selected = bool(urls & selected_urls)
        status = ("selected" if legacy_selected else "not_selected" if production_ids else
                  "shadow_only" if matches else "not_discovered")
        events.append({
            "event_id": event.event_id, "title": event.title,
            "coverage_domain": event.coverage_domain.value, "major": event.major,
            "status": status, "production_source_ids": production_ids,
            "all_source_ids": sorted(matches),
            "publisher_groups": sorted({s.publisher_group for s in matches.values()}),
            "matched_urls": sorted(urls & by_url.keys()), "legacy_selected": legacy_selected,
        })
    return {
        "schema_version": "coverage-evaluation-v1",
        "synthetic": refs.synthetic, "purpose": refs.purpose,
        "measurement": "reviewed_reference_url_coverage_not_fact_or_delivery_acceptance",
        "window": {"from": cutoff.isoformat(), "to": snapshot.captured_at.isoformat()},
        "input_fingerprints": {"audit_sha256": _fingerprint(audit),
                               "reference_sha256": _fingerprint(reference),
                               "preferences_sha256": _fingerprint(preferences.model_dump(mode="json"))},
        "summary": _metrics(events),
        "domains": {domain.value: _metrics([e for e in events if e["coverage_domain"] == domain])
                    for domain in CoverageDomain},
        "events": events, "excluded_references": excluded,
        "legacy_selected_urls": sorted(selected_urls),
        "source_errors": [source.id for source in snapshot.sources if source.status == "error"],
        "release_readiness": {"status": "not_ready", "missing_evidence": [
            "independent_reference_completeness_and_url_alias_review",
            "8_week_reference_set_and_holdout_acceptance",
            "fact_quality_and_actual_delivery_acceptance",
        ]},
    }
