from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from .feeds import FeedSource
from .models import CoverageDomain, EvidenceRole, SourceKind, SourceTier, Topic


class RegisteredSource(BaseModel):
    id: str
    name: str
    tier: SourceTier
    adapter: str
    url: HttpUrl
    topics: list[Topic] = Field(min_length=1)
    coverage_domains: list[CoverageDomain] = Field(min_length=1)
    publisher_group: str
    evidence_role: EvidenceRole | None = None
    source_kind: SourceKind | None = None
    language: str = Field(default="en", pattern=r"^(zh|en)$")
    naive_timezone: str | None = None
    terms_reviewed_at: date | None = None
    access_stability: str = Field(default="rss", min_length=1)
    signal_weight: float = Field(default=0, ge=0, le=1)
    enabled: bool = True
    production_enabled: bool = True
    fetch_interval_hours: int = Field(default=24, ge=1)
    failure_policy: str = "continue"

    @field_validator("naive_timezone")
    @classmethod
    def naive_timezone_must_be_known(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError(f"unknown timezone: {value}") from error
        return value

    @model_validator(mode="after")
    def apply_legacy_role_defaults(self) -> RegisteredSource:
        if self.evidence_role is None:
            self.evidence_role = {
                SourceTier.T0: EvidenceRole.PRIMARY_FACT,
                SourceTier.T1: EvidenceRole.INDEPENDENT_REPORTING,
                SourceTier.T2: EvidenceRole.EXPERT_ANALYSIS,
                SourceTier.T3: EvidenceRole.COMMUNITY_SIGNAL,
            }[self.tier]
        if self.source_kind is None:
            self.source_kind = (
                SourceKind.OFFICIAL if self.tier is SourceTier.T0 else SourceKind.MEDIA
            )
        return self

    @property
    def coverage_domain(self) -> CoverageDomain:
        return self.coverage_domains[0]

    def as_feed_source(self) -> FeedSource:
        if self.adapter != "rss":
            raise ValueError(f"source {self.id} is not an RSS source")
        return FeedSource(
            name=self.name,
            url=self.url,
            language=self.language,
            topic=self.topics[0],
            naive_timezone=self.naive_timezone,
        )

    def candidate_metadata(self) -> tuple[str, SourceTier, CoverageDomain]:
        return self.id, self.tier, self.coverage_domain


class SourceRegistry(BaseModel):
    version: str
    sources: list[RegisteredSource]

    @model_validator(mode="after")
    def validate_unique_source_identities(self) -> SourceRegistry:
        for field in ("id", "name"):
            seen: set[str] = set()
            for source in self.sources:
                value = getattr(source, field)
                if value in seen:
                    raise ValueError(f"duplicate source {field}: {value}")
                seen.add(value)
        return self

    @model_validator(mode="after")
    def validate_enabled_v2_sources(self) -> SourceRegistry:
        if self.version != "source-registry-v2":
            return self
        for source in self.sources:
            if not source.enabled:
                continue
            if source.terms_reviewed_at is None:
                raise ValueError(
                    f"enabled source {source.id} requires terms_reviewed_at"
                )
            if source.access_stability not in {"public_rss", "public_api"}:
                raise ValueError(
                    f"enabled source {source.id} requires approved access_stability"
                )
        return self

    def enabled_feed_sources(self) -> list[RegisteredSource]:
        return [
            source
            for source in self.enabled_sources("rss")
            if source.production_enabled
        ]

    def enabled_sources(self, adapter: str) -> list[RegisteredSource]:
        return [source for source in self.sources if source.enabled and source.adapter == adapter]


def load_source_registry(path: Path) -> SourceRegistry:
    return SourceRegistry.model_validate(yaml.safe_load(path.read_text()))
