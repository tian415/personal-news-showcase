"""Explicit local reading recommendations, separate from news and its feedback."""

import json
from datetime import date, datetime
from hashlib import sha256
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from .coverage_evaluation import NonEmpty, _aware
from .feeds import canonicalize_url
from .models import BackgroundReading, DeliveryRecord
from .state import StateStore


class _ReadingReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: NonEmpty
    url: HttpUrl
    source: NonEmpty
    published_on: date
    kind: Literal["background", "analysis", "opinion"]
    why_read: NonEmpty
    limitation: NonEmpty
    language: Literal["zh", "en"]
    material: NonEmpty
    material_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_excerpt: NonEmpty
    review_scope: Literal["excerpt", "full_text"]


class _ReadingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["background-readings-v1"]
    reviewed_by: NonEmpty
    reviewed_at: datetime
    items: list[_ReadingReview] = Field(max_length=2)

    _aware_time = field_validator("reviewed_at")(_aware)


def _confirmed_urls(store: StateStore, now: datetime) -> set[str]:
    """Only pages the user confirmed locally suppress future recommendations."""
    runs = set()
    for path in (store.root / "deliveries").glob("*/*/deliveries.jsonl"):
        for raw in store._read_jsonl(path):
            record = DeliveryRecord.model_validate(raw)
            if (record.channel == "local" and record.status == "delivered"
                    and record.run_id and record.delivered_at
                    and _aware(record.delivered_at) <= now):
                runs.add(record.run_id)
    urls = set()
    for run_id in runs:
        report_path = store._find_run_directory(run_id) / "reviewed-selection.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            urls.update(canonicalize_url(url) for url in report.get("background_reading_urls", []))
    return urls


def select_background_readings(
    manifest: dict | None, *, now: datetime, store: StateStore, news_urls: list[str],
) -> list[BackgroundReading]:
    if manifest is None:
        return []
    now = _aware(now)
    review = _ReadingManifest.model_validate(manifest)
    if review.reviewed_at > now:
        raise ValueError("background review cannot be in the future")
    cutoff = review.reviewed_at.astimezone(ZoneInfo("Asia/Shanghai")).date()
    seen = set()
    for item in review.items:
        url = canonicalize_url(str(item.url))
        if url in seen or item.published_on > cutoff:
            raise ValueError("duplicate or future background reading")
        seen.add(url)
        if (sha256(item.material.encode()).hexdigest() != item.material_sha256
                or item.evidence_excerpt not in item.material):
            raise ValueError("background evidence binding mismatch")
    excluded = _confirmed_urls(store, now) | {canonicalize_url(url) for url in news_urls}
    output = []
    for item in review.items:
        url = canonicalize_url(str(item.url))
        if url in excluded:
            continue
        limitation = item.limitation
        if item.review_scope == "excerpt":
            limitation = "仅据公开片段推荐，未完成全文审阅。" + limitation
        output.append(BackgroundReading(
            id=sha256(url.encode()).hexdigest()[:16],
            **item.model_dump(include={"title", "url", "source", "published_on", "kind",
                                      "why_read", "language"}),
            limitation=limitation,
        ))
    return output
