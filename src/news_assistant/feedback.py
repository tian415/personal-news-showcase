from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .models import Preferences, Topic


class _FeedbackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def public_event_ref(
    event_id: str, *, run_id: str | None = None, update_id: str | None = None,
) -> str:
    if not isinstance(event_id, str):
        raise TypeError("event_id must be a string")
    binding = (json.dumps([run_id, event_id, update_id], separators=(",", ":"))
               if run_id is not None else f"news-event:{event_id}")
    return sha256(binding.encode()).hexdigest()[:24]


class FeedbackTarget(_FeedbackModel):
    event_id: StrictStr
    topic: Topic
    source: StrictStr


class FeedbackEntry(_FeedbackModel):
    event_ref: StrictStr
    vote: Literal["helpful", "reduce"] | None = None
    reason: Literal[
        "already_known",
        "irrelevant",
        "shallow",
        "insufficient_evidence",
        "duplicate",
    ] | None = None
    tracking: StrictBool
    updated_at: datetime

    @field_validator("updated_at", mode="before")
    @classmethod
    def reject_numeric_timestamps(cls, value: object) -> object:
        if not isinstance(value, (datetime, str)):
            raise PydanticCustomError(
                "strict_datetime", "updated_at must be an ISO datetime string or datetime"
            )
        return value

    @field_validator("updated_at")
    @classmethod
    def updated_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_reason_for_vote(self) -> FeedbackEntry:
        if self.vote == "reduce" and self.reason is None:
            raise ValueError("reduce vote requires a reason")
        if self.vote != "reduce" and self.reason is not None:
            raise ValueError("only reduce vote may include a reason")
        return self


class FeedbackPayload(_FeedbackModel):
    schema_version: Literal["news-feedback-v2"]
    events: list[FeedbackEntry] = Field(max_length=10000)


class FeedbackState(_FeedbackModel):
    schema_version: Literal["feedback-state-v1"] = "feedback-state-v1"
    events: dict[str, FeedbackEntry] = Field(default_factory=dict)
    preferences: Preferences = Field(default_factory=Preferences)
    tracked_event_ids: list[str] = Field(default_factory=list)
    diagnostics: dict[str, int] = Field(default_factory=dict)


def merge_feedback(
    payload: dict,
    known_targets: dict[str, FeedbackTarget],
    *,
    previous: FeedbackState | None = None,
    now: datetime,
) -> FeedbackState:
    parsed = FeedbackPayload.model_validate(payload)
    _ensure_timezone_aware(now, "now")

    for ref, target in known_targets.items():
        if not isinstance(ref, str) or not isinstance(target, FeedbackTarget):
            raise TypeError("known_targets must map string refs to FeedbackTarget values")

    if previous is not None:
        missing_previous = sorted(set(previous.events) - set(known_targets))
        if missing_previous:
            raise ValueError(f"previous event_ref requires target mapping: {missing_previous[0]}")

    unknown_refs = sorted({item.event_ref for item in parsed.events} - set(known_targets))
    if unknown_refs:
        raise ValueError(f"unknown event_ref: {unknown_refs[0]}")

    future_limit = now + timedelta(minutes=5)
    future_refs = sorted(
        item.event_ref for item in parsed.events if item.updated_at > future_limit
    )
    if future_refs:
        raise ValueError(f"updated_at is in the future: {future_refs[0]}")

    merged: dict[str, FeedbackEntry] = dict(previous.events) if previous is not None else {}
    for incoming in parsed.events:
        existing = merged.get(incoming.event_ref)
        if existing is None or incoming.updated_at > existing.updated_at:
            merged[incoming.event_ref] = incoming
        elif incoming.updated_at == existing.updated_at and incoming != existing:
            raise ValueError(f"conflicting feedback with same updated_at: {incoming.event_ref}")

    ordered_events = {ref: merged[ref] for ref in sorted(merged)}
    topic_weights: dict[Topic, float] = {}
    source_weights: dict[str, float] = {}
    diagnostics: Counter[str] = Counter()
    tracked_event_ids: list[str] = []
    tracked_seen: set[str] = set()

    latest_by_event: dict[str, tuple[str, FeedbackEntry]] = {}
    for ref, item in ordered_events.items():
        event_id = known_targets[ref].event_id
        current = latest_by_event.get(event_id)
        if current is None or item.updated_at > current[1].updated_at:
            latest_by_event[event_id] = (ref, item)
    for ref, item in latest_by_event.values():
        target = known_targets[ref]
        if item.tracking and target.event_id not in tracked_seen:
            tracked_event_ids.append(target.event_id)
            tracked_seen.add(target.event_id)
        if item.vote == "helpful":
            topic_weights[target.topic] = topic_weights.get(target.topic, 1.0) + 0.1
            source_weights[target.source] = source_weights.get(target.source, 1.0) + 0.1
        elif item.vote == "reduce":
            if item.reason == "irrelevant":
                topic_weights[target.topic] = topic_weights.get(target.topic, 1.0) - 0.1
                source_weights[target.source] = source_weights.get(target.source, 1.0) - 0.1
            elif item.reason is not None:
                diagnostics[item.reason] += 1

    return FeedbackState(
        events=ordered_events,
        preferences=Preferences(
            topic_weights=topic_weights,
            source_weights=source_weights,
        ),
        tracked_event_ids=tracked_event_ids,
        diagnostics=dict(sorted(diagnostics.items())),
    )


def _ensure_timezone_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
