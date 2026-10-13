from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from itertools import pairwise

from pydantic import BaseModel, Field

from .models import SignalSnapshot


class TrendAssessment(BaseModel):
    snapshot_count: int = Field(ge=0)
    attention_level: float | None = Field(default=None, ge=0, le=1)
    attention_velocity: float | None = Field(default=None, ge=0)
    attention_acceleration: float | None = None
    confidence: float = Field(ge=0, le=1)
    status: str = Field(
        pattern=(
            r"^(no_data|snapshot_only|velocity_available|acceleration_available|"
            r"counter_reset|conflicting_snapshots)$"
        )
    )


def assess_trend(snapshots: Iterable[SignalSnapshot], *, now: datetime) -> TrendAssessment:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    records = list(snapshots)
    identities = {(item.source_id, item.external_id) for item in records}
    if len(identities) > 1:
        raise ValueError("snapshots must use the same source_id and external_id")

    reference = now.astimezone(UTC)
    cutoff = reference - timedelta(hours=24)
    in_window = [
        item
        for item in records
        if cutoff <= item.captured_at.astimezone(UTC) <= reference
    ]
    grouped: dict[datetime, list[SignalSnapshot]] = {}
    for item in in_window:
        grouped.setdefault(item.captured_at.astimezone(UTC), []).append(item)

    selected: list[SignalSnapshot] = []
    has_conflicting_timestamp = False
    for captured_at in sorted(grouped):
        candidates = grouped[captured_at]
        if len({(item.reply_count, item.rank) for item in candidates}) > 1:
            has_conflicting_timestamp = True
        # Lower counts and lower attention rank are conservative. The final
        # fields make the selection stable even when malformed duplicates tie.
        chosen = min(candidates, key=lambda item: (item.reply_count, -item.rank))
        selected.append(chosen)

    count = len(selected)
    if count == 0:
        return TrendAssessment(snapshot_count=0, confidence=0, status="no_data")

    latest = selected[-1]
    attention_level = 1 / latest.rank
    if has_conflicting_timestamp:
        return TrendAssessment(
            snapshot_count=count,
            attention_level=attention_level,
            confidence=0,
            status="conflicting_snapshots",
        )
    if count == 1:
        return TrendAssessment(
            snapshot_count=1,
            attention_level=attention_level,
            confidence=0.25,
            status="snapshot_only",
        )

    decreases = any(
        newer.reply_count < older.reply_count
        for older, newer in pairwise(selected)
    )
    if decreases:
        return TrendAssessment(
            snapshot_count=count,
            attention_level=attention_level,
            confidence=0,
            status="counter_reset",
        )

    velocity = _relative_velocity(selected[-2], latest)
    if count < 3:
        return TrendAssessment(
            snapshot_count=count,
            attention_level=attention_level,
            attention_velocity=velocity,
            confidence=0.5,
            status="velocity_available",
        )

    first_velocity = _relative_velocity(selected[-3], selected[-2])
    second_velocity = _relative_velocity(selected[-2], selected[-1])
    midpoint_spacing = (
        _hours_between(selected[-2].captured_at, selected[-1].captured_at)
        + _hours_between(selected[-3].captured_at, selected[-2].captured_at)
    ) / 2
    acceleration = (second_velocity - first_velocity) / midpoint_spacing
    return TrendAssessment(
        snapshot_count=count,
        attention_level=attention_level,
        attention_velocity=velocity,
        attention_acceleration=acceleration,
        confidence=0.75,
        status="acceleration_available",
    )


def _hours_between(older: datetime, newer: datetime) -> float:
    hours = (newer.astimezone(UTC) - older.astimezone(UTC)).total_seconds() / 3600
    if hours <= 0:
        raise ValueError("snapshots must have distinct, increasing timestamps")
    return hours


def _relative_velocity(older: SignalSnapshot, newer: SignalSnapshot) -> float:
    return (newer.reply_count - older.reply_count) / max(older.reply_count, 1) / _hours_between(
        older.captured_at, newer.captured_at
    )
