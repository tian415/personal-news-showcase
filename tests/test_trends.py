from datetime import UTC, datetime, timedelta, timezone

import pytest

from news_assistant.models import SignalSnapshot
from news_assistant.trends import TrendAssessment, assess_trend


def snapshot(
    captured_at: datetime,
    *,
    reply_count: int,
    rank: int = 1,
    source_id: str = "forum",
    external_id: str = "topic-1",
) -> SignalSnapshot:
    return SignalSnapshot(
        source_id=source_id,
        external_id=external_id,
        captured_at=captured_at,
        rank=rank,
        reply_count=reply_count,
    )


def test_empty_snapshots_have_no_data() -> None:
    result = assess_trend([], now=datetime(2026, 9, 12, 12, tzinfo=UTC))

    assert isinstance(result, TrendAssessment)
    assert result.model_dump() == {
        "snapshot_count": 0,
        "attention_level": None,
        "attention_velocity": None,
        "attention_acceleration": None,
        "confidence": 0,
        "status": "no_data",
    }


def test_one_snapshot_reports_reciprocal_rank_without_growth_claim() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)

    result = assess_trend([snapshot(now - timedelta(hours=2), reply_count=9, rank=4)], now=now)

    assert result.snapshot_count == 1
    assert result.attention_level == pytest.approx(0.25)
    assert result.attention_velocity is None
    assert result.attention_acceleration is None
    assert result.status == "snapshot_only"
    assert 0 <= result.confidence <= 1


def test_window_future_and_duplicate_records_are_filtered_deterministically() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    first = now - timedelta(hours=6)
    result = assess_trend(
        [
            snapshot(now + timedelta(minutes=1), reply_count=100),
            snapshot(now - timedelta(hours=25), reply_count=100),
            snapshot(now, reply_count=4, rank=2),
            snapshot(first, reply_count=0, rank=5),
            snapshot(first, reply_count=0, rank=5),
        ],
        now=now,
    )

    assert result.snapshot_count == 2
    assert result.attention_level == pytest.approx(0.5)
    assert result.attention_velocity == pytest.approx(4 / 6)
    assert result.attention_acceleration is None
    assert result.status == "velocity_available"


def test_velocity_uses_relative_growth_and_zero_baseline() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)

    result = assess_trend(
        [
            snapshot(now - timedelta(hours=3), reply_count=0),
            snapshot(now, reply_count=3),
        ],
        now=now,
    )

    assert result.attention_velocity == pytest.approx(1)
    assert result.status == "velocity_available"


def test_three_uneven_time_points_provide_acceleration() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)

    result = assess_trend(
        [
            snapshot(now - timedelta(hours=9), reply_count=1),
            snapshot(now - timedelta(hours=6), reply_count=3),
            snapshot(now, reply_count=9),
        ],
        now=now,
    )

    # v1 = 2 / 1 / 3; v2 = 6 / 3 / 6; midpoint spacing = (3 + 6) / 2.
    assert result.attention_velocity == pytest.approx((9 - 3) / 3 / 6)
    assert result.attention_acceleration == pytest.approx(((6 / 3 / 6) - (2 / 1 / 3)) / 4.5)
    assert result.status == "acceleration_available"


def test_counter_decrease_does_not_claim_growth() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)

    result = assess_trend(
        [
            snapshot(now - timedelta(hours=6), reply_count=10),
            snapshot(now - timedelta(hours=3), reply_count=5),
            snapshot(now, reply_count=7),
        ],
        now=now,
    )

    assert result.attention_level == pytest.approx(1)
    assert result.attention_velocity is None
    assert result.attention_acceleration is None
    assert result.confidence == 0
    assert result.status == "counter_reset"


def test_conflicting_same_timestamp_cannot_fabricate_growth() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    middle = now - timedelta(hours=3)

    result = assess_trend(
        [
            snapshot(now - timedelta(hours=6), reply_count=1),
            snapshot(middle, reply_count=2),
            snapshot(middle, reply_count=20),
            snapshot(now, reply_count=4),
        ],
        now=now,
    )

    assert result.snapshot_count == 3
    assert result.attention_level == pytest.approx(1)
    assert result.attention_velocity is None
    assert result.attention_acceleration is None
    assert result.confidence == 0
    assert result.status == "conflicting_snapshots"


def test_rank_conflict_at_same_timestamp_is_also_explicitly_unusable() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    middle = now - timedelta(hours=3)

    result = assess_trend(
        [
            snapshot(now - timedelta(hours=6), reply_count=1, rank=3),
            snapshot(middle, reply_count=2, rank=1),
            snapshot(middle, reply_count=2, rank=4),
            snapshot(now, reply_count=4, rank=1),
        ],
        now=now,
    )

    assert result.attention_level == pytest.approx(1)
    assert result.attention_velocity is None
    assert result.attention_acceleration is None
    assert result.confidence == 0
    assert result.status == "conflicting_snapshots"


def test_mixed_source_or_external_ids_are_rejected() -> None:
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    first = snapshot(now - timedelta(hours=1), reply_count=1)
    second = snapshot(now, reply_count=2, external_id="topic-2")

    with pytest.raises(ValueError, match="same source_id and external_id"):
        assess_trend([first, second], now=now)


def test_now_and_snapshot_times_must_be_timezone_aware() -> None:
    aware = datetime(2026, 9, 12, 12, tzinfo=UTC)

    with pytest.raises(ValueError, match="now must be timezone-aware"):
        assess_trend([snapshot(aware, reply_count=1)], now=aware.replace(tzinfo=None))


def test_offset_timezones_are_compared_by_instant() -> None:
    now = datetime(2026, 9, 12, 20, tzinfo=timezone(timedelta(hours=8)))

    result = assess_trend(
        [
            snapshot(now - timedelta(hours=2), reply_count=2),
            snapshot(now, reply_count=4),
        ],
        now=now,
    )

    assert result.snapshot_count == 2
    assert result.attention_velocity == pytest.approx(0.5)
