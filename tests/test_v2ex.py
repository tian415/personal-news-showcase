from datetime import UTC, datetime

from news_assistant.models import CoverageDomain, EvidenceRole, SourceKind, SourceTier, Topic
from news_assistant.source_registry import RegisteredSource


def test_v2ex_adapter_maps_public_topic_and_snapshot() -> None:
    from news_assistant.discovery.v2ex import fetch_v2ex_topics

    source = RegisteredSource(
        id="v2ex_ai",
        name="V2EX AI",
        tier=SourceTier.T3,
        evidence_role=EvidenceRole.COMMUNITY_SIGNAL,
        source_kind=SourceKind.FORUM,
        adapter="v2ex",
        url="https://www.v2ex.com/api/topics/latest.json",
        topics=[Topic.AI],
        coverage_domains=[CoverageDomain.AI_TECHNOLOGY],
        publisher_group="v2ex",
        language="zh",
        terms_reviewed_at="2026-08-17",
        access_stability="public_api",
        signal_weight=0.8,
    )
    payload = (
        b'[{"id": 42, "title": "\xe6\x96\xb0\xe6\xa8\xa1\xe5\x9e\x8b\xe5\x8f\x91\xe5\xb8\x83", '
        b'"url": "https://www.v2ex.com/t/42", "created": 1786900000, "replies": 12}]'
    )
    now = datetime(2026, 8, 17, 0, tzinfo=UTC)

    result = fetch_v2ex_topics(source, loader=lambda _: payload, now=now)

    assert result.articles[0].id == "v2ex_ai:42"
    assert result.articles[0].language == "zh"
    assert result.articles[0].published_at == datetime.fromtimestamp(1786900000, tz=UTC)
    assert result.snapshots[0].source_id == "v2ex_ai"
    assert result.snapshots[0].external_id == "42"
    assert result.snapshots[0].reply_count == 12
    assert result.snapshots[0].rank == 1


def test_v2ex_adapter_skips_invalid_topics_without_orphaning_articles() -> None:
    from news_assistant.discovery.v2ex import fetch_v2ex_topics

    source = RegisteredSource(
        id="v2ex_ai",
        name="V2EX AI",
        tier=SourceTier.T3,
        evidence_role=EvidenceRole.COMMUNITY_SIGNAL,
        source_kind=SourceKind.FORUM,
        adapter="v2ex",
        url="https://www.v2ex.com/api/topics/latest.json",
        topics=[Topic.AI],
        coverage_domains=[CoverageDomain.AI_TECHNOLOGY],
        publisher_group="v2ex",
        language="zh",
        terms_reviewed_at="2026-08-17",
        access_stability="public_api",
        signal_weight=0.8,
    )
    payload = (
        b"["
        b'{"id": 1, "title": "Before", "created": 1786900000, "replies": 1},'
        b'{"id": 2, "title": "Negative replies", "created": 1786900001, "replies": -1},'
        b'{"id": 3, "title": "Overflow time", "created": 99999999999999999999, "replies": 2},'
        b'{"id": 4, "title": null, "created": 1786900003, "replies": 3},'
        b'{"id": null, "title": "Null id", "created": 1786900004, "replies": 4},'
        b'{"id": 6, "title": "After", "created": 1786900005, "replies": 5}'
        b"]"
    )
    now = datetime(2026, 8, 17, 0, tzinfo=UTC)

    result = fetch_v2ex_topics(source, loader=lambda _: payload, now=now)

    assert [article.id for article in result.articles] == ["v2ex_ai:1", "v2ex_ai:6"]
    assert [snapshot.external_id for snapshot in result.snapshots] == ["1", "6"]
    assert [snapshot.rank for snapshot in result.snapshots] == [1, 6]
    assert len(result.articles) == len(result.snapshots)
