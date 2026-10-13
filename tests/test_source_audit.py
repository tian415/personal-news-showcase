from __future__ import annotations

from datetime import UTC, datetime

import pytest

from news_assistant.models import CoverageDomain, EvidenceRole, SourceTier, Topic
from news_assistant.source_audit import audit_sources
from news_assistant.source_registry import RegisteredSource, SourceRegistry

NOW = datetime(2026, 9, 21, 8, tzinfo=UTC)


def _source(
    source_id: str,
    *,
    adapter: str = "rss",
    enabled: bool = True,
    production_enabled: bool = True,
    language: str = "en",
    domain: CoverageDomain = CoverageDomain.AI_TECHNOLOGY,
    publisher_group: str | None = None,
) -> RegisteredSource:
    return RegisteredSource(
        id=source_id,
        name=f"Source {source_id}",
        tier=SourceTier.T1,
        adapter=adapter,
        url=f"https://{source_id}.example/feed",
        topics=[Topic.AI],
        coverage_domains=[domain],
        publisher_group=publisher_group or source_id,
        evidence_role=EvidenceRole.INDEPENDENT_REPORTING,
        language=language,
        enabled=enabled,
        production_enabled=production_enabled,
    )


def _rss(*items: tuple[str, str, str]) -> bytes:
    entries = "".join(
        f"<item><title>{title}</title><link>{url}</link><pubDate>{published}</pubDate></item>"
        for title, url, published in items
    )
    return f"<rss version='2.0'><channel><title>Fixture</title>{entries}</channel></rss>".encode()


def test_audit_uses_real_feed_parsing_and_reports_freshness_language_and_metadata() -> None:
    source = _source(
        "mixed",
        language="zh",
        domain=CoverageDomain.BUSINESS_MACRO,
        publisher_group="mixed-group",
    )
    registry = SourceRegistry(version="source-registry-v1", sources=[source])
    content = _rss(
        ("中文新消息", "https://news.example/fresh", "Sun, 21 Sep 2026 08:00:00 GMT"),
        ("English at cutoff", "https://news.example/cutoff", "Fri, 19 Sep 2026 20:00:00 GMT"),
        ("Old item", "https://news.example/old", "Fri, 19 Sep 2026 19:59:59 GMT"),
        ("Future item", "https://news.example/future", "Sun, 21 Sep 2026 08:00:01 GMT"),
    )

    report = audit_sources(registry, loader=lambda _url: content, now=NOW)

    assert report["schema_version"] == "source-audit-v1"
    assert report["captured_at"] == "2026-09-21T08:00:00+00:00"
    assert report["max_age_hours"] == 36
    assert report["sources"] == [
        {
            "id": "mixed",
            "name": "Source mixed",
            "url": "https://mixed.example/feed",
            "production_enabled": True,
            "evidence_role": "INDEPENDENT_REPORTING",
            "publisher_group": "mixed-group",
            "topics": ["ai"],
            "coverage_domains": ["business_macro"],
            "configured_language": "zh",
            "configured_naive_timezone": None,
            "status": "ok",
            "error_type": None,
            "entry_count": 4,
            "rejected_entry_count": 0,
            "entry_rejection_reasons": {},
            "parsed_count": 4,
            "fresh_count": 2,
            "stale_count": 1,
            "future_count": 1,
            "latest_published_at": "2026-09-21T08:00:01+00:00",
            "oldest_published_at": "2026-09-19T19:59:59+00:00",
            "actual_language_counts": {"zh": 1, "en": 3},
        }
    ]
    assert report["summary"] == {
        "production": {
            "source_count": 1,
            "available_source_count": 1,
            "fresh_candidate_count": 2,
            "coverage_domains_with_fresh": ["business_macro"],
            "publisher_groups_with_fresh": ["mixed-group"],
        },
        "shadow": {
            "source_count": 0,
            "available_source_count": 0,
            "fresh_candidate_count": 0,
            "coverage_domains_with_fresh": [],
            "publisher_groups_with_fresh": [],
        },
    }


def test_audit_fetches_enabled_rss_in_registry_order_and_skips_other_sources() -> None:
    production = _source("production")
    shadow = _source("shadow", production_enabled=False)
    disabled = _source("disabled", enabled=False)
    manual = _source("manual", adapter="manual")
    v2ex = _source("v2ex", adapter="v2ex")
    registry = SourceRegistry(
        version="source-registry-v1",
        sources=[production, disabled, manual, shadow, v2ex],
    )
    requested: list[str] = []

    def loader(url: str) -> bytes:
        requested.append(url)
        return _rss()

    report = audit_sources(registry, loader=loader, now=NOW)

    assert sorted(requested) == [
        "https://production.example/feed",
        "https://shadow.example/feed",
    ]
    assert [item["id"] for item in report["sources"]] == ["production", "shadow"]
    assert [item["status"] for item in report["sources"]] == ["no_recent", "no_recent"]
    assert [item["oldest_published_at"] for item in report["sources"]] == [None, None]
    assert report["summary"]["production"]["available_source_count"] == 1
    assert report["summary"]["shadow"]["available_source_count"] == 1


def test_audit_isolates_source_errors_without_exposing_exception_text(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _source("broken")
    good = _source("good", production_enabled=False)
    registry = SourceRegistry(version="source-registry-v1", sources=[broken, good])
    good_feed = _rss(
        ("Good", "https://news.example/good", "Sun, 21 Sep 2026 07:00:00 GMT"),
    )

    def loader(url: str) -> bytes:
        if "broken" in url:
            raise RuntimeError("secret-token=do-not-report")
        return good_feed

    report = audit_sources(registry, loader=loader, now=NOW)

    assert report["sources"][0]["status"] == "error"
    assert report["sources"][0]["error_type"] == "RuntimeError"
    assert "secret-token" not in str(report)
    assert report["sources"][1]["status"] == "ok"
    assert report["summary"]["production"]["available_source_count"] == 0
    assert report["summary"]["shadow"]["fresh_candidate_count"] == 1

    def invalid_parse(_content: bytes, _source: object, **_kwargs) -> list[object]:
        raise ValueError("private response body")

    monkeypatch.setattr("news_assistant.source_audit.parse_feed", invalid_parse)
    invalid_report = audit_sources(
        SourceRegistry(version="source-registry-v1", sources=[good]),
        loader=lambda _url: b"invalid",
        now=NOW,
    )
    assert invalid_report["sources"][0]["status"] == "error"
    assert invalid_report["sources"][0]["error_type"] == "ValueError"
    assert "private response body" not in str(invalid_report)


@pytest.mark.parametrize(
    ("now", "max_age_hours", "message"),
    [
        (datetime(2026, 9, 21, 8), 36, "timezone-aware"),  # noqa: DTZ001
        (NOW, 0, "max_age_hours"),
        (NOW, -1, "max_age_hours"),
    ],
)
def test_audit_rejects_invalid_time_arguments(
    now: datetime,
    max_age_hours: int,
    message: str,
) -> None:
    registry = SourceRegistry(version="source-registry-v1", sources=[])

    with pytest.raises(ValueError, match=message):
        audit_sources(registry, loader=lambda _url: b"", now=now, max_age_hours=max_age_hours)


def test_coverage_reports_publisher_independence_and_production_gaps() -> None:
    registry = SourceRegistry(version="source-registry-v1", sources=[
        _source("first", publisher_group="shared"),
        _source("second", publisher_group="shared"),
        _source("policy", production_enabled=False,
                domain=CoverageDomain.POLICY_PUBLIC_AFFAIRS),
    ])
    report = audit_sources(registry, loader=lambda _: _rss(), now=NOW)
    ai = report["coverage"]["ai_technology"]["production"]
    assert ai["source_count"] == 2
    assert ai["publisher_group_count"] == 1
    assert "single_publisher" in ai["gaps"]
    assert ai["fresh_source_count"] == 0
    assert ai["available_source_count"] == 2
    assert report["coverage"]["policy_public_affairs"]["production"]["gaps"] == [
        "no_configured_sources",
    ]


def test_candidate_capture_is_explicit_and_uses_only_the_same_fresh_window() -> None:
    source = _source("news")
    registry = SourceRegistry(version="source-registry-v1", sources=[source])
    feed = _rss(
        ("Fresh", "https://news.example/fresh", "Sun, 21 Sep 2026 07:00:00 GMT"),
        ("Old", "https://news.example/old", "Fri, 18 Sep 2026 07:00:00 GMT"),
        ("Future", "https://news.example/future", "Tue, 22 Sep 2026 07:00:00 GMT"),
    )
    normal = audit_sources(registry, loader=lambda _: feed, now=NOW)
    assert "candidates" not in normal["sources"][0]
    captured = audit_sources(registry, loader=lambda _: feed, now=NOW, capture_candidates=True)
    assert captured["candidate_snapshot_available"] is True
    assert [c["title"] for c in captured["sources"][0]["candidates"]] == ["Fresh"]


def test_unavailable_roles_do_not_hide_live_coverage_gaps():
    primary = _source("primary").model_copy(update={"evidence_role": EvidenceRole.PRIMARY_FACT})
    reporting = _source("broken")

    def loader(url):
        if "broken" in url:
            raise OSError("offline")
        return _rss(("Report", "https://primary.example/report", "Sun, 21 Sep 2026 07:00:00 GMT"))

    report = audit_sources(SourceRegistry(version="source-registry-v1", sources=[primary, reporting]),
                           loader=loader, now=NOW)
    coverage = report["coverage"]["ai_technology"]["production"]
    assert "no_available_independent_reporting_source" in coverage["gaps"]
    assert coverage["available_publisher_group_count"] == 1
    assert coverage["fresh_publisher_groups_by_role"]["INDEPENDENT_REPORTING"] == []


def test_all_rejected_feed_entries_are_distinct_from_valid_empty_or_stale_feeds():
    registry = SourceRegistry(version="source-registry-v1", sources=[_source("news")])
    invalid = b'<rss version="2.0"><channel><title>Feed</title><item><title>Missing date</title><link>https://news.example/x</link></item></channel></rss>'
    report = audit_sources(registry, loader=lambda _: invalid, now=NOW, capture_candidates=True)
    source = report["sources"][0]
    assert source["status"] == "error"
    assert source["error_type"] == "NoValidFeedEntriesError"
    assert source["entry_count"] == 1
    assert source["rejected_entry_count"] == 1
    assert source["entry_rejection_reasons"] == {"missing_time": 1}
    assert source["candidates"] == []
    empty = audit_sources(registry, loader=lambda _: _rss(), now=NOW)["sources"][0]
    assert empty["status"] == "no_recent"
    assert empty["entry_count"] == 0
    assert empty["rejected_entry_count"] == 0
