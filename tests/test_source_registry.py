from pathlib import Path

import pytest
from pydantic import ValidationError

from news_assistant.models import CoverageDomain, EvidenceRole, SourceKind, SourceTier
from news_assistant.normalization import normalize_title, normalize_url
from news_assistant.source_registry import RegisteredSource, SourceRegistry, load_source_registry


def test_registry_loads_only_enabled_rss_sources_with_tier_and_domain(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """version: source-registry-v1
sources:
  - id: official
    name: Official feed
    tier: T0
    adapter: rss
    url: https://example.com/feed.xml
    topics: [ai]
    coverage_domains: [ai_technology]
    publisher_group: example
    enabled: true
    fetch_interval_hours: 24
    failure_policy: continue
  - id: disabled
    name: Disabled feed
    tier: T1
    adapter: rss
    url: https://example.com/disabled.xml
    topics: [business]
    coverage_domains: [business_macro]
    publisher_group: example-media
    enabled: false
    fetch_interval_hours: 24
    failure_policy: continue
"""
    )

    source = load_source_registry(path).enabled_feed_sources()[0]

    assert source.id == "official"
    assert source.tier is SourceTier.T0
    assert source.coverage_domain is CoverageDomain.AI_TECHNOLOGY
    assert source.publisher_group == "example"


def test_normalization_removes_tracking_and_site_suffix() -> None:
    assert normalize_url("https://EXAMPLE.com/a/?utm_source=x&id=7#part") == "https://example.com/a?id=7"
    assert normalize_title(" OpenAI launches X — Example News ") == "openai launches x"


def test_registry_preserves_independent_evidence_and_signal_roles(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """version: source-registry-v2
sources:
  - id: v2ex_ai
    name: V2EX AI
    tier: T3
    evidence_role: COMMUNITY_SIGNAL
    source_kind: FORUM
    adapter: v2ex
    url: https://www.v2ex.com/api/topics/latest.json
    topics: [ai]
    coverage_domains: [ai_technology]
    publisher_group: v2ex
    language: zh
    terms_reviewed_at: 2026-08-17
    access_stability: public_api
    signal_weight: 0.8
"""
    )

    source = load_source_registry(path).enabled_sources("v2ex")[0]

    assert source.evidence_role is EvidenceRole.COMMUNITY_SIGNAL
    assert source.source_kind is SourceKind.FORUM
    assert source.language == "zh"
    assert source.signal_weight == 0.8


def test_registry_excludes_shadow_rss_sources_from_production_feed_input(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """version: source-registry-v2
sources:
  - id: shadow-rss
    name: Shadow RSS
    tier: T1
    evidence_role: INDEPENDENT_REPORTING
    source_kind: MEDIA
    adapter: rss
    url: https://example.com/feed.xml
    topics: [business]
    coverage_domains: [business_macro]
    publisher_group: example
    language: zh
    terms_reviewed_at: 2026-08-17
    access_stability: public_rss
    enabled: true
    production_enabled: false
"""
    )

    registry = load_source_registry(path)

    assert registry.enabled_sources("rss") == [registry.sources[0]]
    assert registry.enabled_feed_sources() == []


def test_registry_v2_rejects_enabled_source_without_access_review(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """version: source-registry-v2
sources:
  - id: unreviewed
    name: Unreviewed
    tier: T1
    adapter: rss
    url: https://example.com/feed.xml
    topics: [business]
    coverage_domains: [business_macro]
    publisher_group: example
    language: zh
"""
    )

    with pytest.raises(ValueError, match="terms_reviewed_at"):
        load_source_registry(path)


@pytest.mark.parametrize("field", ["id", "name"])
def test_registry_rejects_ambiguous_source_identity(field: str) -> None:
    first = RegisteredSource(id="first", name="First", tier="T1", adapter="rss",
                             url="https://example.com/feed", topics=["science"],
                             coverage_domains=["science_health"], publisher_group="example")
    values = {"id": "second", "name": "Second", field: getattr(first, field)}
    duplicate = first.model_copy(update=values)
    with pytest.raises(ValueError, match=f"duplicate source {field}"):
        SourceRegistry(version="source-registry-v1", sources=[first, duplicate])


def test_configured_sources_cover_chinese_and_exploration_without_promoting_new_shadow_sources() -> None:
    registry = load_source_registry(Path("config/source_registry.yaml"))
    production = registry.enabled_feed_sources()
    assert {source.language for source in production} == {"zh", "en"}
    assert any(source.topics[0].value == "exploration" for source in production)
    assert len({source.publisher_group for source in production}) >= 6
    shadow = [source for source in registry.enabled_sources("rss") if not source.production_enabled]
    assert any(source.evidence_role == EvidenceRole.PRIMARY_FACT for source in shadow)
    assert any(CoverageDomain.POLICY_PUBLIC_AFFAIRS in source.coverage_domains for source in shadow)
    assert not any(source.evidence_role == EvidenceRole.COMMUNITY_SIGNAL for source in production)


def test_registry_propagates_optional_naive_timezone_to_feed_source() -> None:
    source = RegisteredSource(
        id="local-time-feed",
        name="Local time feed",
        tier="T1",
        adapter="rss",
        url="https://example.com/feed",
        topics=["science"],
        coverage_domains=["science_health"],
        publisher_group="example",
        naive_timezone="Asia/Shanghai",
    )

    assert source.as_feed_source().naive_timezone == "Asia/Shanghai"


def test_registry_rejects_unknown_naive_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        RegisteredSource(
            id="bad-timezone",
            name="Bad timezone",
            tier="T1",
            adapter="rss",
            url="https://example.com/feed",
            topics=["science"],
            coverage_domains=["science_health"],
            publisher_group="example",
            naive_timezone="Mars/Olympus_Mons",
        )


def test_only_sciencenet_configures_naive_source_timezone() -> None:
    registry = load_source_registry(Path("config/source_registry.yaml"))
    configured = [source for source in registry.sources if source.naive_timezone is not None]

    assert [(source.id, source.naive_timezone) for source in configured] == [
        ("sciencenet_news", "Asia/Shanghai")
    ]
