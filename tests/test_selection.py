from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from news_assistant.models import Article, Preferences, Topic
from news_assistant.selection import deduplicate, score_article, select_digest


def article(
    index: int,
    topic: Topic,
    quality: float = 0.8,
    title: str | None = None,
    language: str = "en",
    source: str | None = None,
) -> Article:
    return Article(
        id=str(index),
        title=title or f"Story {index}",
        url=f"https://example.com/{index}",
        source=source or f"Source {index % 3}",
        language=language,
        published_at=datetime(2026, 8, 12, index % 12, tzinfo=UTC),
        topic=topic,
        quality_score=quality,
        importance_score=0.7,
    )


def test_deduplicate_removes_same_story_with_tracking_variants() -> None:
    first = article(1, Topic.AI, title="New AI model released today")
    duplicate = article(2, Topic.AI, title="New AI model released today!")
    duplicate.url = "https://example.com/1?utm_source=email"
    assert deduplicate([first, duplicate]) == [first]


def test_select_digest_preserves_five_three_two_diversity() -> None:
    candidates = [article(i, Topic.AI) for i in range(1, 8)]
    candidates += [article(i, Topic.BUSINESS) for i in range(20, 25)]
    candidates += [article(i, Topic.EXPLORATION) for i in range(40, 44)]
    selected = select_digest(candidates, Preferences())
    assert len(selected) == 10
    assert sum(item.topic in {Topic.AI, Topic.TECH, Topic.SCIENCE} for item in selected) == 5
    assert sum(item.topic in {Topic.BUSINESS, Topic.MARKETS} for item in selected) == 3
    assert sum(item.topic == Topic.EXPLORATION for item in selected) == 2


def test_select_digest_does_not_pad_with_low_quality_content() -> None:
    candidates = [article(1, Topic.AI), article(2, Topic.BUSINESS), article(3, Topic.EXPLORATION, 0.2)]
    assert len(select_digest(candidates, Preferences())) == 2


def test_low_quality_duplicate_does_not_hide_eligible_article() -> None:
    low = article(1, Topic.AI, quality=0.2, title="A new discovery")
    high = article(2, Topic.AI, quality=0.8, title="A new discovery")
    assert select_digest([low, high], Preferences()) == [high]


def test_url_dedup_preserves_meaningful_blank_parameters() -> None:
    plain = article(1, Topic.AI, title="Technology report")
    translated = article(2, Topic.AI, title="Completely different reporting")
    translated.url = "https://example.com/1?translated="
    assert deduplicate([plain, translated]) == [plain, translated]


def test_url_dedup_uses_shared_case_insensitive_tracking_rules() -> None:
    first = article(1, Topic.AI, title="Technology report")
    tracked = article(2, Topic.AI, title="Completely different wording")
    tracked.url = "https://example.com/1?UTM_SOURCE=rss"
    assert deduplicate([first, tracked]) == [first]


def test_freshness_spans_the_feed_window_and_does_not_depend_on_calendar_epoch() -> None:
    now = datetime(2026, 9, 21, 0, tzinfo=UTC)
    fresh = article(1, Topic.AI).model_copy(update={"published_at": now})
    old = article(2, Topic.AI).model_copy(update={"published_at": now - timedelta(hours=36)})
    assert score_article(fresh, Preferences(), now=now) == pytest.approx(0.785)
    assert score_article(old, Preferences(), now=now) == pytest.approx(0.685)
    future_year = now + timedelta(days=365)
    shifted = fresh.model_copy(update={"published_at": future_year})
    assert score_article(shifted, Preferences(), now=future_year) == pytest.approx(0.785)


def test_selection_respects_explicit_build_time() -> None:
    now = datetime(2026, 9, 21, 0, tzinfo=UTC)
    fresh = article(1, Topic.AI, quality=0.8, title="Today's technology launch")
    fresh.published_at = now
    old = article(2, Topic.AI, quality=0.9, title="Yesterday's scientific study")
    old.published_at = now - timedelta(hours=36)
    assert select_digest([old, fresh], Preferences(), limit=1, now=now) == [fresh]


def test_selection_rejects_negative_limits() -> None:
    with pytest.raises(ValueError, match="limit"):
        select_digest([article(1, Topic.AI), article(2, Topic.BUSINESS)], Preferences(), -1)


def test_chinese_target_does_not_break_core_business_exploration_quotas() -> None:
    candidates = [article(i, Topic.AI, language="zh") for i in range(1, 6)]
    candidates += [article(i, Topic.BUSINESS) for i in range(20, 24)]
    candidates += [article(i, Topic.EXPLORATION) for i in range(40, 43)]

    selected = select_digest(candidates, Preferences(), chinese_ratio=0.70)

    assert len(selected) == 10
    assert sum(item.topic in {Topic.AI, Topic.TECH, Topic.SCIENCE} for item in selected) == 5
    assert sum(item.topic in {Topic.BUSINESS, Topic.MARKETS} for item in selected) == 3
    assert sum(item.topic == Topic.EXPLORATION for item in selected) == 2


def test_source_soft_cap_distributes_high_scoring_single_source() -> None:
    dominant = [article(i, Topic.AI, quality=0.95, source="Dominant") for i in range(1, 7)]
    alternatives = [
        article(20, Topic.AI, quality=0.8, source="Other A"),
        article(21, Topic.AI, quality=0.8, source="Other B"),
        article(22, Topic.AI, quality=0.8, source="Other C"),
        article(23, Topic.AI, quality=0.8, source="Other D"),
    ]

    selected = select_digest(dominant + alternatives, Preferences(), limit=6, source_soft_cap=2)

    counts = Counter(item.source for item in selected)
    assert len(selected) == 6
    assert counts["Dominant"] == 2
    assert len(counts) >= 4


def test_source_soft_cap_does_not_drop_qualified_articles_when_sources_are_few() -> None:
    candidates = [article(i, Topic.AI, source="Only source") for i in range(1, 4)]

    selected = select_digest(candidates, Preferences(), limit=5, source_soft_cap=2)

    assert len(selected) == 3
    assert all(item.quality_score >= 0.5 for item in selected)


def test_chinese_target_does_not_fill_with_low_quality_articles() -> None:
    candidates = [article(i, Topic.AI, quality=0.8, language="en") for i in range(1, 4)]
    candidates += [article(i, Topic.AI, quality=0.2, language="zh") for i in range(20, 23)]

    selected = select_digest(candidates, Preferences(), limit=5, chinese_ratio=0.70)

    assert len(selected) == 3
    assert all(item.quality_score >= 0.5 for item in selected)
    assert all(item.language == "en" for item in selected)


def test_core_topics_include_science_before_more_similar_ai_and_tech_reports() -> None:
    tech = [article(i, Topic.TECH, quality=0.95, language="zh", source=f"Tech {i}")
            for i in range(1, 6)]
    ai = article(6, Topic.AI, quality=0.99, language="zh", source="AI source")
    science = [article(i, Topic.SCIENCE, quality=0.8, source="Science") for i in (20, 21)]
    selected = select_digest([ai, *tech, *science], Preferences(), limit=5)
    assert sum(item.topic in {Topic.AI, Topic.TECH} for item in selected) == 3
    assert sum(item.topic == Topic.SCIENCE for item in selected) == 2


def test_promotional_candidate_does_not_replace_useful_story_or_pad_count() -> None:
    promo = article(1, Topic.EXPLORATION, quality=0.99, title="品牌合作｜新品体验招募")
    useful = article(2, Topic.EXPLORATION, title="研究团队公开数据集与复现方法")
    assert select_digest([promo, useful], Preferences()) == [useful]
