from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from math import ceil

from .content_quality import promotional_reason
from .models import Article, Preferences, Topic
from .normalization import normalize_url

CORE_TOPICS = {Topic.AI, Topic.TECH, Topic.SCIENCE}
BUSINESS_TOPICS = {Topic.BUSINESS, Topic.MARKETS}
QUALITY_FLOOR = 0.5


def _topic_group(topic: Topic) -> str:
    if topic in {Topic.AI, Topic.TECH}:
        return "ai_technology"
    if topic in BUSINESS_TOPICS:
        return "business_macro"
    return topic.value


def _clean_title(value: str) -> str:
    return re.sub(r"[^\w\s]", "", value.casefold()).strip()


def deduplicate(articles: Iterable[Article]) -> list[Article]:
    unique: list[Article] = []
    urls: set[str] = set()
    titles: list[str] = []
    for article in articles:
        url = normalize_url(str(article.url))
        title = _clean_title(article.title)
        if url in urls or any(
            SequenceMatcher(None, existing, title).ratio() >= 0.94
            for existing in titles
        ):
            continue
        unique.append(article)
        urls.add(url)
        titles.append(title)
    return unique


def score_article(
    article: Article, preferences: Preferences, *, now: datetime | None = None,
) -> float:
    topic_weight = preferences.topic_weights.get(article.topic, 1.0)
    source_weight = preferences.source_weights.get(article.source, 1.0)
    reference = now or datetime.now(UTC)
    age_hours = max(0, (reference - article.published_at).total_seconds() / 3600)
    recency = max(0, 1 - age_hours / 36)
    return (article.quality_score * 0.55 + article.importance_score * 0.35 + recency * 0.1) * topic_weight * source_weight


def select_digest(
    articles: Iterable[Article], preferences: Preferences, limit: int = 10,
    *, now: datetime | None = None, chinese_ratio: float = 0.70, source_soft_cap: int = 2,
    topic_soft_cap: int = 3, fill_shortfalls: bool = True,
) -> list[Article]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not 0 <= chinese_ratio <= 1:
        raise ValueError("chinese_ratio must be between 0 and 1")
    if source_soft_cap < 0:
        raise ValueError("source_soft_cap must be non-negative")
    if topic_soft_cap < 0:
        raise ValueError("topic_soft_cap must be non-negative")
    reference = now or datetime.now(UTC)
    eligible = deduplicate(
        article for article in articles
        if article.quality_score >= QUALITY_FLOOR and promotional_reason(article) is None
    )
    ranked = sorted(
        eligible, key=lambda item: score_article(item, preferences, now=reference), reverse=True,
    )
    quotas = (
        (CORE_TOPICS, min(5, limit)),
        (BUSINESS_TOPICS, min(3, max(0, limit - 5))),
        ({Topic.EXPLORATION}, min(2, max(0, limit - 8))),
    )
    chinese_quota = ceil(limit * chinese_ratio)
    selected: list[Article] = []

    source_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()

    def ordered(items: list[Article]) -> list[Article]:
        if sum(item.language == "zh" for item in selected) < chinese_quota:
            return [item for item in items if item.language == "zh"] + [
                item for item in items if item.language != "zh"
            ]
        return items

    def append_for_quota(items: list[Article], quota: int) -> None:
        candidates = ordered(items)
        start = len(selected)
        # Prefer distinct fields as well as distinct publishers. AI and general
        # tech share a group so relabelling them cannot bypass diversity.
        for enforce_source, enforce_topic in (
            (True, True), (False, True), (True, False), (False, False),
        ):
            for item in candidates:
                if len(selected) - start >= quota:
                    return
                group = _topic_group(item.topic)
                if item in selected:
                    continue
                if enforce_source and source_counts[item.source] >= source_soft_cap:
                    continue
                if enforce_topic and topic_counts[group] >= topic_soft_cap:
                    continue
                selected.append(item)
                source_counts[item.source] += 1
                topic_counts[group] += 1

    for topics, quota in quotas:
        matching = [item for item in ranked if item.topic in topics and item not in selected]
        append_for_quota(matching, quota)
    if fill_shortfalls and len(selected) < limit:
        remaining = [item for item in ranked if item not in selected]
        append_for_quota(remaining, limit - len(selected))
    return selected
