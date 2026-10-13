from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from news_assistant.models import Article, Preferences, Topic


def test_article_requires_http_url() -> None:
    with pytest.raises(ValidationError):
        Article(
            id="bad",
            title="Bad URL",
            url="javascript:alert(1)",
            source="Example",
            language="en",
            published_at=datetime.now(UTC),
            topic=Topic.TECH,
        )


def test_article_requires_timezone_aware_publication_time() -> None:
    with pytest.raises(ValidationError):
        Article(
            id="naive",
            title="Naive time",
            url="https://example.com/item",
            source="Example",
            language="en",
            published_at=datetime(2026, 8, 12, 8, 0),  # noqa: DTZ001 - invalid input
            topic=Topic.TECH,
        )


def test_preferences_clamp_weights_to_safe_range() -> None:
    preferences = Preferences(topic_weights={Topic.AI: 9, Topic.BUSINESS: -8})
    assert preferences.topic_weights[Topic.AI] == 2.0
    assert preferences.topic_weights[Topic.BUSINESS] == 0.5
