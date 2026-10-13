from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from news_assistant.feeds import FeedSource, canonicalize_url, fetch_articles, parse_feed
from news_assistant.models import Topic


def test_parse_feed_normalizes_tracking_url() -> None:
    source = FeedSource(name="Example", url="https://example.com/rss", language="en", topic=Topic.AI)
    articles = parse_feed(Path("tests/fixtures/sample_feed.xml").read_bytes(), source)
    assert str(articles[0].url) == "https://example.com/story?id=7"
    assert articles[0].language == "en"


def test_feed_url_canonicalization_uses_shared_case_insensitive_tracking_rules() -> None:
    assert canonicalize_url("https://EXAMPLE.com/story/?UTM_SOURCE=rss&id=7") == "https://example.com/story?id=7"


def test_parse_feed_skips_invalid_entries_without_dropping_neighbors() -> None:
    source = FeedSource(name="Example", url="https://example.com/rss", language="en", topic=Topic.AI)
    content = b"""
    <rss version="2.0"><channel><title>Example</title>
      <item><title>Before</title><link>https://example.com/before</link>
        <pubDate>Wed, 12 Aug 2026 00:00:00 GMT</pubDate></item>
      <item><title>Broken URL</title><link>not a url</link>
        <pubDate>Wed, 12 Aug 2026 00:01:00 GMT</pubDate></item>
      <item><title>   </title><link>https://example.com/blank-title</link>
        <pubDate>Wed, 12 Aug 2026 00:02:00 GMT</pubDate></item>
      <item><title>After</title><link>https://example.com/after</link>
        <pubDate>Wed, 12 Aug 2026 00:03:00 GMT</pubDate></item>
    </channel></rss>
    """

    articles = parse_feed(content, source)

    assert [article.title for article in articles] == ["Before", "After"]


def test_parse_feed_skips_invalid_date_without_dropping_neighbor(monkeypatch) -> None:
    source = FeedSource(
        name="Example",
        url="https://example.com/rss",
        language="en",
        topic=Topic.AI,
    )
    monkeypatch.setattr(
        "news_assistant.feeds.feedparser.parse",
        lambda _: SimpleNamespace(
            version="rss20",
            entries=[
                {
                    "title": "Bad date",
                    "link": "https://example.com/bad-date",
                    "published": "10000-01-01 00:00",
                    "published_parsed": (10000, 1, 1, 0, 0, 0, 0, 1, 0),
                },
                {
                    "title": "Good date",
                    "link": "https://example.com/good-date",
                    "published": "2026-08-12 00:00",
                    "published_parsed": (2026, 8, 12, 0, 0, 0, 2, 224, 0),
                },
            ],
        ),
    )

    articles = parse_feed(b"ignored", source)

    assert [article.title for article in articles] == ["Good date"]


def test_parse_feed_uses_atom_updated_when_published_is_missing() -> None:
    source = FeedSource(name="Example", url="https://example.com/atom", language="en", topic=Topic.AI)
    content = b"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Example</title>
      <entry>
        <title>Updated only</title>
        <link href="https://example.com/updated" />
        <updated>2026-08-12T11:00:00Z</updated>
      </entry>
    </feed>
    """

    articles = parse_feed(content, source)

    assert len(articles) == 1
    assert articles[0].published_at == datetime(2026, 8, 12, 11, tzinfo=UTC)


def test_fetch_articles_keeps_working_sources_when_one_fails() -> None:
    sources = [
        FeedSource(name="Broken", url="https://bad.example/rss", language="en", topic=Topic.TECH),
        FeedSource(name="Good", url="https://good.example/rss", language="en", topic=Topic.AI),
    ]
    content = Path("tests/fixtures/sample_feed.xml").read_bytes()

    def loader(url: str) -> bytes:
        if "bad" in url:
            raise TimeoutError("offline")
        return content

    articles = fetch_articles(sources, loader=loader, now=datetime(2026, 8, 12, 12, tzinfo=UTC))
    assert [article.source for article in articles] == ["Good"]


def test_html_error_page_is_not_a_successful_empty_feed() -> None:
    source = FeedSource(name="Example", url="https://example.com/rss", language="en", topic=Topic.AI)
    with pytest.raises(ValueError, match="RSS|Atom"):
        parse_feed(b"<html><body>Temporarily unavailable</body></html>", source)


def test_valid_empty_feed_can_have_no_recent_publications() -> None:
    source = FeedSource(name="Example", url="https://example.com/rss", language="en", topic=Topic.AI)
    assert parse_feed(b'<rss version="2.0"><channel><title>Empty</title></channel></rss>', source) == []


def test_fetch_can_surface_source_failure_for_health_collection() -> None:
    source = FeedSource(name="Broken", url="https://example.com/rss", language="en", topic=Topic.AI)
    with pytest.raises(ValueError, match="RSS|Atom"):
        fetch_articles([source], loader=lambda _: b"<html>Unavailable</html>", raise_on_error=True)


def test_chinese_source_english_entries_keep_translation_link_language() -> None:
    source = FeedSource(name="Mixed", url="https://example.com/rss", language="zh", topic=Topic.TECH)
    content = '''<rss version="2.0"><channel><title>Mixed feed</title>
      <item><title>Technology update</title><link>https://example.com/english</link>
        <pubDate>Mon, 21 Sep 2026 00:00:00 GMT</pubDate><description>English summary</description></item>
      <item><title>中文新闻</title><link>https://example.com/chinese</link>
        <pubDate>Mon, 21 Sep 2026 00:00:00 GMT</pubDate></item>
    </channel></rss>'''.encode()
    assert [item.language for item in parse_feed(content, source)] == ["en", "zh"]


def test_source_timezone_applies_only_to_naive_iso_publication_dates() -> None:
    source = FeedSource(
        name="Example",
        url="https://example.com/rss",
        language="zh",
        topic=Topic.SCIENCE,
        naive_timezone="Asia/Shanghai",
    )
    content = b"""
    <rss version="2.0"><channel><title>Example</title>
      <item><title>Naive ISO</title><link>https://example.com/naive</link>
        <pubDate>2026-09-21 16:47</pubDate></item>
      <item><title>Offset ISO</title><link>https://example.com/offset-iso</link>
        <pubDate>2026-09-21T16:47:00+02:00</pubDate></item>
      <item><title>Offset RFC</title><link>https://example.com/offset-rfc</link>
        <pubDate>Mon, 21 Sep 2026 16:47:00 +0200</pubDate></item>
    </channel></rss>
    """

    articles = parse_feed(content, source)

    assert [article.published_at for article in articles] == [
        datetime(2026, 9, 21, 8, 47, tzinfo=UTC),
        datetime(2026, 9, 21, 14, 47, tzinfo=UTC),
        datetime(2026, 9, 21, 14, 47, tzinfo=UTC),
    ]


def test_naive_iso_without_source_timezone_keeps_existing_utc_interpretation() -> None:
    source = FeedSource(
        name="Example",
        url="https://example.com/rss",
        language="zh",
        topic=Topic.SCIENCE,
    )
    content = b"""
    <rss version="2.0"><channel><title>Example</title>
      <item><title>Naive ISO</title><link>https://example.com/naive</link>
        <pubDate>2026-09-21 16:47</pubDate></item>
    </channel></rss>
    """

    assert parse_feed(content, source)[0].published_at == datetime(
        2026, 9, 21, 16, 47, tzinfo=UTC
    )


def test_source_timezone_applies_to_naive_atom_updated_fallback() -> None:
    source = FeedSource(
        name="Example",
        url="https://example.com/atom",
        language="zh",
        topic=Topic.SCIENCE,
        naive_timezone="Asia/Shanghai",
    )
    content = b"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Example</title>
      <entry><title>Updated only</title><link href="https://example.com/updated" />
        <updated>2026-09-21 16:47</updated></entry>
    </feed>
    """

    assert parse_feed(content, source)[0].published_at == datetime(
        2026, 9, 21, 8, 47, tzinfo=UTC
    )


def test_configured_source_timezone_is_applied_before_future_filtering() -> None:
    content = b"""
    <rss version="2.0"><channel><title>Example</title>
      <item><title>Current local story</title><link>https://example.com/current</link>
        <pubDate>2026-09-21 16:47</pubDate></item>
    </channel></rss>
    """
    configured = FeedSource(
        name="Configured",
        url="https://example.com/rss",
        language="zh",
        topic=Topic.SCIENCE,
        naive_timezone="Asia/Shanghai",
    )
    default = configured.model_copy(update={"name": "Default", "naive_timezone": None})
    now = datetime(2026, 9, 21, 9, tzinfo=UTC)

    assert [article.source for article in fetch_articles(
        [configured, default], loader=lambda _: content, now=now
    )] == ["Configured"]
