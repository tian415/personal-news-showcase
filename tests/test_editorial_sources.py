from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from news_assistant.editorial_sources import (
    jiemian_article_urls,
    parse_jiemian_article,
    parse_sixthtone_listing,
)
from news_assistant.feeds import FeedSource
from news_assistant.models import Topic


def source(name: str = "Sixth Tone") -> FeedSource:
    return FeedSource(
        name=name,
        url="https://www.sixthtone.com/",
        language="en" if name == "Sixth Tone" else "zh",
        topic=Topic.TECH,
        naive_timezone="Asia/Shanghai",
    )


def sixthtone(content: str) -> list:
    return parse_sixthtone_listing(content.encode(), source())


def next_data(items: str) -> str:
    return f'<script id="__NEXT_DATA__" type="application/json">{items}</script>'


def test_sixthtone_uses_exact_millisecond_timestamp_and_canonical_url_hash() -> None:
    content = next_data(
        '{"props":{"pageProps":{"stories":[{"contId":123,"name":"Title",'
        '"pubTimeLong":1791516129585,"summary":"Source summary"}]}}}'
    )

    article = sixthtone(content)[0]

    url = "https://www.sixthtone.com/news/123"
    assert str(article.url) == url
    assert article.id == hashlib.sha256(url.encode()).hexdigest()[:16]
    assert article.published_at == datetime.fromtimestamp(1791516129.585, UTC)
    assert article.title == "Title"
    assert article.summary == "Source summary"


@pytest.mark.parametrize(
    "item",
    [
        '{"contId":1,"name":"No time","summary":"summary"}',
        (
            '{"contId":1,"name":"Date only","pubTimeLong":"2026-10-09",'
            '"summary":"summary"}'
        ),
    ],
)
def test_sixthtone_rejects_missing_or_inexact_publication_time(item: str) -> None:
    with pytest.raises(ValueError):
        sixthtone(next_data(f'{{"items":[{item}]}}'))


def test_sixthtone_keeps_future_timestamp_for_downstream_freshness_audit() -> None:
    item = '{"contId":2,"name":"Future","pubTimeLong":4102444800000,"summary":""}'

    assert len(sixthtone(next_data(f'{{"items":[{item}]}}'))) == 1


def test_sixthtone_deduplicates_identical_ids_but_rejects_conflicts() -> None:
    item = '{"contId":3,"name":"Same","pubTimeLong":1791516129585,"summary":"x"}'
    content = next_data(f'{{"items":[{item},{item}]}}')
    assert len(sixthtone(content)) == 1

    conflicting = item.replace('"name":"Same"', '"name":"Different"')
    with pytest.raises(ValueError, match="conflicting"):
        sixthtone(next_data(f'{{"items":[{item},{conflicting}]}}'))


@pytest.mark.parametrize(
    ("link", "expected_count"),
    [
        ("https://outside.example/news/4", 0),
        ("https://interaction.sixthtone.com/news/4", 0),
        ("https://www.sixthtone.com/interaction/4", 0),
        ("https://www.sixthtone.com/news/5", 0),
        ("https://www.sixthtone.com/news/4", 1),
        ("http://www.sixthtone.com/news/4/a-story", 1),
        ("/news/4/a-story", 1),
        ("", 1),
    ],
)
def test_sixthtone_only_uses_matching_article_links(link: str, expected_count: int) -> None:
    item = (
        '{"contId":4,"name":"Title","pubTimeLong":1791516129585,'
        f'"summary":"Summary","link":{json.dumps(link)}}}'
    )
    fallback = (
        '{"contId":99,"name":"Fallback","pubTimeLong":1791516129585,'
        '"summary":"Summary"}'
    )
    content = next_data(f'{{"items":[{item},{fallback}]}}')

    articles = sixthtone(content)

    assert len(articles) == expected_count + 1
    if expected_count:
        assert str(articles[0].url) == "https://www.sixthtone.com/news/4"


def test_sixthtone_rejects_duplicate_id_with_conflicting_valid_article_link() -> None:
    base = (
        '{"contId":4,"name":"Title","pubTimeLong":1791516129585,'
        '"summary":"Summary","link":'
    )
    first = base + '"/news/4"}'
    second = base + '"/news/4/a-story"}'

    with pytest.raises(ValueError, match="conflicting"):
        sixthtone(next_data(f'{{"items":[{first},{second}]}}'))


def test_sixthtone_skips_daily_tones_card_mode_eight() -> None:
    item = (
        '{"contId":1019111,"name":"Old story reused as a card",'
        '"pubTimeLong":1791529431825,"summary":"Card copy",'
        '"link":"","cardMode":"8","nodeInfo":{"name":"20261009"}}'
    )

    with pytest.raises(ValueError, match="no articles"):
        sixthtone(next_data(f'{{"items":[{item}]}}'))


@pytest.mark.parametrize("content", [b"", b"<html><body>No listing</body></html>"])
def test_sixthtone_fails_closed_on_empty_or_unrecognized_document(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_sixthtone_listing(content, source())


def test_jiemian_listing_keeps_order_deduplicates_and_ignores_external_links() -> None:
    content = b"""<ul>
      <li><a href="https://www.jiemian.com/article/123.html"> First title </a></li>
      <li><a href="https://outside.example/article/9.html">Outside title</a></li>
      <li><a href="/article/456.html">Second title</a></li>
      <li><a href="/article/123.html">Duplicate title</a></li>
      <li><a href="/article/789.html">   </a></li>
    </ul>"""

    assert jiemian_article_urls(content) == [
        "https://www.jiemian.com/article/123.html",
        "https://www.jiemian.com/article/456.html",
    ]
    assert jiemian_article_urls(content, max_articles=1) == [
        "https://www.jiemian.com/article/123.html"
    ]


@pytest.mark.parametrize(
    "content",
    [
        b'<a href="https://outside.example/article/2.html">Outside</a>',
        b'<a href="/article/2.html"> </a>',
    ],
)
def test_jiemian_listing_requires_a_titled_local_article_link(content: bytes) -> None:
    with pytest.raises(ValueError):
        jiemian_article_urls(content)


def test_jiemian_article_uses_article_info_timestamp_and_shanghai_timezone() -> None:
    content = """<html><head><meta charset="utf-8"><meta name="description" content="摘要"></head><body>
      <h1>文章标题</h1><div class="article-info">作者 · 2026/10/09 12:12 来源：界面新闻</div>
      <article><p>正文提及 2024/01/02 03:04，不能当作发布时间。</p></article>
    </body></html>""".encode()

    article = parse_jiemian_article(
        content, "https://www.jiemian.com/article/15171179.html", source("界面新闻")
    )

    assert article.title == "文章标题"
    assert article.summary == "摘要"
    assert article.published_at == datetime(2026, 10, 9, 4, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    "html",
    [
        '<h1>标题</h1><div class="article-info">作者</div>',
        '<h1>标题</h1><div class="article-info">2026/10/09</div>',
        '<h1>标题</h1><article>2026/10/09 12:12</article>',
    ],
)
def test_jiemian_article_rejects_missing_or_date_only_article_info_time(html: str) -> None:
    with pytest.raises(ValueError):
        parse_jiemian_article(
            html.encode(), "https://www.jiemian.com/article/1.html", source("界面新闻")
        )
