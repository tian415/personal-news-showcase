"""Bounded parsers for isolated public editorial-source research samples."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from lxml import html
from pydantic import ValidationError

from .feeds import FeedSource, canonicalize_url
from .models import Article

_SIXTHTONE_FIELDS = {"contId", "name", "pubTimeLong", "summary"}
_JIEMIAN_PATH = re.compile(r"^/article/[0-9]+\.html$")
_JIEMIAN_TIME = re.compile(r"(?<!\d)(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})(?!\d)")
_SIXTHTONE_ARTICLE_PATH = re.compile(r"^/news/([0-9]+)(?:/[^/]+)?$")


def _document(content: bytes):
    if not content.strip():
        raise ValueError("empty HTML document")
    try:
        return html.fromstring(content)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid HTML document") from error


def _dictionary_list_entries(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from _dictionary_list_entries(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict):
                if _SIXTHTONE_FIELDS <= child.keys():
                    yield child
                yield from _dictionary_list_entries(child)
            else:
                yield from _dictionary_list_entries(child)


def _sixthtone_link_path(value: object, article_id: str) -> str | None:
    """Return an allowed article path, or None for an empty/invalid link."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return f"/news/{article_id}"
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    if parts.query or parts.fragment:
        return None
    if parts.scheme or parts.netloc:
        if (
            parts.scheme.lower() not in {"http", "https"}
            or parts.hostname != "www.sixthtone.com"
            or parts.username is not None
            or parts.password is not None
            or port is not None
        ):
            return None
    elif not value.strip().startswith("/"):
        return None
    match = _SIXTHTONE_ARTICLE_PATH.fullmatch(parts.path)
    if not match or match.group(1) != article_id:
        return None
    return parts.path


def parse_sixthtone_listing(content: bytes, source: FeedSource) -> list[Article]:
    """Parse exact publication times from public Sixth Tone Next.js data."""
    document = _document(content)
    scripts = document.xpath('//script[@id="__NEXT_DATA__"]')
    if len(scripts) != 1 or not scripts[0].text:
        raise ValueError("Sixth Tone listing has no supported __NEXT_DATA__ script")
    try:
        payload = json.loads(scripts[0].text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Sixth Tone __NEXT_DATA__ is not valid JSON") from error

    entries = list(_dictionary_list_entries(payload))
    if not entries:
        raise ValueError("Sixth Tone listing structure has no article entries")

    articles: dict[str, Article] = {}
    records: dict[str, tuple[object, ...]] = {}
    for entry in entries:
        if str(entry.get("cardMode", "")) == "8":
            continue
        raw_id, title, raw_time, summary = (entry.get(key) for key in ("contId", "name", "pubTimeLong", "summary"))
        if isinstance(raw_id, bool) or not str(raw_id).isdigit():
            continue
        article_id = str(raw_id)
        link_path = _sixthtone_link_path(entry.get("link"), article_id)
        if link_path is None:
            continue
        record = (title, raw_time, summary, link_path)
        if article_id in records:
            if records[article_id] != record:
                raise ValueError(f"conflicting Sixth Tone entries for contId {article_id}")
            continue
        records[article_id] = record

        if not isinstance(title, str) or not title.strip() or not isinstance(summary, str):
            continue
        if isinstance(raw_time, bool) or not isinstance(raw_time, int) or raw_time <= 0:
            continue
        try:
            published = datetime.fromtimestamp(raw_time // 1000, UTC).replace(
                microsecond=(raw_time % 1000) * 1000
            )
        except (OverflowError, OSError, ValueError):
            continue

        url = canonicalize_url(f"https://www.sixthtone.com/news/{article_id}")
        identifier = hashlib.sha256(url.encode()).hexdigest()[:16]
        try:
            article = Article(
                id=identifier,
                title=title.strip(),
                url=url,
                source=source.name,
                language="en",
                published_at=published,
                topic=source.topic,
                summary=summary,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise ValueError(f"invalid Sixth Tone article {article_id}") from error
        articles[article_id] = article

    if not articles:
        raise ValueError("Sixth Tone listing contains no articles with exact timestamps")
    return list(articles.values())


def jiemian_article_urls(content: bytes, *, max_articles: int = 12) -> list[str]:
    """Extract bounded, titled Jiemian article links in document order."""
    if isinstance(max_articles, bool) or not isinstance(max_articles, int) or max_articles < 1:
        raise ValueError("max_articles must be a positive integer")
    document = _document(content)
    urls: list[str] = []
    seen: set[str] = set()
    for anchor in document.xpath("//a[@href]"):
        title = "".join(anchor.itertext()).strip()
        if not title:
            continue
        href = anchor.get("href", "").strip()
        if href.startswith("https://www.jiemian.com"):
            href = href.removeprefix("https://www.jiemian.com")
        elif href.startswith("http://www.jiemian.com"):
            href = href.removeprefix("http://www.jiemian.com")
        if not _JIEMIAN_PATH.fullmatch(href):
            continue
        url = f"https://www.jiemian.com{href}"
        if url not in seen:
            seen.add(url)
            urls.append(url)
            if len(urls) == max_articles:
                break
    if not urls:
        raise ValueError("Jiemian listing contains no titled article links")
    return urls


def parse_jiemian_article(content: bytes, url: str, source: FeedSource) -> Article:
    """Parse a Jiemian article using only its heading and article-info time."""
    document = _document(content)
    headings = document.xpath("//h1")
    title = " ".join("".join(headings[0].itertext()).split()) if headings else ""
    if not title:
        raise ValueError("Jiemian article is missing an h1 title")

    info_nodes = document.xpath(
        '//*[contains(concat(" ", normalize-space(@class), " "), " article-info ")]'
    )
    info_text = " ".join(" ".join("".join(node.itertext()).split()) for node in info_nodes)
    match = _JIEMIAN_TIME.search(info_text)
    if not match:
        raise ValueError("Jiemian article-info is missing a full publication timestamp")
    try:
        local_time = datetime.strptime(
            f"{match.group(1)} {match.group(2)}", "%Y/%m/%d %H:%M"
        ).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    except ValueError as error:
        raise ValueError("Jiemian article-info has an invalid publication timestamp") from error

    descriptions = document.xpath(
        '//meta[translate(@name,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="description"]/@content'
    )
    summary = descriptions[0].strip() if descriptions and descriptions[0].strip() else ""
    canonical_url = canonicalize_url(url)
    identifier = hashlib.sha256(canonical_url.encode()).hexdigest()[:16]
    try:
        return Article(
            id=identifier,
            title=title,
            url=canonical_url,
            source=source.name,
            language="zh",
            published_at=local_time.astimezone(UTC),
            topic=source.topic,
            summary=summary,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ValueError("invalid Jiemian article") from error
