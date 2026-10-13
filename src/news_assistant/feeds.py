from __future__ import annotations

import hashlib
import html
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import feedparser
import httpx
from pydantic import BaseModel, HttpUrl, ValidationError, field_validator

from .models import Article, Topic
from .normalization import normalize_url

logger = logging.getLogger(__name__)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NAIVE_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?$"
)


class FeedSource(BaseModel):
    name: str
    url: HttpUrl
    language: str
    topic: Topic
    naive_timezone: str | None = None

    @field_validator("naive_timezone")
    @classmethod
    def naive_timezone_must_be_known(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError(f"unknown timezone: {value}") from error
        return value


def canonicalize_url(value: str) -> str:
    return normalize_url(value)


def _article_language(title: str, summary: str, source_language: str) -> str:
    if source_language != "zh":
        return source_language
    plain_text = html.unescape(_HTML_TAG_RE.sub(" ", f"{title} {summary}"))
    return "zh" if _CJK_RE.search(plain_text) else "en"


def _entry_publication_time(entry: dict, source: FeedSource) -> datetime | None:
    if entry.get("published_parsed"):
        raw_value = entry.get("published")
        date_parts = entry["published_parsed"]
    elif entry.get("updated_parsed"):
        raw_value = entry.get("updated")
        date_parts = entry["updated_parsed"]
    else:
        return None
    if source.naive_timezone and isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if _NAIVE_ISO_DATETIME_RE.fullmatch(raw_value):
            try:
                local_time = datetime.fromisoformat(raw_value).replace(
                    tzinfo=ZoneInfo(source.naive_timezone)
                )
            except ValueError:
                pass
            else:
                return local_time.astimezone(UTC)
    return datetime(*date_parts[:6], tzinfo=UTC)


@dataclass
class FeedDiagnostics:
    entry_count: int | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1


def parse_feed(
    content: bytes, source: FeedSource, *, diagnostics: FeedDiagnostics | None = None,
) -> list[Article]:
    diagnostics = diagnostics if diagnostics is not None else FeedDiagnostics()
    parsed = feedparser.parse(content)
    diagnostics.entry_count = len(parsed.entries)
    if not parsed.version:
        raise ValueError("response is not an RSS or Atom feed")
    articles: list[Article] = []
    for entry in parsed.entries:
        title = entry.get("title")
        link = entry.get("link")
        if not isinstance(title, str) or not title.strip() or not isinstance(link, str) or not link.strip():
            diagnostics.reject("missing_title_or_link")
            continue
        try:
            published = _entry_publication_time(entry, source)
            if published is None:
                diagnostics.reject("missing_time")
                continue
            url = canonicalize_url(link)
            identifier = hashlib.sha256(url.encode()).hexdigest()[:16]
            summary = entry.get("summary", "")
            summary = summary.strip() if isinstance(summary, str) else ""
            articles.append(
                Article(
                    id=identifier,
                    title=title.strip(),
                    url=url,
                    source=source.name,
                    language=_article_language(title, summary, source.language),
                    published_at=published,
                    topic=source.topic,
                    summary=summary,
                )
            )
        except (TypeError, ValueError, ValidationError):
            diagnostics.reject("invalid_entry")
            continue
    return articles


def _http_loader(url: str) -> bytes:
    response = httpx.get(url, timeout=15, follow_redirects=True)
    response.raise_for_status()
    return response.content


def fetch_articles(
    sources: Iterable[FeedSource],
    *,
    loader: Callable[[str], bytes] = _http_loader,
    now: datetime | None = None,
    max_age_hours: int = 36,
    raise_on_error: bool = False,
) -> list[Article]:
    reference = now or datetime.now(UTC)
    cutoff = reference - timedelta(hours=max_age_hours)
    articles: list[Article] = []
    for source in sources:
        try:
            parsed = parse_feed(loader(str(source.url)), source)
            articles.extend(article for article in parsed if cutoff <= article.published_at <= reference)
        except (httpx.HTTPError, TimeoutError, OSError, ValueError, ValidationError) as error:
            if raise_on_error:
                raise
            logger.warning("feed_failed source=%s error=%s", source.name, type(error).__name__)
    return articles
