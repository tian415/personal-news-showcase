from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError

from ..models import Article, SignalSnapshot
from ..source_registry import RegisteredSource


class V2exFetchResult(BaseModel):
    articles: list[Article] = Field(default_factory=list)
    snapshots: list[SignalSnapshot] = Field(default_factory=list)


def fetch_v2ex_topics(
    source: RegisteredSource, *, loader: Callable[[str], bytes], now: datetime
) -> V2exFetchResult:
    if source.adapter != "v2ex":
        raise ValueError(f"source {source.id} is not a V2EX source")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    payload = json.loads(loader(str(source.url)))
    if not isinstance(payload, list):
        raise TypeError("V2EX response must be a list")

    articles: list[Article] = []
    snapshots: list[SignalSnapshot] = []
    for rank, topic in enumerate(payload, start=1):
        if not isinstance(topic, dict):
            continue
        try:
            raw_id = topic["id"]
            raw_title = topic["title"]
            if raw_id is None or raw_title is None:
                continue
            external_id = str(raw_id).strip()
            title = str(raw_title).strip()
            if not external_id or not title:
                continue
            published_at = datetime.fromtimestamp(int(topic["created"]), tz=UTC)
            url = topic.get("url") or f"https://www.v2ex.com/t/{external_id}"
            article = Article(
                id=f"{source.id}:{external_id}",
                title=title,
                url=url,
                source=source.name,
                language=source.language,
                published_at=published_at,
                topic=source.topics[0],
            )
            snapshot = SignalSnapshot(
                source_id=source.id,
                external_id=external_id,
                captured_at=now,
                rank=rank,
                reply_count=int(topic.get("replies", 0)),
            )
            articles.append(article)
            snapshots.append(snapshot)
        except (KeyError, TypeError, ValueError, OverflowError, OSError, ValidationError):
            continue
    return V2exFetchResult(articles=articles, snapshots=snapshots)
