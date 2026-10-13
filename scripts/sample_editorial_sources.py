"""Capture a private, bounded editorial-source sample for offline review."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml

from news_assistant.article_content import fetch_public_html
from news_assistant.editorial_sources import (
    jiemian_article_urls,
    parse_jiemian_article,
    parse_sixthtone_listing,
)
from news_assistant.feeds import FeedDiagnostics, FeedSource, parse_feed
from news_assistant.models import Article, CoverageDomain
from news_assistant.source_audit import (
    _coverage_for,
    _is_chinese,
    _source_metadata,
    _summary_for,
)
from news_assistant.source_registry import RegisteredSource, SourceRegistry

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_JIEMIAN_ARTICLES = 20
MAX_CONCURRENCY = 4
SUPPORTED_ADAPTERS = {"rss", "sixthtone_listing", "jiemian_listing"}


class CacheReplayError(ValueError):
    """A requested response is absent or fails its local cache binding."""


def _aware_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {label} timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _load_registry(path: Path) -> SourceRegistry:
    return SourceRegistry.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _load_cache_index(path: Path) -> tuple[datetime, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        captured_at = _aware_utc(payload["captured_at"], "cache captured_at")
        entries = payload["entries"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid replay cache index: {path}") from error
    if not isinstance(entries, dict):
        raise TypeError("cache entries must be an object")
    return captured_at, entries


def _replay_entry(index_path: Path, entries: dict[str, Any], url: str) -> tuple[bytes, datetime]:
    entry = entries.get(url)
    if not isinstance(entry, dict):
        raise CacheReplayError("cache entry missing")
    try:
        relative = Path(entry["path"])
        root = index_path.parent.resolve()
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise CacheReplayError("cache path escapes index directory")
        body = path.read_bytes()
        expected = entry["sha256"]
        retrieved_at = _aware_utc(entry["retrieved_at"], "retrieved_at")
    except CacheReplayError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise CacheReplayError("cache entry invalid") from error
    if len(body) > MAX_RESPONSE_BYTES:
        raise CacheReplayError("cached response exceeds 2 MiB")
    actual = hashlib.sha256(body).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise CacheReplayError("cache response hash mismatch")
    return body, retrieved_at


def _fetch_live_bytes(url: str, *, is_rss: bool, dns_mode: str) -> tuple[bytes, datetime]:
    if is_rss:
        with (
            httpx.Client(timeout=15, follow_redirects=True, trust_env=False) as client,
            client.stream(
                "GET",
                url,
                headers={
                    "User-Agent": "PersonalNewsAssistant/0.1",
                    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
                    "Accept-Encoding": "identity",
                },
            ) as response,
        ):
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise ValueError("RSS response exceeds 2 MiB")
                body.extend(chunk)
        return bytes(body), datetime.now(UTC)
    page = fetch_public_html(url, max_bytes=MAX_RESPONSE_BYTES, dns_mode=dns_mode)
    return page.html, datetime.now(UTC)


def _source_feed_source(source: RegisteredSource) -> FeedSource:
    return FeedSource(
        name=source.name,
        url=source.url,
        language=source.language,
        topic=source.topics[0],
        naive_timezone=source.naive_timezone,
    )


def _validate_paths(output_path: Path, replay_cache_path: Path | None, fetch: bool) -> Path:
    output = output_path.resolve()
    if "public" in output.parts:
        raise ValueError("snapshot output must be private and outside public")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("snapshot output already exists")
    if fetch and replay_cache_path is not None:
        raise ValueError("--fetch and --replay-cache are mutually exclusive")
    if not fetch and replay_cache_path is None:
        raise ValueError("choose exactly one of --fetch or --replay-cache")
    return output


def _metadata_result(
    source: RegisteredSource,
    articles: list[Article],
    *,
    captured_at: datetime,
    cutoff: datetime,
    fatal_error: Exception | None = None,
    parse_failures: list[dict[str, str]] | None = None,
    partial: bool = False,
    parse_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = parse_failures or []
    metadata = _source_metadata(source)
    diagnostics = parse_diagnostics or {}
    if fatal_error is not None:
        return {
            **metadata,
            **diagnostics,
            "status": "error",
            "error_type": type(fatal_error).__name__,
            "parsed_count": 0,
            "fresh_count": 0,
            "stale_count": 0,
            "future_count": 0,
            "latest_published_at": None,
            "oldest_published_at": None,
            "actual_language_counts": {"zh": 0, "en": 0},
            "candidates": [],
            "partial": bool(failures),
            "parse_failures": failures,
        }

    fresh = [article for article in articles if cutoff <= article.published_at <= captured_at]
    stale_count = sum(article.published_at < cutoff for article in articles)
    future_count = sum(article.published_at > captured_at for article in articles)
    zh_count = sum(_is_chinese(article) for article in articles)
    latest = max((article.published_at for article in articles), default=None)
    oldest = min((article.published_at for article in articles), default=None)
    return {
        **metadata,
        **diagnostics,
        "status": "ok" if fresh else "no_recent",
        "error_type": None,
        "parsed_count": len(articles),
        "fresh_count": len(fresh),
        "stale_count": stale_count,
        "future_count": future_count,
        "latest_published_at": latest.isoformat() if latest else None,
        "oldest_published_at": oldest.isoformat() if oldest else None,
        "actual_language_counts": {"zh": zh_count, "en": len(articles) - zh_count},
        "candidates": [article.model_dump(mode="json") for article in fresh],
        "partial": partial or bool(failures),
        "parse_failures": failures,
    }


def sample_editorial_sources(
    *,
    sources_path: Path,
    output_path: Path,
    baseline_sources_path: Path | None = None,
    replay_cache_path: Path | None = None,
    fetch: bool = False,
    dns_mode: str = "system",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build and save a source-audit-v1 snapshot from replay or bounded public fetches."""
    output = _validate_paths(output_path, replay_cache_path, fetch)
    if dns_mode not in {"system", "cloudflare"}:
        raise ValueError("dns_mode must be system or cloudflare")
    editorial_registry = _load_registry(sources_path)
    editorial_sources = [source for source in editorial_registry.sources if source.enabled]
    if any(source.production_enabled for source in editorial_sources):
        raise ValueError("enabled editorial sources must set production_enabled: false")
    unsupported = [source.adapter for source in editorial_sources if source.adapter not in SUPPORTED_ADAPTERS]
    if unsupported:
        raise ValueError(f"unsupported editorial adapter: {unsupported[0]}")

    baseline_sources: list[RegisteredSource] = []
    if baseline_sources_path is not None:
        baseline_sources = _load_registry(baseline_sources_path).enabled_feed_sources()
    sources = [*baseline_sources, *editorial_sources]
    ids = [source.id for source in sources]
    names = [source.name for source in sources]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("baseline and editorial sources must have unique IDs and names")

    if replay_cache_path is not None:
        captured_at, cache_entries = _load_cache_index(replay_cache_path)
        cache_root = replay_cache_path
    else:
        reference = now or datetime.now(UTC)
        if reference.tzinfo is None or reference.utcoffset() is None:
            raise ValueError("captured time must be timezone-aware")
        captured_at = reference.astimezone(UTC)
        cache_entries = {}
        cache_root = None
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("captured time must be timezone-aware")
    cutoff = captured_at - timedelta(hours=36)

    raw_dir = output.with_name(f"{output.stem}.source-cache")
    if fetch and (raw_dir.exists() or raw_dir.is_symlink()):
        raise ValueError("live response directory already exists")

    live_index: dict[str, dict[str, str]] = {}
    live_index_lock = threading.Lock()

    def acquire(url: str, *, is_rss: bool) -> tuple[bytes, datetime]:
        if cache_root is not None:
            return _replay_entry(cache_root, cache_entries, url)
        body, retrieved_at = _fetch_live_bytes(url, is_rss=is_rss, dns_mode=dns_mode)
        digest = hashlib.sha256(body).hexdigest()
        relative = Path("responses") / f"{hashlib.sha256(url.encode()).hexdigest()}.bin"
        response_path = raw_dir / relative
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(body)
        with live_index_lock:
            live_index[url] = {
                "path": relative.as_posix(),
                "sha256": digest,
                "retrieved_at": retrieved_at.isoformat(),
            }
        return body, retrieved_at

    initial_urls = list(dict.fromkeys(str(source.url) for source in sources))
    initial_types = {
        str(source.url): source.adapter == "rss"
        for source in sources
    }
    initial_results: dict[str, tuple[bytes, datetime] | Exception] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = {
            executor.submit(acquire, url, is_rss=initial_types[url]): url
            for url in initial_urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                initial_results[url] = future.result()
            except Exception as error:  # noqa: BLE001 - isolate failures by source URL
                initial_results[url] = error

    parsed_articles: dict[str, list[Article]] = {}
    source_diagnostics: dict[str, dict[str, Any]] = {}
    fatal_errors: dict[str, Exception] = {}
    parse_failures: dict[str, list[dict[str, str]]] = {}
    jiemian_urls_by_source: dict[str, list[str]] = {}
    article_owners: dict[str, set[str]] = {}

    for source in sources:
        url = str(source.url)
        if source.adapter == "rss":
            source_diagnostics[source.id] = {
                "entry_count": None,
                "rejected_entry_count": 0,
                "entry_rejection_reasons": {},
            }
        result = initial_results[url]
        if isinstance(result, Exception):
            fatal_errors[source.id] = result
            parsed_articles[source.id] = []
            parse_failures[source.id] = [{"url": url, "error_type": type(result).__name__}]
            continue
        content, _retrieved_at = result
        try:
            feed_source = _source_feed_source(source)
            if source.adapter == "rss":
                diagnostics = FeedDiagnostics()
                articles = parse_feed(content, feed_source, diagnostics=diagnostics)
                source_diagnostics[source.id] = {
                    "entry_count": diagnostics.entry_count,
                    "rejected_entry_count": sum(diagnostics.rejection_reasons.values()),
                    "entry_rejection_reasons": diagnostics.rejection_reasons,
                }
                if diagnostics.entry_count and not articles:
                    raise ValueError("RSS entries contained no valid articles")
                parsed_articles[source.id] = articles
            elif source.adapter == "sixthtone_listing":
                parsed_articles[source.id] = parse_sixthtone_listing(content, feed_source)
            else:
                urls = jiemian_article_urls(content, max_articles=MAX_JIEMIAN_ARTICLES + 1)
                jiemian_urls_by_source[source.id] = urls[:MAX_JIEMIAN_ARTICLES]
                if len(urls) > MAX_JIEMIAN_ARTICLES:
                    parse_failures.setdefault(source.id, []).append(
                        {"url": url, "error_type": "ArticleLimitReached"}
                    )
                parsed_articles[source.id] = []
                for article_url in jiemian_urls_by_source[source.id]:
                    article_owners.setdefault(article_url, set()).add(source.id)
        except Exception as error:  # noqa: BLE001 - isolate failures by source
            fatal_errors[source.id] = error
            parsed_articles[source.id] = []
            parse_failures.setdefault(source.id, []).append(
                {"url": url, "error_type": type(error).__name__}
            )

    article_results: dict[str, tuple[bytes, datetime] | Exception] = {}
    article_urls = list(article_owners)
    if article_urls:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
            futures = {
                executor.submit(acquire, url, is_rss=False): url
                for url in article_urls
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    article_results[url] = future.result()
                except Exception as error:  # noqa: BLE001 - isolate article page failures
                    article_results[url] = error

    for article_url, owners in article_owners.items():
        result = article_results[article_url]
        for source_id in owners:
            source = next(item for item in sources if item.id == source_id)
            if isinstance(result, Exception):
                parse_failures.setdefault(source_id, []).append(
                    {"url": article_url, "error_type": type(result).__name__}
                )
                continue
            content, _retrieved_at = result
            try:
                parsed_articles[source_id].append(
                    parse_jiemian_article(content, article_url, _source_feed_source(source))
                )
            except Exception as error:  # noqa: BLE001 - preserve other readable articles
                parse_failures.setdefault(source_id, []).append(
                    {"url": article_url, "error_type": type(error).__name__}
                )

    source_results = []
    for source in sources:
        source_results.append(
            _metadata_result(
                source,
                parsed_articles.get(source.id, []),
                captured_at=captured_at,
                cutoff=cutoff,
                fatal_error=fatal_errors.get(source.id),
                parse_failures=parse_failures.get(source.id, []),
                partial=bool(parse_failures.get(source.id)),
                parse_diagnostics=source_diagnostics.get(source.id),
            )
        )

    production = [source for source in source_results if source["production_enabled"]]
    shadow = [source for source in source_results if not source["production_enabled"]]
    partial = any(source["status"] == "error" or source["partial"] for source in source_results)
    report = {
        "schema_version": "source-audit-v1",
        "captured_at": captured_at.isoformat(),
        "max_age_hours": 36,
        "candidate_snapshot_available": True,
        "partial": partial,
        "discovery_limits": {
            "response_bytes": MAX_RESPONSE_BYTES,
            "max_concurrency": MAX_CONCURRENCY,
            "max_jiemian_articles_per_listing": MAX_JIEMIAN_ARTICLES,
            "window_start": cutoff.isoformat(),
            "window_end": captured_at.isoformat(),
        },
        "parse_failures": parse_failures,
        "sources": source_results,
        "coverage": {
            domain.value: {
                group: _coverage_for(
                    [source for source in subset if domain.value in source["coverage_domains"]]
                )
                for group, subset in (("production", production), ("shadow", shadow))
            }
            for domain in CoverageDomain
        },
        "summary": {"production": _summary_for(production), "shadow": _summary_for(shadow)},
    }

    if fetch:
        raw_dir.mkdir(parents=True, exist_ok=True)
        index = {"captured_at": captured_at.isoformat(), "entries": live_index}
        (raw_dir / "cache-index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("snapshot output already exists")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("config/editorial_sources.yaml"))
    parser.add_argument("--baseline-sources", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay-cache", type=Path)
    mode.add_argument("--fetch", action="store_true")
    parser.add_argument("--dns-mode", choices=("system", "cloudflare"), default="system")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        sample_editorial_sources(
            sources_path=args.sources,
            baseline_sources_path=args.baseline_sources,
            replay_cache_path=args.replay_cache,
            fetch=args.fetch,
            dns_mode=args.dns_mode,
            output_path=args.output,
        )
    except (OSError, ValueError, yaml.YAMLError, httpx.HTTPError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
