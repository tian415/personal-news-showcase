from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

from .feeds import FeedDiagnostics, _http_loader, parse_feed
from .models import Article, CoverageDomain, EvidenceRole
from .source_registry import RegisteredSource, SourceRegistry

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class NoValidFeedEntriesError(ValueError):
    """The feed had entries, but none satisfied the article input contract."""


def _parse_diagnostics(diagnostics: FeedDiagnostics) -> dict[str, Any]:
    return {
        "entry_count": diagnostics.entry_count,
        "rejected_entry_count": sum(diagnostics.rejection_reasons.values()),
        "entry_rejection_reasons": diagnostics.rejection_reasons,
    }


def _is_chinese(article: Article) -> bool:
    return bool(_CJK_PATTERN.search(f"{article.title} {article.summary}"))


def _error_result(source: RegisteredSource, error: Exception) -> dict[str, Any]:
    return {
        **_source_metadata(source),
        "status": "error",
        "error_type": type(error).__name__,
        "parsed_count": 0,
        "fresh_count": 0,
        "stale_count": 0,
        "future_count": 0,
        "latest_published_at": None,
        "oldest_published_at": None,
        "actual_language_counts": {"zh": 0, "en": 0},
    }


def _source_metadata(source: RegisteredSource) -> dict[str, Any]:
    return {
        "id": source.id,
        "name": source.name,
        "url": str(source.url),
        "production_enabled": source.production_enabled,
        "evidence_role": source.evidence_role.value,
        "publisher_group": source.publisher_group,
        "topics": [topic.value for topic in source.topics],
        "coverage_domains": [domain.value for domain in source.coverage_domains],
        "configured_language": source.language,
        "configured_naive_timezone": source.naive_timezone,
    }


def _audit_source(
    source: RegisteredSource,
    *,
    loader: Callable[[str], bytes],
    reference: datetime,
    cutoff: datetime,
    capture_candidates: bool = False,
) -> dict[str, Any]:
    diagnostics = FeedDiagnostics()
    try:
        articles = parse_feed(loader(str(source.url)), source.as_feed_source(), diagnostics=diagnostics)
        if diagnostics.entry_count and not articles:
            raise NoValidFeedEntriesError("all feed entries were rejected")
    except Exception as error:  # noqa: BLE001 - isolate every source failure
        result = {**_error_result(source, error), **_parse_diagnostics(diagnostics)}
        if capture_candidates:
            result["candidates"] = []
        return result

    fresh_count = sum(cutoff <= article.published_at <= reference for article in articles)
    stale_count = sum(article.published_at < cutoff for article in articles)
    future_count = sum(article.published_at > reference for article in articles)
    zh_count = sum(_is_chinese(article) for article in articles)
    latest = max((article.published_at for article in articles), default=None)
    oldest = min((article.published_at for article in articles), default=None)
    result = {
        **_source_metadata(source),
        **_parse_diagnostics(diagnostics),
        "status": "ok" if fresh_count else "no_recent",
        "error_type": None,
        "parsed_count": len(articles),
        "fresh_count": fresh_count,
        "stale_count": stale_count,
        "future_count": future_count,
        "latest_published_at": latest.isoformat() if latest else None,
        "oldest_published_at": oldest.isoformat() if oldest else None,
        "actual_language_counts": {"zh": zh_count, "en": len(articles) - zh_count},
    }
    if capture_candidates:
        result["candidates"] = [article.model_dump(mode="json") for article in articles
                                if cutoff <= article.published_at <= reference]
    return result


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _summary_for(sources: list[dict[str, Any]]) -> dict[str, Any]:
    fresh_sources = [source for source in sources if source["fresh_count"] > 0]
    return {
        "source_count": len(sources),
        "available_source_count": sum(source["status"] != "error" for source in sources),
        "fresh_candidate_count": sum(source["fresh_count"] for source in sources),
        "coverage_domains_with_fresh": _stable_unique(
            [
                domain
                for source in fresh_sources
                for domain in source["coverage_domains"]
            ]
        ),
        "publisher_groups_with_fresh": _stable_unique(
            [source["publisher_group"] for source in fresh_sources]
        ),
    }


def _groups_by_role(sources: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {role.value: sorted({source["publisher_group"] for source in sources
                                if source["evidence_role"] == role.value})
             for role in EvidenceRole}


def _coverage_for(sources: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({source["publisher_group"] for source in sources})
    available = [source for source in sources if source["status"] != "error"]
    fresh = [source for source in sources if source["fresh_count"] > 0]
    roles = _groups_by_role(sources)
    available_roles = _groups_by_role(available)
    gaps = []
    if not sources:
        gaps.append("no_configured_sources")
    else:
        if len(groups) == 1:
            gaps.append("single_publisher")
        if not roles[EvidenceRole.PRIMARY_FACT]:
            gaps.append("no_primary_source")
        if not roles[EvidenceRole.INDEPENDENT_REPORTING]:
            gaps.append("no_independent_reporting_source")
        for role, label in ((EvidenceRole.PRIMARY_FACT, "primary"),
                            (EvidenceRole.INDEPENDENT_REPORTING, "independent_reporting")):
            if roles[role] and not available_roles[role]:
                gaps.append(f"no_available_{label}_source")
        if all(source["status"] == "error" for source in sources):
            gaps.append("all_sources_unavailable")
    return {
        **_summary_for(sources),
        "fresh_source_count": sum(source["fresh_count"] > 0 for source in sources),
        "publisher_group_count": len(groups),
        "publisher_groups": groups,
        "publisher_groups_by_role": roles,
        "available_publisher_group_count": len({s["publisher_group"] for s in available}),
        "available_publisher_groups_by_role": available_roles,
        "fresh_publisher_groups_by_role": _groups_by_role(fresh),
        "gaps": gaps,
    }


def audit_sources(
    registry: SourceRegistry,
    *,
    loader: Callable[[str], bytes] = _http_loader,
    now: datetime | None = None,
    max_age_hours: int = 36,
    capture_candidates: bool = False,
) -> dict[str, Any]:
    """Fetch enabled RSS sources and return a read-only freshness report."""
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive")
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    cutoff = reference - timedelta(hours=max_age_hours)
    registered = registry.enabled_sources("rss")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                _audit_source,
                source,
                loader=loader,
                reference=reference,
                cutoff=cutoff,
                capture_candidates=capture_candidates,
            )
            for source in registered
        ]
        sources = [future.result() for future in futures]

    production = [source for source in sources if source["production_enabled"]]
    shadow = [source for source in sources if not source["production_enabled"]]
    return {
        "schema_version": "source-audit-v1",
        "captured_at": reference.isoformat(),
        "max_age_hours": max_age_hours,
        "candidate_snapshot_available": capture_candidates,
        "sources": sources,
        # Registered roles describe coverage, not independent verification of every claim.
        "coverage": {domain.value: {
            name: _coverage_for([source for source in subset
                                 if domain.value in source["coverage_domains"]])
            for name, subset in (("production", production), ("shadow", shadow))
        } for domain in CoverageDomain},
        "summary": {
            "production": _summary_for(production),
            "shadow": _summary_for(shadow),
        },
    }
