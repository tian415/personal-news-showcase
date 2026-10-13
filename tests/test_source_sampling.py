from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from news_assistant.reviewed_selection import prepare_review
from scripts import sample_editorial_sources as sampler

CAPTURED_AT = "2026-10-10T12:00:00+00:00"
RETRIEVED_AT = "2026-10-10T12:00:05+00:00"


def registered_source(source_id: str, adapter: str, url: str, *, production: bool) -> dict:
    return {
        "id": source_id,
        "name": source_id,
        "tier": "T1",
        "adapter": adapter,
        "url": url,
        "topics": ["exploration"],
        "coverage_domains": ["exploration"],
        "publisher_group": source_id,
        "evidence_role": "INDEPENDENT_REPORTING",
        "source_kind": "MEDIA",
        "language": "zh" if source_id.startswith("jiemian") else "en",
        "naive_timezone": "Asia/Shanghai" if source_id.startswith("jiemian") else None,
        "enabled": True,
        "production_enabled": production,
    }


def write_registry(path: Path, sources: list[dict], version: str = "editorial-local-v2") -> Path:
    path.write_text(yaml.safe_dump({"version": version, "sources": sources}, allow_unicode=True))
    return path


def write_cache(root: Path, entries: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = {
        "captured_at": CAPTURED_AT,
        "entries": {},
    }
    for number, (url, content) in enumerate(entries.items()):
        relative = f"response-{number}.bin"
        (root / relative).write_bytes(content)
        index["entries"][url] = {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "retrieved_at": RETRIEVED_AT,
        }
    path = root / "cache-index.json"
    path.write_text(json.dumps(index))
    return path


def article_xml(title: str, link: str, published: str) -> str:
    return (
        "<item><title>" + title + "</title><link>" + link + "</link>"
        "<pubDate>" + published + "</pubDate><description>来源摘要</description></item>"
    )


def test_replay_uses_one_36_hour_window_and_every_candidate_prepares_for_review(tmp_path: Path) -> None:
    rss_url = "https://rss.example.test/feed"
    sixth_url = "https://www.sixthtone.com/"
    jiemian_url = "https://www.jiemian.com/lists/130.html"
    jiemian_article_url = "https://www.jiemian.com/article/123.html"
    rss = (
        "<rss><channel>"
        + article_xml("Old", "https://rss.example.test/old", "2026-10-08T23:59:59Z")
        + article_xml("At cutoff", "https://rss.example.test/boundary", "2026-10-09T00:00:00Z")
        + article_xml("Future", "https://rss.example.test/future", "2026-10-10T12:00:01Z")
        + "</channel></rss>"
    ).encode()
    sixthtone = (
        b'<script id="__NEXT_DATA__">{"items":[{"contId":42,"name":"Sixth",'
        b'"pubTimeLong":1791504000000,"summary":"Sixth summary"}]}</script>'
    )
    jiemian_listing = b'<a href="/article/123.html">A titled article</a>'
    jiemian_article = (
        '<html><head><meta charset="utf-8"><meta name="description" content="摘要">'
        '</head><body><h1>界面文章</h1><div class="article-info">2026/10/09 12:00</div>'
        '</body></html>'
    ).encode()
    editorial = write_registry(
        tmp_path / "editorial.yaml",
        [
            registered_source("sixthtone", "sixthtone_listing", sixth_url, production=False),
            registered_source("jiemian", "jiemian_listing", jiemian_url, production=False),
        ],
    )
    baseline_row = registered_source("rss", "rss", rss_url, production=True)
    baseline_row.update(terms_reviewed_at="2026-01-01", access_stability="public_rss")
    shadow_baseline = registered_source(
        "baseline-shadow", "rss", "https://shadow.example.test/feed", production=False
    )
    shadow_baseline.update(terms_reviewed_at="2026-01-01", access_stability="public_rss")
    disabled_baseline = registered_source(
        "baseline-disabled", "rss", "https://disabled.example.test/feed", production=True
    )
    disabled_baseline["enabled"] = False
    baseline = write_registry(
        tmp_path / "baseline.yaml",
        [baseline_row, shadow_baseline, disabled_baseline],
        "source-registry-v2",
    )
    cache = write_cache(
        tmp_path / "cache",
        {
            rss_url: rss,
            sixth_url: sixthtone,
            jiemian_url: jiemian_listing,
            jiemian_article_url: jiemian_article,
        },
    )
    output = tmp_path / "private" / "snapshot.json"

    snapshot = sampler.sample_editorial_sources(
        sources_path=editorial,
        baseline_sources_path=baseline,
        replay_cache_path=cache,
        output_path=output,
    )

    assert snapshot["captured_at"] == CAPTURED_AT
    by_id = {source["id"]: source for source in snapshot["sources"]}
    assert by_id["rss"]["fresh_count"] == 1
    assert by_id["rss"]["stale_count"] == 1
    assert by_id["rss"]["future_count"] == 1
    assert by_id["sixthtone"]["fresh_count"] == 1
    assert by_id["jiemian"]["fresh_count"] == 1
    assert by_id["rss"]["production_enabled"] is True
    assert set(by_id) == {"rss", "sixthtone", "jiemian"}
    assert len(prepare_review(snapshot)["items"]) == 3


def test_cache_miss_and_hash_mismatch_fail_without_network(tmp_path: Path, monkeypatch) -> None:
    missing_url = "https://www.sixthtone.com/"
    mismatch_url = "https://www.jiemian.com/lists/130.html"
    editorial = write_registry(
        tmp_path / "editorial.yaml",
        [
            registered_source("missing", "sixthtone_listing", missing_url, production=False),
            registered_source("mismatch", "jiemian_listing", mismatch_url, production=False),
        ],
    )
    cache = write_cache(tmp_path / "cache", {mismatch_url: b"<html></html>"})
    index = json.loads(cache.read_text())
    index["entries"][mismatch_url]["sha256"] = "0" * 64
    cache.write_text(json.dumps(index))
    monkeypatch.setattr(sampler, "_fetch_live_bytes", lambda *a, **k: pytest.fail("network attempted"))

    snapshot = sampler.sample_editorial_sources(
        sources_path=editorial,
        replay_cache_path=cache,
        output_path=tmp_path / "snapshot.json",
    )

    assert [source["status"] for source in snapshot["sources"]] == ["error", "error"]
    assert [source["error_type"] for source in snapshot["sources"]] == ["CacheReplayError"] * 2


def test_enabled_editorial_production_source_is_rejected_before_output(tmp_path: Path) -> None:
    editorial = write_registry(
        tmp_path / "editorial.yaml",
        [registered_source("bad", "sixthtone_listing", "https://www.sixthtone.com/", production=True)],
    )
    output = tmp_path / "snapshot.json"

    with pytest.raises(ValueError, match="production_enabled"):
        sampler.sample_editorial_sources(
            sources_path=editorial,
            replay_cache_path=tmp_path / "missing-cache.json",
            output_path=output,
        )

    assert not output.exists()


def test_one_jiemian_article_failure_keeps_other_parsed_article_and_marks_partial(tmp_path: Path) -> None:
    listing_url = "https://www.jiemian.com/lists/130.html"
    good_url = "https://www.jiemian.com/article/123.html"
    missing_url = "https://www.jiemian.com/article/456.html"
    editorial = write_registry(
        tmp_path / "editorial.yaml",
        [registered_source("jiemian", "jiemian_listing", listing_url, production=False)],
    )
    listing = (
        b'<a href="/article/123.html">Good title</a>'
        b'<a href="/article/456.html">Missing page</a>'
    )
    good_article = (
        b'<html><head><meta charset="utf-8"></head><body><h1>Good</h1>'
        b'<div class="article-info">2026/10/10 10:00</div></body></html>'
    )
    cache = write_cache(tmp_path / "cache", {listing_url: listing, good_url: good_article})

    snapshot = sampler.sample_editorial_sources(
        sources_path=editorial,
        replay_cache_path=cache,
        output_path=tmp_path / "snapshot.json",
    )

    source = snapshot["sources"][0]
    assert source["status"] == "ok"
    assert source["fresh_count"] == 1
    assert source["partial"] is True
    assert source["parse_failures"] == [{"url": missing_url, "error_type": "CacheReplayError"}]
    assert len(prepare_review(snapshot)["items"]) == 1


def test_private_output_is_never_written_under_public(tmp_path: Path) -> None:
    editorial = write_registry(tmp_path / "editorial.yaml", [])

    with pytest.raises(ValueError, match="public"):
        sampler.sample_editorial_sources(
            sources_path=editorial,
            replay_cache_path=tmp_path / "cache-index.json",
            output_path=tmp_path / "public" / "snapshot.json",
        )


def test_live_fetch_saves_bounded_response_and_replay_index_without_network(tmp_path: Path, monkeypatch) -> None:
    rss_url = "https://rss.example.test/feed"
    editorial = write_registry(
        tmp_path / "editorial.yaml",
        [registered_source("rss", "rss", rss_url, production=False)],
    )
    response = (
        "<rss><channel>"
        + article_xml("Fresh", "https://rss.example.test/fresh", "2026-10-10T10:00:00Z")
        + "</channel></rss>"
    ).encode()
    retrieved = datetime(2026, 10, 10, 12, 0, 5, tzinfo=UTC)
    monkeypatch.setattr(
        sampler,
        "_fetch_live_bytes",
        lambda url, *, is_rss, dns_mode: (response, retrieved),
    )
    output = tmp_path / "snapshot.json"

    snapshot = sampler.sample_editorial_sources(
        sources_path=editorial,
        fetch=True,
        output_path=output,
        now=datetime.fromisoformat(CAPTURED_AT),
    )

    raw_dir = tmp_path / "snapshot.source-cache"
    cache_index = json.loads((raw_dir / "cache-index.json").read_text())
    entry = cache_index["entries"][rss_url]
    saved = (raw_dir / entry["path"]).read_bytes()
    assert snapshot["sources"][0]["fresh_count"] == 1
    assert cache_index["captured_at"] == CAPTURED_AT
    assert saved == response
    assert entry["sha256"] == hashlib.sha256(response).hexdigest()
    assert entry["retrieved_at"] == retrieved.isoformat()
