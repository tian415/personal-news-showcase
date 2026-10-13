import json
from datetime import UTC, datetime, timedelta

import pytest

from news_assistant.cli import build_digest, collect_signals
from news_assistant.deepseek import fallback_enrichment
from news_assistant.models import (
    Article,
    CandidateArticle,
    EvidenceRole,
    Preferences,
    SignalSnapshot,
    Topic,
)
from news_assistant.publication import audit_public_directory
from news_assistant.state import StateStore

NOW = datetime(2026, 9, 12, 7, tzinfo=UTC)


class OfflineEnricher:
    enrich = staticmethod(fallback_enrichment)


def test_private_community_candidates_reach_layers_without_entering_public_digest(tmp_path):
    state = tmp_path / "private-state"
    candidates = [
        CandidateArticle(
            id=name,
            candidate_id=name,
            source_id=name,
            publisher_group=name,
            title="Private community launch",
            source=name,
            signal_external_id="42",
            url=f"https://example.com/{name}",
            language="en",
            published_at=NOW,
            discovered_at=NOW,
            topic=Topic.AI,
            evidence_role=EvidenceRole.COMMUNITY_SIGNAL,
        )
        for name in ("forum-a", "forum-b")
    ]
    snapshots = [
        SignalSnapshot(
            source_id=name, external_id="42", captured_at=when, rank=1, reply_count=count
        )
        for name in ("forum-a", "forum-b")
        for when, count in ((NOW - timedelta(hours=6), 10), (NOW, 50))
    ]
    StateStore(state).write_signal_collection(NOW, snapshots, [], candidates)
    public_article = Article(
        id="public",
        title="Public official launch",
        source="Official",
        url="https://example.com/public",
        language="en",
        published_at=NOW,
        topic=Topic.AI,
    )
    digest = build_digest(
        [public_article],
        Preferences(),
        OfflineEnricher(),
        tmp_path / "public",
        selection_engine="layered_shadow_v1",
        state_dir=state,
        run_id="test",
        now=NOW,
    )
    assert [item.id for item in digest.articles] == ["public"]
    audit_public_directory(tmp_path / "public")
    html = (tmp_path / "public" / "index.html").read_text()
    assert "Private community launch" not in html
    report = json.loads(next(state.glob("runs/*/*/*/*/layered-shadow.json")).read_text())
    assert report["layer_counts"]["signal_radar"] == 1
    assert set(report["layered_only"]) == {"forum-a", "forum-b"}


def test_layered_build_rejects_private_state_inside_public_before_writing(tmp_path):
    public = tmp_path / "public"
    with pytest.raises(ValueError, match="state"):
        build_digest(
            [],
            Preferences(),
            OfflineEnricher(),
            public,
            selection_engine="layered_shadow_v1",
            state_dir=public / "state",
            now=NOW,
        )
    assert not public.exists()


def test_collect_signals_persists_explicit_candidate_snapshot_link(monkeypatch, tmp_path):
    from news_assistant import cli
    from news_assistant.discovery.v2ex import V2exFetchResult
    from news_assistant.source_registry import RegisteredSource, SourceRegistry

    source = RegisteredSource(
        id="v2ex",
        name="Forum",
        tier="T3",
        adapter="v2ex",
        url="https://example.com/api",
        topics=["ai"],
        coverage_domains=["ai_technology"],
        publisher_group="forum",
        production_enabled=False,
    )
    article = Article(
        id="v2ex:42",
        title="Forum title",
        source="Forum",
        url="https://example.com/42",
        language="en",
        published_at=NOW,
        topic="ai",
    )
    snapshot = SignalSnapshot(source_id="v2ex", external_id="42", captured_at=NOW, rank=1)
    monkeypatch.setattr(
        cli,
        "fetch_v2ex_topics",
        lambda *a, **kw: V2exFetchResult(articles=[article], snapshots=[snapshot]),
    )
    collect_signals(
        SourceRegistry(version="source-registry-v1", sources=[source]), tmp_path, now=NOW
    )
    saved = StateStore(tmp_path).load_recent_signal_candidates(NOW)
    assert len(saved) == 1
    assert saved[0].signal_external_id == "42"
    assert saved[0].source_id == "v2ex"
    assert saved[0].evidence_role is EvidenceRole.COMMUNITY_SIGNAL
