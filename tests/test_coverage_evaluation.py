from copy import deepcopy
from datetime import UTC, datetime

import pytest

from news_assistant.coverage_evaluation import evaluate_coverage
from news_assistant.models import CoverageDomain, Preferences, SourceTier, Topic
from news_assistant.source_audit import audit_sources
from news_assistant.source_registry import RegisteredSource, SourceRegistry

NOW = datetime(2026, 9, 28, 8, tzinfo=UTC)


def snapshot():
    sources = [RegisteredSource(
        id=name, name=name, tier=SourceTier.T1, adapter="rss",
        url=f"https://{name}.example/feed", topics=[Topic.TECH],
        coverage_domains=[CoverageDomain.AI_TECHNOLOGY], publisher_group=name,
        production_enabled=production,
    ) for name, production in [("production", True), ("shadow", False)]]

    def loader(url):
        name = "production" if "production" in url else "shadow"
        return (
            f'<rss version="2.0"><channel><title>{name}</title><item>'
            f'<title>{name} research update</title><link>https://{name}.example/event</link>'
            '<pubDate>Mon, 28 Sep 2026 07:00:00 GMT</pubDate>'
            '</item></channel></rss>'
        ).encode()

    return audit_sources(SourceRegistry(version="source-registry-v1", sources=sources),
                         loader=loader, now=NOW, capture_candidates=True)


def reference():
    return {
        "schema_version": "coverage-reference-v1", "synthetic": True,
        "purpose": "evaluation", "reviewed_by": "test reviewer",
        "reviewed_at": "2026-09-28T08:00:00Z",
        "reference_origin": "independently reviewed fixture; not a quality result",
        "events": [{
            "event_id": name, "title": f"{name} fixture", "coverage_domain": "ai_technology",
            "available_at": "2026-09-28T07:00:00Z", "major": True,
            "verified": True, "verification_note": "synthetic regression scenario",
            "evidence_urls": [f"https://{name}.example/event"],
        } for name in ["production", "shadow", "missing"]],
    }


def test_replay_distinguishes_discovery_gaps_from_shadow_only_candidates():
    result = evaluate_coverage(snapshot(), reference(), Preferences())
    assert result["summary"]["reference_event_count"] == 3
    assert result["summary"]["production_discovered_count"] == 1
    assert result["summary"]["all_sources_discovered_count"] == 2
    assert result["summary"]["legacy_selected_count"] == 1
    assert result["summary"]["production_major_discovery_recall"] == pytest.approx(1 / 3)
    assert {e["event_id"]: e["status"] for e in result["events"]} == {
        "production": "selected", "shadow": "shadow_only", "missing": "not_discovered",
    }
    assert result["release_readiness"]["status"] == "not_ready"
    assert result["synthetic"] is True
    assert result["domains"]["policy_public_affairs"]["production_major_discovery_recall"] is None


def test_low_quality_reference_is_discovered_but_not_selected():
    captured = snapshot()
    captured["sources"][0]["candidates"][0]["quality_score"] = 0.1
    result = evaluate_coverage(captured, reference(), Preferences())
    assert result["events"][0]["status"] == "not_selected"
    assert result["summary"]["legacy_selected_count"] == 0


def test_unverified_and_out_of_window_references_are_excluded_not_counted_as_misses():
    refs = reference()
    refs["events"][0]["verified"] = False
    refs["events"][1]["available_at"] = "2026-09-20T07:00:00Z"
    refs["events"][2]["available_at"] = "2026-09-29T07:00:00Z"
    result = evaluate_coverage(snapshot(), refs, Preferences())
    assert result["summary"]["reference_event_count"] == 0
    assert result["summary"]["production_major_discovery_recall"] is None
    assert len(result["excluded_references"]) == 3


def test_unambiguous_canonical_urls_match_without_inventing_semantic_matches():
    refs = reference()
    refs["events"][0]["evidence_urls"] = ["https://production.example/event?utm_source=test"]
    refs["events"][2]["title"] = "production research update"
    result = evaluate_coverage(snapshot(), refs, Preferences())
    assert result["events"][0]["status"] == "selected"
    assert result["events"][2]["status"] == "not_discovered"


@pytest.mark.parametrize("mutation", ["duplicate_id", "ambiguous_url", "naive_time"])
def test_rejects_ambiguous_or_invalid_reference_sets(mutation):
    refs = reference()
    if mutation == "duplicate_id":
        refs["events"][1]["event_id"] = refs["events"][0]["event_id"]
    elif mutation == "ambiguous_url":
        refs["events"][1]["evidence_urls"] = refs["events"][0]["evidence_urls"]
    else:
        refs["events"][0]["available_at"] = "2026-09-28T07:00:00"
    with pytest.raises(ValueError):
        evaluate_coverage(snapshot(), refs, Preferences())


@pytest.mark.parametrize("mutation", ["missing_capture", "count_mismatch", "future_candidate",
                                     "duplicate_source", "conflicting_article"])
def test_rejects_incomplete_or_inconsistent_candidate_snapshots(mutation):
    captured = snapshot()
    if mutation == "missing_capture":
        del captured["candidate_snapshot_available"]
    elif mutation == "count_mismatch":
        captured["sources"][0]["fresh_count"] = 9
    elif mutation == "future_candidate":
        captured["sources"][0]["candidates"][0]["published_at"] = "2026-10-01T07:00:00Z"
    elif mutation == "duplicate_source":
        captured["sources"].append(deepcopy(captured["sources"][0]))
    else:
        captured["sources"][1]["candidates"][0]["id"] = captured["sources"][0]["candidates"][0]["id"]
    with pytest.raises(ValueError):
        evaluate_coverage(captured, reference(), Preferences())
