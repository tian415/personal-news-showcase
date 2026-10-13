"""Synthetic contract checks, never evidence of real-world accuracy."""

import hashlib
from copy import deepcopy

import pytest

from experiments.multisource_validation.engine import corpus_hash, run_comparison


def inputs(count=2):
    corpus = {
        "schema": "trial-v1",
        "reference_utc": "2026-10-10T08:00:00+00:00",
        "window_hours": 36,
        "sources": [],
        "articles": [],
    }
    reviews = {"reviewer": "synthetic tester", "items": []}
    for i in range(count):
        body = "On October 10 the agency announced a rule. It affects local laboratories."
        digest = hashlib.sha256(body.encode()).hexdigest()
        corpus["articles"].append(
            {
                "id": str(i),
                "source_id": f"s{i}",
                "source": f"Source {i}",
                "publisher_group": f"g{i}",
                "language": "en",
                "domain_hint": "science_health",
                "title": f"Laboratory rule development {i}",
                "url": f"https://example.org/{i}",
                "published_at": "2026-10-10T07:00:00+00:00",
                "summary": "",
                "body": body,
                "body_sha256": digest,
                "html_sha256": hashlib.sha256(("<p>" + body + "</p>").encode()).hexdigest(),
                "body_status": "extracted",
                "retrieved_at": "2026-10-10T07:30:00+00:00",
                "body_final_url": f"https://example.org/final/{i}",
                "extractor_version": "synthetic-fixture-1",
                "appearances": [],
            }
        )
        reviews["items"].append(
            {
                "id": str(i),
                "body_sha256": digest,
                "html_sha256": hashlib.sha256(("<p>" + body + "</p>").encode()).hexdigest(),
                "status": "reviewed",
                "reason": "Explicit synthetic judgment",
                "domain": "science_health",
                "region": "global",
                "headline": f"实验室规则变化{i}",
                "identity": {
                    "actor": "agency",
                    "action": "announce",
                    "object": f"rule{i}",
                    "place": "global",
                    "development": "new rule",
                },
                "time_status": "confirmed",
                "event_at": "2026-10-10T06:00:00+00:00",
                "time_basis": "event",
                "time_quote": "On October 10",
                "why_today": "Rule announced today",
                "impact": "Affects laboratories",
                "impact_quote": "It affects local laboratories.",
                "importance": 0.7,
                "quality": 0.8,
                "story_shape": "single_event",
                "origin_group": f"g{i}",
                "claims": [
                    {
                        "text": "Rule announced",
                        "quote": "the agency announced a rule.",
                        "origin_key": f"original{i}",
                        "kind": "direct_reporting",
                        "supports_core": True,
                    }
                ],
                "conflict": False,
                "conflict_reason": None,
            }
        )
    reviews["corpus_sha256"] = corpus_hash(corpus)
    return corpus, reviews


def test_semantic_cross_language_cluster_old_title_misses():
    c, r = inputs()
    c["articles"][1]["title"] = "机构公布新的实验室规则"
    c["articles"][1]["language"] = "zh"
    r["items"][1]["identity"] = deepcopy(r["items"][0]["identity"])
    r["items"][1]["domain"] = "policy_public_affairs"
    r["corpus_sha256"] = corpus_hash(c)
    result = run_comparison(c, r)
    assert len(result["arms"]["legacy"]["selected"]) == 2
    assert result["arms"]["proposed"]["clusters"] == [["0", "1"]]
    assert result["arms"]["proposed"]["selected"][0]["confidence"]["independent_chains"] == 2


def test_different_object_and_date_do_not_merge():
    c, r = inputs(3)
    r["items"][2]["identity"] = deepcopy(r["items"][0]["identity"])
    r["items"][2]["event_at"] = "2026-10-09T06:00:00+00:00"
    assert len(run_comparison(c, r)["arms"]["proposed"]["clusters"]) == 3


@pytest.mark.parametrize(
    "change,reason",
    [
        ("pending", "pending_review"),
        ("missing", "pending_review"),
        ("conflict", "core_conflict"),
        ("assertion", "insufficient_core_evidence"),
        ("old", "old_event"),
    ],
)
def test_admission_gates_cannot_be_rescued_by_scores(change, reason):
    c, r = inputs(1)
    item = r["items"][0]
    item["importance"] = 1
    item["hotness"] = 1000000
    if change == "pending":
        item["status"] = "pending"
        item.pop("quality")
    elif change == "missing":
        r["items"] = []
    elif change == "conflict":
        item.update(conflict=True, conflict_reason="Core reports disagree")
    elif change == "assertion":
        item["claims"][0]["kind"] = "assertion"
    else:
        item["event_at"] = "2025-01-01T00:00:00+00:00"
    arm = run_comparison(c, r)["arms"]["proposed"]
    assert arm["selected"] == []
    assert arm["excluded"]["0"] == reason


def test_duplicate_upstream_does_not_raise_value_or_confidence():
    c, r = inputs(2)
    baseline = run_comparison(c, r, limit=1)["arms"]["proposed"]
    c["articles"].append(
        {
            **c["articles"][0],
            "id": "copy",
            "source_id": "copy",
            "source": "Copy host",
            "url": "https://example.org/copy",
        }
    )
    r["items"].append({**deepcopy(r["items"][0]), "id": "copy", "importance": 1})
    r["corpus_sha256"] = corpus_hash(c)
    after = run_comparison(c, r, limit=1)["arms"]["proposed"]
    assert after["selected"][0]["id"] == baseline["selected"][0]["id"]
    assert after["selected"][0]["importance"] == baseline["selected"][0]["importance"]
    assert after["selected"][0]["confidence"]["independent_chains"] == 1
    assert after["selected"][0]["headline"] == baseline["selected"][0]["headline"]


def test_major_is_in_same_coverage_set():
    c, r = inputs(3)
    r["items"][0]["importance"] = 0.9
    r["items"][2]["domain"] = "culture_entertainment"
    result = run_comparison(c, r, limit=2)["arms"]["proposed"]
    assert [x["member_ids"] for x in result["selected"]] == [["0"], ["2"]]


def test_input_order_does_not_change_proposed():
    c, r = inputs(4)
    before = run_comparison(c, r)["arms"]["proposed"]
    c["articles"].reverse()
    r["items"].reverse()
    r["corpus_sha256"] = corpus_hash(c)
    assert run_comparison(c, r)["arms"]["proposed"] == before


@pytest.mark.parametrize("change", ["binding", "body", "quote", "impact", "time"])
def test_binding_or_quote_mismatch_rejects(change):
    c, r = inputs(1)
    if change == "binding":
        r["corpus_sha256"] = "0" * 64
    elif change == "body":
        r["items"][0]["body_sha256"] = "0" * 64
    elif change == "quote":
        r["items"][0]["claims"][0]["quote"] = "invented"
    else:
        r["items"][0][f"{change}_quote"] = "invented"
    with pytest.raises(ValueError):
        run_comparison(c, r)


def test_same_publisher_different_claims_are_one_independent_chain():
    c, r = inputs(2)
    r["items"][1]["identity"] = deepcopy(r["items"][0]["identity"])
    r["items"][1]["origin_group"] = r["items"][0]["origin_group"]
    selected = run_comparison(c, r)["arms"]["proposed"]["selected"]
    assert selected[0]["confidence"]["independent_chains"] == 1


def test_same_upstream_different_hosts_are_one_chain():
    c, r = inputs(2)
    r["items"][1]["identity"] = deepcopy(r["items"][0]["identity"])
    r["items"][1]["claims"][0]["origin_key"] = r["items"][0]["claims"][0]["origin_key"]
    selected = run_comparison(c, r)["arms"]["proposed"]["selected"]
    assert selected[0]["confidence"]["independent_chains"] == 1


def test_unknown_identity_stays_singleton_and_flagged():
    c, r = inputs()
    for item in r["items"]:
        item["identity"] = {}
    arm = run_comparison(c, r)["arms"]["proposed"]
    assert arm["clusters"] == [["0"], ["1"]]
    assert "incomplete identity" in " ".join(arm["selected"][0]["confidence"]["limitations"])


def test_attributed_support_has_no_independent_bonus():
    c, r = inputs(1)
    r["items"][0]["claims"][0]["kind"] = "attributed_report"
    record = run_comparison(c, r)["arms"]["proposed"]["selected"][0]
    assert record["confidence"]["level"] == "attributed"
    assert record["confidence"]["independent_chains"] == 0


def test_title_only_and_disable_diversity_ablations():
    c, r = inputs(3)
    r["items"][1]["identity"] = deepcopy(r["items"][0]["identity"])
    assert (
        len(run_comparison(c, r, title_only_clustering=True)["arms"]["proposed"]["clusters"]) == 3
    )
    r["items"][0]["importance"] = 0.9
    r["items"][1]["identity"]["object"] = "separate"
    r["items"][2]["domain"] = "culture_entertainment"
    assert run_comparison(c, r, limit=2, disable_diversity=True)["arms"]["proposed"]["selected"][1][
        "member_ids"
    ] == ["1"]


def test_major_budget_never_exceeds_limit():
    c, r = inputs(4)
    for item in r["items"]:
        item["importance"] = 0.9
    assert len(run_comparison(c, r, limit=2)["arms"]["proposed"]["selected"]) == 2


def test_adapter_requires_real_html_capture_hash():
    c, r = inputs(1)
    del c["articles"][0]["html_sha256"]
    r["corpus_sha256"] = corpus_hash(c)
    with pytest.raises(ValueError, match="real HTML capture hash"):
        run_comparison(c, r)


def test_oversize_body_requires_explicit_common_pool_decision():
    c, r = inputs(1)
    c["articles"][0]["body"] += "x" * 20000
    digest = hashlib.sha256(c["articles"][0]["body"].encode()).hexdigest()
    c["articles"][0]["body_sha256"] = r["items"][0]["body_sha256"] = digest
    r["corpus_sha256"] = corpus_hash(c)
    with pytest.raises(ValueError, match="common-pool decision required"):
        run_comparison(c, r)


def test_adapter_preserves_actual_capture_provenance(monkeypatch):
    from experiments.multisource_validation import engine

    c, r = inputs(1)
    original = engine.prepare_review
    observed = {}

    def inspect(audit):
        observed.update(audit)
        return original(audit)

    monkeypatch.setattr(engine, "prepare_review", inspect)
    result = engine.run_comparison(c, r)
    content = observed["article_content"]["items"]["s0:0"]
    article = c["articles"][0]
    for actual, expected in [
        ("retrieved_at", "retrieved_at"),
        ("final_url", "body_final_url"),
        ("extractor_version", "extractor_version"),
    ]:
        assert content[actual] == article[expected]
    assert observed["sources"][0]["candidates"][0]["summary"] == article["summary"]
    assert not any("appends" in note for note in result["adapter_notes"])


def test_existing_value_members_are_event_full_set_and_no_ids_disappear():
    c, r = inputs(3)
    r["items"][1]["headline"] = r["items"][0]["headline"]
    r["items"][1]["identity"] = deepcopy(r["items"][0]["identity"])
    r["items"][2]["importance"] = 0.6
    result = run_comparison(c, r, limit=1)
    selected = result["arms"]["reviewed_value"]["selected"]
    assert selected[0]["member_ids"] == ["0", "1"]
    for arm in result["arms"].values():
        represented = {i for event in arm["selected"] for i in event["member_ids"]}
        assert represented | arm["excluded"].keys() == {"0", "1", "2"}


def test_missing_old_selector_reasons_are_explicit_not_inferred(monkeypatch):
    from types import SimpleNamespace

    from experiments.multisource_validation import engine

    c, r = inputs(1)
    monkeypatch.setattr(
        engine,
        "select_reviewed",
        lambda *a, **kw: SimpleNamespace(selected=[], events=[], excluded={}, value_details={}),
    )
    arm = engine.run_comparison(c, r)["arms"]["reviewed_value"]
    assert arm["excluded"] == {"0": "legacy_reason_unavailable"}


@pytest.mark.parametrize("explicit_pending", [True, False])
def test_metadata_only_null_hash_remains_pending_without_default_scores(explicit_pending):
    c, r = inputs(2)
    c["articles"][1].update(body="", body_sha256=None, body_status="pending", html_sha256=None)
    if explicit_pending:
        r["items"][1] = {"id": "1", "body_sha256": None, "status": "pending"}
    else:
        r["items"] = r["items"][:1]
    r["corpus_sha256"] = corpus_hash(c)
    result = run_comparison(c, r)
    assert result["pool_ids"] == ["0"]
    for arm in result["arms"].values():
        assert arm["excluded"]["1"] == "pending_review"
        assert all("1" not in event["member_ids"] for event in arm["selected"])
        represented = {i for event in arm["selected"] for i in event["member_ids"]}
        assert represented | arm["excluded"].keys() == {"0", "1"}


@pytest.mark.parametrize(
    "body,body_hash",
    [("", None), ("", hashlib.sha256(b"").hexdigest()), ("captured evidence", None)],
)
def test_ok_capture_cannot_be_empty_or_unhashed_even_when_pending(body, body_hash):
    c, r = inputs(1)
    c["articles"][0].update(body=body, body_sha256=body_hash, body_status="ok")
    r["items"][0] = {"id": "0", "body_sha256": body_hash, "status": "pending"}
    r["corpus_sha256"] = corpus_hash(c)
    with pytest.raises(ValueError):
        run_comparison(c, r)


@pytest.mark.parametrize("difference", ["object", "actor", "date"])
def test_equal_titles_cannot_override_known_semantic_conflict(difference):
    c, r = inputs(2)
    c["articles"][1]["title"] = c["articles"][0]["title"]
    r["items"][1]["identity"] = deepcopy(r["items"][0]["identity"])
    if difference == "date":
        r["items"][1]["event_at"] = "2026-10-09T06:00:00+00:00"
    else:
        r["items"][1]["identity"][difference] = "different normalized value"
    r["corpus_sha256"] = corpus_hash(c)
    assert run_comparison(c, r)["arms"]["proposed"]["clusters"] == [["0"], ["1"]]
    assert run_comparison(c, r, title_only_clustering=True)["arms"]["proposed"]["clusters"] == [
        ["0", "1"]
    ]


def test_unknown_title_bridge_cannot_join_conflicting_identity_clusters():
    c, r = inputs(3)
    for article in c["articles"]:
        article["title"] = "公司宣布新投资"
    r["items"][1]["identity"] = {}
    r["items"][2]["identity"] = deepcopy(r["items"][0]["identity"])
    r["items"][2]["identity"]["object"] = "distinct investment target"
    r["corpus_sha256"] = corpus_hash(c)
    arm = run_comparison(c, r)["arms"]["proposed"]
    assert len(arm["clusters"]) == 2
    assert not any({"0", "2"} <= set(cluster) for cluster in arm["clusters"])


def test_same_url_duplicate_is_retained_despite_identity_annotation_conflict():
    c, r = inputs(2)
    c["articles"][1]["url"] = c["articles"][0]["url"]
    r["corpus_sha256"] = corpus_hash(c)
    assert run_comparison(c, r)["arms"]["proposed"]["clusters"] == [["0", "1"]]


def test_later_capture_requires_real_replay_clock_without_bypassing_old_gate():
    c, r = inputs(1)
    c["articles"][0]["retrieved_at"] = "2026-10-10T08:04:00+00:00"
    r["corpus_sha256"] = corpus_hash(c)
    with pytest.raises(ValueError, match="capture is in the future"):
        run_comparison(c, r)
    result = run_comparison(c, r, evaluation_utc="2026-10-10T08:19:00+00:00")
    assert result["collection_reference_utc"] == c["reference_utc"]
    assert result["evaluation_utc"] == "2026-10-10T08:19:00+00:00"
    assert result["collection_to_evaluation_seconds"] == 19 * 60
    assert result["corpus_sha256"] == r["corpus_sha256"]
    assert result["arms"]["reviewed_value"]["selected"]


def test_elapsed_publication_window_is_excluded_from_all_three_pools():
    c, r = inputs(2)
    c["articles"][1]["published_at"] = "2026-10-08T20:01:00+00:00"
    r["corpus_sha256"] = corpus_hash(c)
    result = run_comparison(c, r, evaluation_utc="2026-10-10T08:19:00+00:00")
    assert result["pool_ids"] == ["0"]
    for arm in result["arms"].values():
        assert arm["excluded"]["1"] == "comparison_window_elapsed"
        assert all("1" not in event["member_ids"] for event in arm["selected"])
        represented = {i for event in arm["selected"] for i in event["member_ids"]}
        assert represented | arm["excluded"].keys() == {"0", "1"}


def test_review_adapter_preserves_per_article_domain_in_mixed_source(monkeypatch):
    from experiments.multisource_validation import engine

    c, r = inputs(3)
    for article in c["articles"]:
        article.update(source_id="mixed", source="Mixed publisher", publisher_group="mixed")
    r["items"][0]["domain"] = "ai_technology"
    r["items"][1]["domain"] = "business_macro"
    r["items"][2]["domain"] = "culture_entertainment"
    r["corpus_sha256"] = corpus_hash(c)
    observed = {}
    original = engine.select_reviewed

    def inspect(audit, review, *args, **kwargs):
        observed.update({i["candidate_id"]: i for i in review["items"]})
        return original(audit, review, *args, **kwargs)

    monkeypatch.setattr(engine, "select_reviewed", inspect)
    engine.run_comparison(c, r)
    assert [
        (observed[f"mixed:{i}"]["coverage_domain"], observed[f"mixed:{i}"]["topic"])
        for i in range(3)
    ] == [("ai_technology", "tech"), ("business_macro", "business"), ("exploration", "exploration")]
