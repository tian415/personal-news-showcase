"""V3 packet boundaries and evidence independence, using synthetic fixtures."""

from copy import deepcopy

import pytest
from test_multisource_v2 import fact, fixture

from experiments.multisource_validation.engine import corpus_hash
from experiments.multisource_validation.engine_v3 import run_v3


def inputs(count=2):
    corpus, reviews, facts = fixture(count)
    packet = {
        "schema": "event-packets-v3", "reviewer": "synthetic packet review",
        "corpus_sha256": corpus_hash(corpus), "facts_sha256": corpus_hash(facts),
        "value_basis": "within-corpus incremental information; no delivery history",
        "packets": [{"packet_id": "rule-episode", "title": "规则及其进展",
                     "scope": "One specific rule, not all agency news",
                     "domain": "science_health", "region": "global",
                     "anchor": {"article_id": "0", "quote": "the agency announced a rule."}}],
        "items": [],
    }
    for index in range(count):
        f = fact(facts, index)
        f["identity"]["object"] = "part" + str(index)
        packet["items"].append({
            "fact_id": f["fact_id"], "article_id": str(index), "packet_id": "rule-episode",
            "delta_key": "delta" + str(index), "incremental_value": [0.82, 0.64, 0.71][index % 3],
            "delta_kind": "decision", "value_reason": "substantive new rule",
            "relation_reason": "different provision of the same rule",
            "quote": f["quote"],
        })
    packet["facts_sha256"] = corpus_hash(facts)
    return corpus, reviews, facts, packet


def bind(c, r, f, p):
    r["corpus_sha256"] = f["corpus_sha256"] = p["corpus_sha256"] = corpus_hash(c)
    p["facts_sha256"] = corpus_hash(f)


def test_one_event_one_slot_keeps_two_distinct_evidence_records():
    c, r, f, p = inputs()
    result = run_v3(c, r, f, p)
    assert len(result["selected"]) == 1
    packet = result["selected"][0]
    assert len(packet["developments"]) == 2
    assert "confidence" not in packet
    assert packet["importance"] == 0.82
    assert [d["importance"] for d in packet["developments"]] == [0.82, 0.64]
    assert all(d["confidence"]["independent_chains"] == 1 for d in packet["developments"])


def test_packet_does_not_transfer_date_between_different_facts():
    c, r, f, p = inputs()
    fact(f, 1)["times"] = []
    bind(c, r, f, p)
    result = run_v3(c, r, f, p)
    assert result["temporal_ledger"]["fact_1"]["status"] == "unknown"
    assert result["selected"][0]["fact_ids"] == ["fact_0"]


def test_secondary_material_progress_can_qualify_without_dating_old_primary():
    c, r, f, p = inputs(1)
    secondary = deepcopy(fact(f))
    secondary.update(fact_id="resignation", primary=False)
    secondary["identity"].update(action="resign", stage="response")
    fact(f)["times"][0].update(start="2026-09-01T00:00:00+00:00",
                              end="2026-09-01T23:59:59+00:00")
    f["items"][0]["facts"].append(secondary)
    p["items"].append({**p["items"][0], "fact_id": "resignation", "delta_key": "resignation",
                       "incremental_value": 0.79, "delta_kind": "response"})
    bind(c, r, f, p)
    result = run_v3(c, r, f, p)
    assert result["selected"][0]["fact_ids"] == ["resignation"]
    assert result["temporal_ledger"]["fact_0"]["status"] == "old"
    assert run_v3(c, r, f, p, include_secondary=False)["selected"] == []


def test_incremental_score_can_demote_repetitive_high_scoring_article():
    c, r, f, p = inputs()
    r["items"][1]["importance"] = 0.99
    p["items"][1]["incremental_value"] = 0.2
    result = run_v3(c, r, f, p)
    assert result["selected"][0]["fact_ids"] == ["fact_0"]
    assert result["excluded_facts"]["fact_1"] == "low_news_value"
    old = run_v3(c, r, f, p, incremental_value=False)
    assert old["selected"][0]["developments"][0]["fact_ids"] == ["fact_1"]


def supplement(c, f, target="fact_0", basis="event", start="2026-10-10T01:00:00+00:00",
               end="2026-10-10T02:00:00+00:00"):
    import hashlib

    text = "The rule was announced between 1 and 2 on October 10."
    return {"schema": "time-supplements-v3", "reviewer": "synthetic supplement",
            "corpus_sha256": corpus_hash(c), "facts_sha256": corpus_hash(f),
            "investigations": [], "evidence": [{
                "evidence_id": "date-proof", "target_fact_id": target,
                "url": "https://example.org/official-time", "title": "Official time",
                "retrieved_at": "2026-10-11T00:00:00+00:00",
                "source_published_at": "2026-10-10T03:00:00+00:00",
                "source_date_precision": "instant", "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "identity_quote": "The rule", "identity_reason": "same rule instance",
                "quote": "between 1 and 2 on October 10", "start": start, "end": end,
                "time_basis": basis, "source_role": "primary",
                "as_of_status": "source_claimed_before", "status": "supported",
                "reason": "bounded source date only; no new fact confidence"}]}


def test_supplement_changes_only_time_not_claim_confidence_or_source_weight():
    c, r, f, p = inputs(1)
    fact(f)["times"] = []
    bind(c, r, f, p)
    s = supplement(c, f)
    original = deepcopy((c, r, f, p, s))
    assert run_v3(c, r, f, p)["selected"] == []
    result = run_v3(c, r, f, p, supplements=s)
    dev = result["selected"][0]["developments"][0]
    assert dev["confidence"]["independent_chains"] == 1
    assert dev["confidence"]["evidence_families"] == 1
    assert result["selected"][0]["origin_groups"] == ["g0"]
    assert result["temporal_ledger"]["fact_0"]["citations"][0]["evidence_id"] == "date-proof"
    assert (c, r, f, p, s) == original


@pytest.mark.parametrize("bad", ["basis", "after", "unknown_as_of", "invalid_hash", "invalid_quote"])
def test_incompatible_or_unbound_external_date_not_used(bad):
    c, r, f, p = inputs(1)
    fact(f)["times"] = []
    bind(c, r, f, p)
    s = supplement(c, f)
    ev = s["evidence"][0]
    if bad == "basis":
        ev["time_basis"] = "first_disclosure"
    elif bad == "after":
        ev["source_published_at"] = "2026-10-11T00:00:00+00:00"
    elif bad == "unknown_as_of":
        ev["as_of_status"] = "unknown"
    elif bad == "invalid_hash":
        ev["text_sha256"] = "not actual hash"
    else:
        ev["quote"] = "fabricated evidence"
    if bad in {"invalid_hash", "invalid_quote"}:
        with pytest.raises(ValueError):
            run_v3(c, r, f, p, supplements=s)
    else:
        result = run_v3(c, r, f, p, supplements=s)
        assert result["selected"] == []
        assert result["supplement_audit"][0]["used"] is False


def test_date_only_supplement_crossing_cutoff_cannot_rescue_candidate():
    c, r, f, p = inputs(1)
    fact(f)["times"] = []
    bind(c, r, f, p)
    s = supplement(c, f, start="2026-10-09T00:00:00+08:00", end="2026-10-10T00:00:00+08:00")
    result = run_v3(c, r, f, p, supplements=s)
    assert result["selected"] == []
    assert result["temporal_ledger"]["fact_0"]["status"] == "boundary_uncertain"


def test_borrowed_packet_evidence_does_not_create_corroboration():
    c, r, f, p = inputs(3)
    # First two nodes support the same fact through two independent chains;
    # the third is another fact in the same packet, with a single chain.
    fact(f, 1)["identity"] = deepcopy(fact(f)["identity"])
    p["items"][1].update(delta_key=p["items"][0]["delta_key"], incremental_value=0.82)
    bind(c, r, f, p)
    result = run_v3(c, r, f, p)
    levels = {tuple(d["fact_ids"]): d["confidence"]["independent_chains"]
              for d in result["selected"][0]["developments"]}
    assert levels[("fact_0", "fact_1")] == 2
    assert levels[("fact_2",)] == 1


def test_unsupported_fact_cannot_supply_headline_or_count_as_supporting_source():
    c, r, f, p = inputs()
    fact(f, 1)["identity"] = deepcopy(fact(f)["identity"])
    fact(f)["evidence"]["kind"] = "assertion"
    fact(f)["claim"] = "Unsupported extra detail must not become the headline"
    p["items"][1].update(delta_key="delta0", incremental_value=0.82)
    bind(c, r, f, p)
    result = run_v3(c, r, f, p)
    child = result["selected"][0]["developments"][0]
    assert child["fact_ids"] == ["fact_1"]
    assert child["headline"] == fact(f, 1)["claim"]
    assert child["origin_groups"] == ["g1"]
    assert result["excluded_facts"]["fact_0"] == "insufficient_core_evidence"


@pytest.mark.parametrize("bad", ["missing", "duplicate", "score", "quote", "binding"])
def test_packet_review_contract(bad):
    c, r, f, p = inputs()
    if bad == "missing":
        p["items"].pop()
    elif bad == "duplicate":
        p["items"].append(deepcopy(p["items"][0]))
    elif bad == "score":
        p["items"][0]["incremental_value"] = float("nan")
    elif bad == "quote":
        p["items"][0]["quote"] = "not in original fact"
    else:
        p["facts_sha256"] = "stale"
    with pytest.raises(ValueError):
        run_v3(c, r, f, p)


def test_order_independence():
    c, r, f, p = inputs()
    result = run_v3(c, r, f, p)
    c["articles"].reverse()
    r["items"].reverse()
    f["items"].reverse()
    p["items"].reverse()
    bind(c, r, f, p)
    assert run_v3(c, r, f, p) == result


def test_repeated_evidence_and_additional_low_value_progress_do_not_inflate_packet():
    c, r, f, p = inputs(3)
    fact(f, 1)["identity"] = deepcopy(fact(f)["identity"])
    fact(f, 1)["evidence"] = deepcopy(fact(f)["evidence"])
    p["items"][1].update(delta_key="delta0", incremental_value=0.82)
    bind(c, r, f, p)
    result = run_v3(c, r, f, p)
    packet = result["selected"][0]
    assert packet["importance"] == 0.82
    assert len(packet["developments"]) == 2
    duplicate = next(d for d in packet["developments"] if "fact_0" in d["fact_ids"])
    assert duplicate["confidence"]["independent_chains"] == 1
    assert duplicate["confidence"]["evidence_families"] == 1


def test_coverage_audit_requires_the_specific_fact_not_its_article(monkeypatch):
    from experiments.multisource_validation import evaluate_v3

    c, r, f, p = inputs(1)
    arm = run_v3(c, r, f, p)
    reference = {"pairs": [], "important_events": [{"event_id": "other-action",
                                                    "eligible_in_window": "yes"}]}
    # Another action in this very same article must not count as shown.
    monkeypatch.setattr(evaluate_v3, "MAPPING", {"other-action": ["unshown_fact"]})
    measured = evaluate_v3.measure(arm, reference, {a["id"]: a for a in c["articles"]},
                                  {"fact_0": fact(f)}, {"fact_0": "0"},
                                  {"fact_0": p["items"][0]})
    assert measured["reference_fact_audit"]["covered"] == 0
