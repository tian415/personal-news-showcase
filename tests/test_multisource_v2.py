"""Mechanism regressions; synthetic evidence is not an accuracy benchmark."""

from copy import deepcopy

import pytest
from test_multisource_engine import inputs

from experiments.multisource_validation.engine import corpus_hash, run_comparison
from experiments.multisource_validation.engine_v2 import run_v2


def fixture(count=2):
    corpus, reviews = inputs(count)
    facts = {
        "schema": "event-facts-v2",
        "reviewer": "synthetic fact reviewer",
        "corpus_sha256": corpus_hash(corpus),
        "aliases": [],
        "items": [],
    }
    for article in corpus["articles"]:
        identifier = article["id"]
        facts["items"].append({
            "id": identifier, "body_sha256": article["body_sha256"],
            "reason": "synthetic", "facts": [{
                "fact_id": "fact_" + identifier, "primary": True,
                "time_basis": "event",
                "identity": {"actor": "agency", "action": "announce", "object": "rule",
                             "stage": "announcement", "occurrence": "October rule"},
                "claim": "The agency announced a rule.",
                "quote": "the agency announced a rule.", "reason": "explicit action",
                "evidence": {"kind": "direct_reporting", "origin_key": "original" + identifier,
                             "origin_group": "g" + identifier, "reason": "direct synthetic quote"},
                "times": [{"start": "2026-10-10T00:00:00+00:00",
                           "end": "2026-10-10T23:59:59+00:00", "basis": "event",
                           "precision": "day", "quote": "On October 10",
                           "rationale": "Day interval, not an invented exact instant."}],
            }],
        })
    return corpus, reviews, facts


def fact(facts, index=0):
    return facts["items"][index]["facts"][0]


def test_same_event_missing_time_borrows_bound_citation_without_score_boost():
    c, r, f = fixture()
    fact(f, 1)["times"] = []
    r["items"][1].update(time_status="unknown", event_at=None)
    before = deepcopy((c, r, f))
    result = run_v2(c, r, f)
    assert result["clusters"] == [["0", "1"]]
    assert result["selected"][0]["member_ids"] == ["0", "1"]
    assert result["selected"][0]["importance"] == 0.7
    assert result["temporal_ledger"]["1"]["borrowed_from"] == ["0"]
    assert result["temporal_ledger"]["1"]["citations"][0]["quote"] == "On October 10"
    assert (c, r, f) == before
    without = run_v2(c, r, f, transfer_time=False)
    assert without["selected"][0]["member_ids"] == ["0"]


def test_aliases_grounded_in_body_enable_entity_match():
    c, r, f = fixture()
    # Two actual forms in the fixture text, mapped by the semantic proposal.
    fact(f, 1)["identity"]["actor"] = "the agency"
    f["aliases"] = [{"canonical": "agency", "variants": ["agency", "the agency"],
                     "quotes": [{"article_id": "0", "quote": "the agency announced a rule."}]}]
    assert run_v2(c, r, f)["clusters"] == [["0", "1"]]
    assert run_v2(c, r, f, use_aliases=False)["clusters"] == [["0"], ["1"]]


@pytest.mark.parametrize("field,value", [("object", "another rule"), ("stage", "implementation"),
                                        ("occurrence", "November rule"), ("action", "criticize")])
def test_same_headline_cannot_bridge_different_action_target_or_stage(field, value):
    c, r, f = fixture()
    c["articles"][1]["title"] = c["articles"][0]["title"]
    r["corpus_sha256"] = f["corpus_sha256"] = corpus_hash(c)
    fact(f, 1)["identity"][field] = value
    assert run_v2(c, r, f)["clusters"] == [["0"], ["1"]]


def test_undated_report_cannot_bridge_disjoint_occurrences():
    c, r, f = fixture(3)
    fact(f, 1)["times"][0].update(start="2026-10-09T06:00:00+00:00",
                                end="2026-10-09T07:00:00+00:00")
    fact(f, 2)["times"] = []
    result = run_v2(c, r, f)
    assert result["clusters"] == [["0"], ["1"], ["2"]]
    assert result["temporal_ledger"]["2"]["status"] == "ambiguous_occurrence"


def test_interval_crossing_cutoff_stays_pending_until_same_fact_narrows_it():
    c, r, f = fixture()
    fact(f)["times"][0].update(start="2026-10-09T00:00:00+08:00",
                              end="2026-10-09T23:59:59+08:00")
    fact(f, 1)["times"] = []
    assert run_v2(c, r, f)["selected"] == []
    fact(f, 1)["times"] = [{**fact(f)["times"][0],
                           "start": "2026-10-09T12:00:00+08:00",
                           "end": "2026-10-09T13:00:00+08:00"}]
    result = run_v2(c, r, f)
    assert len(result["selected"]) == 1
    assert result["temporal_ledger"]["0"]["status"] == "confirmed"
    assert result["temporal_ledger"]["0"]["borrowed_from"] == ["1"]


def test_published_today_does_not_supply_event_date():
    c, r, f = fixture(1)
    fact(f)["times"] = []
    result = run_v2(c, r, f)
    assert result["selected"] == []
    assert result["temporal_ledger"]["0"]["status"] == "unknown"


def test_future_implementation_does_not_date_announcement():
    c, r, f = fixture()
    fact(f)["times"] = []
    fact(f, 1)["identity"]["stage"] = "implementation"
    fact(f, 1)["times"][0].update(start="2026-11-01T00:00:00+00:00",
                                end="2026-11-01T23:59:59+00:00")
    result = run_v2(c, r, f)
    assert result["selected"] == []
    assert result["temporal_ledger"]["0"]["status"] == "unknown"
    assert result["temporal_ledger"]["1"]["status"] == "future"


@pytest.mark.parametrize("bad", ["pending", "quality", "assertion", "roundup", "conflict"])
def test_ineligible_donor_cannot_rescue_unknown_time(bad):
    c, r, f = fixture()
    fact(f)["times"] = []
    if bad == "pending":
        r["items"][1]["status"] = "pending"
    elif bad == "quality":
        r["items"][1]["quality"] = 0.1
    elif bad == "assertion":
        fact(f, 1)["evidence"]["kind"] = "assertion"
    elif bad == "roundup":
        r["items"][1]["story_shape"] = "roundup"
    else:
        r["items"][1].update(conflict=True, conflict_reason="contradictory core")
    result = run_v2(c, r, f)
    assert "0" not in [i for e in result["selected"] for i in e["member_ids"]]
    assert result["temporal_ledger"]["0"]["status"] == "unknown"


def test_secondary_fact_can_supply_time_but_never_own_output_slot():
    c, r, f = fixture()
    secondary = deepcopy(fact(f, 1))
    secondary.update(fact_id="secondary", primary=False)
    fact(f, 1)["identity"]["object"] = "other rule"
    fact(f)["times"] = []
    f["items"][1]["facts"].append(secondary)
    result = run_v2(c, r, f)
    assert result["temporal_ledger"]["0"]["borrowed_from"] == ["1"]
    assert len(result["selected"]) == 2


@pytest.mark.parametrize("invalid", ["body", "quote", "time_quote", "alias_quote", "alias_collision"])
def test_unbound_or_ambiguous_semantic_proposal_rejected(invalid):
    c, r, f = fixture()
    if invalid == "body":
        f["items"][0]["body_sha256"] = "wrong"
    elif invalid == "quote":
        fact(f)["quote"] = "fabricated quote"
    elif invalid == "time_quote":
        fact(f)["times"][0]["quote"] = "yesterday at noon"
    else:
        f["aliases"] = [{"canonical": "one", "variants": ["agency"],
                         "quotes": [{"article_id": "0", "quote": "the agency announced a rule."}]}]
        if invalid == "alias_quote":
            f["aliases"][0]["quotes"][0]["quote"] = "not in body"
        else:
            f["aliases"].append({**deepcopy(f["aliases"][0]), "canonical": "two"})
    with pytest.raises(ValueError):
        run_v2(c, r, f)


def test_order_invariance_and_no_language_quota():
    c, r, f = fixture(3)
    for index in range(3):
        fact(f, index)["identity"]["object"] = str(index)
    c["articles"][2]["language"] = "zh"
    r["items"][2]["importance"] = 0.56
    r["corpus_sha256"] = f["corpus_sha256"] = corpus_hash(c)
    result = run_v2(c, r, f, limit=2)
    assert {i for e in result["selected"] for i in e["member_ids"]} == {"0", "1"}
    c["articles"].reverse()
    r["items"].reverse()
    f["items"].reverse()
    r["corpus_sha256"] = f["corpus_sha256"] = corpus_hash(c)
    assert run_v2(c, r, f, limit=2) == result


def test_secondary_wrong_stage_never_transfers_time():
    c, r, f = fixture(1)
    secondary = deepcopy(fact(f))
    secondary.update(primary=False, fact_id="secondary")
    secondary["identity"]["stage"] = "implementation"
    fact(f)["times"] = []
    f["items"][0]["facts"].append(secondary)
    assert run_v2(c, r, f)["temporal_ledger"]["0"]["status"] == "unknown"


def test_conflicting_secondary_dates_fail_closed():
    c, r, f = fixture(3)
    fact(f)["times"] = []
    for index in (1, 2):
        secondary = deepcopy(fact(f, index))
        secondary.update(primary=False, fact_id="secondary" + str(index))
        if index == 2:
            secondary["times"][0].update(start="2026-10-09T00:00:00+00:00",
                                         end="2026-10-09T23:59:59+00:00")
        fact(f, index)["identity"]["object"] = "other" + str(index)
        f["items"][index]["facts"].append(secondary)
    result = run_v2(c, r, f)
    assert result["temporal_ledger"]["0"]["status"] == "conflict"
    assert "0" not in [i for e in result["selected"] for i in e["member_ids"]]


def test_syndicated_copies_do_not_become_independent_chains():
    c, r, f = fixture()
    fact(f, 1)["evidence"] = deepcopy(fact(f)["evidence"])
    result = run_v2(c, r, f)
    assert result["selected"][0]["confidence"]["independent_chains"] == 1
    assert result["selected"][0]["confidence"]["evidence_families"] == 1


@pytest.mark.parametrize("disable_diversity", [True, False])
def test_unchanged_ranking_for_unchanged_single_facts(disable_diversity):
    c, r, f = fixture(5)
    for index in range(5):
        fact(f, index)["identity"]["object"] = "rule" + str(index)
        r["items"][index]["importance"] = [0.7, 0.72, 0.86, 0.59, 0.68][index]
        r["items"][index]["domain"] = ["science_health", "ai_technology", "business_macro",
                                       "society_life", "culture_entertainment"][index]
    old = run_comparison(c, r, limit=3, disable_diversity=disable_diversity)["arms"]["proposed"]
    new = run_v2(c, r, f, limit=3, disable_diversity=disable_diversity)
    assert [e["member_ids"] for e in old["selected"]] == [e["member_ids"] for e in new["selected"]]


def test_base_subset_cannot_receive_time_from_diagnostic_addition():
    c, r, f = fixture()
    fact(f)["times"] = []
    result = run_v2(c, r, f, active_ids={"0"})
    assert result["clusters"] == [["0"]]
    assert result["selected"] == []


def test_occurrence_and_disclosure_dates_must_be_separate_facts():
    c, r, f = fixture(1)
    fact(f)["times"].append({**fact(f)["times"][0], "basis": "first_disclosure"})
    with pytest.raises(ValueError, match="different time bases"):
        run_v2(c, r, f)


def test_disclosure_fact_cannot_narrow_occurrence_even_with_same_identity():
    c, r, f = fixture(1)
    fact(f)["times"][0].update(start="2026-10-08T00:00:00+00:00",
                              end="2026-10-08T23:59:59+00:00")
    secondary = deepcopy(fact(f))
    secondary.update(primary=False, fact_id="disclosure", time_basis="first_disclosure")
    secondary["times"][0].update(start="2026-10-08T22:00:00+00:00",
                                end="2026-10-08T22:00:00+00:00", basis="first_disclosure")
    f["items"][0]["facts"].append(secondary)
    for transfer in (True, False):
        result = run_v2(c, r, f, transfer_time=transfer)
        assert result["temporal_ledger"]["0"]["status"] == "boundary_uncertain"
        assert result["selected"] == []


def test_undated_occurrence_cannot_borrow_disclosure_date():
    c, r, f = fixture()
    fact(f)["times"] = []
    fact(f, 1)["time_basis"] = "first_disclosure"
    fact(f, 1)["times"][0]["basis"] = "first_disclosure"
    result = run_v2(c, r, f)
    assert result["temporal_ledger"]["0"]["status"] == "unknown"
    assert result["clusters"] == [["0"], ["1"]]


@pytest.mark.parametrize("value", [None, "invalid", "first_disclosure"])
def test_missing_or_inconsistent_fact_time_basis_rejected(value):
    c, r, f = fixture(1)
    if value is None:
        del fact(f)["time_basis"]
    else:
        fact(f)["time_basis"] = value
    with pytest.raises(ValueError):
        run_v2(c, r, f)


def test_v2_measurement_never_labels_old_annotations_as_new_time_audit():
    from experiments.multisource_validation.evaluate_v2 import measure_v2

    c, r, f = fixture(1)
    r["items"][0].update(time_status="unknown", event_at=None)
    arm = run_v2(c, r, f)
    measured = measure_v2(arm, {"pairs": [], "important_events": []},
                          {x["id"]: x for x in r["items"]},
                          {x["id"]: x for x in c["articles"]})
    assert "selection_audit" not in measured
    assert measured["selection_audit_under_v1_annotations"]["without_confirmed_time"] == 1
    assert measured["v2_temporal_audit"]["selected_nonconfirmed_ids"] == []


def test_subset_audit_does_not_call_out_of_pool_items_regressions():
    from experiments.multisource_validation.evaluate_v2 import measure_v2

    c, r, f = fixture()
    arm = run_v2(c, r, f, active_ids={"0"})
    measured = measure_v2(arm, {"pairs": [], "important_events": []},
                          {x["id"]: x for x in r["items"]},
                          {x["id"]: x for x in c["articles"]})
    assert measured["v2_temporal_audit"]["no_longer_confirmed_vs_v1"] == []
