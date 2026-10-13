"""Mechanism regressions, independent of any real news identity or outcome."""

from copy import deepcopy

import pytest
from test_multisource_v3 import bind, fact, inputs, supplement

from experiments.multisource_validation.engine_v4 import run_v4


def test_important_event_survives_small_increment_without_changing_scores():
    c, r, f, p = inputs(1)
    r["items"][0]["importance"] = 0.8
    p["items"][0]["incremental_value"] = 0.2
    original = deepcopy((c, r, f, p))
    result = run_v4(c, r, f, p)
    assert len(result["selected"]) == 1
    event = result["selected"][0]
    assert event["event_importance"] == 0.8
    assert event["incremental_value"] == 0.2
    assert (c, r, f, p) == original


def test_novel_but_unimportant_event_does_not_pass_importance_floor():
    c, r, f, p = inputs(1)
    r["items"][0]["importance"] = 0.2
    p["items"][0]["incremental_value"] = 0.99
    result = run_v4(c, r, f, p)
    assert result["selected"] == []
    assert result["excluded_facts"]["fact_0"] == "event_importance_below_floor"


@pytest.mark.parametrize("blocker", ["old", "unknown", "quality", "unsupported"])
def test_ineligible_fact_cannot_lend_its_importance_to_packet(blocker):
    c, r, f, p = inputs()
    r["items"][0]["importance"] = 0.99
    r["items"][1]["importance"] = 0.3
    if blocker == "old":
        fact(f)["times"][0].update(start="2026-09-01T00:00:00+00:00", end="2026-09-02T00:00:00+00:00")
    elif blocker == "unknown":
        fact(f)["times"] = []
    elif blocker == "quality":
        r["items"][0]["quality"] = 0.2
    else:
        fact(f)["evidence"]["kind"] = "assertion"
    bind(c, r, f, p)
    assert run_v4(c, r, f, p)["selected"] == []


def two_events(importance, novelty):
    c, r, f, p = inputs()
    p["packets"].append({**p["packets"][0], "packet_id": "other-event",
                         "anchor": {"article_id": "1", "quote": fact(f, 1)["quote"]}})
    p["items"][1]["packet_id"] = "other-event"
    for i in range(2):
        r["items"][i]["importance"] = importance[i]
        p["items"][i]["incremental_value"] = novelty[i]
    return c, r, f, p


def test_novelty_ranks_within_importance_band_but_cannot_cross_it():
    data = two_events([0.79, 0.71], [0.1, 0.9])
    assert run_v4(*data, limit=1)["selected"][0]["id"] == "other-event"
    data = two_events([0.71, 0.69], [0.1, 0.99])
    assert run_v4(*data, limit=1)["selected"][0]["id"] == "rule-episode"


def test_packet_admission_and_child_order_use_different_dimensions():
    c, r, f, p = inputs()
    r["items"][0]["importance"] = 0.8
    r["items"][1]["importance"] = 0.3
    p["items"][0]["incremental_value"] = 0.2
    p["items"][1]["incremental_value"] = 0.9
    packet = run_v4(c, r, f, p)["selected"][0]
    assert packet["event_importance"] == 0.8
    assert packet["incremental_value"] == 0.9
    assert [d["fact_ids"] for d in packet["developments"]] == [["fact_1"], ["fact_0"]]
    assert [d["importance"] for d in packet["developments"]] == [0.3, 0.8]
    assert "confidence" not in packet
    assert all(d["confidence"]["independent_chains"] == 1 for d in packet["developments"])


def test_reprints_keep_original_family_conservative_importance():
    c, r, f, p = inputs()
    fact(f, 1)["identity"] = deepcopy(fact(f)["identity"])
    fact(f, 1)["evidence"] = deepcopy(fact(f)["evidence"])
    p["items"][1].update(delta_key="delta0", incremental_value=0.82)
    r["items"][0]["importance"] = 0.9
    r["items"][1]["importance"] = 0.7
    bind(c, r, f, p)
    packet = run_v4(c, r, f, p)["selected"][0]
    assert packet["event_importance"] == 0.7
    assert len(packet["developments"]) == 1
    assert packet["developments"][0]["confidence"]["independent_chains"] == 1


def test_external_time_proof_does_not_increase_importance_novelty_or_confidence():
    c, r, f, p = inputs(1)
    r["items"][0]["importance"] = 0.75
    p["items"][0]["incremental_value"] = 0.1
    fact(f)["times"] = []
    bind(c, r, f, p)
    assert run_v4(c, r, f, p)["selected"] == []
    packet = run_v4(c, r, f, p, supplements=supplement(c, f))["selected"][0]
    assert packet["event_importance"] == 0.75
    assert packet["incremental_value"] == 0.1
    assert packet["developments"][0]["confidence"]["independent_chains"] == 1


def test_original_invalid_importance_cannot_be_hidden_by_qualification_adapter():
    c, r, f, p = inputs(1)
    r["items"][0]["importance"] = float("nan")
    with pytest.raises(ValueError):
        run_v4(c, r, f, p)
