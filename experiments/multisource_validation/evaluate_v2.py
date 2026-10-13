"""Replay the frozen corpus; write only new v2 experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.multisource_validation.engine import corpus_hash, run_comparison
from experiments.multisource_validation.engine_v2 import run_v2
from experiments.multisource_validation.evaluate import dump, measure, read, validate_reference

OLD = Path("work/multisource-validation-2026-10-10")
NEW = Path("work/multisource-validation-v2-2026-10-10")


def language_audit(arm, articles, reviews):
    selected = arm["selected"]
    members = {i for event in selected for i in event["member_ids"]}
    return {
        "events_with_zh_member": sum(any(articles[i]["language"] == "zh" for i in e["member_ids"])
                                     for e in selected),
        "zh_member_ids": sorted(i for i in members if articles[i]["language"] == "zh"),
        "zh_representative_count": sum(articles[min(e["member_ids"])]["language"] == "zh"
                                       for e in selected),
        "selected_regions": dict(Counter(e["region"] for e in selected)),
        "zh_decisions": [{"id": i, "title": a["title"], "url": a["url"],
                          "importance": reviews[i]["importance"],
                          "selected": i in members, "reason": arm["excluded"].get(i),
                          "time": arm.get("temporal_ledger", {}).get(i)}
                         for i, a in sorted(articles.items())
                         if a["language"] == "zh" and i in reviews
                         and i in arm.get("temporal_ledger", {})],
    }


def measure_v2(result, reference, entries, articles):
    """Every v2 arm must distinguish old review labels from current time evidence."""
    measured = measure(result, reference, entries, articles)
    # Keep the old annotation audit explicitly labelled: it cannot see v2 dates.
    measured["selection_audit_under_v1_annotations"] = measured.pop("selection_audit")
    selected_ids = {i for e in result["selected"] for i in e["member_ids"]}
    measured["v2_temporal_audit"] = {
        "selected_nonconfirmed_ids": [i for i in sorted(selected_ids)
                                      if result["temporal_ledger"][i]["status"] != "confirmed"],
        "status_distribution": dict(Counter(result["temporal_ledger"][i]["status"]
                                             for i in entries if i in result["temporal_ledger"])),
        "newly_confirmed_vs_v1": [i for i in sorted(entries)
                                  if entries[i]["time_status"] != "confirmed"
                                  and result["temporal_ledger"].get(i, {}).get("status") == "confirmed"],
        "no_longer_confirmed_vs_v1": [i for i in sorted(entries)
                                     if i in result["temporal_ledger"]
                                     and entries[i]["time_status"] == "confirmed"
                                     and result["temporal_ledger"].get(i, {}).get("status") != "confirmed"],
    }
    measured["language_audit"] = language_audit(result, articles, entries)
    return measured


def evaluate():
    corpus = read(OLD / "corpus/corpus.json")
    reviews = read(OLD / "results/reviews-combined.json")
    proposals = read(NEW / "semantic-facts.json")
    reference = read(OLD / "reference/independent.json")
    clock = read(OLD / "evaluation-clock.json")["evaluation_utc"]
    articles = {a["id"]: a for a in corpus["articles"]}
    entries = {r["id"]: r for r in reviews["items"]}
    validate_reference(reference, articles)
    old = run_comparison(corpus, reviews, limit=12, evaluation_utc=clock)
    if old != read(OLD / "results/comparison.json"):
        raise ValueError("v1 frozen replay changed; investigate before comparing")
    result = run_v2(corpus, reviews, proposals, evaluation_utc=clock)
    measured = measure_v2(result, reference, entries, articles)
    ablations = {}
    for name, options in (
        ("without_aliases", {"use_aliases": False}),
        ("without_time_transfer", {"transfer_time": False}),
        ("without_aliases_or_transfer", {"use_aliases": False, "transfer_time": False}),
        ("without_diversity", {"disable_diversity": True}),
    ):
        arm = run_v2(corpus, reviews, proposals, evaluation_utc=clock, **options)
        ablations[name] = {"result": arm, "measurement": measure_v2(arm, reference, entries, articles),
                           "language_audit": language_audit(arm, articles, entries)}
    base_ids = {a["id"] for a in corpus["articles"] if a.get("sample_stratum") == "balanced_latest"}
    base_reference = {**reference,
                      "pairs": [p for p in reference["pairs"]
                                if {p["left_id"], p["right_id"]} <= base_ids],
                      "important_events": [e for e in reference["important_events"]
                                           if set(e["member_ids"]) & base_ids]}
    base = run_v2(corpus, reviews, proposals, evaluation_utc=clock, active_ids=base_ids)
    output = NEW / "results"
    dump(output / "comparison.json", {"v1_replay_identical": True, "v2": result})
    dump(output / "evaluation.json", {"v1": read(OLD / "results/evaluation.json")["arms"],
                                      "v2": measured})
    dump(output / "ablations.json", ablations)
    dump(output / "baseline-only.json", {
        "v1": read(OLD / "results/baseline-only.json"),
        "v2": {"result": base, "measurement": measure_v2(base, base_reference, entries, articles)},
        "scope": "52 base articles; no temporal/score donors from diagnostic additions; shared alias registry",
    })
    paths = [OLD / "corpus/corpus.json", OLD / "results/reviews-combined.json",
             OLD / "reference/independent.json", OLD / "evaluation-clock.json",
             NEW / "semantic-facts.json", *Path("experiments/multisource_validation").glob("*v2*"),
             Path("tests/test_multisource_v2.py")]
    dump(output / "manifest.json", {
        "evaluation_utc": clock, "corpus_sha256": corpus_hash(corpus),
        "proposals_sha256": corpus_hash(proposals),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
    })
    print(json.dumps({"v1_replay_identical": True,
                      "merged_pairs": measured["pair_evaluation"]["correctly_merged"],
                      "incorrect_pairs": measured["pair_evaluation"]["incorrectly_merged"],
                      "important_hits": measured["important_reference"]["hit_count"],
                      "cross_merges": measured["important_reference"]["cross_reference_merges"],
                      "zh_event_count": measured["language_audit"]["events_with_zh_member"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    evaluate()
