"""Frozen, staged offline comparison. New mappings are audits, not blind gold."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.multisource_validation.engine import corpus_hash
from experiments.multisource_validation.engine_v2 import run_v2
from experiments.multisource_validation.engine_v3 import run_v3
from experiments.multisource_validation.evaluate import dump, read, validate_reference

OLD = Path("work/multisource-validation-2026-10-10")
V2 = Path("work/multisource-validation-v2-2026-10-10")
NEW = Path("work/multisource-validation-v3-2026-10-11")

# Explicitly match developments, never count every claim in a member article.
# This is a post-design audit of the old reference, not new independent gold.
MAPPING = {
    "indref-diesel-announcement": ["efv2-044-primary", "efv2-045-primary",
                                  "efv2-047-primary", "efv2-052-secondary-1"],
    "indref-malaysia-budget": ["efv2-024-primary", "efv2-026-primary", "efv2-027-primary"],
    "indref-eu-china-hybrid-agreement": ["efv2-019-primary", "efv2-053-primary"],
    "indref-delhi-transport-internet-curbs": ["efv2-023-primary"],
    "indref-delhi-founder-detention": ["efv2-022-primary", "efv2-072-secondary-1"],
    "indref-anthropic-police-tip-disclosure": ["efv2-033-primary", "efv2-073-primary"],
    # Only the company-response component; the letter's date remains unproven.
    "indref-openai-public-letter-response": ["efv2-035-primary", "efv2-048-secondary-1"],
    # The approval is NOT covered merely by showing a related resignation.
    "indref-guinea-bissau-trial-approval": ["efv2-049-primary"],
    "indref-china-fiscal-report": ["efv2-064-primary"],
    "indref-choice-hearing-rct": ["efv2-017-primary"],
    "indref-rna-thermal-formulation": ["efv2-012-primary"],
    "indref-pillay-peace-prize": ["efv2-020-primary", "efv2-050-primary"],
    "indref-panama-earthquake": ["efv2-051-primary", "efv2-075-primary"],
    "indref-renoir-recovery": ["efv2-074-primary"],
    "indref-carson-literature-prize": ["efv2-061-primary", "efv2-062-primary"],
}


def children(arm):
    return [child for event in arm["selected"] for child in event.get("developments", [event])]


def fractional_distribution(selected, articles, field):
    counts = Counter()
    for event in selected:
        groups = {articles[i][field] for i in event["member_ids"]}
        for group in groups:
            counts[group] += 1 / len(groups)
    total = sum(counts.values())
    return {"distribution": dict(sorted(counts.items())),
            "max_share": max((v / total for v in counts.values()), default=0)}


def measure(arm, reference, articles, facts, sources, judgments, *, primary_only=False):
    selected = arm["selected"]
    shown = children(arm)
    shown_facts = {fid for child in shown for fid in child["fact_ids"]}
    packet_slots = Counter()
    for event in selected:
        for pid in {judgments[fid]["packet_id"] for fid in event["fact_ids"]}:
            packet_slots[pid] += 1
    if "fact_clusters" in arm:
        clusters = [{sources[fid] for fid in group if not primary_only or facts[fid]["primary"]}
                    for group in arm["fact_clusters"]]
    else:
        clusters = [set(group) for group in arm["clusters"]]
    pairs = []
    for pair in reference["pairs"]:
        together = any({pair["left_id"], pair["right_id"]} <= group for group in clusters)
        pairs.append({"left_id": pair["left_id"], "right_id": pair["right_id"],
                      "expected_same_fact": pair["same_event"], "shared_fact": together})
    coverage = []
    for ref in reference["important_events"]:
        matching = sorted(set(MAPPING[ref["event_id"]]) & shown_facts)
        coverage.append({"reference_id": ref["event_id"], "eligible": ref["eligible_in_window"],
                         "matching_fact_ids": matching, "covered": bool(matching)})
    eligible = [x for x in coverage if x["eligible"] == "yes"]
    members = {i for e in selected for i in e["member_ids"]}
    language = {
        "events_with_zh_member": sum(any(articles[i]["language"] == "zh" for i in e["member_ids"])
                                     for e in selected),
        "zh_member_ids": sorted(i for i in members if articles[i]["language"] == "zh"),
        "zh_representative_links": sum(any(articles[i]["url"] == e["url"]
                                           and articles[i]["language"] == "zh"
                                           for i in e["member_ids"]) for e in selected),
    }
    return {
        "reading_slots": len(selected), "distinct_packets": len(packet_slots),
        "duplicate_reading_slots": sum(n - 1 for n in packet_slots.values()),
        "duplicate_packets": {p: n for p, n in packet_slots.items() if n > 1},
        "shown_developments": len(shown), "shown_fact_ids": sorted(shown_facts),
        "shown_delta_keys": sorted({judgments[fid]["delta_key"] for fid in shown_facts}),
        "confidence_distribution": dict(Counter(c["confidence"]["level"] for c in shown)),
        "selected_facts_without_support_quote": sorted(shown_facts - {
            q["fact_id"] for c in shown for q in c["confidence"]["support_quotes"]}),
        "pair_audit": {"positive_total": sum(p["expected_same_fact"] for p in pairs),
                       "correctly_shared": sum(p["expected_same_fact"] and p["shared_fact"] for p in pairs),
                       "negative_total": sum(not p["expected_same_fact"] for p in pairs),
                       "incorrectly_shared": sum(not p["expected_same_fact"] and p["shared_fact"] for p in pairs),
                       "pairs": pairs},
        "reference_fact_audit": {"eligible_total": len(eligible),
                                 "covered": sum(x["covered"] for x in eligible), "items": coverage,
                                 "scope": "new explicit mapping; not blind recall; OpenAI response component only"},
        "language": language, "distributions": arm["metrics"],
        "hosting_publishers": fractional_distribution(selected, articles, "publisher_group"),
        "domain_reclassification": [
            {"packet": e["id"], "child_domain": c["domain"], "packet_domain": e["domain"]}
            for e in selected for c in e.get("developments", []) if e["domain"] != c["domain"]],
    }


def evaluate():
    corpus = read(OLD / "corpus/corpus.json")
    reviews = read(OLD / "results/reviews-combined.json")
    proposals = read(V2 / "semantic-facts.json")
    packet_review = read(NEW / "packet-review.json")
    supplements = read(NEW / "time-supplements.json")
    reference = read(OLD / "reference/independent.json")
    clock = read(OLD / "evaluation-clock.json")["evaluation_utc"]
    articles = {a["id"]: a for a in corpus["articles"]}
    facts = {f["fact_id"]: f for item in proposals["items"] for f in item["facts"]}
    sources = {f["fact_id"]: item["id"] for item in proposals["items"] for f in item["facts"]}
    judgments = {x["fact_id"]: x for x in packet_review["items"]}
    validate_reference(reference, articles)
    assert set(MAPPING) == {r["event_id"] for r in reference["important_events"]}
    for ref in reference["important_events"]:
        assert all(sources[fid] in ref["member_ids"] + ref.get("supporting_article_ids", [])
                   for fid in MAPPING[ref["event_id"]])
    baseline = run_v2(corpus, reviews, proposals, evaluation_utc=clock)
    if baseline != read(V2 / "results/comparison.json")["v2"]:
        raise ValueError("v2 frozen replay changed")
    arms = {"v2": baseline}
    for name, options in (
        ("packet_only", {"include_secondary": False, "incremental_value": False}),
        ("secondary_original_scores", {"incremental_value": False}),
        ("incremental", {}),
        ("incremental_supplemented", {"supplements": supplements}),
        ("incremental_without_diversity", {"disable_diversity": True}),
    ):
        arms[name] = run_v3(corpus, reviews, proposals, packet_review, evaluation_utc=clock, **options)
    measurements = {
        name: measure(arm, reference, articles, facts, sources, judgments, primary_only=name == "packet_only")
        for name, arm in arms.items()
    }
    base_ids = {i for i, a in articles.items() if a.get("sample_stratum") == "balanced_latest"}
    base_reference = {**reference, "pairs": [p for p in reference["pairs"]
                                            if {p["left_id"], p["right_id"]} <= base_ids],
                      "important_events": [r for r in reference["important_events"]
                                           if set(r["member_ids"]) & base_ids]}
    base_arms = {"v2": run_v2(corpus, reviews, proposals, evaluation_utc=clock, active_ids=base_ids)}
    for name, extra in (("incremental", None), ("incremental_supplemented", supplements)):
        base_arms[name] = run_v3(corpus, reviews, proposals, packet_review, evaluation_utc=clock,
                                 active_ids=base_ids, supplements=extra)
    before, after = arms["incremental"], arms["incremental_supplemented"]
    changed = [{"fact_id": fid, "before": before["temporal_ledger"][fid]["status"],
                "after": after["temporal_ledger"][fid]["status"]}
               for fid in before["temporal_ledger"]
               if before["temporal_ledger"][fid]["status"] != after["temporal_ledger"][fid]["status"]]
    old_deltas = set(measurements["v2"]["shown_delta_keys"])
    new_deltas = set(measurements["incremental"]["shown_delta_keys"])
    proof_audit = {
        "status_changes": changed,
        "selected_fact_ids_identical": measurements["incremental"]["shown_fact_ids"]
        == measurements["incremental_supplemented"]["shown_fact_ids"],
        "source_distribution_identical": before["metrics"] == after["metrics"],
        "selected_confidence_identical": [d["confidence"] for d in children(before)]
        == [d["confidence"] for d in children(after)],
        "used_and_skipped_evidence": after["supplement_audit"],
        "retained_v2_developments": sorted(old_deltas & new_deltas),
        "lost_v2_developments": sorted(old_deltas - new_deltas),
        "added_developments": sorted(new_deltas - old_deltas),
        "historical_limit": "retrieved Oct 11; page-claimed dates, not demonstrated historical fetch availability",
    }
    output = NEW / "results"
    dump(output / "reference-fact-mapping.json", {
        "corpus_sha256": corpus_hash(corpus), "facts_sha256": corpus_hash(proposals),
        "method": "post-design explicit fact audit; do not call blind gold or overall recall",
        "references": [{"reference_id": rid, "facts": [{"fact_id": fid, "article_id": sources[fid],
                         "claim": facts[fid]["claim"], "quote": facts[fid]["quote"]} for fid in fids]}
                       for rid, fids in MAPPING.items()],
    })
    dump(output / "comparison.json", {"v2_replay_identical": True, "arms": arms})
    dump(output / "evaluation.json", measurements)
    dump(output / "supplement-and-development-audit.json", proof_audit)
    dump(output / "baseline-only.json", {
        "scope": "52 base articles only; no diagnostic date donors; semantic judgments authored on full corpus",
        "arms": {name: {"result": arm, "measurement": measure(arm, base_reference, articles,
                                                              facts, sources, judgments)}
                 for name, arm in base_arms.items()},
    })
    paths = [OLD / "corpus/corpus.json", OLD / "results/reviews-combined.json",
             OLD / "reference/independent.json", OLD / "evaluation-clock.json",
             V2 / "semantic-facts.json", NEW / "packet-review.json", NEW / "time-supplements.json",
             *Path("experiments/multisource_validation").glob("*v3*"),
             Path("tests/test_multisource_v3.py")]
    dump(output / "manifest.json", {"evaluation_utc": clock, "budget": 12,
                                    "corpus_sha256": corpus_hash(corpus),
                                    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(paths)}})
    print(json.dumps({name: {key: m[key] for key in ("reading_slots", "distinct_packets",
                                                  "duplicate_reading_slots", "shown_developments")}
                      for name, m in measurements.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
