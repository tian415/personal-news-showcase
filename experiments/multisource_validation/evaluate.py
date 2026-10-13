"""Offline comparisons against a separately authored, explicitly bounded reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.multisource_validation.engine import corpus_hash, run_comparison


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def review_union(corpus, documents):
    items = [item for document in documents for item in document["items"]]
    identifiers = [item["id"] for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("overlapping reviewers must be adjudicated explicitly")
    expected = {a["id"] for a in corpus["articles"] if a["body_status"] == "ok"}
    if set(identifiers) != expected:
        raise ValueError(f"review coverage differs: {set(identifiers) ^ expected}")
    return {
        "corpus_sha256": corpus_hash(corpus),
        "reviewer": "; ".join(document["reviewer"] for document in documents),
        "items": sorted(items, key=lambda item: item["id"]),
    }


def validate_reference(reference, articles):
    """Prevent a reference from silently citing unavailable or different material."""
    pairs = set()
    for pair in reference["pairs"]:
        key = tuple(sorted((pair["left_id"], pair["right_id"])))
        if key in pairs or key[0] == key[1]:
            raise ValueError("duplicate or self reference pair")
        pairs.add(key)
        if type(pair["same_event"]) is not bool:
            raise ValueError("reference relation must be boolean")
        for side in ("left", "right"):
            quote = pair[side + "_quote"]
            if not quote.strip() or quote not in articles[pair[side + "_id"]]["body"]:
                raise ValueError("reference pair quotation mismatch")
    assigned = set()
    for event in reference["important_events"]:
        if not event["member_ids"] or not set(event["member_ids"]) <= articles.keys():
            raise ValueError("reference event members unavailable")
        if assigned & set(event["member_ids"]):
            raise ValueError("reference events overlap")
        assigned.update(event["member_ids"])
        for citation in event["date_evidence"]:
            quote = citation["quote"]
            if not quote.strip() or quote not in articles[citation["article_id"]]["body"]:
                raise ValueError("reference date quotation mismatch")


def hosting_distribution(selected, articles):
    result = {}
    for field in ("publisher_group", "language"):
        counts = Counter()
        for event in selected:
            groups = {articles[i][field] for i in event["member_ids"]}
            for group in groups:
                counts[group] += 1 / len(groups)
        total = sum(counts.values())
        result[field] = {
            "distribution": dict(sorted(counts.items())),
            "max_share": max(counts.values(), default=0) / total if total else 0,
            "hhi": sum((n / total) ** 2 for n in counts.values()) if total else 0,
        }
    return result


def measure(arm, reference, reviews, articles=None):
    clusters = [set(group) for group in arm["clusters"]]
    pair_results = []
    for pair in reference["pairs"]:
        related = any({pair["left_id"], pair["right_id"]} <= c for c in clusters)
        pair_results.append({
            **pair, "merged": related, "correct": related == pair["same_event"],
        })
    positives = [p for p in pair_results if p["same_event"]]
    negatives = [p for p in pair_results if not p["same_event"]]
    hits = []
    selected = arm["selected"]
    for event in reference["important_events"]:
        matches = [index for index, item in enumerate(selected)
                   if set(event["member_ids"]) & set(item["member_ids"])]
        hits.append({**event, "selected_slots": matches, "hit": bool(matches),
                     "member_exclusions": {i: arm["excluded"].get(i)
                                           for i in event["member_ids"]}})
    eligible = [e for e in hits if e["eligible_in_window"] == "yes"
                and e["importance"] in {"must", "high"}]
    cross_merges = [
        {"selected_slot": slot, "reference_events": [
            e.get("event_id", str(index)) for index, e in enumerate(hits)
            if slot in e["selected_slots"]]}
        for slot in range(len(selected))
        if sum(slot in e["selected_slots"] for e in hits) > 1
    ]
    # Maximum one-to-one matching: a broad predicted cluster cannot receive
    # several event hits just because it swallowed multiple reference groups.
    slot_matches = {}

    def match(index, visited):
        for slot in eligible[index]["selected_slots"]:
            if slot in visited:
                continue
            visited.add(slot)
            if slot not in slot_matches or match(slot_matches[slot], visited):
                slot_matches[slot] = index
                return True
        return False

    for index in range(len(eligible)):
        match(index, set())
    merged_slots = {m["selected_slot"] for m in cross_merges}
    audit = []
    for event in selected:
        members = [reviews[i] for i in event["member_ids"]]
        # These are independent annotation audits, not inferred confidence from
        # whether a particular algorithm emits its own confidence object.
        core = [c for r in members for c in r["claims"] if c["supports_core"]
                and c["kind"] in {"document", "direct_reporting", "attributed_report"}]
        audit.append({
            "headline": event["headline"], "member_ids": event["member_ids"],
            "has_bound_support": bool(core),
            "has_confirmed_time": any(r["time_status"] == "confirmed" for r in members),
            "has_core_conflict": any(r["conflict"] for r in members),
            "contains_single_event": any(r["story_shape"] == "single_event" for r in members),
            "confidence": event["confidence"],
        })
    return {
        "selected_count": len(selected),
        "pair_evaluation": {
            "same_event_pairs": len(positives),
            "correctly_merged": sum(p["merged"] for p in positives),
            "different_event_pairs": len(negatives),
            "incorrectly_merged": sum(p["merged"] for p in negatives),
            "details": pair_results,
        },
        "important_reference": {
            "eligible_count": len(eligible), "hit_count": len(slot_matches),
            "article_overlap_hit_count": sum(e["hit"] for e in eligible),
            "unambiguous_hit_count": sum(any(s not in merged_slots for s in e["selected_slots"])
                                         for e in eligible),
            "cross_reference_merges": cross_merges,
            "hit_method": "maximum one-to-one reference-event to output-slot matching",
            "duplicate_slots_in_reference_events": sum(
                max(0, len(e["selected_slots"]) - 1) for e in hits),
            "details": hits,
            "scope": "bounded separately authored reference; not web-wide recall",
        },
        "selection_audit": {
            "without_bound_core_support": sum(not a["has_bound_support"] for a in audit),
            "without_confirmed_time": sum(not a["has_confirmed_time"] for a in audit),
            "core_conflict": sum(a["has_core_conflict"] for a in audit),
            "non_single_event": sum(not a["contains_single_event"] for a in audit),
            "details": audit,
        },
        "independent_reference_time_audit": {
            "uncertain_event_hits": [e.get("event_id") for e in hits
                                     if e["hit"] and e["eligible_in_window"] == "uncertain"],
            "old_event_hits": [e.get("event_id") for e in hits
                               if e["hit"] and e["eligible_in_window"] == "no"],
        },
        "distribution": arm["metrics"],
        "hosting_distribution": hosting_distribution(selected, articles) if articles else None,
    }


def evaluate(root):
    corpus = read(root / "corpus/corpus.json")
    reference = read(root / "reference/independent.json")
    evaluation_utc = read(root / "evaluation-clock.json")["evaluation_utc"]
    documents = [read(root / "reviews" / name)
                 for name in ("domestic-science.json", "international.json")]
    reviews = review_union(corpus, documents)
    articles = {a["id"]: a for a in corpus["articles"]}
    validate_reference(reference, articles)
    entries = {r["id"]: r for r in reviews["items"]}
    result = run_comparison(corpus, reviews, limit=12, evaluation_utc=evaluation_utc)
    measured = {name: measure(arm, reference, entries, articles)
                for name, arm in result["arms"].items()}
    ablations = {}
    for name, options in (("title_only", {"title_only_clustering": True}),
                          ("without_diversity", {"disable_diversity": True})):
        arm = run_comparison(corpus, reviews, limit=12,
                             evaluation_utc=evaluation_utc, **options)["arms"]["proposed"]
        ablations[name] = {"result": arm, "measurement": measure(arm, reference, entries, articles)}
    base_ids = {a["id"] for a in corpus["articles"]
                if a.get("sample_stratum") == "balanced_latest"}
    base_reviews = {**reviews, "items": [r for r in reviews["items"] if r["id"] in base_ids]}
    base_reference = {**reference,
                      "pairs": [p for p in reference["pairs"]
                                if {p["left_id"], p["right_id"]} <= base_ids],
                      "important_events": [e for e in reference["important_events"]
                                           if set(e["member_ids"]) & base_ids]}
    base_result = run_comparison(corpus, base_reviews, limit=12, evaluation_utc=evaluation_utc)
    base_evaluation = {name: measure(arm, base_reference, entries, articles)
                       for name, arm in base_result["arms"].items()}
    output = root / "results"
    dump(output / "reviews-combined.json", reviews)
    dump(output / "comparison.json", result)
    dump(output / "ablations.json", ablations)
    dump(output / "baseline-only.json", {
        "result": base_result, "measurement": base_evaluation,
        "scope": "same 52 source-balanced base articles; no targeted diagnostic additions",
    })
    dump(output / "evaluation.json", {
        "corpus_sha256": corpus_hash(corpus), "reviews_sha256": corpus_hash(reviews),
        "collection_reference_utc": corpus["reference_utc"], "evaluation_utc": evaluation_utc,
        "reference_sha256": corpus_hash(reference), "arms": measured,
        "source_status": dict(Counter(s["status"] for s in corpus["sources"])),
        "limitations": [
            "same-session independent reference, not external gold",
            "reviewer-normalized semantics; not unattended extraction accuracy",
            "52 baseline plus 23 purposive samples; distribution is not natural feed traffic",
            "article-level news value scores supplied equally to all arms",
        ],
    })
    files = [*output.glob("*.json"), root / "corpus/corpus.json", root / "evaluation-clock.json",
             root / "reference/independent.json", *(root / "reviews").glob("*.json")]
    dump(output / "manifest.json", {
        "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(files) if p.name != "manifest.json"},
        "code_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(__file__).parent.glob("*.py"))},
    })
    print(json.dumps({name: {"selected": m["selected_count"],
                            "merged_pairs": m["pair_evaluation"]["correctly_merged"],
                            "important_hits": m["important_reference"]["hit_count"]}
                      for name, m in measured.items()}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("work/multisource-validation-2026-10-10"))
    evaluate(parser.parse_args().root)


if __name__ == "__main__":
    main()
