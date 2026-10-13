"""Replay unchanged inputs, audit Chinese exclusions and dual-score tradeoffs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

from experiments.multisource_validation.engine import _time
from experiments.multisource_validation.engine_v2 import SUPPORTED, _range, run_v2
from experiments.multisource_validation.engine_v3 import run_v3
from experiments.multisource_validation.engine_v4 import run_v4
from experiments.multisource_validation.evaluate import dump, read
from experiments.multisource_validation.evaluate_v3 import OLD, V2, children, measure

V3 = Path("work/multisource-validation-v3-2026-10-11")
NEW = Path("work/multisource-validation-v4-2026-10-11")


def chinese_audit(arm, previous, articles, entries, facts, sources, judgments, clock):
    shown = {fid for child in children(arm) for fid in child["fact_ids"]}
    qualified = {fid for d in arm["eligible_developments"] for fid in d["fact_ids"]}
    importance = {}
    for d in arm["eligible_developments"]:
        pid = d["packet_id"]
        importance[pid] = max(importance.get(pid, 0), d["importance"])
    rows = []
    for fid, fact in sorted(facts.items()):
        article, review = articles[sources[fid]], entries[sources[fid]]
        if article["language"] != "zh":
            continue
        j = judgments[fid]
        failures = []
        if review["status"] != "reviewed":
            failures.append("pending_review")
        else:
            if review["quality"] < 0.5:
                failures.append("below_quality_floor")
            if review["story_shape"] != "single_event":
                failures.append("content_type_" + review["story_shape"])
            if review["conflict"]:
                failures.append("core_conflict")
        if fact["evidence"]["kind"] not in SUPPORTED:
            failures.append("insufficient_core_evidence")
        temporal = arm["temporal_ledger"].get(fid, {}).get("status", "outside_active_feed")
        own_range = _range(fact, article, _time(clock))
        own_time = own_range if isinstance(own_range, str) else "unknown"
        if isinstance(own_range, tuple):
            cutoff = _time(clock) - timedelta(hours=36)
            own_time = ("old" if own_range[1] < cutoff else "boundary_uncertain"
                        if own_range[0] < cutoff else "confirmed")
        if temporal != "confirmed":
            failures.append("time_" + temporal)
        if fid in qualified and fid not in shown:
            failures.append(arm["excluded_facts"][fid])
        elif fid not in shown:
            reason = arm["excluded_facts"].get(fid)
            if reason and reason not in failures:
                failures.append(reason)
        rows.append({
            "fact_id": fid, "article_id": sources[fid], "primary": fact["primary"],
            "title": article["title"], "claim": fact["claim"], "url": article["url"],
            "source_id": article["source_id"], "packet_id": j["packet_id"],
            "frozen_article_importance": review.get("importance"),
            "event_importance_from_qualified_facts": importance.get(j["packet_id"]),
            "incremental_value": j["incremental_value"],
            "temporal_status": temporal, "shown": fid in shown,
            "own_frozen_time_status_before_quality_gates": own_time,
            "time_status_note": "ledger unknown can mean donor rejected by quality/type, not absence of dates",
            "blockers": failures, "v3_exclusion": previous["excluded_facts"].get(fid),
        })
    return {"rows": rows, "primary_count": sum(r["primary"] for r in rows),
            "shown_primary_count": sum(r["primary"] and r["shown"] for r in rows),
            "primary_blocker_counts_nonexclusive": dict(Counter(
                reason for r in rows if r["primary"] for reason in r["blockers"])),
            "scope": "all frozen Chinese facts; simultaneous blockers retained; no novelty cutoff"}


def evaluate():
    corpus = read(OLD / "corpus/corpus.json")
    reviews = read(OLD / "results/reviews-combined.json")
    facts_doc = read(V2 / "semantic-facts.json")
    packets = read(V3 / "packet-review.json")
    supplement = read(V3 / "time-supplements.json")
    reference = read(OLD / "reference/independent.json")
    clock = read(OLD / "evaluation-clock.json")["evaluation_utc"]
    articles = {a["id"]: a for a in corpus["articles"]}
    entries = {r["id"]: r for r in reviews["items"]}
    facts = {f["fact_id"]: f for item in facts_doc["items"] for f in item["facts"]}
    sources = {f["fact_id"]: item["id"] for item in facts_doc["items"] for f in item["facts"]}
    judgments = {j["fact_id"]: j for j in packets["items"]}
    v2 = run_v2(corpus, reviews, facts_doc, evaluation_utc=clock)
    v3 = run_v3(corpus, reviews, facts_doc, packets, evaluation_utc=clock)
    if (v2 != read(V2 / "results/comparison.json")["v2"]
            or v3 != read(V3 / "results/comparison.json")["arms"]["incremental"]):
        raise ValueError("previous replay changed")
    arms = {"v2": v2, "v3": v3}
    for name, options in (
        ("v4", {}), ("v4_supplemented", {"supplements": supplement}),
        ("v4_without_novelty_ranking", {"disable_novelty": True}),
        ("v4_without_coverage_ranking", {"disable_diversity": True}),
    ):
        arms[name] = run_v4(corpus, reviews, facts_doc, packets, evaluation_utc=clock, **options)
    measured = {name: measure(arm, reference, articles, facts, sources, judgments)
                for name, arm in arms.items()}
    base_ids = {i for i, a in articles.items() if a.get("sample_stratum") == "balanced_latest"}
    base_ref = {**reference, "pairs": [p for p in reference["pairs"]
                                      if {p["left_id"], p["right_id"]} <= base_ids],
                "important_events": [r for r in reference["important_events"]
                                     if set(r["member_ids"]) & base_ids]}
    base_arms = {}
    for name, engine in (("v3", run_v3), ("v4", run_v4)):
        arm = engine(corpus, reviews, facts_doc, packets, evaluation_utc=clock, active_ids=base_ids)
        base_arms[name] = {"result": arm, "measurement": measure(arm, base_ref, articles,
                                                                 facts, sources, judgments)}
    before, after = arms["v4"], arms["v4_supplemented"]
    audit = {
        "selected_facts_unchanged_by_supplements": measured["v4"]["shown_fact_ids"]
        == measured["v4_supplemented"]["shown_fact_ids"],
        "confidence_unchanged_by_supplements": [d["confidence"] for d in children(before)]
        == [d["confidence"] for d in children(after)],
        "time_status_changes": [{"fact_id": fid, "before": row["status"],
                                 "after": after["temporal_ledger"][fid]["status"]}
                                for fid, row in before["temporal_ledger"].items()
                                if row["status"] != after["temporal_ledger"][fid]["status"]],
    }
    for name in ("v2", "v3", "v4_without_novelty_ranking"):
        old = set(measured[name]["shown_delta_keys"])
        new = set(measured["v4"]["shown_delta_keys"])
        audit[name + "_to_v4"] = {"retained": sorted(old & new), "added": sorted(new - old),
                                    "lost": sorted(old - new)}
    output = NEW / "results"
    dump(output / "comparison.json", {"v2_v3_replay_identical": True, "arms": arms})
    dump(output / "evaluation.json", measured)
    dump(output / "changes.json", audit)
    dump(output / "baseline-only.json", base_arms)
    dump(output / "chinese-audit.json", {
        name: chinese_audit(arms[name], v3, articles, entries, facts, sources, judgments, clock)
        for name in ("v4", "v4_supplemented")})
    paths = [OLD / "corpus/corpus.json", OLD / "results/reviews-combined.json",
             OLD / "reference/independent.json", OLD / "evaluation-clock.json",
             V2 / "semantic-facts.json", V3 / "packet-review.json", V3 / "time-supplements.json",
             *Path("experiments/multisource_validation").glob("*v4*"), Path("tests/test_multisource_v4.py")]
    dump(output / "manifest.json", {"evaluation_utc": clock, "limit": 12,
                                    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(paths)}})
    print(json.dumps({name: {"events": m["distinct_packets"], "duplicates": m["duplicate_reading_slots"],
                             "developments": m["shown_developments"],
                             "important": m["reference_fact_audit"]["covered"],
                             "zh_events": m["language"]["events_with_zh_member"]}
                      for name, m in measured.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
