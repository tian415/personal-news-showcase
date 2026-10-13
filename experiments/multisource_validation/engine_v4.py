"""Offline two-dimensional selection, reusing frozen v3 evidence admission.

Original reviewed article importance is a proxy, not a new semantic annotation.
The qualification adapter never escapes as an output score.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from statistics import median

from experiments.multisource_validation.engine import _metrics, _time, _validate
from experiments.multisource_validation.engine_v3 import run_v3


def _rank_packets(candidates, limit, disable_novelty, disable_diversity):
    selected, remaining = [], list(candidates)

    def coverage(event):
        value = 0
        for field in ("domain", "region"):
            count = sum(e[field] == event[field] for e in selected)
            value += 1 if count == 0 else -0.2 * count
        used = Counter(o for e in selected for o in e["origin_groups"])
        value += sum(1 if used[o] == 0 else -0.3 * used[o]
                     for o in event["origin_groups"]) / len(event["origin_groups"])
        return value

    while remaining and len(selected) < limit:
        majors = [e for e in remaining if e["_major"]]
        if majors:
            chosen = min(majors, key=lambda e: (-e["event_importance"], -e["_time"], e["member_ids"]))
            reason = "major_first_within_shared_budget"
        else:
            band = max(int(e["event_importance"] * 10) for e in remaining)
            pool = [e for e in remaining if int(e["event_importance"] * 10) == band]
            chosen = min(pool, key=lambda e: (
                0 if disable_novelty else -e["incremental_value"],
                0 if disable_diversity else -coverage(e),
                -e["event_importance"], -e["_time"], e["member_ids"],
            ))
            reason = "importance_band_then_" + ("coverage" if disable_novelty else "increment_then_coverage")
        chosen["reasons"] = ["fact_gates_then_event_importance_floor", reason]
        selected.append(chosen)
        remaining.remove(chosen)
    for event in candidates:
        event.pop("_major")
        event.pop("_time")
    return selected, remaining


def run_v4(corpus, reviews, proposals, packet_review, *, evaluation_utc=None, limit=12,
           supplements=None, active_ids=None, disable_novelty=False, disable_diversity=False):
    # Validate the actual scores first: a neutral adapter must not hide invalid,
    # missing, or pending review inputs. All mutations below are on deep copies.
    _, articles, original_reviews = _validate(corpus, reviews)
    qualification_reviews = deepcopy(reviews)
    for review in qualification_reviews["items"]:
        if review["status"] == "reviewed":
            review["importance"] = 1.0
    base = run_v3(corpus, qualification_reviews, proposals, packet_review,
                  evaluation_utc=evaluation_utc, limit=limit, incremental_value=False,
                  supplements=supplements, active_ids=active_ids)
    metadata = {p["packet_id"]: p for p in packet_review["packets"]}
    judgments = {j["fact_id"]: j for j in packet_review["items"]}
    grouped = defaultdict(list)
    # Discard the adapter's dummy ranking exclusions; retain all actual gates.
    excluded = {fid: reason for fid, reason in base["excluded_facts"].items()
                if reason != "lost_to_higher_value_or_same_band_coverage"}
    for child in base["eligible_developments"]:
        families, evidence = defaultdict(list), []
        for quote in child["confidence"]["support_quotes"]:
            article_id = quote["article_id"]
            score = original_reviews[article_id]["importance"]
            families[quote["evidence_family"]].append(score)
            evidence.append({"fact_id": quote["fact_id"], "article_id": article_id,
                             "body_sha256": articles[article_id]["body_sha256"],
                             "frozen_article_importance": score,
                             "evidence_family": quote["evidence_family"]})
        child["importance"] = median(min(scores) for scores in families.values())
        child["incremental_value"] = judgments[child["fact_ids"][0]]["incremental_value"]
        child["importance_evidence"] = evidence
        child["importance_basis"] = "frozen article proxy; minimum per family, median across families"
        grouped[child["packet_id"]].append(child)
    candidates = []
    for pid, children in sorted(grouped.items()):
        children.sort(key=lambda c: (-c["incremental_value"], -c["importance"], c["id"]))
        importance = max(c["importance"] for c in children)
        fids = sorted({fid for c in children for fid in c["fact_ids"]})
        if importance < 0.55:
            excluded.update(dict.fromkeys(fids, "event_importance_below_floor"))
            continue
        packet = metadata[pid]
        origins = sorted({o for c in children for o in c["origin_groups"]})
        candidates.append({
            "id": pid, "headline": packet["title"], "scope": packet["scope"],
            "domain": packet["domain"], "region": packet["region"],
            "importance": importance, "event_importance": importance,
            "incremental_value": max(c["incremental_value"] for c in children),
            "score_rule": "importance admits event; incremental value orders its developments; no sums",
            "member_ids": sorted({i for c in children for i in c["member_ids"]}),
            "fact_ids": fids, "origin_groups": origins, "sources": origins,
            "source_ids": sorted({s for c in children for s in c["source_ids"]}),
            "url": children[0]["url"], "developments": children, "reasons": [],
            "_major": any(c["importance"] >= 0.85 and c["confidence"]["independent_chains"] > 0
                          for c in children),
            "_time": max(_time(base["temporal_ledger"][fid]["range"][0]).timestamp() for fid in fids),
        })
    selected, remaining = _rank_packets(candidates, limit, disable_novelty, disable_diversity)
    for packet in remaining:
        excluded.update(dict.fromkeys(packet["fact_ids"], "lost_to_budget_ranking"))
    return {
        "selected": selected, "eligible_packets": candidates,
        "eligible_developments": base["eligible_developments"],
        "fact_clusters": base["fact_clusters"], "excluded_facts": dict(sorted(excluded.items())),
        "temporal_ledger": base["temporal_ledger"], "supplement_audit": base["supplement_audit"],
        "metrics": _metrics(selected),
        "options": {"importance_basis": "frozen original article reviews; not re-scored",
                    "increment_basis": "frozen v3 packet review; not re-scored",
                    "disable_novelty": disable_novelty, "disable_diversity": disable_diversity,
                    "supplements": supplements is not None},
    }
