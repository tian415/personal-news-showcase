"""Offline event packets with fact-local confidence and separately bound dates.

No semantic extraction is claimed: packet boundaries and incremental judgments
are attributed, frozen review inputs. No network, model, or publication calls.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta

from experiments.multisource_validation.engine import (
    DOMAINS,
    _metrics,
    _proposed,
    _time,
    _validate,
    corpus_hash,
)
from experiments.multisource_validation.engine_v2 import (
    SUPPORTED,
    _eligible,
    _groups,
    _intersect,
    _rank,
    _signature,
    validate_facts,
)


def validate_packets(corpus, proposals, review, facts, source_of, articles):
    if review.get("schema") != "event-packets-v3" or not review.get("reviewer"):
        raise ValueError("attributed event packet review required")
    if review.get("corpus_sha256") != corpus_hash(corpus):
        raise ValueError("packet corpus binding mismatch")
    if review.get("facts_sha256") != corpus_hash(proposals):
        raise ValueError("packet facts binding mismatch")
    packets = {p["packet_id"]: p for p in review["packets"]}
    entries = {x["fact_id"]: x for x in review["items"]}
    if len(packets) != len(review["packets"]) or len(entries) != len(review["items"]):
        raise ValueError("duplicate packet or fact review")
    if set(entries) != set(facts):
        raise ValueError("packet review must cover all facts exactly once")
    deltas, occupied = {}, set()
    for fid, item in entries.items():
        if item["packet_id"] not in packets or item["article_id"] != source_of[fid]:
            raise ValueError("unbound packet member")
        value = item["incremental_value"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("explicit bounded incremental value required")
        if item["delta_kind"] not in {"decision", "implementation", "consequence",
                                      "disclosure", "response", "context"}:
            raise ValueError("invalid delta kind")
        if not item.get("value_reason") or not item.get("relation_reason"):
            raise ValueError("value and relationship explanations required")
        if not item.get("quote") or item["quote"] not in facts[fid]["quote"]:
            raise ValueError("packet quotation absent from bound fact")
        key = item["delta_key"]
        state = (item["packet_id"], value)
        if not key or (key in deltas and deltas[key] != state):
            raise ValueError("one delta must have one packet and one value judgment")
        deltas[key] = state
        occupied.add(item["packet_id"])
    if occupied != set(packets):
        raise ValueError("empty packet")
    for key, packet in packets.items():
        if packet["domain"] not in DOMAINS or not all(packet.get(k) for k in ("title", "scope", "region")):
            raise ValueError("packet metadata required")
        anchor = packet["anchor"]
        members = {source_of[fid] for fid, x in entries.items() if x["packet_id"] == key}
        if anchor["article_id"] not in members or not anchor.get("quote"):
            raise ValueError("packet anchor must belong to a member body")
        if anchor["quote"] not in articles[anchor["article_id"]]["body"]:
            raise ValueError("packet anchor quotation mismatch")
    return packets, entries


def validate_supplements(corpus, proposals, supplements, facts, now):
    if supplements is None:
        return {}, []
    if supplements.get("schema") != "time-supplements-v3" or not supplements.get("reviewer"):
        raise ValueError("attributed time supplements required")
    if (supplements.get("corpus_sha256") != corpus_hash(corpus)
            or supplements.get("facts_sha256") != corpus_hash(proposals)):
        raise ValueError("supplement input binding mismatch")
    usable, audit, seen = defaultdict(list), [], set()
    for ev in sorted(supplements["evidence"], key=lambda e: e["evidence_id"]):
        eid, fid = ev["evidence_id"], ev["target_fact_id"]
        if eid in seen or fid not in facts:
            raise ValueError("duplicate or unbound supplement")
        seen.add(eid)
        if hashlib.sha256(ev["text"].encode()).hexdigest() != ev["text_sha256"]:
            raise ValueError("external evidence hash mismatch")
        for field in ("quote", "identity_quote"):
            if not ev.get(field) or ev[field] not in ev["text"]:
                raise ValueError("external evidence quotation mismatch")
        if not ev.get("identity_reason") or not ev.get("reason") or not ev.get("url"):
            raise ValueError("external provenance and identity rationale required")
        _time(ev["retrieved_at"])
        reason = None
        if ev["status"] != "supported":
            reason = "not_supported"
        elif ev["time_basis"] != facts[fid]["time_basis"]:
            reason = "time_basis_mismatch"
        elif ev["as_of_status"] != "source_claimed_before":
            reason = "historical_publication_unestablished"
        elif ev.get("source_published_at") and _time(ev["source_published_at"]) > now:
            reason = "published_after_evaluation"
        elif not ev.get("start") or not ev.get("end"):
            reason = "date_unresolved"
        elif _time(ev["start"]) > _time(ev["end"]):
            raise ValueError("invalid external interval")
        if reason is None:
            usable[fid].append({"start": ev["start"], "end": ev["end"],
                                "basis": ev["time_basis"], "quote": ev["quote"],
                                "evidence_id": eid, "url": ev["url"],
                                "text_sha256": ev["text_sha256"], "kind": "external_date",
                                "retrieved_at": ev["retrieved_at"],
                                "availability": "source_claimed_before; retrieved retrospectively"})
        audit.append({"evidence_id": eid, "target_fact_id": fid, "used": reason is None,
                      "reason": reason or "date_only; no new candidate or confidence chain",
                      "url": ev["url"]})
    return usable, audit


def _resolved_range(citations, article, now):
    if not citations:
        return None
    lo = max(_time(t["start"]) for t in citations)
    hi = min(_time(t["end"]) for t in citations)
    if lo > hi:
        return "conflict"
    cap = min(_time(article["published_at"]), now)
    if lo > cap:
        return "future"
    return lo, min(hi, cap)


def _temporal_ledger(groups, ambiguous, facts, sources, articles, reviews, citations, ranges, now):
    ledger = {}
    for group in groups:
        donors = [fid for fid in group if isinstance(ranges[fid], tuple)
                  and _eligible(articles[sources[fid]], reviews.get(sources[fid]), facts[fid])]
        interval = _intersect([ranges[fid] for fid in donors]) if donors else None
        for fid in group:
            if fid in ambiguous:
                status = "ambiguous_occurrence"
            elif isinstance(ranges[fid], str):
                status = ranges[fid]
            elif not interval:
                status = "unknown"
            elif interval[0] > interval[1]:
                status = "conflict"
            elif interval[1] < now - timedelta(hours=36):
                status = "old"
            elif interval[0] < now - timedelta(hours=36):
                status = "boundary_uncertain"
            else:
                status = "confirmed"
            ledger[fid] = {"status": status, "time_basis": facts[fid]["time_basis"],
                           "range": [v.isoformat() for v in interval] if interval else None,
                           "citations": [citation for donor in donors for citation in citations[donor]],
                           "borrowed_from_facts": [d for d in donors if d != fid]}
    return ledger


def run_v3(corpus, reviews, proposals, packet_review, *, evaluation_utc=None, limit=12,
           include_secondary=True, incremental_value=True, supplements=None,
           disable_diversity=False, active_ids=None):
    collection_now, articles, entries = _validate(corpus, reviews)
    items, aliases = validate_facts(corpus, proposals)
    facts = {f["fact_id"]: f for x in items.values() for f in x["facts"]}
    sources = {f["fact_id"]: i for i, x in items.items() for f in x["facts"]}
    packets, judgments = validate_packets(corpus, proposals, packet_review, facts, sources, articles)
    now = _time(evaluation_utc) if evaluation_utc else collection_now
    if now < collection_now or type(limit) is not int or not 1 <= limit <= 15:
        raise ValueError("invalid evaluation clock or budget")
    external, supplement_audit = validate_supplements(corpus, proposals, supplements, facts, now)
    active = {fid: f for fid, f in facts.items()
              if (active_ids is None or sources[fid] in active_ids)
              and now - timedelta(hours=36) <= _time(articles[sources[fid]]["published_at"]) <= now}
    node_articles = {fid: {**articles[sources[fid]], "id": fid} for fid in active}
    signatures = {fid: _signature(f, aliases) for fid, f in active.items()}
    citations = {
        fid: [{**t, "kind": "frozen_body", "article_id": sources[fid], "fact_id": fid,
               "body_sha256": articles[sources[fid]]["body_sha256"],
               "url": articles[sources[fid]]["url"]} for t in f["times"]]
        + [{**t, "fact_id": fid} for t in external.get(fid, [])]
        for fid, f in active.items()
    }
    ranges = {fid: _resolved_range(citations[fid], node_articles[fid], now) for fid in active}
    groups, ambiguous = _groups(node_articles, active, ranges, signatures)
    ledger = _temporal_ledger(groups, ambiguous, active, sources, articles, entries, citations, ranges, now)
    effective, excluded, developments = {}, {}, []
    for group in groups:
        assignments = {(judgments[fid]["packet_id"], judgments[fid]["delta_key"]) for fid in group}
        if len(assignments) != 1:
            raise ValueError(f"same fact assigned to different packets/deltas: {group}")
        packet_id, delta_key = next(iter(assignments))
        times = [ledger[fid]["range"][0] for fid in group if ledger[fid]["status"] == "confirmed"]
        anchor = max(times, key=_time) if times else None
        token = corpus_hash([signatures[group[0]], group])
        for fid in group:
            source = sources[fid]
            if source not in entries:
                continue
            f, j = active[fid], judgments[fid]
            r = deepcopy(entries[source])
            if not include_secondary and not f["primary"]:
                excluded[fid] = "secondary_disabled"
                r["status"] = "pending"
            elif f["evidence"]["kind"] not in SUPPORTED:
                # _proposed filters unsupported claims from its chain count but
                # retains admitted members for its headline/source record. Stop
                # such a fact here so another member cannot lend it credibility.
                excluded[fid] = "insufficient_core_evidence"
                r["status"] = "pending"
            status = ledger[fid]["status"]
            r.update(importance=j["incremental_value"] if incremental_value else r.get("importance"),
                     headline=f["claim"], time_status="confirmed" if status == "confirmed" else
                     "old" if status == "old" else "unknown",
                     event_at=anchor if status == "confirmed" else None,
                     identity=dict.fromkeys(("actor", "action", "object", "place", "development"), token),
                     origin_group=f["evidence"].get("origin_group"),
                     claims=[{"text": f["claim"], "quote": f["quote"], "supports_core": True,
                              "kind": f["evidence"]["kind"], "origin_key": f["evidence"].get("origin_key"),
                              "fact_id": fid, "article_id": source,
                              "attribution_reason": f["evidence"]["reason"]}])
            effective[fid] = r
        # v1 gates and evidence-family aggregation operate inside an already
        # resolved atomic fact. The packet never enters this confidence calculation.
        arm = _proposed({fid: node_articles[fid] for fid in group}, effective, now, 15, True, False)
        for fid, reason in arm["excluded"].items():
            if fid not in excluded:
                excluded[fid] = "time_" + ledger[fid]["status"] if reason in {
                    "event_time_unknown", "old_event"} else reason
        for event in arm["selected"]:
            fids = event["member_ids"]
            event.update(id="development_" + corpus_hash([signatures[fids[0]],
                                                         ledger[fids[0]]["range"]])[:16],
                         fact_ids=fids, member_ids=sorted({sources[fid] for fid in fids}),
                         packet_id=packet_id, delta_key=delta_key,
                         value_judgments=[judgments[fid] for fid in fids],
                         temporal_evidence={fid: ledger[fid] for fid in fids})
            event["confidence"]["limitations"].append(
                "confidence applies only to this development; external date proof adds no fact chain")
            developments.append(event)
    grouped = defaultdict(list)
    for event in developments:
        grouped[event["packet_id"]].append(event)
    candidates = []
    for pid, children in sorted(grouped.items()):
        children.sort(key=lambda e: (-e["importance"], e["id"]))
        metadata = packets[pid]
        origins = sorted({o for child in children for o in child["origin_groups"]})
        fids = sorted({fid for child in children for fid in child["fact_ids"]})
        candidates.append({
            "id": pid, "headline": metadata["title"], "scope": metadata["scope"],
            "domain": metadata["domain"], "region": metadata["region"],
            "importance": max(c["importance"] for c in children),
            "score_rule": "maximum qualified development; no sum or volume bonus",
            "member_ids": sorted({i for child in children for i in child["member_ids"]}),
            "fact_ids": fids, "origin_groups": origins, "sources": origins,
            "source_ids": sorted({s for child in children for s in child["source_ids"]}),
            "url": children[0]["url"], "developments": children, "reasons": [],
            "_major": any(c["importance"] >= 0.85 and c["confidence"]["independent_chains"] > 0
                          for c in children),
            "_time": max(_time(ledger[fid]["range"][0]).timestamp() for fid in fids),
        })
    selected, remaining = _rank(candidates, limit, disable_diversity)
    for packet in remaining:
        excluded.update({fid: "lost_to_higher_value_or_same_band_coverage" for fid in packet["fact_ids"]})
    return {
        "selected": selected, "eligible_packets": candidates,
        "eligible_developments": developments, "fact_clusters": groups,
        "excluded_facts": dict(sorted(excluded.items())), "temporal_ledger": ledger,
        "supplement_audit": supplement_audit, "metrics": _metrics(selected),
        "options": {"include_secondary": include_secondary, "incremental_value": incremental_value,
                    "supplements": supplements is not None, "disable_diversity": disable_diversity},
    }
