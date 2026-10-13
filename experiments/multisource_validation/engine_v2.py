"""Isolated event-fact experiment; no fetching, model calls, or production writes.

Semantic proposals are reviewed inputs, not automatic extraction accuracy. The
v1 scoring/evidence-family policy is reused, with narrower fact-bound quotations.
Only primary facts define output units; secondary facts can supply dated evidence.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta

from experiments.multisource_validation.engine import (
    _metrics,
    _proposed,
    _time,
    _validate,
    corpus_hash,
)

ROLES = ("actor", "action", "object", "stage", "occurrence")
SUPPORTED = {"document", "direct_reporting", "attributed_report"}


def normalized(value):
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", value).casefold())


def validate_facts(corpus, proposals):
    if proposals.get("corpus_sha256") != corpus_hash(corpus):
        raise ValueError("fact corpus binding mismatch")
    if proposals.get("schema") != "event-facts-v2" or not proposals.get("reviewer"):
        raise ValueError("attributed event-facts-v2 required")
    articles = {a["id"]: a for a in corpus["articles"]}
    expected = {i for i, a in articles.items() if a["body_status"] in {"ok", "extracted", "captured"}}
    items = {r["id"]: r for r in proposals["items"]}
    if set(items) != expected or len(items) != len(proposals["items"]):
        raise ValueError("fact proposals must cover each captured body exactly once")

    def bound(identifier, quote):
        if identifier not in articles or not isinstance(quote, str) or not quote.strip():
            raise ValueError("missing quotation")
        if quote not in articles[identifier]["body"]:
            raise ValueError("fact quotation absent from exact body")

    aliases = {}
    for alias in proposals["aliases"]:
        canonical = normalized(alias["canonical"])
        if not canonical or not alias["quotes"]:
            raise ValueError("alias needs canonical name and body evidence")
        for citation in alias["quotes"]:
            bound(citation["article_id"], citation["quote"])
        quoted = [normalized(c["quote"]) for c in alias["quotes"]]
        for variant in alias["variants"]:
            if not normalized(variant) or not any(normalized(variant) in q for q in quoted):
                raise ValueError(f"alias variant not grounded in quotations: {variant}")
        for variant in [alias["canonical"], *alias["variants"]]:
            key = normalized(variant)
            if key in aliases and aliases[key] != canonical:
                raise ValueError(f"ambiguous entity alias: {variant}")
            aliases[key] = canonical
    fact_ids = set()
    for identifier, item in items.items():
        if item["body_sha256"] != articles[identifier]["body_sha256"]:
            raise ValueError("fact body hash mismatch")
        if sum(f["primary"] is True for f in item["facts"]) != bool(item["facts"]):
            raise ValueError("exactly one primary fact required when facts present")
        for fact in item["facts"]:
            if not fact["fact_id"] or fact["fact_id"] in fact_ids:
                raise ValueError("duplicate fact id")
            fact_ids.add(fact["fact_id"])
            if type(fact["primary"]) is not bool:
                raise ValueError("primary must be boolean")
            if not all(isinstance(fact["identity"].get(k), str) and
                       normalized(fact["identity"][k]) for k in ROLES):
                raise ValueError("all event identity roles required")
            if not fact.get("claim") or not fact.get("reason"):
                raise ValueError("fact explanation required")
            bound(identifier, fact["quote"])
            evidence = fact["evidence"]
            if evidence["kind"] not in SUPPORTED | {"assertion", "analysis"}:
                raise ValueError("invalid fact evidence kind")
            if not evidence.get("reason"):
                raise ValueError("fact attribution explanation required")
            if evidence["kind"] in SUPPORTED and not all(
                evidence.get(k) for k in ("origin_key", "origin_group")
            ):
                raise ValueError("supported fact requires original evidence chain")
            if len({t["basis"] for t in fact["times"]}) > 1:
                raise ValueError("different time bases must be separate facts, not intersected dates")
            if fact.get("time_basis") not in {"event", "first_disclosure", "material_update"}:
                raise ValueError("explicit time basis required even for undated facts")
            if any(t["basis"] != fact["time_basis"] for t in fact["times"]):
                raise ValueError("fact time basis mismatch")
            for time in fact["times"]:
                if time["basis"] not in {"event", "first_disclosure", "material_update"}:
                    raise ValueError("invalid fact time basis")
                if time["precision"] not in {"instant", "day", "daypart"}:
                    raise ValueError("invalid time precision")
                if _time(time["start"]) > _time(time["end"]) or not time.get("rationale"):
                    raise ValueError("invalid time interval")
                bound(identifier, time["quote"])
    return items, aliases


def _signature(fact, aliases):
    values = []
    for role in ROLES:
        value = normalized(fact["identity"][role])
        values.append(aliases.get(value, value) if role in {"actor", "object"} else value)
    return (*values, fact["time_basis"])


def _range(fact, article, now):
    if not fact["times"]:
        return None
    # A published description of an already-observed fact bounds the upper end;
    # it cannot provide the missing lower bound or turn a future action into news.
    start = max(_time(t["start"]) for t in fact["times"])
    end = min(_time(t["end"]) for t in fact["times"])
    if start > end:
        return "conflict"
    cap = min(_time(article["published_at"]), now)
    if start > cap:
        return "future"
    return start, min(end, cap)


def _intersect(ranges):
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def _compatible(left, right):
    return max(left[0], right[0]) <= min(left[1], right[1])


def _eligible(article, review, fact):
    return bool(review and review.get("status") == "reviewed"
                and review["quality"] >= 0.5 and review["story_shape"] == "single_event"
                and not review["conflict"] and article["body"].strip()
                and article["body_status"] in {"ok", "extracted", "captured"}
                and fact["evidence"]["kind"] in SUPPORTED)


def _groups(articles, primaries, ranges, signatures):
    buckets, groups, ambiguous = defaultdict(list), [], set()
    for identifier in sorted(articles):
        if identifier in primaries:
            buckets[signatures[identifier]].append(identifier)
        else:
            groups.append([identifier])
    for members in buckets.values():
        known = sorted((i for i in members if isinstance(ranges[i], tuple)),
                       key=lambda i: (ranges[i], i))
        local = []
        for identifier in known:
            compatible = [g for g in local
                          if _compatible(_intersect([ranges[i] for i in g]), ranges[identifier])]
            if len(compatible) == 1:
                compatible[0].append(identifier)
            else:
                local.append([identifier])
        unknown = [i for i in members if ranges[i] is None]
        if len(local) == 1:
            local[0].extend(unknown)
        elif not local:
            if unknown:
                local.append(unknown)
        else:
            ambiguous.update(unknown)
            local.extend([i] for i in unknown)
        # Invalid/conflicting dates are never bridges for valid clusters.
        local.extend([i] for i in members if isinstance(ranges[i], str))
        groups.extend(local)
    return sorted(sorted(g) for g in groups), ambiguous


def _rank(candidates, limit, disable_diversity):
    """Same major-first/value-decile/marginal-coverage policy as v1."""
    chosen, remaining = [], list(candidates)

    def cover(record):
        result = 0
        for field in ("domain", "region"):
            count = sum(e[field] == record[field] for e in chosen)
            result += 1 if count == 0 else -0.2 * count
        used = Counter(o for e in chosen for o in e["origin_groups"])
        result += sum(1 if used[o] == 0 else -0.3 * used[o]
                      for o in record["origin_groups"]) / len(record["origin_groups"])
        return result

    while remaining and len(chosen) < min(limit, 15):
        majors = [e for e in remaining if e["_major"]]
        pool = majors or remaining
        if majors:
            picked = min(pool, key=lambda e: (-e["importance"], -e["_time"], e["member_ids"]))
            reason = "major_first_within_shared_budget"
        else:
            band = max(int(e["importance"] * 10) for e in pool)
            pool = [e for e in pool if int(e["importance"] * 10) == band]
            picked = min(pool, key=lambda e: (0 if disable_diversity else -cover(e),
                                            -e["importance"], -e["_time"], e["member_ids"]))
            reason = "diversity_disabled" if disable_diversity else "value_band_then_marginal_coverage"
        picked["reasons"] = ["evidence_time_content_gates_passed", reason]
        chosen.append(picked)
        remaining.remove(picked)
    for event in candidates:
        event.pop("_major")
        event.pop("_time")
    return chosen, remaining


def run_v2(corpus, reviews, proposals, *, evaluation_utc=None, limit=12,
           use_aliases=True, transfer_time=True, disable_diversity=False, active_ids=None):
    collection_now, articles, entries = _validate(corpus, reviews)
    items, aliases = validate_facts(corpus, proposals)
    if not use_aliases:
        aliases = {}
    now = _time(evaluation_utc) if evaluation_utc else collection_now
    if now < collection_now or not 1 <= limit <= 15:
        raise ValueError("invalid evaluation clock or output budget")
    elapsed = {i: "comparison_window_elapsed" for i, a in articles.items()
               if _time(a["published_at"]) < now - timedelta(hours=36)}
    articles = {i: a for i, a in articles.items()
                if active_ids is None or i in active_ids}
    articles = {i: a for i, a in articles.items()
                if now - timedelta(hours=36) <= _time(a["published_at"]) <= now}
    primaries = {i: next(f for f in x["facts"] if f["primary"])
                 for i, x in items.items() if x["facts"] and i in articles}
    signatures = {i: _signature(f, aliases) for i, f in primaries.items()}
    ranges = {i: _range(f, articles[i], now) for i, f in primaries.items()}
    groups, ambiguous = _groups(articles, primaries, ranges, signatures)
    group_of = {i: index for index, g in enumerate(groups) for i in g}
    signature_groups = defaultdict(set)
    for i, signature in signatures.items():
        signature_groups[signature].add(group_of[i])
    donors = defaultdict(list)
    for identifier in sorted(items):
        if identifier not in articles:
            continue
        for fact in items[identifier]["facts"]:
            if not _eligible(articles[identifier], entries.get(identifier), fact):
                continue
            interval = _range(fact, articles[identifier], now)
            if not isinstance(interval, tuple):
                continue
            signature = _signature(fact, aliases)
            targets = []
            for group_id in sorted(signature_groups[signature]):
                members = groups[group_id]
                if any(i in ambiguous or isinstance(ranges[i], str) for i in members):
                    continue
                known = [ranges[i] for i in members if isinstance(ranges[i], tuple)]
                if not known or _compatible(_intersect(known), interval):
                    targets.append(group_id)
            if len(targets) == 1:
                donors[targets[0]].append((identifier, fact, interval))
    ledger, excluded, candidates = {}, dict(elapsed), []
    effective = deepcopy(entries)
    for group_id, members in enumerate(groups):
        group_token = corpus_hash([signatures.get(i, (i,)) for i in members])
        for identifier in members:
            fact = primaries.get(identifier)
            evidence = [d for d in donors[group_id] if transfer_time or d[0] == identifier]
            own_range = ranges.get(identifier)
            if identifier in ambiguous:
                status = "ambiguous_occurrence"
            elif isinstance(own_range, str):
                status = own_range
            elif not fact:
                status = "no_primary_fact"
            elif not evidence:
                status = "unknown"
            else:
                combined = _intersect([d[2] for d in evidence])
                if combined[0] > combined[1]:
                    status = "conflict"
                elif combined[1] < now - timedelta(hours=36):
                    status = "old"
                elif combined[0] < now - timedelta(hours=36):
                    status = "boundary_uncertain"
                else:
                    status = "confirmed"
            interval = _intersect([d[2] for d in evidence]) if evidence else None
            citations = [{"article_id": i, "fact_id": f["fact_id"], **t}
                         for i, f, _ in evidence for t in f["times"]]
            ledger[identifier] = {
                "status": status,
                "range": [d.isoformat() for d in interval] if interval else None,
                "borrowed_from": sorted({i for i, _, _ in evidence if i != identifier}),
                "citations": citations,
                "primary_fact_id": fact["fact_id"] if fact else None,
                "time_basis": fact["time_basis"] if fact else None,
                "identity": list(signatures[identifier]) if fact else None,
            }
            if identifier not in effective:
                continue
            review = effective[identifier]
            review.update(time_status="confirmed" if status == "confirmed" else
                          "old" if status == "old" else "unknown",
                          event_at=interval[0].isoformat() if status == "confirmed" else None,
                          identity=dict.fromkeys(("actor", "action", "object", "place", "development"),
                                                 group_token))
            if fact:
                provenance = fact["evidence"]
                review.update(headline=fact["claim"], origin_group=provenance.get("origin_group"),
                              claims=[{"text": fact["claim"], "quote": fact["quote"],
                                       "supports_core": True, "kind": provenance["kind"],
                                       "origin_key": provenance.get("origin_key"),
                                       "fact_id": fact["fact_id"],
                                       "attribution_reason": provenance["reason"]}])
        admitted_times = [ledger[i]["range"][0] for i in members
                          if ledger[i]["status"] == "confirmed"]
        if admitted_times:
            # v1's internal clusterer contains a UTC-date key. Admission is already
            # settled by each ledger above; supply one adapter timestamp so that
            # overlapping ranges on different days do not re-split this fixed group.
            # This is never displayed as an event instant or as borrowed evidence.
            anchor = max(admitted_times, key=_time)
            for identifier in members:
                if identifier in effective and ledger[identifier]["status"] == "confirmed":
                    effective[identifier]["event_at"] = anchor
        # Reuse v1 family/confidence/quality/value rules within each fixed group.
        # Per-group invocation prevents its title fallback from bridging v2 facts.
        arm = _proposed({i: articles[i] for i in members}, effective, now, 15, True, False)
        excluded.update(arm["excluded"])
        for identifier in members:
            if excluded.get(identifier) in {"event_time_unknown", "old_event"}:
                excluded[identifier] = "time_" + ledger[identifier]["status"]
        for event in arm["selected"]:
            event["id"] = "fact_event_" + corpus_hash([
                signatures[event["member_ids"][0]],
                ledger[event["member_ids"][0]]["range"],
            ])[:16]
            event["fact_ids"] = [primaries[i]["fact_id"] for i in event["member_ids"]]
            event["confidence"]["limitations"].append(
                "fact-level source attribution is reviewed input, not calibrated truth probability")
            event["_major"] = (event["importance"] >= 0.85
                               and event["confidence"]["independent_chains"] > 0)
            event["_time"] = max(_time(ledger[i]["range"][0]).timestamp()
                                 for i in event["member_ids"])
            candidates.append(event)
    selected, remaining = _rank(candidates, limit, disable_diversity)
    for event in remaining:
        excluded.update({i: "lost_to_higher_value_or_same_band_coverage" for i in event["member_ids"]})
    return {"selected": selected, "excluded": excluded, "clusters": groups,
            "metrics": _metrics(selected), "temporal_ledger": ledger,
            "eligible_candidates": candidates,
            "options": {"use_aliases": use_aliases, "transfer_time": transfer_time,
                        "disable_diversity": disable_diversity}}
