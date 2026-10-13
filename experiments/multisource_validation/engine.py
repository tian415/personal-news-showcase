"""Offline frozen-pool comparison. No reference/gold files or network access.

Identity uses strictly equal reviewer-normalized tuples, not embeddings or automatic
cross-language understanding. Value is an experimental median across evidence
families, not a fresh semantic event valuation. No historical state is consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from statistics import median

from news_assistant.coverage_evaluation import _fingerprint
from news_assistant.models import Article, Preferences
from news_assistant.reviewed_selection import prepare_review, select_reviewed
from news_assistant.selection import deduplicate, select_digest

POLICY_VERSION = "multisource-experiment-v1"
DOMAINS = {
    "ai_technology",
    "business_macro",
    "policy_public_affairs",
    "science_health",
    "society_life",
    "culture_entertainment",
}
TOPICS = {
    "ai_technology": "tech",
    "business_macro": "business",
    "science_health": "science",
    "policy_public_affairs": "exploration",
    "society_life": "exploration",
    "culture_entertainment": "exploration",
}


def corpus_hash(value):
    """SHA256 of UTF-8 JSON, sorted keys, compact separators, no ASCII escaping."""
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamps require a timezone")
    return result.astimezone(UTC)


def _norm(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _title(value):
    return re.sub(r"[^\w]", "", _norm(value))


def _validate(corpus, reviews):
    if reviews.get("corpus_sha256") != corpus_hash(corpus):
        raise ValueError("corpus review binding mismatch")
    if corpus.get("window_hours") != 36:
        raise ValueError("experiment requires frozen 36-hour feed window")
    now = _time(corpus["reference_utc"])
    articles = {a["id"]: a for a in corpus["articles"]}
    entries = {r["id"]: r for r in reviews["items"]}
    if len(articles) != len(corpus["articles"]) or len(entries) != len(reviews["items"]):
        raise ValueError("duplicate input id")
    if not entries.keys() <= articles.keys():
        raise ValueError("review contains unknown ids")
    if (
        any(r.get("status") == "reviewed" for r in entries.values())
        and not reviews.get("reviewer", "").strip()
    ):
        raise ValueError("reviewer attribution required")
    for identifier, article in articles.items():
        body = article["body"]
        if not isinstance(body, str):
            raise TypeError("corpus body must be text")
        item = entries.get(identifier)
        if body.strip():
            if hashlib.sha256(body.encode()).hexdigest() != article["body_sha256"]:
                raise ValueError("corpus body hash mismatch")
        elif article["body_status"] in {"ok", "extracted", "captured"}:
            raise ValueError("successful body capture cannot be empty")
        elif article["body_sha256"] is not None:
            raise ValueError("metadata-only article must have null body hash")
        elif item and item.get("status") == "reviewed":
            raise ValueError("reviewed article requires captured body")
        if not now - timedelta(hours=36) <= _time(article["published_at"]) <= now:
            raise ValueError("raw article outside frozen feed window")
        item = entries.get(identifier)
        if item is None:
            continue
        if item["body_sha256"] != article["body_sha256"]:
            raise ValueError("review body hash mismatch")
        if item.get("status") not in {"pending", "reviewed"}:
            raise ValueError("invalid review status")
        if item["status"] == "pending":
            continue
        for key in ("quality", "importance"):
            value = item.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("review needs explicit bounded scores")
        if item["domain"] not in DOMAINS:
            raise ValueError("unknown domain")
        if item["time_status"] not in {"confirmed", "unknown", "old"}:
            raise ValueError("invalid time status")
        if item["time_basis"] not in {"event", "first_disclosure", "material_update"}:
            raise ValueError("invalid time basis")
        if item["story_shape"] not in {"single_event", "roundup", "background", "promotional"}:
            raise ValueError("invalid story shape")
        for key in ("reason", "headline", "region", "why_today", "impact"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"missing review {key}")
        if item["event_at"] is not None:
            _time(item["event_at"])
        if item["time_status"] == "confirmed" and item["event_at"] is None:
            raise ValueError("confirmed time requires event_at")
        for quote in [
            item["time_quote"],
            item["impact_quote"],
            *[c["quote"] for c in item["claims"]],
        ]:
            if not isinstance(quote, str) or not quote.strip() or quote not in body:
                raise ValueError("quotation absent from exact captured body")
        for claim in item["claims"]:
            if claim["kind"] not in {
                "document",
                "direct_reporting",
                "attributed_report",
                "assertion",
                "analysis",
            }:
                raise ValueError("invalid claim kind")
            if type(claim["supports_core"]) is not bool:
                raise ValueError("core support must be boolean")
        if type(item["conflict"]) is not bool:
            raise ValueError("conflict must be boolean")
        if item["conflict"] and not item.get("conflict_reason"):
            raise ValueError("conflict explanation required")
    return now, articles, entries


def _clusters(articles, entries, title_only=False):
    identifiers = sorted(articles)
    parents = {i: i for i in identifiers}
    known = {}
    for identifier in identifiers:
        item = entries.get(identifier, {})
        identity = item.get("identity") or {}
        values = tuple(
            _norm(identity.get(k)) for k in ("actor", "action", "object", "place", "development")
        )
        known[identifier] = (
            {(values, _time(item["event_at"]).date().isoformat())}
            if all(values) and item.get("event_at")
            else set()
        )

    def root(identifier):
        while parents[identifier] != identifier:
            identifier = parents[identifier]
        return identifier

    edges = []
    for position, left in enumerate(identifiers):
        for right in identifiers[position + 1 :]:
            if articles[left]["url"] == articles[right]["url"]:
                edges.append((0, left, right))
            elif not title_only and known[left] and known[left] == known[right]:
                edges.append((1, left, right))
            elif _title(articles[left]["title"]) == _title(articles[right]["title"]):
                edges.append((2, left, right))
    # Exact URL identity is retained. Semantic edges precede title fallbacks;
    # a title bridge may never unite clusters with different known identities.
    for priority, left, right in sorted(edges):
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            continue
        combined = known[left_root] | known[right_root]
        if not title_only and priority != 0 and len(combined) > 1:
            continue
        first, second = sorted((left_root, right_root))
        parents[second] = first
        known[first] = combined
    groups = defaultdict(list)
    for identifier in identifiers:
        groups[root(identifier)].append(identifier)
    return sorted(groups.values())


def _metrics(selected):
    distributions = {k: Counter() for k in ("source", "origin_group", "domain", "region")}
    for event in selected:
        # Equal fractional credit per distinct original reporting group; never use
        # representative link or number of syndicated copies as distribution weight.
        for key, field in [("source", "sources"), ("origin_group", "origin_groups")]:
            values = event[field]
            for value in values:
                distributions[key][value] += 1 / len(values)
        distributions["domain"][event["domain"]] += 1
        distributions["region"][event["region"]] += 1
    result = {}
    for key, counts in distributions.items():
        total = sum(counts.values())
        shares = [v / total for v in counts.values()] if total else []
        result[key] = {
            "distribution": dict(sorted(counts.items())),
            "max_share": max(shares, default=0),
            "hhi": sum(v * v for v in shares),
        }
    result["distribution_method"] = (
        "equal_fraction_per_distinct_original_group; source=original_reporting_group"
    )
    return result


def _record(members, articles, entries, importance=None, confidence=None, reasons=None):
    representative = min(members)
    item = entries[representative]
    identity = item.get("identity") or {}
    values = tuple(
        _norm(identity.get(k)) for k in ("actor", "action", "object", "place", "development")
    )
    identity_key = (
        (values, _time(item["event_at"]).date().isoformat())
        if all(values) and item.get("event_at")
        else (articles[representative]["url"],)
    )
    origins = sorted(
        {entries[i].get("origin_group") or articles[i]["publisher_group"] for i in members}
    )
    return {
        "id": "event_" + corpus_hash(identity_key)[:16],
        "member_ids": members,
        "headline": item["headline"],
        "domain": item["domain"],
        "region": item["region"],
        "importance": item["importance"] if importance is None else importance,
        "origin_groups": origins,
        "sources": origins,
        "source_ids": sorted({articles[i]["source_id"] for i in members}),
        "url": articles[representative]["url"],
        "confidence": confidence or {},
        "reasons": reasons or [],
    }


def _proposed(articles, entries, now, limit, disable_diversity, title_only):
    clusters = _clusters(articles, entries, title_only)
    excluded, candidates = {}, []
    for members in clusters:
        admitted = []
        for identifier in members:
            a, r = articles[identifier], entries.get(identifier)
            reason = None
            if not r or r["status"] == "pending":
                reason = "pending_review"
            elif not a["body"].strip() or a["body_status"] not in {"extracted", "captured", "ok"}:
                reason = "body_unavailable"
            elif r["quality"] < 0.5:
                reason = "below_quality_floor"
            elif r["story_shape"] != "single_event":
                reason = "content_type_" + r["story_shape"]
            elif r["time_status"] == "old":
                reason = "old_event"
            elif r["time_status"] != "confirmed":
                reason = "event_time_unknown"
            elif _time(r["event_at"]) > now:
                reason = "future_event"
            elif _time(r["event_at"]) < now - timedelta(hours=36):
                reason = "old_event"
            if reason:
                excluded[identifier] = reason
            else:
                admitted.append(identifier)
        if not admitted:
            continue
        if any(
            entries[i]["conflict"]
            for i in members
            if i in entries and entries[i]["status"] == "reviewed"
        ):
            excluded.update({i: "core_conflict" for i in admitted})
            continue
        families, quotes = defaultdict(dict), []
        independent, attributed = set(), set()
        chain_groups = defaultdict(set)
        for identifier in admitted:
            r, a = entries[identifier], articles[identifier]
            for claim in r["claims"]:
                if not claim["supports_core"]:
                    continue
                origin = claim.get("origin_key") or "publisher:" + a["publisher_group"]
                if claim["kind"] in {"document", "direct_reporting"}:
                    independent.add(origin)
                    chain_groups[origin].add(r.get("origin_group") or a["publisher_group"])
                elif claim["kind"] == "attributed_report" and claim.get("origin_key"):
                    attributed.add(origin)
                else:
                    continue
                # One deterministic original-family judgment; duplicated body/copies
                # cannot change it. IDs are frozen provenance, not ranking features.
                families[origin][identifier] = r["importance"]
                quotes.append({"member_id": identifier, **claim, "evidence_family": origin})
        if not families:
            excluded.update({i: "insufficient_core_evidence" for i in admitted})
            continue
        family_values = [min(values.values()) for _, values in sorted(families.items())]
        # Shared publisher groups and shared upstream keys connect one chain.
        # This deliberately conservative connected-component count prevents a
        # syndicated cross-post bridge from creating independent corroboration.
        chain_components = []
        for origin in sorted(independent):
            groups = chain_groups[origin]
            overlap = [c for c in chain_components if c & groups]
            combined = set(groups)
            for component in overlap:
                combined.update(component)
                chain_components.remove(component)
            chain_components.append(combined)
        chain_count = len(chain_components)
        importance = median(family_values)
        if importance < 0.55:
            excluded.update({i: "low_news_value" for i in admitted})
            continue
        limitations = [
            "median of conservative minimum review per upstream family; no fresh event reassessment"
        ]
        if not independent:
            limitations.append("attributed report only; state source attribution explicitly")
        elif chain_count == 1:
            limitations.append("single evidence chain; narrow statement to supported claim")
        if any(
            not all(
                _norm((entries[i].get("identity") or {}).get(k))
                for k in ("actor", "action", "object", "place", "development")
            )
            for i in admitted
        ):
            limitations.append("incomplete identity; semantic merging unavailable")
        confidence = {
            "level": "corroborated"
            if chain_count >= 2
            else "single_chain"
            if independent
            else "attributed",
            "independent_chains": chain_count,
            "evidence_families": len(families),
            "support_quotes": quotes,
            "limitations": limitations,
        }
        record = _record(
            admitted,
            articles,
            entries,
            importance,
            confidence,
            ["evidence_time_content_gates_passed"],
        )
        record["_major"] = importance >= 0.85 and bool(independent)
        record["_time"] = max(_time(entries[i]["event_at"]).timestamp() for i in admitted)
        candidates.append(record)
    chosen = []

    def cover(record):
        result = 0
        for field in ("domain", "region"):
            count = sum(e[field] == record[field] for e in chosen)
            result += 1 if count == 0 else -0.2 * count
        used = Counter(o for e in chosen for o in e["origin_groups"])
        result += sum(1 if used[o] == 0 else -0.3 * used[o] for o in record["origin_groups"]) / len(
            record["origin_groups"]
        )
        return result

    remaining = list(candidates)
    while remaining and len(chosen) < min(limit, 15):
        majors = [e for e in remaining if e["_major"]]
        pool = majors or remaining
        if majors:
            picked = min(pool, key=lambda e: (-e["importance"], -e["_time"], e["member_ids"]))
            picked["reasons"].append("major_first_within_shared_budget")
        else:
            band = max(int(e["importance"] * 10) for e in pool)
            pool = [e for e in pool if int(e["importance"] * 10) == band]
            picked = min(
                pool,
                key=lambda e: (
                    0 if disable_diversity else -cover(e),
                    -e["importance"],
                    -e["_time"],
                    e["member_ids"],
                ),
            )
            picked["reasons"].append(
                "value_band_then_marginal_coverage"
                if not disable_diversity
                else "diversity_disabled"
            )
        chosen.append(picked)
        remaining.remove(picked)
    for e in remaining:
        excluded.update({i: "lost_to_higher_value_or_same_band_coverage" for i in e["member_ids"]})
    for e in chosen:
        del e["_major"], e["_time"]
    return {
        "selected": chosen,
        "excluded": excluded,
        "clusters": clusters,
        "metrics": _metrics(chosen),
    }


def _existing_arms(articles, entries, now, limit):
    pool = sorted(i for i, r in entries.items() if r["status"] == "reviewed")
    base_excluded = {i: "pending_review" for i in articles if i not in pool}
    oversized = {i: len(articles[i]["body"]) for i in pool if len(articles[i]["body"]) > 20_000}
    if oversized:
        raise ValueError(
            f"review body exceeds 20000 characters; common-pool decision required: {oversized}"
        )
    missing_html = [
        i for i in pool if not re.fullmatch(r"[0-9a-f]{64}", articles[i].get("html_sha256") or "")
    ]
    if missing_html:
        raise ValueError(
            f"real HTML capture hash required for existing reviewed adapter: {missing_html}"
        )
    models, source_groups, content = [], defaultdict(list), {}
    for identifier in pool:
        a, r = articles[identifier], entries[identifier]
        article = Article(
            id=identifier,
            title=a["title"],
            url=a["url"],
            source=a["source"],
            language=a["language"],
            published_at=_time(a["published_at"]),
            topic=TOPICS[r["domain"]],
            summary=a["summary"],
            quality_score=r["quality"],
            importance_score=r["importance"],
        )
        models.append(article)
        source_groups[a["source_id"]].append(article)
        cid = a["source_id"] + ":" + identifier
        content[cid] = {
            "status": "extracted",
            "requested_url": str(article.url),
            "final_url": a["body_final_url"],
            "retrieved_at": a["retrieved_at"],
            "extractor": "trafilatura",
            "extractor_version": a["extractor_version"],
            "text": a["body"],
            "text_sha256": a["body_sha256"],
            "html_sha256": a["html_sha256"],
        }
    audit = {
        "schema_version": "source-audit-v1",
        "candidate_snapshot_available": True,
        "captured_at": min(
            (_time(a["retrieved_at"]) for i, a in articles.items() if i in pool), default=now
        ).isoformat(),
        "max_age_hours": 36,
        "sources": [],
        "article_content": {"schema_version": "article-content-v1", "items": content},
    }
    for sid, group in sorted(source_groups.items()):
        a = articles[group[0].id]
        domains = sorted({entries[x.id]["domain"] for x in group})
        audit["sources"].append(
            {
                "id": sid,
                "name": a["source"],
                "publisher_group": a["publisher_group"],
                "production_enabled": True,
                "status": "ok",
                "evidence_role": "INDEPENDENT_REPORTING",
                "coverage_domains": [
                    d
                    if d
                    in {
                        "ai_technology",
                        "business_macro",
                        "science_health",
                        "policy_public_affairs",
                    }
                    else "exploration"
                    for d in domains
                ],
                "parsed_count": len(group),
                "fresh_count": len(group),
                "stale_count": 0,
                "future_count": 0,
                "candidates": [x.model_dump(mode="json") for x in group],
            }
        )
    editorial = prepare_review(audit)
    editorial.update(reviewed_by="frozen experiment adapter", reviewed_at=now.isoformat())
    values = []
    idmap = {}
    for item in editorial["items"]:
        cid = item["candidate_id"]
        identifier = next(i for i in pool if articles[i]["source_id"] + ":" + i == cid)
        idmap[cid] = identifier
        r = entries[identifier]
        supported = any(
            c["supports_core"]
            and c["kind"] in {"document", "direct_reporting", "attributed_report"}
            for c in r["claims"]
        )
        item.update(
            coverage_domain=r["domain"]
            if r["domain"]
            in {"ai_technology", "business_macro", "science_health", "policy_public_affairs"}
            else "exploration",
            topic=TOPICS[r["domain"]],
            status="accepted",
            quality_score=r["quality"],
            importance_score=r["importance"],
            evidence_excerpt=r["time_quote"],
            fact_status="reported" if supported else "unverified",
            story_shape="roundup" if r["story_shape"] == "roundup" else "single_event",
            origin_publisher_group=r.get("origin_group"),
            rationale=r["reason"],
            major_reason=r["impact"] if r["importance"] >= 0.85 else None,
        )
        values.append(
            {
                "candidate_id": cid,
                "event_title": r["headline"],
                "timing": "confirmed"
                if r["time_status"] in {"confirmed", "old"} and r["event_at"]
                else "unknown",
                "event_at": r["event_at"],
                "time_basis": r["time_basis"],
                "time_quote": r["time_quote"],
                "why_today": r["why_today"],
                "impact": r["impact"],
                "impact_quote": r["impact_quote"],
                "importance": r["importance"],
            }
        )
    value_review = {
        "schema_version": "news-value-review-v1",
        "snapshot_sha256": _fingerprint(audit),
        "review_sha256": _fingerprint(editorial),
        "reviewed_by": "frozen experiment adapter",
        "reviewed_at": now.isoformat(),
        "items": values,
    }
    old = select_digest(models, Preferences(), limit=limit, now=now)
    reviewed = select_reviewed(
        audit, editorial, Preferences(), now=now, limit=limit, value_review=value_review
    )
    old_ids = {a.id for a in old}
    legacy_clusters = []
    # Reuse real legacy duplicate decisions; do not approximate them with gold.
    for article in models:
        for group in legacy_clusters:
            if len(deduplicate([group[0], article])) == 1:
                group.append(article)
                break
        else:
            legacy_clusters.append([article])
    legacy = [_record([a.id], articles, entries, reasons=["actual_select_digest"]) for a in old]
    event_members = {
        e.primary_candidate_id: sorted(idmap[i] for i in e.candidate_ids) for e in reviewed.events
    }
    selected = [
        _record(
            event_members[a.id],
            articles,
            entries,
            importance=reviewed.value_details[a.id]["importance"],
            reasons=["actual_select_reviewed_value"],
        )
        for a in reviewed.selected[:limit]
    ]
    reviewed_excluded = {
        **base_excluded,
        **{idmap[i]: reason for i, reason in reviewed.excluded.items()},
        **{idmap[a.id]: "comparison_budget_truncation" for a in reviewed.selected[limit:]},
    }
    represented = {i for event in selected for i in event["member_ids"]}
    for identifier in pool:
        if identifier not in represented and identifier not in reviewed_excluded:
            reviewed_excluded[identifier] = "legacy_reason_unavailable"
    return {
        "legacy": {
            "selected": legacy,
            "clusters": [[a.id for a in group] for group in legacy_clusters],
            "excluded": {
                **base_excluded,
                **{i: "legacy_reason_unavailable" for i in pool if i not in old_ids},
            },
            "metrics": _metrics(legacy),
        },
        "reviewed_value": {
            "selected": selected,
            "clusters": [sorted(idmap[i] for i in e.candidate_ids) for e in reviewed.events],
            "excluded": reviewed_excluded,
            "raw_selected_count": len(reviewed.selected),
            "metrics": _metrics(selected),
        },
    }


def run_comparison(
    corpus,
    reviews,
    limit=12,
    *,
    disable_diversity=False,
    title_only_clustering=False,
    evaluation_utc=None,
):
    """Run actual existing algorithms and proposed policy on one frozen reviewed pool."""
    if type(limit) is not int or not 0 <= limit <= 15:
        raise ValueError("limit must be 0..15")
    collection_reference, articles, entries = _validate(corpus, reviews)
    now = _time(evaluation_utc) if evaluation_utc is not None else collection_reference
    if now < collection_reference:
        raise ValueError("evaluation clock precedes collection reference")
    elapsed = {
        i: "comparison_window_elapsed"
        for i, a in articles.items()
        if _time(a["published_at"]) < now - timedelta(hours=36)
    }
    active_articles = {i: a for i, a in articles.items() if i not in elapsed}
    active_entries = {i: r for i, r in entries.items() if i in active_articles}
    arms = _existing_arms(active_articles, active_entries, now, limit)
    arms["proposed"] = _proposed(
        active_articles, active_entries, now, limit, disable_diversity, title_only_clustering
    )
    for arm in arms.values():
        arm["excluded"].update(elapsed)
    return {
        "policy_version": POLICY_VERSION,
        "corpus_sha256": corpus_hash(corpus),
        "reviews_sha256": corpus_hash(reviews),
        "limit": limit,
        "collection_reference_utc": corpus["reference_utc"],
        "evaluation_utc": now.isoformat(),
        "collection_to_evaluation_seconds": (now - collection_reference).total_seconds(),
        "pool_ids": sorted(i for i, r in active_entries.items() if r["status"] == "reviewed"),
        "adapter_notes": [
            "society_life/culture_entertainment map to exploration in old schemas",
            "old source evidence role is reporting; origin_group comes from review",
            "reviewed_value uses exact captured body and real HTML capture provenance in CapturedContent sidecar; original summary retained",
            "adapter audit captured_at is earliest actual body retrieval; original capture timestamps and corpus hash are preserved",
            "adapter reviewed_at is the common evaluation replay binding time, not semantic author's original annotation creation time",
            "collection window validates against collection reference; all arms use common evaluation clock and exclude elapsed publication-window items",
            "existing reviewed major overflow is explicitly truncated to common comparison budget; raw_selected_count retained",
            "no historical tracking interface; no automatic semantic normalization",
        ],
        "arms": arms,
    }
