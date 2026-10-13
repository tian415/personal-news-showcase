"""Attributed news-value judgments, separate from evidence admission and model receipts.

Quotes bind judgments to captured material; they do not prove the interpretation.
No popularity score is inferred by a language model or from missing measurements.
"""
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from .coverage_evaluation import NonEmpty, _aware, _fingerprint
from .models import CandidateArticle, NewsValueDisplay, Preferences, SignalSnapshot
from .trends import assess_trend


class AttentionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: HttpUrl
    snapshots: list[SignalSnapshot] = Field(min_length=1, max_length=100)


class NewsValueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: NonEmpty
    event_title: NonEmpty | None = None
    previous_event_id: NonEmpty | None = None
    previous_update_id: NonEmpty | None = None
    development: NonEmpty | None = None
    development_quote: NonEmpty | None = None
    timing: Literal["confirmed", "unknown", "conflicting"]
    event_at: datetime | None = None
    time_basis: Literal["event", "first_disclosure", "material_update"]
    time_quote: NonEmpty
    why_today: NonEmpty
    impact: NonEmpty
    impact_quote: NonEmpty
    importance: float = Field(ge=0, le=1, allow_inf_nan=False)
    attention: AttentionEvidence | None = None

    @field_validator("event_at")
    @classmethod
    def aware_time(cls, value):
        return _aware(value) if value is not None else None

    @model_validator(mode="after")
    def confirmed_time(self):
        if self.timing == "confirmed" and self.event_at is None:
            raise ValueError("confirmed event time requires event_at")
        return self


class NewsValueReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["news-value-review-v1"]
    snapshot_sha256: str
    review_sha256: str
    reviewed_by: NonEmpty
    reviewed_at: datetime
    items: list[NewsValueItem]

    @field_validator("reviewed_at")
    @classmethod
    def aware_time(cls, value):
        return _aware(value)


def validate_value_review(value: dict, audit: dict, review: dict, now: datetime):
    assessed = NewsValueReview.model_validate(value)
    if (assessed.snapshot_sha256 != _fingerprint(audit)
            or assessed.review_sha256 != _fingerprint(review)):
        raise ValueError("news value review binding mismatch")
    earliest = datetime.fromisoformat(review.get("reviewed_at") or audit["captured_at"])
    if not earliest <= assessed.reviewed_at <= now:
        raise ValueError("news value review time outside review and selection")
    entries = {item.candidate_id: item for item in assessed.items}
    if len(entries) != len(assessed.items):
        raise ValueError("duplicate news value item")
    if not entries.keys() <= {item['candidate_id'] for item in review['items']}:
        raise ValueError("unknown news value candidate")
    for item in entries.values():
        if item.attention and any(s.captured_at > assessed.reviewed_at for s in item.attention.snapshots):
            raise ValueError("attention capture must precede review")
    return entries


def evaluate_value(item: NewsValueItem, text: str, now: datetime) -> tuple[str | None, dict]:
    if item.time_quote not in text or item.impact_quote not in text:
        raise ValueError("news value quotation is absent from captured content")
    if item.development_quote is not None and item.development_quote not in text:
        raise ValueError("development quotation is absent from captured content")
    trend = assess_trend(item.attention.snapshots if item.attention else [], now=now)
    detail = {
        **item.model_dump(mode="json", exclude={'attention'}),
        'attention_status': trend.status, 'attention_level': trend.attention_level,
        'attention_velocity': trend.attention_velocity,
        'attention_url': str(item.attention.url) if item.attention else None,
        'independent_reporting_count': 0,
    }
    if item.timing != 'confirmed':
        return f'event_time_{item.timing}', detail
    if item.event_at > now:
        return 'future_event', detail
    if item.event_at < now - timedelta(hours=36):
        return 'old_event', detail
    return None, detail


def has_attention(detail: dict) -> bool:
    return (detail['independent_reporting_count'] >= 2 or (
        detail['attention_status'] in {'snapshot_only', 'velocity_available', 'acceleration_available'}
        and (detail['attention_level'] or 0) >= 0.1))


def rank_value(candidates: list[CandidateArticle], details: dict, preferences: Preferences):
    """Value bands dominate; preference and coverage only reorder comparable value."""
    remaining = list(candidates)
    selected = []
    domains: dict[str, int] = {}
    publishers: dict[str, int] = {}
    while remaining:
        def key(candidate):
            detail = details[candidate.id]
            value = max(detail['importance'], 0.6 if has_attention(detail) else 0)
            relevance = (preferences.topic_weights.get(candidate.topic, 1)
                         * preferences.source_weights.get(candidate.source, 1))
            return (int(value * 10 + 1e-8), relevance,
                    -domains.get(candidate.coverage_domain, 0),
                    -publishers.get(candidate.publisher_group, 0),
                    detail['event_at'], candidate.id)
        chosen = max(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
        domains[chosen.coverage_domain] = domains.get(chosen.coverage_domain, 0) + 1
        publishers[chosen.publisher_group] = publishers.get(chosen.publisher_group, 0) + 1
    return selected


def value_display(detail: dict) -> NewsValueDisplay:
    if detail['independent_reporting_count'] >= 2:
        attention = f"{detail['independent_reporting_count']} 家独立报道来源关注；不代表热度上升"
    elif detail['attention_status'] == 'snapshot_only':
        attention = '有排行快照；单次观测不能证明热度上升'
    elif detail['attention_status'] in {'velocity_available', 'acceleration_available'}:
        attention = '有连续讨论快照；讨论量变化不等于事实重要性'
    else:
        attention = '热点未知：缺少有效的独立关注或讨论数据'
    return NewsValueDisplay(why_today=detail['why_today'], impact=detail['impact'],
                            date_label={'event': '事件／进展日期', 'first_disclosure': '本次披露日期',
                                        'material_update': '实质更新日期'}[detail['time_basis']],
                            event_time=detail['event_at'], attention_note=attention,
                            attention_url=detail['attention_url'],
                            value_basis_url=detail.get('value_basis_url'),
                            value_basis_source=detail.get('value_basis_source'))


def assign_value_events(candidates, values, historical_events, reported_ids, material_hashes):
    """Require a version-bound, content-changing and specifically supported delta."""
    from hashlib import sha256
    from uuid import NAMESPACE_URL, uuid5

    from .events import assign_events
    from .feeds import canonicalize_url
    from .models import EventUpdate, EvidenceRole
    from .normalization import normalize_title

    history_by_id = {}
    for old in historical_events:
        if (old.event_id not in history_by_id
                or old.latest_event_time > history_by_id[old.event_id].latest_event_time):
            history_by_id[old.event_id] = old
    history = {(e.coverage_domain, normalize_title(e.title)): e for e in history_by_id.values()}
    excluded, canonical = {}, []
    for c in candidates:
        v = values[c.id]
        title = v.event_title or c.title
        if v.previous_event_id:
            old = history_by_id.get(v.previous_event_id)
            if old is None or old.coverage_domain != c.coverage_domain:
                excluded[c.id] = 'unknown_previous_event'
                continue
            if v.previous_update_id != old.latest_update_id:
                excluded[c.id] = 'event_version_conflict'
                continue
            title = old.title  # Explicit identity cannot be bypassed by rewording.
        canonical.append(c.model_copy(update={'title': title, 'event_time': v.event_at}))
    assignment = assign_events(canonical, [])
    by_id = {c.id: c for c in candidates}
    seen_hashes = {h for old in history_by_id.values() for h in old.editorial_content_fingerprints}
    seen_urls = {u for old in history_by_id.values() for u in old.editorial_source_urls}
    kept = []
    for event in assignment.events:
        old = history.get((event.coverage_domain, normalize_title(event.title)))
        hashes = [material_hashes[cid] for cid in event.candidate_ids]
        urls = [canonicalize_url(str(by_id[cid].url)) for cid in event.candidate_ids]
        if old is None:
            if set(hashes) & seen_hashes or set(urls) & seen_urls:
                excluded.update({cid: 'event_duplicate_no_delta' for cid in event.candidate_ids})
                continue
            initial = values[event.primary_candidate_id]
            kept.append(event.model_copy(update={
                'editorial_development': initial.development,
                'editorial_content_fingerprints': hashes, 'editorial_source_urls': urls,
                'editorial_evidence_sha256': sha256((initial.development_quote or initial.time_quote).encode()).hexdigest(),
            }))
            continue
        updates = [values[cid] for cid in event.candidate_ids if (
            cid in reported_ids and values[cid].time_basis == 'material_update'
            and values[cid].previous_event_id == old.event_id
            and values[cid].previous_update_id == old.latest_update_id
            and values[cid].development and values[cid].development_quote
            and values[cid].event_at > old.latest_event_time
            and old.editorial_content_fingerprints
            and material_hashes[cid] not in seen_hashes
            and values[cid].development != old.editorial_development
            and sha256(values[cid].development_quote.encode()).hexdigest() != old.editorial_evidence_sha256)]
        groups = {}
        for v in updates:
            groups.setdefault(v.development, []).append(v)
        supported = []
        for group in groups.values():
            supporting = [by_id[v.candidate_id] for v in group]
            reporting = [c for c in supporting if c.evidence_role == EvidenceRole.INDEPENDENT_REPORTING]
            strong = (any(c.evidence_role == EvidenceRole.PRIMARY_FACT for c in supporting)
                      or (len({c.publisher_group for c in reporting}) >= 2
                          and len({canonicalize_url(str(c.url)) for c in reporting}) >= 2))
            if strong:
                supported.append(group)
        if not supported:
            excluded.update({cid: 'event_duplicate_no_delta' for cid in event.candidate_ids})
            continue
        group = max(supported, key=lambda g: max(v.event_at for v in g))
        development = max(group, key=lambda v: (v.event_at, v.candidate_id))
        update_id = 'upd_' + str(uuid5(NAMESPACE_URL, '|'.join([
            old.event_id, old.latest_update_id or '', development.development,
            development.event_at.isoformat(), development.development_quote,
        ])))
        update = EventUpdate(update_id=update_id, event_id=old.event_id,
            previous_update_id=old.latest_update_id, delta_fields=['reviewed_development'],
            material=True, material_update_reason='CONTENT_BOUND_EDITORIAL_UPDATE',
            occurred_at=development.event_at)
        support_ids = [v.candidate_id for v in group]
        excluded.update({cid: 'unsupported_event_development' for cid in event.candidate_ids
                         if cid not in support_ids})
        kept.append(old.model_copy(update={
            'candidate_ids': support_ids, 'primary_candidate_id': support_ids[0],
            'latest_event_time': development.event_at, 'latest_update_id': update_id,
            'editorial_development': development.development,
            'editorial_content_fingerprints': sorted(set(old.editorial_content_fingerprints + [material_hashes[cid] for cid in support_ids])),
            'editorial_source_urls': sorted(set(old.editorial_source_urls + [canonicalize_url(str(by_id[cid].url)) for cid in support_ids])),
            'editorial_evidence_sha256': sha256(development.development_quote.encode()).hexdigest(),
            'update_history': [*old.update_history, update], 'state': 'reviewed_material_update',
        }))
    return kept, excluded


def discover_supplement(registry, plan: dict, *, now: datetime, max_sources: int = 6,
                        include_shadow: bool = False, loader=None) -> dict:
    """One bounded public RSS pass, prioritizing gaps; output still requires review."""
    from .models import CoverageDomain, EvidenceRole
    from .source_audit import audit_sources

    if type(max_sources) is not int or not 1 <= max_sources <= 20:
        raise ValueError('max_sources must be 1..20')
    needed = plan['needed']
    if type(needed) is not int or not 0 <= needed <= 10:
        raise ValueError('invalid discovery shortfall')
    domains = {CoverageDomain(d) for d in plan['priority_domains']}
    sources = [s for s in registry.enabled_sources('rss')
               if (s.production_enabled or include_shadow)
               and s.evidence_role != EvidenceRole.COMMUNITY_SIGNAL]
    sources.sort(key=lambda s: (not bool(set(s.coverage_domains) & domains),
                               s.evidence_role != EvidenceRole.PRIMARY_FACT, s.id))
    chosen = sources[:max_sources]
    options = {'loader': loader} if loader is not None else {}
    result = audit_sources(registry.model_copy(update={'sources': chosen}),
                           now=now, max_age_hours=36, capture_candidates=True, **options)
    result['discovery_context'] = {
        'shortfall_before': needed, 'priority_domains': sorted(d.value for d in domains),
        'source_ids': [s.id for s in chosen], 'include_shadow': include_shadow,
        'status': 'unreviewed_candidates', 'paid_calls': 0,
    }
    return result


def prepare_value_review(audit: dict, review: dict) -> dict:
    """An unjudged worksheet: no identity/time/value is inferred from admission."""
    return {
        'schema_version': 'news-value-review-v1',
        'snapshot_sha256': _fingerprint(audit), 'review_sha256': _fingerprint(review),
        'reviewed_by': None, 'reviewed_at': None,
        'items': [{
            'candidate_id': item['candidate_id'], 'event_title': None,
            'timing': 'unknown', 'event_at': None, 'time_basis': 'first_disclosure',
            'time_quote': item['evidence_excerpt'], 'why_today': '待核对近期事件或实质进展',
            'impact': '待说明影响对象、范围、变化和限制',
            'impact_quote': item['evidence_excerpt'], 'importance': 0,
        } for item in review['items'] if item['status'] == 'accepted'],
    }
