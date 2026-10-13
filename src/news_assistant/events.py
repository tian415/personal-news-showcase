from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field

from .models import (
    CandidateArticle,
    DecisionRecord,
    Event,
    EventUpdate,
    EvidenceRole,
    ProcessingStage,
    ReasonCode,
)
from .normalization import normalize_title


class EventAssignment(BaseModel):
    events: list[Event] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)


def assign_events(
    candidates: Iterable[CandidateArticle], historical_events: Iterable[Event]
) -> EventAssignment:
    historical = list(historical_events)
    historical_by_title: dict[str, Event] = {}
    for event in historical:
        title = normalize_title(event.title)
        previous = historical_by_title.get(title)
        if previous is None or (event.latest_event_time, event.event_id) > (
            previous.latest_event_time,
            previous.event_id,
        ):
            historical_by_title[title] = event
    grouped: dict[tuple[str, str], list[CandidateArticle]] = {}
    decisions: list[DecisionRecord] = []
    events: list[Event] = []
    for candidate in candidates:
        title = normalize_title(candidate.title)
        matching_updates = [event for event in historical if _is_material_update(candidate, event)]
        update_of = (
            max(
                matching_updates,
                key=lambda event: (
                    event.latest_event_time,
                    len(event.update_history),
                    event.latest_update_id or "",
                    event.event_id,
                ),
            )
            if matching_updates
            else None
        )
        if update_of:
            if _is_replayed_confirmation(candidate, update_of):
                decisions.append(
                    DecisionRecord(
                        candidate_id=candidate.candidate_id,
                        event_id=update_of.event_id,
                        stage=ProcessingStage.EVENT_ASSIGNED,
                        selected=False,
                        primary_reason=ReasonCode.EVENT_DUPLICATE_NO_DELTA,
                    )
                )
                continue

            occurred_at = candidate.event_time or candidate.published_at
            state = "official_confirmation"
            update_id = _material_update_id(update_of, candidate, state)
            update = EventUpdate(
                update_id=update_id,
                event_id=update_of.event_id,
                previous_update_id=update_of.latest_update_id,
                delta_fields=_delta_fields(update_of, candidate, occurred_at, state),
                material=True,
                material_update_reason="PRIMARY_FACT_OFFICIAL_CONFIRMATION",
                occurred_at=occurred_at,
            )
            events.append(
                update_of.model_copy(
                    update={
                        "title": candidate.title,
                        "latest_event_time": occurred_at,
                        "candidate_ids": [candidate.candidate_id],
                        "primary_candidate_id": candidate.candidate_id,
                        "state": state,
                        "latest_update_id": update_id,
                        "update_history": [*update_of.update_history, update],
                        "identity_terms": update_of.identity_terms or _identity_terms(update_of.title),
                    }
                )
            )
            decisions.append(
                DecisionRecord(
                    candidate_id=candidate.candidate_id,
                    event_id=update_of.event_id,
                    stage=ProcessingStage.EVENT_ASSIGNED,
                    selected=False,
                    primary_reason=ReasonCode.MATERIAL_UPDATE_INCLUDED,
                )
            )
            continue
        duplicate_of = historical_by_title.get(title)
        if duplicate_of is not None:
            decisions.append(
                DecisionRecord(
                    candidate_id=candidate.candidate_id,
                    event_id=duplicate_of.event_id,
                    stage=ProcessingStage.EVENT_ASSIGNED,
                    selected=False,
                    primary_reason=ReasonCode.EVENT_DUPLICATE_NO_DELTA,
                )
            )
            continue
        grouped.setdefault((candidate.coverage_domain.value, title), []).append(candidate)

    for (domain, normalized_title), members in grouped.items():
        primary = members[0]
        event = Event(
            event_id=f"evt_{uuid5(NAMESPACE_URL, f'{domain}:{normalized_title}')}",
            coverage_domain=primary.coverage_domain,
            event_type=_event_type(normalized_title),
            title=primary.title,
            first_event_time=min(item.event_time or item.published_at for item in members),
            latest_event_time=max(item.event_time or item.published_at for item in members),
            candidate_ids=[item.candidate_id for item in members],
            primary_candidate_id=primary.candidate_id,
            independent_source_count=len({item.source_id for item in members}),
            identity_terms=_identity_terms(primary.title),
        )
        events.append(event)
        decisions.extend(
            DecisionRecord(
                candidate_id=item.candidate_id,
                event_id=event.event_id,
                stage=ProcessingStage.EVENT_ASSIGNED,
                selected=False,
                primary_reason=ReasonCode.LOST_TO_HIGHER_VALUE,
            )
            for item in members
        )
    return EventAssignment(events=events, decisions=decisions)


def _is_material_update(candidate: CandidateArticle, event: Event) -> bool:
    """Return true only for a conservative primary-fact event continuation."""
    if candidate.evidence_role is not EvidenceRole.PRIMARY_FACT:
        return False
    if candidate.coverage_domain != event.coverage_domain:
        return False
    candidate_title = normalize_title(candidate.title)
    historical_title = normalize_title(event.title)
    if candidate_title == historical_title or not _has_confirmation_marker(candidate_title):
        return False
    occurred_at = candidate.event_time or candidate.published_at
    if occurred_at < event.latest_event_time:
        return False

    candidate_type = _event_type(candidate_title)
    historical_type = event.event_type
    if historical_type == "news":
        historical_type = _event_type(historical_title)
    if candidate_type == "news" or candidate_type != historical_type:
        return False

    historical_terms = set(event.identity_terms) or set(_identity_terms(event.title))
    candidate_terms = set(_identity_terms(candidate.title))
    if not historical_terms or not candidate_terms:
        return False
    return historical_terms == candidate_terms


def _is_replayed_confirmation(candidate: CandidateArticle, event: Event) -> bool:
    if event.state != "official_confirmation":
        return False
    expected_id = _material_update_id(event, candidate, event.state)
    return event.latest_update_id == expected_id or normalize_title(event.title) == normalize_title(candidate.title)


def _material_update_id(event: Event, candidate: CandidateArticle, state: str) -> str:
    candidate_key = candidate.signal_external_id or candidate.candidate_id or candidate.id
    canonical = f"{event.event_id}|{candidate_key}|{state}"
    return f"upd_{uuid5(NAMESPACE_URL, canonical)}"


def _delta_fields(
    event: Event, candidate: CandidateArticle, occurred_at: datetime, state: str
) -> list[str]:
    fields: list[str] = []
    if normalize_title(event.title) != normalize_title(candidate.title):
        fields.append("title")
    if event.latest_event_time != occurred_at:
        fields.append("latest_event_time")
    if event.primary_candidate_id != candidate.candidate_id:
        fields.append("primary_candidate_id")
    if event.state != state:
        fields.append("state")
    if event.candidate_ids != [candidate.candidate_id]:
        fields.append("candidate_ids")
    return fields


_GENERIC_TERMS = {
    "a",
    "an",
    "and",
    "ai",
    "announced",
    "announce",
    "announcement",
    "approved",
    "approval",
    "agreement",
    "buyout",
    "company",
    "complete",
    "completed",
    "confirm",
    "confirmed",
    "confirmation",
    "clearance",
    "closed",
    "deal",
    "firm",
    "group",
    "inc",
    "industry",
    "ltd",
    "launch",
    "launched",
    "merger",
    "official",
    "officially",
    "publish",
    "published",
    "publication",
    "regulation",
    "regulatory",
    "release",
    "released",
    "rumor",
    "said",
    "service",
    "sign",
    "signed",
    "signs",
    "takeover",
    "the",
    "tech",
    "technology",
    "unveil",
    "unveiled",
    "acquire",
    "acquired",
    "acquisition",
    "authorize",
    "authorized",
    "buy",
    "buys",
    "purchase",
    "platform",
    "product",
    "software",
    "system",
    "market",
    "business",
    "收购",
    "并购",
    "合并",
    "公司",
    "集团",
    "宣布",
    "公告",
    "发布",
    "推出",
    "上线",
    "公布",
    "确认",
    "正式",
    "签署",
    "协议",
    "批准",
    "监管",
    "许可",
    "通过",
    "完成",
    "传闻",
    "消息",
    "表示",
    "称",
    "部门",
    "将",
    "拟",
    "已",
    "被",
    "与",
    "对",
}
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[\u3400-\u9fff]+")


def _identity_terms(title: str) -> list[str]:
    terms: list[str] = []
    for token in _WORD_RE.findall(title.casefold()):
        if token.isascii() and (token in _GENERIC_TERMS or len(token) < 2):
            continue
        if not token.isascii():
            cleaned = token
            for generic in sorted(
                (term for term in _GENERIC_TERMS if not term.isascii()),
                key=len,
                reverse=True,
            ):
                cleaned = cleaned.replace(generic, "")
            if len(cleaned) < 2:
                continue
            token = cleaned
        if token not in terms:
            terms.append(token)
    return terms


def _has_confirmation_marker(title: str) -> bool:
    tokens = _WORD_RE.findall(title)
    if any(
        token.startswith(marker)
        for token in tokens
        for marker in (
            "official",
            "confirm",
            "sign",
            "agreement",
            "announc",
            "approv",
            "authoriz",
            "clearance",
            "complet",
            "clos",
            "finaliz",
            "definitive",
        )
    ):
        return True
    return any(
        marker in title
        for marker in (
            "正式",
            "确认",
            "签署",
            "协议",
            "公告",
            "批准",
            "许可",
            "监管",
            "通过",
            "完成",
        )
    )


def _event_type(title: str) -> str:
    for markers, event_type in (
        (("acquisition", "acquire", "acquired", "takeover", "buyout", "merger", "收购", "并购", "合并"), "acquisition"),
        (("launch", "release", "unveil", "publish", "publication", "发布", "推出", "上线", "公布"), "release"),
        (("approve", "approval", "regulation", "regulatory", "authorize", "clearance", "监管", "批准", "许可", "通过"), "regulation"),
    ):
        if any(marker in title for marker in markers):
            return event_type
    return "news"
