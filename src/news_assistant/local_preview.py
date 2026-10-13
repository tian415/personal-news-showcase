"""Render an explicitly selected, content-reviewed local edition and private audit."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .article_content import captured_contents, write_content_snapshot
from .background_readings import select_background_readings
from .coverage_evaluation import _AuditSnapshot, _aware, _fingerprint
from .deepseek import fallback_enrichment
from .delivery_recovery import record_publication
from .feedback import FeedbackState, public_event_ref
from .models import CoverageDomain, Digest, Preferences, RunRecord
from .news_value import value_display
from .publication import audit_public_directory, validate_public_output
from .render import daily_digest_url, render_digest, render_email_digest
from .reviewed_selection import select_reviewed
from .selection import BUSINESS_TOPICS, CORE_TOPICS, select_digest
from .state import StateStore


def build_reviewed_preview(
    audit: dict, review: dict, preferences: Preferences, output_dir: Path, state_dir: Path,
    *, now: datetime, run_id: str, include_shadow: bool = False,
    email_base_url: str | None = None, value_review: dict | None = None,
    background_readings: dict | None = None,
) -> dict:
    now = _aware(now)
    if email_base_url is not None:
        daily_digest_url(email_base_url, now.astimezone(ZoneInfo("Asia/Shanghai")).date())
    validate_public_output(output_dir)
    if (state_dir.resolve().is_relative_to(output_dir.resolve())
            or output_dir.resolve().is_relative_to(state_dir.resolve())):
        raise ValueError("private state and public output must stay outside each other")
    run = RunRecord(run_id=run_id, started_at=now, selection_engine="reviewed_local_v1")
    store = StateStore(state_dir)
    if store._find_run_directory(run_id, required=False):
        raise ValueError("run_id already exists; use a new run_id")
    snapshot = _AuditSnapshot.model_validate(audit)
    feedback_path = state_dir / "local-feedback-state.json"
    feedback = (FeedbackState.model_validate_json(feedback_path.read_text())
                if feedback_path.exists() else FeedbackState())
    effective = Preferences(
        topic_weights={**preferences.topic_weights, **{
            topic: preferences.topic_weights.get(topic, 1.0) * weight
            for topic, weight in feedback.preferences.topic_weights.items()}},
        source_weights={**preferences.source_weights, **{
            source: preferences.source_weights.get(source, 1.0) * weight
            for source, weight in feedback.preferences.source_weights.items()}},
    )
    history = store.load_recent_events(now, channel="local")
    baseline = select_reviewed(audit, review, preferences, now=now,
                               historical_events=history, include_shadow=include_shadow, value_review=value_review)
    selection = select_reviewed(audit, review, effective, now=now,
                               historical_events=history, include_shadow=include_shadow,
                               tracked_event_ids=feedback.tracked_event_ids, value_review=value_review)
    readings = select_background_readings(
        background_readings, now=now, store=store,
        news_urls=[str(article.url) for article in selection.selected],
    )
    refs = {
        candidate_id: public_event_ref(event.event_id, run_id=run_id,
                                       update_id=event.latest_update_id)
        for event in selection.events for candidate_id in event.candidate_ids
    }
    contents = captured_contents(audit, snapshot)
    review_items = {item["candidate_id"]: item for item in review["items"]}
    topic_counts = {
        "core": sum(a.topic in CORE_TOPICS for a in selection.selected),
        "business": sum(a.topic in BUSINESS_TOPICS for a in selection.selected),
        "exploration": sum(a.topic.value == "exploration" for a in selection.selected),
    }
    pending = sum(item["status"] == "pending" for item in review_items.values())
    pending_by_group = {
        "core": sum(i["status"] == "pending" and i["topic"] in CORE_TOPICS for i in review_items.values()),
        "business": sum(i["status"] == "pending" and i["topic"] in BUSINESS_TOPICS for i in review_items.values()),
        "exploration": sum(i["status"] == "pending" and i["topic"] == "exploration" for i in review_items.values()),
    }
    coverage = {"total": len(review_items), "pending": pending,
                "pending_by_group": pending_by_group,
                "selected_by_group": topic_counts}
    coverage_notes = [
        f"候选 {len(review_items)} 条 · 待审 {pending} 条；当前结果仅代表已通过审阅的材料。",
        (f"科技／科学 {topic_counts['core']}/5 · 商业／市场 {topic_counts['business']}/3 · "
        f"探索 {topic_counts['exploration']}/2（普通条目目标）"),
        (f"待审分布：科技／科学 {pending_by_group['core']} · 商业／市场 {pending_by_group['business']} · "
        f"探索 {pending_by_group['exploration']}。缺少条目不代表该领域没有新闻。"),
        "主题不足时不以其他主题补位；重大事件和已追踪的重要更新仍优先保留。",
    ]
    discovery_plan = None
    if value_review is not None:
        selected_domains = {a.coverage_domain for a in selection.selected}
        missing_domains = [d.value for d in CoverageDomain if d not in selected_domains]
        shortfall = max(0, 10 - len(selection.selected))
        coverage.update(target_min=10, target_max=15, shortfall=shortfall,
                        missing_domains=missing_domains)
        discovery_plan = {
            "needed": shortfall, "priority_domains": missing_domains,
            "pending_candidate_ids": [identifier for identifier, reason in selection.excluded.items()
                                      if reason in {"pending_review", "pending_news_value"}],
            "source_ids": [s['id'] for s in audit['sources']
                           if (s['production_enabled'] or include_shadow)
                           and any(d in missing_domains for d in s['coverage_domains'])],
            "next_action": "supplement_discovery_and_review" if shortfall else "review_coverage",
            "automatic_paid_calls": False,
        }
        domain_labels = {"ai_technology": "AI 与科技", "business_macro": "商业与宏观",
                         "science_health": "科学与健康", "policy_public_affairs": "政策与公共事务",
                         "exploration": "探索"}
        coverage_notes = [
            "覆盖缺口：" + ("、".join(domain_labels[d] for d in missing_domains)
                           if missing_domains else "所列领域均有条目，仍可能漏报") + "。",
            f"每日目标 10–15 条不同事件 · 本期 {len(selection.selected)} 条 · 距最低目标缺 {shortfall} 条。",
            "按事件时效、重要影响及可核对关注信号选文；主题不设固定名额，审核通过不是入选理由。",
            f"候选 {len(review_items)} 条 · 待审 {pending} 条；新闻时间与影响是具名编辑判断，尚非独立事实核验。",
            ("数量不足，优先补充发现并审核缺失领域；不以旧闻或低价值稿凑数。" if shortfall else
             "仍需检查跨领域漏报；达到数量目标不表示完整覆盖。"),
        ]
    else:
        coverage_notes.append("兼容审阅预览：尚未按事件时间与新闻价值完成审核，不代表完整日报。")
    digest = Digest(
        digest_date=now.astimezone(ZoneInfo("Asia/Shanghai")).date(), local_preview=True,
        coverage_notes=coverage_notes,
        background_readings=readings,
        articles=[fallback_enrichment(article).model_copy(update={
            **({"chinese_title": review_items[article.id]["chinese_title"],
                "chinese_summary": review_items[article.id]["chinese_summary"],
                "beginner_context": "",
                "degraded": False}
               if review_items[article.id].get("chinese_title") else {}),
            "event_ref": refs[article.id],
            "news_value": value_display(selection.value_details[article.id]) if value_review is not None else None,
            "key_points": [review_items[article.id]["evidence_excerpt"]]
            if article.id in contents and contents[article.id].status == "extracted" else [],
            "claim_type": "opinion" if review_items[article.id]["fact_status"] == "opinion" else "fact",
        })
                  for article in selection.selected],
    )
    selected_ids = [article.id for article in selection.selected]
    before_ids = [article.id for article in baseline.selected]
    legacy = select_digest([
        article.model_copy(update={"id": f"{source.id}:{article.id}"})
        for source in snapshot.sources if source.production_enabled
        for article in source.candidates
        if 0 <= (now - article.published_at).total_seconds() <= 36 * 3600
    ], preferences, now=now)
    report = {
        "schema_version": "reviewed-local-preview-v1", "run_id": run_id,
        "selection_engine": "reviewed_local_v1", "include_shadow": include_shadow,
        "input_fingerprints": {"snapshot_sha256": _fingerprint(audit), "review_sha256": _fingerprint(review)},
        "selected_ids": selected_ids, "legacy_selected_ids": [a.id for a in legacy],
        "background_reading_count": len(readings),
        "background_reading_urls": [str(item.url) for item in readings],
        "without_feedback_ids": before_ids,
        "feedback_changed_selection": selected_ids != before_ids,
        "feedback_records": len(feedback.events),
        "tracked_event_ids": feedback.tracked_event_ids,
        "effective_preferences": effective.model_dump(mode="json"),
        "coverage": coverage, "discovery_plan": discovery_plan,
        "news_value_policy": "news-value-v1" if value_review is not None else "evidence-preview-legacy",
        "news_value_details": selection.value_details,
        "excluded": selection.excluded, "review_reasons": selection.reasons,
        "protected_major_event_ids": [event.event_id for event in selection.events if event.major],
        "delivery_status": "not_confirmed", "release_readiness": "not_ready",
        "limitations": [
            "Content scores and classifications are attributed reviewer judgments, not measured quality.",
            "A matching quotation proves attribution, not factual truth or a complete evidence chain.",
            "Conservative title-based event grouping is not complete multilingual event resolution.",
            "Local feedback affects this edition; production and web/email delivery remain unchanged.",
        ],
    }
    store.write_run(run, selection.candidates, [], selection.events)
    store.write_private_report(run, "reviewed-selection.json", report)
    store.write_private_report(run, "editorial-review.json", review)
    if background_readings is not None:
        store.write_private_report(run, "background-readings.json", background_readings)
    if value_review is not None:
        store.write_private_report(run, "news-value-review.json", value_review)
        store.write_private_report(run, "discovery-plan.json", discovery_plan)
    if email_base_url is not None:
        email_path = store._find_run_directory(run_id) / "email-preview.html"
        write_content_snapshot(email_path, "")
        render_email_digest(digest, email_path, email_base_url)
    page = render_digest(digest, output_dir)
    audit_public_directory(output_dir)
    record_publication(store, run, digest.digest_date, selected_ids, page)
    return report
