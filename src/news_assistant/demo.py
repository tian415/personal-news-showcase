"""Completely offline, synthetic end-to-end product demonstration."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .delivery_recovery import reconcile_publications
from .feedback_store import import_feedback_file
from .layered import LayeredSelectionResult, run_layered_selection
from .layered_report import evaluate_layers
from .models import (
    Article,
    CandidateArticle,
    CoverageDomain,
    EnrichedArticle,
    EvidenceRole,
    Preferences,
    SourceTier,
    Topic,
)
from .publication import audit_public_directory
from .source_registry import RegisteredSource, SourceRegistry
from .state import StateStore


class _DemoEnricher:
    """Deterministic enrichment that never invokes an external provider."""

    def enrich(self, item: Article) -> EnrichedArticle:
        return EnrichedArticle(
            **item.model_dump(),
            chinese_title=item.title,
            chinese_summary=f"合成演示摘要：{item.summary}",
            key_points=["仅用于验证产品流程", "内容与来源均为合成演示数据"],
            tags=["演示数据", item.topic.value],
            beginner_context="这是完全离线的合成场景，不代表真实新闻或线上投递。",
        )


def run_demo(output_dir: Path, *, now: datetime | None = None) -> dict:
    """Run a two-round local demo without network, paid APIs, or real delivery."""

    # Imported here so CLI can expose this function without creating an import cycle.
    from .cli import _candidate_from_article, build_digest

    root = Path(output_dir)
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    _prepare_empty_output(root)

    public = root / "public"
    public_next = root / "public-next"
    published = root / "published"
    state_dir = root / "private-state"
    email_output = root / "email.html"
    feedback_path = root / "simulation-feedback.json"
    articles, registry = _demo_inputs(reference)
    preferences = Preferences()

    first = build_digest(
        articles,
        preferences,
        _DemoEnricher(),
        public,
        email_output=email_output,
        digest_base_url="https://example.com/offline-demo",
        selection_engine="layered_shadow_v1",
        state_dir=state_dir,
        source_registry=registry,
        run_id="demo-round-1",
        now=reference,
        demo_mode=True,
    )
    store = StateStore(state_dir)
    first_feedback_effect = json.loads(
        (store._find_run_directory("demo-round-1") / "feedback-effect.json").read_text()
    )

    # This local byte copy represents publication only inside the demo root.
    shutil.copytree(public, published)
    recovered = reconcile_publications(store, published, now=reference)
    if recovered != ["demo-round-1"]:
        raise RuntimeError("simulated publication did not match the first-round page hash")

    refs = [item.event_ref for item in first.articles if item.event_ref]
    if len(refs) < 2:
        raise RuntimeError("demo requires at least two published event references")
    payload = {
        "schema_version": "news-feedback-v2",
        "events": [
            {
                "event_ref": refs[0],
                "vote": "helpful",
                "reason": None,
                "tracking": False,
                "updated_at": reference.isoformat(),
            },
            {
                "event_ref": refs[1],
                "vote": None,
                "reason": None,
                "tracking": True,
                "updated_at": reference.isoformat(),
            },
        ],
    }
    StateStore._atomic_write(
        feedback_path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
    )
    feedback = import_feedback_file(feedback_path, state_dir, now=reference)
    feedback_state_path = state_dir / "feedback-state.json"
    first_feedback_hash = _sha256(feedback_state_path)
    import_feedback_file(feedback_path, state_dir, now=reference)
    if _sha256(feedback_state_path) != first_feedback_hash:
        raise RuntimeError("simulated feedback replay was not idempotent")

    second_time = reference + timedelta(minutes=1)
    helpful_article = next(item for item in articles if item.id == first.articles[0].id)
    private_candidate = _private_feedback_probe(helpful_article, second_time)
    store.write_signal_collection(
        second_time,
        snapshots=[],
        health_records=[],
        candidates=[private_candidate],
    )
    historical = store.load_recent_events(second_time)
    control = run_layered_selection(
        [
            *(
                _candidate_from_article(item, second_time, registry)
                for item in articles
            ),
            private_candidate,
        ],
        historical,
        Preferences(),
        snapshots=store.load_signal_snapshots(second_time),
        now=second_time,
    )
    control_probe = _find_layered_candidate(control, private_candidate.candidate_id)
    second = build_digest(
        articles,
        preferences,
        _DemoEnricher(),
        public_next,
        selection_engine="layered_shadow_v1",
        state_dir=state_dir,
        source_registry=registry,
        run_id="demo-round-2",
        now=second_time,
        demo_mode=True,
    )
    first_ids = [item.id for item in first.articles]
    second_ids = [item.id for item in second.articles]
    feedback_effect_path = (
        store._find_run_directory("demo-round-2") / "feedback-effect.json"
    )
    feedback_effect = json.loads(feedback_effect_path.read_text())
    actual_layered = LayeredSelectionResult.model_validate(
        json.loads(
            (store._find_run_directory("demo-round-2") / "layered-shadow.json").read_text()
        )
    )
    actual_probe = _find_layered_candidate(actual_layered, private_candidate.candidate_id)
    before = control_probe.event.scores.relevance
    after = actual_probe.event.scores.relevance
    if after <= before:
        raise RuntimeError("simulated helpful feedback did not increase probe relevance")
    for rendered in (public / "index.html", public_next / "index.html"):
        if private_candidate.title in rendered.read_text():
            raise RuntimeError("private feedback probe leaked into public demo output")
    evaluation = evaluate_layers(state_dir, now=second_time)

    audited = {
        "public": len(audit_public_directory(public)),
        "public_next": len(audit_public_directory(public_next)),
        "published": len(audit_public_directory(published)),
    }
    report = {
        "schema_version": "offline-demo-v1",
        "labels": {
            "content": "mock",
            "feedback": "simulated",
            "publication": "simulated_local_copy",
        },
        "counts": {
            "input_articles": len(articles),
            "round_1_selected": len(first.articles),
            "round_2_selected": len(second.articles),
            "feedback_records": len(feedback.events),
            "tracked_events": len(feedback.tracked_event_ids),
            "confirmed_simulated_publications": len(recovered),
        },
        "stages": {
            "round_1_build": "completed",
            "simulated_publication": "hash_confirmed",
            "feedback_import": "idempotent",
            "round_2_build": "completed",
            "email_preview": "rendered_not_sent",
            "public_audit": "passed",
            "layered_evaluation": evaluation["release_readiness"]["status"],
        },
        "legacy_selection_unchanged": first_ids == second_ids,
        "feedback_effect": feedback_effect,
        "feedback_effect_change": {
            "round_1_feedback_records": first_feedback_effect["feedback_records"],
            "round_2_feedback_records": feedback_effect["feedback_records"],
            "effective_preferences_changed": (
                first_feedback_effect["effective_preferences"]
                != feedback_effect["effective_preferences"]
            ),
            "tracked_events_delta": (
                len(feedback_effect["tracked_event_ids"])
                - len(first_feedback_effect["tracked_event_ids"])
            ),
        },
        "feedback_scoring_probe": {
            "visibility": "private_demo_report",
            "candidate_id": private_candidate.candidate_id,
            "event_id": actual_probe.event.event_id,
            "control_relevance_before": before,
            "actual_relevance_after": after,
            "proof_scope": "feedback_entered_scoring_not_recommendation_quality",
        },
        "evaluation": evaluation,
        "layer_coverage": {
            "counts": evaluation["layer_counts"],
            "empty_layers": sorted(
                layer
                for layer, count in evaluation["layer_counts"].items()
                if count == 0
            ),
        },
        "public_audit_file_counts": audited,
        "artifacts": {
            "visibility": "private_demo_root",
            "public_round_1": "public/index.html",
            "public_round_2": "public-next/index.html",
            "simulated_published_copy": "published/index.html",
            "private_state": "private-state/",
            "email_preview": "email.html",
            "simulated_feedback": "simulation-feedback.json",
            "report": "report.json",
        },
        "limitations": [
            "所有文章、来源、摘要和反馈均为合成演示数据。",
            "published 目录只是本地精确字节副本，不代表真实线上发布或邮件投递。",
            "反馈由程序模拟，不代表真实用户意见。",
            "分层评估仍为 not_ready，尚缺人工事实核验、召回证据和持续观察。",
            "演示不保证三个内容层均有样本，空层应按报告如实解释。",
        ],
    }
    StateStore._atomic_write(
        root / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def _prepare_empty_output(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("demo output must not be a symlink")
    if root.exists():
        if not root.is_dir():
            raise ValueError("demo output must be a directory")
        if any(root.iterdir()):
            raise ValueError("demo output must be empty; refusing to overwrite non-empty output")
        return
    root.mkdir(parents=True)


def _demo_inputs(now: datetime) -> tuple[list[Article], SourceRegistry]:
    specs = [
        ("ai-1", "合成演示：模型实验室发布离线推理工具", Topic.AI, CoverageDomain.AI_TECHNOLOGY),
        ("ai-2", "合成演示：芯片团队发布节能计算架构", Topic.TECH, CoverageDomain.AI_TECHNOLOGY),
        ("ai-3", "合成演示：研究机构批准诊断模型试验", Topic.SCIENCE, CoverageDomain.SCIENCE_HEALTH),
        ("ai-4", "合成演示：开源社区发布数据检查套件", Topic.AI, CoverageDomain.AI_TECHNOLOGY),
        ("ai-5", "合成演示：工程团队发布隐私计算方案", Topic.TECH, CoverageDomain.AI_TECHNOLOGY),
        ("biz-1", "合成演示：零售样本发布季度经营数据", Topic.BUSINESS, CoverageDomain.BUSINESS_MACRO),
        ("biz-2", "合成演示：市场样本发布供应链观察", Topic.MARKETS, CoverageDomain.BUSINESS_MACRO),
        ("biz-3", "合成演示：企业样本批准研发预算", Topic.BUSINESS, CoverageDomain.BUSINESS_MACRO),
        ("exp-1", "合成演示：材料团队发布低能耗实验", Topic.EXPLORATION, CoverageDomain.EXPLORATION),
        ("exp-2", "合成演示：海洋团队发布传感研究", Topic.EXPLORATION, CoverageDomain.EXPLORATION),
    ]
    articles: list[Article] = []
    sources: list[RegisteredSource] = []
    for index, (identifier, title, topic, domain) in enumerate(specs, start=1):
        source = f"演示来源{index}"
        articles.append(
            Article(
                id=identifier,
                title=title,
                url=f"https://example.com/offline-demo/{identifier}",
                source=source,
                language="zh",
                published_at=now - timedelta(minutes=index),
                topic=topic,
                summary="该条目用于离线验证选择、渲染、模拟发布和反馈闭环。",
                quality_score=0.9,
                importance_score=0.9 - index / 100,
            )
        )
        sources.append(
            RegisteredSource(
                id=f"demo-source-{index}",
                name=source,
                tier=SourceTier.T0,
                adapter="demo",
                url=f"https://example.com/offline-demo/source-{index}.xml",
                topics=[topic],
                coverage_domains=[domain],
                publisher_group=f"demo-publisher-{index}",
                language="zh",
                enabled=False,
                production_enabled=False,
            )
        )
    return articles, SourceRegistry(version="source-registry-v1", sources=sources)


def _private_feedback_probe(article: Article, now: datetime) -> CandidateArticle:
    return CandidateArticle(
        **article.model_dump(
            exclude={"id", "title", "url", "summary", "published_at", "importance_score"}
        ),
        id="private-feedback-probe",
        candidate_id="private-feedback-probe",
        title="合成演示：私有候选发布反馈验证工具",
        url="https://example.com/offline-demo/private-feedback-probe",
        summary="仅进入私有影子选择，用于比较反馈前后的相关性分数。",
        published_at=now,
        importance_score=0.7,
        source_id="demo-source-1",
        source_tier=SourceTier.T0,
        evidence_role=EvidenceRole.PRIMARY_FACT,
        publisher_group="demo-publisher-1",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        canonical_url="https://example.com/offline-demo/private-feedback-probe",
        discovered_at=now,
        event_time=now,
        adapter_version="demo-private-v1",
    )


def _find_layered_candidate(
    result: LayeredSelectionResult, candidate_id: str
):
    for item in result.items:
        if candidate_id in item.event.candidate_ids:
            return item
    raise RuntimeError(f"private layered candidate was not selected: {candidate_id}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
