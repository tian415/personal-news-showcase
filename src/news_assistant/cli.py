from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import httpx
import yaml

from . import credential_store
from .coverage_evaluation import evaluate_coverage
from .deepseek import DeepSeekClient, fallback_enrichment
from .delivery_recovery import reconcile_publications, record_publication
from .discovery.v2ex import fetch_v2ex_topics
from .emailer import EmailSettings, send_html
from .feedback import FeedbackState, public_event_ref
from .feedback_store import import_feedback_file
from .feeds import FeedSource, _http_loader, fetch_articles
from .history import preserve_history
from .layered import run_layered_selection
from .layered_report import write_layered_evaluation
from .models import (
    Article,
    CandidateArticle,
    CoverageDomain,
    Digest,
    EnrichedArticle,
    EvidenceRole,
    Preferences,
    RunRecord,
    SignalSnapshot,
    SourceHealthRecord,
    SourceTier,
    Topic,
)
from .publication import audit_public_directory, validate_public_output
from .render import daily_digest_url, render_archive, render_digest, render_email_digest
from .selection import select_digest
from .shadow import run_shadow_selection
from .source_audit import audit_sources
from .source_registry import SourceRegistry, load_source_registry
from .state import StateStore
from .state_report import write_state_evaluation


class Enricher(Protocol):
    def enrich(self, item: Article) -> EnrichedArticle: ...


class SourceOnlyEnricher:
    """Explicitly render source summaries without invoking a paid model."""

    enrich = staticmethod(fallback_enrichment)


def _validate_build_options(
    output_dir: Path,
    *,
    email_output: Path | None,
    digest_base_url: str | None,
    selection_engine: str,
    state_dir: Path | None,
    delivery_manifest: Path | None,
    run_id: str | None = None,
) -> None:
    validate_public_output(output_dir)
    if (email_output is None) != (digest_base_url is None):
        raise ValueError("email_output and digest_base_url must be provided together")
    if selection_engine not in {"legacy", "shadow_event_v1", "layered_shadow_v1"}:
        raise ValueError("unknown selection_engine")
    if selection_engine != "legacy" and state_dir is None:
        raise ValueError("state_dir is required for shadow selection")
    if run_id is not None:
        RunRecord(run_id=run_id, started_at=datetime.now(UTC), selection_engine=selection_engine)
    if digest_base_url is not None:
        daily_digest_url(digest_base_url, datetime.now(UTC).date())
    public_root = output_dir.resolve()
    for label, path in (
        ("email_output", email_output),
        ("delivery_manifest", delivery_manifest),
        ("state_dir", state_dir),
    ):
        if path is None:
            continue
        private_path = path.resolve()
        if private_path.is_relative_to(public_root):
            raise ValueError(f"{label} must be outside public output")
        if public_root.is_relative_to(private_path):
            raise ValueError(f"public output must be outside {label}")
    if email_output and delivery_manifest and email_output.resolve() == delivery_manifest.resolve():
        raise ValueError("email_output and delivery_manifest must be distinct")


def build_digest(articles: list[Article], preferences: Preferences, enricher: Enricher, output_dir: Path, *, digest_date: date | None = None, email_output: Path | None = None, digest_base_url: str | None = None, selection_engine: str = "legacy", state_dir: Path | None = None, source_registry: SourceRegistry | None = None, run_id: str | None = None, delivery_manifest: Path | None = None, now: datetime | None = None, demo_mode: bool = False) -> Digest:
    build_time = now or datetime.now(UTC)
    if build_time.tzinfo is None or build_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    _validate_build_options(
        output_dir, email_output=email_output, digest_base_url=digest_base_url,
        selection_engine=selection_engine, state_dir=state_dir,
        delivery_manifest=delivery_manifest, run_id=run_id,
    )
    unique_articles: dict[str, Article] = {}
    for article in articles:
        previous = unique_articles.get(article.id)
        if previous is not None and previous != article:
            raise ValueError(f"conflicting article id: {article.id}")
        unique_articles[article.id] = article
    articles = list(unique_articles.values())
    store = StateStore(state_dir) if state_dir and selection_engine != "legacy" else None
    historical = store.load_recent_events(build_time) if store else []
    candidates = [_candidate_from_article(item, build_time, source_registry) for item in articles]
    shadow = run_shadow_selection(candidates, historical, preferences, now=build_time) if store else None
    run = RunRecord(
        run_id=run_id or build_time.strftime("%Y%m%dT%H%M%S%fZ"),
        started_at=build_time, selection_engine=selection_engine,
    ) if store else None
    if store and store._find_run_directory(run.run_id, required=False):
        raise ValueError("run_id already exists; use a new run_id for each build")
    selected = select_digest(articles, preferences, now=build_time)
    publication_events = list(shadow.events) if shadow else []
    if shadow:
        represented = {i for event in publication_events for i in event.candidate_ids}
        old = {event.event_id: event for event in historical}
        duplicate_ids: dict[str, list[str]] = {}
        for decision in shadow.decisions:
            if (decision.candidate_id and decision.candidate_id not in represented
                    and decision.event_id in old):
                duplicate_ids.setdefault(decision.event_id, []).append(decision.candidate_id)
        for event_id, identifiers in duplicate_ids.items():
            publication_events.append(old[event_id].model_copy(update={
                "candidate_ids": sorted(set(identifiers)), "primary_candidate_id": identifiers[0],
            }))
    event_refs = {
        identifier: public_event_ref(event.event_id, run_id=run.run_id,
                                     update_id=event.latest_update_id)
        for event in publication_events for identifier in event.candidate_ids
    }
    reports = {}
    if store and selection_engine == "layered_shadow_v1":
        feedback_path = state_dir / "feedback-state.json"
        feedback = (FeedbackState.model_validate_json(feedback_path.read_text())
                    if feedback_path.exists() else FeedbackState())
        shadow_preferences = preferences.model_copy(update={
            "topic_weights": {**preferences.topic_weights, **{
                k: min(2.0, max(0.5, preferences.topic_weights.get(k, 1.0) * v))
                for k, v in feedback.preferences.topic_weights.items()}},
            "source_weights": {**preferences.source_weights, **{
                k: min(2.0, max(0.5, preferences.source_weights.get(k, 1.0) * v))
                for k, v in feedback.preferences.source_weights.items()}},
        })
        layered = run_layered_selection(
            [*candidates, *store.load_recent_signal_candidates(build_time)],
            historical, shadow_preferences, snapshots=store.load_signal_snapshots(build_time),
            now=build_time, tracked_event_ids=feedback.tracked_event_ids,
        )
        reports["layered-shadow.json"] = layered.comparison(item.id for item in selected)
        reports["feedback-effect.json"] = {
            "feedback_records": len(feedback.events),
            "effective_preferences": shadow_preferences.model_dump(mode="json"),
            "tracked_event_ids": feedback.tracked_event_ids,
            "public_selection": "legacy_unchanged",
        }
    local_date = build_time.astimezone(ZoneInfo("Asia/Shanghai")).date()
    digest = Digest(digest_date=digest_date or local_date, demo_mode=demo_mode, articles=[
        enricher.enrich(item).model_copy(update={"event_ref": event_refs.get(item.id)})
        for item in selected
    ])
    if store:
        store.write_run(run, candidates, shadow.decisions, publication_events)
        for name, report in reports.items():
            store.write_private_report(run, name, report)
    render_digest(digest, output_dir)
    if email_output and digest_base_url:
        render_email_digest(digest, email_output, digest_base_url)
    audit_public_directory(output_dir)
    if store:
        record_publication(store, run, digest.digest_date, [item.id for item in selected],
                           output_dir / f"{digest.digest_date.isoformat()}.html")
    if delivery_manifest:
        StateStore._atomic_write(delivery_manifest, json.dumps([item.id for item in selected]))
    return digest


def _candidate_from_article(article: Article, discovered_at: datetime, registry: SourceRegistry | None = None) -> CandidateArticle:
    domain = {
        Topic.AI: CoverageDomain.AI_TECHNOLOGY,
        Topic.TECH: CoverageDomain.AI_TECHNOLOGY,
        Topic.SCIENCE: CoverageDomain.SCIENCE_HEALTH,
        Topic.BUSINESS: CoverageDomain.BUSINESS_MACRO,
        Topic.MARKETS: CoverageDomain.BUSINESS_MACRO,
        Topic.EXPLORATION: CoverageDomain.EXPLORATION,
    }[article.topic]
    registered = next((source for source in (registry.sources if registry else []) if source.name == article.source), None)
    return CandidateArticle(
        **article.model_dump(),
        candidate_id=article.id,
        source_id=registered.id if registered else article.source.casefold().replace(" ", "-"),
        source_tier=registered.tier if registered else SourceTier.T2,
        evidence_role=(
            registered.evidence_role if registered else EvidenceRole.EXPERT_ANALYSIS
        ),
        publisher_group=registered.publisher_group if registered else article.source,
        coverage_domain=registered.coverage_domain if registered else domain,
        canonical_url=article.url,
        discovered_at=discovered_at,
    )


def _load_sources(path: Path) -> list[FeedSource]:
    data = yaml.safe_load(path.read_text())
    return [FeedSource.model_validate(item) for item in data["sources"]]


def _load_preferences(path: Path | None) -> Preferences:
    return Preferences.model_validate(json.loads(path.read_text())) if path else Preferences()


def _email_settings_from_env() -> EmailSettings:
    return EmailSettings(
        gmail_address=os.environ["GMAIL_ADDRESS"],
        gmail_app_password=os.environ["GMAIL_APP_PASSWORD"],
        recipient=os.environ["NEWS_RECIPIENT"],
    )


def _send_existing_digest(input_path: Path, digest_date: date, digest_url: str) -> None:
    send_html(
        input_path.read_text(),
        f"今日新闻合集：{daily_digest_url(digest_url, digest_date)}",
        digest_date,
        _email_settings_from_env(),
    )


def collect_signals(registry: SourceRegistry, state_dir: Path, *, now: datetime | None = None) -> None:
    captured_at = now or datetime.now(UTC)
    snapshots = []
    health_records = []
    signal_candidates = []
    for source in registry.enabled_sources("rss"):
        if source.production_enabled:
            continue
        try:
            articles = fetch_articles(
                [source.as_feed_source()], loader=_http_loader, now=captured_at,
                raise_on_error=True,
            )
        except (OSError, TimeoutError, ValueError, TypeError, httpx.HTTPError) as error:
            health_records.append(SourceHealthRecord(
                source_id=source.id, captured_at=captured_at, status="error",
                candidate_count=0, error_type=type(error).__name__,
            ))
        else:
            signal_candidates.extend(
                _candidate_from_article(article, captured_at, registry).model_copy(update={
                    "candidate_id": f"{source.id}:{article.id}", "signal_external_id": article.id,
                }) for article in articles
            )
            snapshots.extend(SignalSnapshot(
                source_id=source.id, external_id=article.id, captured_at=captured_at, rank=rank,
            ) for rank, article in enumerate(articles, start=1))
            health_records.append(SourceHealthRecord(
                source_id=source.id, captured_at=captured_at, status="ok",
                candidate_count=len(articles),
            ))
    for source in registry.enabled_sources("v2ex"):
        try:
            result = fetch_v2ex_topics(source, loader=_http_loader, now=captured_at)
        except (OSError, TimeoutError, ValueError, TypeError, httpx.HTTPError) as error:
            health_records.append(
                SourceHealthRecord(
                    source_id=source.id,
                    captured_at=captured_at,
                    status="error",
                    candidate_count=0,
                    error_type=type(error).__name__,
                )
            )
        else:
            signal_candidates.extend(
                _candidate_from_article(article, captured_at, registry).model_copy(update={
                    "candidate_id": f"{source.id}:{article.id}",
                    "signal_external_id": article.id.removeprefix(f"{source.id}:"),
                }) for article in result.articles
            )
            snapshots.extend(result.snapshots)
            health_records.append(
                SourceHealthRecord(
                    source_id=source.id,
                    captured_at=captured_at,
                    status="ok",
                    candidate_count=len(result.articles),
                )
            )
    StateStore(state_dir).write_signal_collection(captured_at, snapshots, health_records, signal_candidates)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize an independent local workspace")
    init_parser.add_argument("--directory", type=Path, required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument(
        "--sources", type=Path, default=Path("config/source_registry.yaml")
    )
    build_parser.add_argument("--preferences", type=Path)
    build_parser.add_argument("--source-only", action="store_true",
                              help="Fetch public RSS and use source summaries without model calls")
    build_parser.add_argument("--output", type=Path, default=Path("public"))
    build_parser.add_argument("--email-output", type=Path)
    build_parser.add_argument("--digest-url")
    build_parser.add_argument(
        "--selection-engine", choices=["legacy", "shadow_event_v1", "layered_shadow_v1"], default="legacy"
    )
    build_parser.add_argument("--state-dir", type=Path)
    build_parser.add_argument("--run-id")
    build_parser.add_argument("--delivery-manifest", type=Path)

    delivered_parser = subparsers.add_parser("mark-delivered")
    delivered_parser.add_argument("--state-dir", type=Path, required=True)
    delivered_parser.add_argument("--run-id", required=True)
    delivered_parser.add_argument("--delivery-manifest", type=Path, required=True)

    reconcile_parser = subparsers.add_parser("reconcile-deliveries")
    reconcile_parser.add_argument("--state-dir", type=Path, required=True)
    reconcile_parser.add_argument("--published", type=Path, required=True)

    signals_parser = subparsers.add_parser("collect-signals")
    signals_parser.add_argument("--sources", type=Path, default=Path("config/source_registry.yaml"))
    signals_parser.add_argument("--state-dir", type=Path, required=True)

    audit_parser = subparsers.add_parser("audit-public")
    audit_parser.add_argument("--output", type=Path, default=Path("public"))

    source_audit_parser = subparsers.add_parser("audit-sources")
    source_audit_parser.add_argument("--sources", type=Path, default=Path("config/source_registry.yaml"))
    source_audit_parser.add_argument("--output", type=Path, required=True)
    source_audit_parser.add_argument("--public-dir", type=Path, default=Path("public"))
    source_audit_parser.add_argument("--max-age-hours", type=int, default=36)
    source_audit_parser.add_argument("--capture-candidates", action="store_true",
                                     help="Save private candidate inputs for offline coverage replay")

    coverage_parser = subparsers.add_parser("evaluate-coverage", help="Replay a reviewed reference set offline")
    coverage_parser.add_argument("--snapshot", type=Path, required=True)
    coverage_parser.add_argument("--reference", type=Path, required=True)
    coverage_parser.add_argument("--preferences", type=Path)
    coverage_parser.add_argument("--output", type=Path, required=True)
    coverage_parser.add_argument("--public-dir", type=Path, default=Path("public"))

    content_parser = subparsers.add_parser(
        "extract-content", help="Fetch public article bodies into a new private snapshot; no model")
    content_parser.add_argument("--snapshot", type=Path, required=True)
    content_parser.add_argument("--output", type=Path, required=True)
    content_parser.add_argument("--public-dir", type=Path, default=Path("public"))
    content_parser.add_argument("--max-articles", type=int, default=20)
    content_parser.add_argument("--dns-mode", choices=("system", "cloudflare"), default="system",
                                help="Explicit HTTPS public DNS; sends domains to Cloudflare")

    review_parser = subparsers.add_parser("prepare-review", help="Prepare a private, unscored editorial worksheet")
    review_parser.add_argument("--snapshot", type=Path, required=True)
    review_parser.add_argument("--output", type=Path, required=True)
    review_parser.add_argument("--public-dir", type=Path, default=Path("public"))

    batches_parser = subparsers.add_parser(
        "prepare-review-batches", help="Split all candidates into bounded private review batches; offline")
    batches_parser.add_argument("--snapshot", type=Path, required=True)
    batches_parser.add_argument("--output-dir", type=Path, required=True)
    batches_parser.add_argument("--max-chars", type=int, default=60000)
    batches_parser.add_argument("--public-dir", type=Path, default=Path("public"))
    merge_parser = subparsers.add_parser(
        "merge-review-batches", help="Validate complete batch reviews and bind to the original snapshot")
    merge_parser.add_argument("--snapshot", type=Path, required=True)
    merge_parser.add_argument("--batch-dir", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--public-dir", type=Path, default=Path("public"))

    subparsers.add_parser("configure-key", help="Store a DeepSeek API key in macOS Keychain; no model call")

    assessment_parser = subparsers.add_parser("assess-review", help="Preflight model review; --execute makes one paid request")
    assessment_parser.add_argument("--snapshot", type=Path, required=True)
    assessment_parser.add_argument("--output", type=Path, required=True)
    assessment_parser.add_argument("--public-dir", type=Path, default=Path("public"))
    assessment_parser.add_argument("--max-candidates", type=int, default=100)
    assessment_parser.add_argument("--model", default="deepseek-v4-flash")
    assessment_mode = assessment_parser.add_mutually_exclusive_group()
    assessment_mode.add_argument("--execute", action="store_true")
    assessment_mode.add_argument("--replay-response", type=Path,
                                 help="Validate a retained response offline without credentials or API calls")

    preview_parser = subparsers.add_parser("preview-reviewed", help="Render a content-reviewed local edition without model calls")
    preview_parser.add_argument("--snapshot", type=Path, required=True)
    preview_parser.add_argument("--review", type=Path, required=True)
    preview_parser.add_argument("--preferences", type=Path)
    preview_parser.add_argument("--value-review", type=Path, help="Bound news-value judgments; enables value-first 10–15 edition")
    preview_parser.add_argument("--background-readings", type=Path, help="Private reviewed reading list; up to 2, outside news quotas")
    preview_parser.add_argument("--output", type=Path, required=True)
    preview_parser.add_argument("--state-dir", type=Path, required=True)
    preview_parser.add_argument("--run-id", required=True)
    preview_parser.add_argument("--include-shadow", action="store_true")
    preview_parser.add_argument("--email-base-url", help="Also render a private email preview; never sends")

    value_parser = subparsers.add_parser("prepare-value-review", help="Prepare unjudged news-value worksheet; no model calls")
    value_parser.add_argument("--snapshot", type=Path, required=True)
    value_parser.add_argument("--review", type=Path, required=True)
    value_parser.add_argument("--output", type=Path, required=True)
    value_parser.add_argument("--public-dir", type=Path, default=Path("public"))

    supplement_parser = subparsers.add_parser("supplement-discovery", help="One bounded public RSS pass prioritizing local edition gaps")
    supplement_parser.add_argument("--plan", type=Path, required=True)
    supplement_parser.add_argument("--sources", type=Path, default=Path("config/source_registry.yaml"))
    supplement_parser.add_argument("--output", type=Path, required=True)
    supplement_parser.add_argument("--public-dir", type=Path, default=Path("public"))
    supplement_parser.add_argument("--max-sources", type=int, default=6)
    supplement_parser.add_argument("--include-shadow", action="store_true")

    local_parser = subparsers.add_parser("confirm-local", help="Confirm a local edition was read; never records online delivery")
    local_parser.add_argument("--state-dir", type=Path, required=True)
    local_parser.add_argument("--run-id", required=True)
    local_parser.add_argument("--page", type=Path, required=True)

    history_parser = subparsers.add_parser("preserve-history")
    history_parser.add_argument("--published", type=Path, required=True)
    history_parser.add_argument("--output", type=Path, required=True)
    history_parser.add_argument("--digest-date", type=date.fromisoformat)

    feedback_parser = subparsers.add_parser("import-feedback")
    feedback_parser.add_argument("--input", type=Path, required=True)
    feedback_parser.add_argument("--state-dir", type=Path, required=True)
    feedback_parser.add_argument("--public-dir", type=Path, default=Path("public"))
    feedback_parser.add_argument("--channel", choices=["web", "local"], default="web")

    layers_parser = subparsers.add_parser("evaluate-layers")
    layers_parser.add_argument("--state-dir", type=Path, required=True)
    layers_parser.add_argument("--output", type=Path, required=True)
    layers_parser.add_argument("--days", type=int, default=7)
    layers_parser.add_argument("--public-dir", type=Path, default=Path("public"))

    evaluation_parser = subparsers.add_parser("evaluate-state")
    evaluation_parser.add_argument("--state-dir", type=Path, required=True)
    evaluation_parser.add_argument("--output", type=Path, required=True)
    evaluation_parser.add_argument("--public-dir", type=Path, default=Path("public"))

    demo_parser = subparsers.add_parser("demo", help="Run an offline product demonstration")
    demo_parser.add_argument("--output", type=Path, default=Path("work/demo"))

    send_parser = subparsers.add_parser("send-email")
    send_parser.add_argument("--input", type=Path, required=True)
    send_parser.add_argument("--digest-date", type=date.fromisoformat, required=True)
    send_parser.add_argument("--digest-url", required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "init":
        from .workspace import initialize_workspace
        try:
            report = initialize_workspace(args.directory)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "extract-content":
        from .article_content import extract_snapshot_content, write_content_snapshot
        if any(path.resolve().is_relative_to(args.public_dir.resolve())
               for path in (args.snapshot, args.output)):
            parser.error("article content input and output must be outside public output")
        if args.output.exists() or args.output.is_symlink():
            parser.error("content output already exists; refusing to overwrite")
        try:
            report = extract_snapshot_content(
                json.loads(args.snapshot.read_text(encoding="utf-8")),
                max_articles=args.max_articles, dns_mode=args.dns_mode,
            )
        except (ValueError, OSError, KeyError, IndexError) as error:
            parser.error(f"invalid content capture ({type(error).__name__})")
        try:
            write_content_snapshot(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        except OSError as error:
            parser.error(f"content output not saved ({type(error).__name__}); existing files preserved")
        statuses = [item["status"] for item in report["article_content"]["items"].values()]
        print(json.dumps({"extracted": statuses.count("extracted"),
                          "rss_fallback": statuses.count("rss_fallback"),
                          "skipped": statuses.count("skipped")}, ensure_ascii=False))
        return 0
    if args.command in {"prepare-review-batches", "merge-review-batches"}:
        from .article_content import write_content_snapshot
        from .review_batches import merge_review_batches, split_review_batches
        from .reviewed_selection import prepare_review

        paths = [args.snapshot]
        if args.command == "prepare-review-batches":
            paths.append(args.output_dir)
        else:
            paths.extend([args.batch_dir, args.output])
        if any(path.resolve().is_relative_to(args.public_dir.resolve()) for path in paths):
            parser.error("review batches and inputs must stay outside public output")
        try:
            audit = json.loads(args.snapshot.read_text(encoding="utf-8"))
            template = prepare_review(audit)
            now = datetime.now(UTC)
            if args.command == "prepare-review-batches":
                if args.output_dir.exists() or args.output_dir.is_symlink():
                    raise ValueError("batch directory already exists")
                batches = split_review_batches(audit, now=now, max_chars=args.max_chars)
                args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
                for index, batch in enumerate(batches, 1):
                    write_content_snapshot(args.output_dir / f"batch-{index:03d}.json",
                                           json.dumps(batch, ensure_ascii=False, indent=2))
                    write_content_snapshot(args.output_dir / f"pending-{index:03d}.json",
                                           json.dumps(prepare_review(batch), ensure_ascii=False, indent=2))
                report = {"schema_version": "review-batches-v1", "batch_count": len(batches),
                          "snapshot_sha256": template["snapshot_sha256"],
                          "candidate_count": len(template["items"]), "max_chars": args.max_chars}
                write_content_snapshot(args.output_dir / "manifest.json", json.dumps(report, indent=2))
            else:
                if args.output.exists() or args.output.is_symlink() or args.batch_dir.is_symlink():
                    raise ValueError("existing output or symlink batch directory")
                manifest_path = args.batch_dir / "manifest.json"
                if manifest_path.is_symlink():
                    raise ValueError("manifest must not be a symlink")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("invalid batch manifest")
                count = manifest.get("batch_count")
                if (manifest.get("schema_version") != "review-batches-v1"
                        or manifest.get("snapshot_sha256") != template["snapshot_sha256"]
                        or type(count) is not int or not 1 <= count <= len(template["items"])):
                    raise ValueError("invalid batch manifest")
                batches, reviews = [], []
                for index in range(1, count + 1):
                    for prefix, destination in (("batch", batches), ("review", reviews)):
                        path = args.batch_dir / f"{prefix}-{index:03d}.json"
                        if path.is_symlink():
                            raise ValueError("batch inputs must not be symlinks")
                        destination.append(json.loads(path.read_text(encoding="utf-8")))
                merged = merge_review_batches(audit, batches, reviews, now=now)
                write_content_snapshot(args.output, json.dumps(merged, ensure_ascii=False, indent=2))
                report = {"merged_batches": count, "candidate_count": len(merged["items"])}
        except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
            parser.error(f"review batch operation failed ({type(error).__name__}); existing files preserved")
        print(json.dumps(report, ensure_ascii=False))
        return 0
    if args.command == "prepare-review":
        from .reviewed_selection import prepare_review
        if any(path.resolve().is_relative_to(args.public_dir.resolve())
               for path in [args.snapshot, args.output]):
            parser.error("editorial input and review must be outside public output")
        if args.output.exists() or args.output.is_symlink():
            parser.error("review output already exists; refusing to overwrite")
        try:
            report = prepare_review(json.loads(args.snapshot.read_text(encoding="utf-8")))
        except (ValueError, OSError, KeyError, IndexError) as error:
            parser.error(f"invalid source snapshot ({type(error).__name__})")
        StateStore._atomic_write(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.command == "configure-key":
        try:
            status = credential_store.configure_deepseek_key()
        except credential_store.CredentialStoreError as error:
            parser.error(str(error))
        print("钥匙串已配置，后续模型命令会自动读取。" if status == "configured"
              else "程序专用钥匙串条目已存在，无需重复输入。")
        return 0
    if args.command == "assess-review":
        from .editorial_model import EditorialModelClient
        attempt_path = args.output.with_name(args.output.name + ".attempt.json")
        response_path = args.output.with_name(args.output.name + ".response.json")
        validation_path = args.output.with_name(args.output.name + ".validation.json")
        private_paths = [args.snapshot, args.output, attempt_path, response_path, validation_path]
        if args.replay_response:
            private_paths.append(args.replay_response)
        if any(path.resolve().is_relative_to(args.public_dir.resolve()) for path in private_paths):
            parser.error("editorial inputs and model receipts must be outside public output")
        if any(path.exists() or path.is_symlink()
               for path in [args.output, attempt_path, response_path, validation_path]):
            parser.error("review or attempt already exists; refusing overwrite or automatic paid retry")
        now = datetime.now(UTC)
        try:
            snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
            client = EditorialModelClient("", model=args.model)
            plan = client.preflight(snapshot, now=now, max_candidates=args.max_candidates)
        except (ValueError, OSError, KeyError, IndexError) as error:
            parser.error(f"invalid model review input ({type(error).__name__})")
        if args.replay_response:
            replay_receipt = {"mode": "offline_replay", "paid_requests": 0,
                              "request_sha256": plan["request_sha256"]}
            try:
                # Bound local artifacts as well as network responses before decoding.
                with args.replay_response.open("rb") as stream:
                    raw = stream.read(20 * 1024 * 1024 + 1)
                if len(raw) > 20 * 1024 * 1024:
                    raise ValueError("saved response too large")
                saved = json.loads(raw)
                assessed = client.replay(snapshot, saved, now=now, max_candidates=args.max_candidates)
                StateStore._atomic_write(args.output, json.dumps(assessed, ensure_ascii=False, indent=2) + "\n")
            except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
                StateStore._atomic_write(validation_path, json.dumps({**replay_receipt,
                    "status": "failed", "validation": client.last_validation,
                    "error_type": type(error).__name__}, indent=2) + "\n")
                parser.error("offline review validation failed; inspect private validation receipt; no API call made")
            StateStore._atomic_write(validation_path, json.dumps({**replay_receipt,
                "status": "succeeded", "validation": client.last_validation}, indent=2) + "\n")
            print(json.dumps({"status": "offline_proposal_saved", "paid_requests": 0,
                              "fact_verification": "not_established"}))
            return 0
        if not args.execute:
            print(json.dumps({**plan, "status": "preflight_only", "paid_requests": 0}, indent=2))
            return 0
        try:
            api_key = credential_store.load_deepseek_key()
        except credential_store.CredentialStoreError as error:
            parser.error(str(error))
        if not api_key:
            parser.error("configure-key（macOS）或 DEEPSEEK_API_KEY 配置后才能执行 --execute")
        receipt = {**plan, "status": "started", "started_at": now.isoformat()}
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation survives an interrupted call and prevents accidental retry billing.
        with attempt_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        client = EditorialModelClient(api_key, model=args.model)
        failure = None
        try:
            assessed = client.assess(snapshot, now=now, max_candidates=args.max_candidates)
        except (ValueError, OSError, KeyError, IndexError, httpx.HTTPError) as error:
            failure = type(error).__name__
        response_retained = False
        try:
            if client.last_response is not None:
                StateStore._atomic_write(response_path, json.dumps(
                    client.last_response, ensure_ascii=False, indent=2) + "\n")
                response_retained = True
            if failure is None:
                StateStore._atomic_write(args.output, json.dumps(assessed, ensure_ascii=False, indent=2) + "\n")
        except OSError as error:
            failure = type(error).__name__
        StateStore._atomic_write(attempt_path, json.dumps({**receipt,
            "status": "failed" if failure else "succeeded", "usage": client.last_usage,
            "transport": client.last_transport, "validation": client.last_validation,
            "response_retained": response_retained,
            **({"error_type": failure} if failure else {})}, indent=2) + "\n")
        if failure:
            parser.error("model review failed; inspect private validation/response receipt; "
                         "attempt retained and no retry made; billing status requires provider review")
        print(json.dumps({"status": "model_proposal_saved", "request_count": 1,
                          "fact_verification": "not_established"}))
        return 0
    if args.command == "preview-reviewed":
        from .local_preview import build_reviewed_preview
        inputs = [args.snapshot, args.review, *([args.preferences] if args.preferences else []),
                  *([args.value_review] if args.value_review else []),
                  *([args.background_readings] if args.background_readings else [])]
        if any(path.resolve().is_relative_to(args.output.resolve()) for path in inputs):
            parser.error("editorial inputs must be outside public output")
        try:
            report = build_reviewed_preview(
                json.loads(args.snapshot.read_text(encoding="utf-8")),
                json.loads(args.review.read_text(encoding="utf-8")),
                _load_preferences(args.preferences), args.output, args.state_dir,
                now=datetime.now(UTC), run_id=args.run_id, include_shadow=args.include_shadow,
                email_base_url=args.email_base_url,
                value_review=json.loads(args.value_review.read_text(encoding="utf-8")) if args.value_review else None,
                background_readings=json.loads(args.background_readings.read_text(encoding="utf-8")) if args.background_readings else None,
            )
        except (ValueError, OSError, KeyError, IndexError) as error:
            parser.error(f"invalid local preview ({type(error).__name__}); check review bindings, paths and run_id")
        print(json.dumps({"run_id": report["run_id"], "selected_count": len(report["selected_ids"]),
                          "background_reading_count": report["background_reading_count"],
                          "feedback_changed_selection": report["feedback_changed_selection"],
                          "delivery_status": report["delivery_status"]}))
        return 0
    if args.command == "prepare-value-review":
        from .article_content import write_content_snapshot
        from .news_value import prepare_value_review
        from .reviewed_selection import select_reviewed
        if args.output.exists() or any(p.resolve().is_relative_to(args.public_dir.resolve())
                for p in [args.snapshot, args.review, args.output]):
            parser.error("news-value inputs/output must stay private; output must be new")
        try:
            audit = json.loads(args.snapshot.read_text())
            review = json.loads(args.review.read_text())
            select_reviewed(audit, review, Preferences(), now=datetime.now(UTC))
            write_content_snapshot(args.output, json.dumps(prepare_value_review(audit, review),
                                                          ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid news-value inputs ({type(error).__name__})")
        return 0
    if args.command == "supplement-discovery":
        from .article_content import write_content_snapshot
        from .news_value import discover_supplement
        if (args.output.exists() or args.output.resolve().is_relative_to(args.public_dir.resolve())
                or any(p.resolve().is_relative_to(args.public_dir.resolve()) for p in [args.plan, args.sources])):
            parser.error("discovery inputs/output must stay private; output must be new")
        try:
            result = discover_supplement(load_source_registry(args.sources),
                json.loads(args.plan.read_text()), now=datetime.now(UTC),
                max_sources=args.max_sources, include_shadow=args.include_shadow)
            write_content_snapshot(args.output, json.dumps(result, ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid discovery request ({type(error).__name__})")
        print(json.dumps(result['discovery_context']))
        return 0
    if args.command == "confirm-local":
        from .delivery_recovery import confirm_local_reading
        if args.state_dir.resolve().is_relative_to(args.page.parent.resolve()):
            parser.error("private state must be outside the reading directory")
        try:
            confirmed = confirm_local_reading(StateStore(args.state_dir), args.run_id,
                                              args.page, now=datetime.now(UTC))
        except (ValueError, OSError) as error:
            parser.error(f"local reading confirmation failed ({type(error).__name__})")
        print(json.dumps({"channel": "local", "new_confirmation": confirmed}))
        return 0
    if args.command == "evaluate-coverage":
        inputs = [args.snapshot, args.reference, *([args.preferences] if args.preferences else [])]
        if any(path.resolve().is_relative_to(args.public_dir.resolve())
               for path in [*inputs, args.output]):
            parser.error("coverage inputs and reports must be outside public output")
        if any(args.output.resolve() == path.resolve() for path in inputs):
            parser.error("coverage output must not overwrite its inputs")
        try:
            report = evaluate_coverage(
                json.loads(args.snapshot.read_text(encoding="utf-8")),
                json.loads(args.reference.read_text(encoding="utf-8")),
                _load_preferences(args.preferences),
            )
        except (ValueError, OSError) as error:
            parser.error(f"invalid coverage input ({type(error).__name__}); check snapshot and reference schema")
        StateStore._atomic_write(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.command == "demo":
        from .demo import run_demo
        report = run_demo(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"import-feedback", "evaluate-layers", "evaluate-state"}:
        paths = [args.state_dir]
        if args.command != "import-feedback":
            paths.append(args.output)
        if any(path.resolve().is_relative_to(args.public_dir.resolve()) for path in paths):
            parser.error("private feedback and reports must be outside public output")
        if args.public_dir.resolve().is_relative_to(args.state_dir.resolve()):
            parser.error("public output must be outside private state")
        if args.command == "import-feedback":
            result = import_feedback_file(args.input, args.state_dir, now=datetime.now(UTC),
                                          channel=args.channel)
            print(json.dumps({"imported_records": len(result.events),
                              "tracked_events": len(result.tracked_event_ids)}))
        elif args.command == "evaluate-state":
            write_state_evaluation(args.state_dir, args.output)
        else:
            write_layered_evaluation(args.state_dir, args.output, now=datetime.now(UTC), days=args.days)
        return 0
    if args.command == "audit-sources":
        if args.output.resolve().is_relative_to(args.public_dir.resolve()):
            parser.error("source audit report must be outside public output")
        if args.output.resolve() == args.sources.resolve():
            parser.error("source audit output must not overwrite the source registry")
        if args.max_age_hours <= 0:
            parser.error("max-age-hours must be positive")
        report = audit_sources(load_source_registry(args.sources), max_age_hours=args.max_age_hours,
                               capture_candidates=args.capture_candidates)
        StateStore._atomic_write(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.command == "preserve-history":
        preserve_history(args.published, args.output, before_date=args.digest_date)
        render_archive(args.output, through=args.digest_date)
        audit_public_directory(args.output)
        return 0
    if args.command == "audit-public":
        audit_public_directory(args.output)
        return 0
    if args.command == "send-email":
        _send_existing_digest(args.input, args.digest_date, args.digest_url)
        return 0
    if args.command == "mark-delivered":
        StateStore(args.state_dir).mark_candidates_delivered(
            args.run_id, json.loads(args.delivery_manifest.read_text()), datetime.now(UTC)
        )
        return 0
    if args.command == "reconcile-deliveries":
        reconcile_publications(StateStore(args.state_dir), args.published, now=datetime.now(UTC))
        return 0
    if args.command == "collect-signals":
        collect_signals(load_source_registry(args.sources), args.state_dir)
        return 0

    try:
        _validate_build_options(
            args.output, email_output=args.email_output, digest_base_url=args.digest_url,
            selection_engine=args.selection_engine, state_dir=args.state_dir,
            delivery_manifest=args.delivery_manifest, run_id=args.run_id,
        )
        preferences = _load_preferences(args.preferences)
        registry = load_source_registry(args.sources)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    api_key = None
    if not args.source_only:
        try:
            api_key = credential_store.load_deepseek_key()
        except credential_store.CredentialStoreError as error:
            parser.error(str(error))
        if not api_key:
            parser.error("configure-key（macOS）或 DEEPSEEK_API_KEY 配置后才能调用模型；免模型使用 --source-only")
    build_time = datetime.now(UTC)
    if args.selection_engine == "layered_shadow_v1":
        collect_signals(registry, args.state_dir, now=build_time)
    articles = fetch_articles(
        [source.as_feed_source() for source in registry.enabled_feed_sources()], now=build_time,
    )
    enricher = (SourceOnlyEnricher() if args.source_only else
                DeepSeekClient(api_key,
                               model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")))
    build_digest(articles, preferences, enricher, args.output, email_output=args.email_output, digest_base_url=args.digest_url, selection_engine=args.selection_engine, state_dir=args.state_dir, source_registry=registry, run_id=args.run_id, delivery_manifest=args.delivery_manifest, now=build_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
