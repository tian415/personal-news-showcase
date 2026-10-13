from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from news_assistant.models import Preferences, Topic
from news_assistant.reviewed_selection import article_fingerprint, prepare_review, select_reviewed

NOW = datetime(2026, 9, 28, 8, tzinfo=UTC)


def inputs(count=12):
    sources = []
    titles = ["黑洞合并产生引力波", "海洋酸化影响珊瑚生长", "新型材料改善电池寿命",
              "古代植物化石揭示气候变迁", "天文望远镜识别行星大气", "细胞蛋白影响记忆形成",
              "地震波描绘地球内部结构", "可降解塑料进入海洋实验", "数学团队研究素数分布",
              "鸟类迁徙受到城市灯光影响", "新方法测量冰川融化速度", "多国更新实验室公共安全规定"]
    for i in range(count):
        name = f"source-{i}"
        article = {
            "id": str(i), "title": titles[i],
            "summary": "该研究仅在实验室观察到结果，尚需后续验证。",
            "source": name, "url": f"https://example.org/{i}", "language": "zh",
            "published_at": (NOW - timedelta(hours=1)).isoformat(), "topic": "science",
        }
        sources.append({
            "id": name, "name": name, "publisher_group": name,
            "production_enabled": True, "status": "ok", "evidence_role": "PRIMARY_FACT",
            "coverage_domains": ["science_health"], "parsed_count": 1,
            "fresh_count": 1, "stale_count": 0, "future_count": 0, "candidates": [article],
        })
    audit = {"schema_version": "source-audit-v1", "candidate_snapshot_available": True,
             "captured_at": NOW.isoformat(), "max_age_hours": 36, "sources": sources}
    review = prepare_review(audit)
    review.update(reviewed_by="test reviewer", reviewed_at=NOW.isoformat())
    for item in review["items"]:
        item.update(status="accepted", quality_score=0.8, importance_score=0.6,
                    fact_status="reported", rationale="研究方法和局限有明确来源，供背景阅读。")
    return audit, review


def test_draft_never_silently_fills_default_quality_scores():
    audit, _ = inputs()
    result = select_reviewed(audit, prepare_review(audit), Preferences(), now=NOW)
    assert result.selected == []
    assert len(result.excluded) == 12
    assert set(result.excluded.values()) == {"pending_review"}


def test_feedback_changes_actual_selection_while_major_floor_is_protected():
    audit, review = inputs()
    review["items"][-1].update(importance_score=0.95, major_reason="实验规则影响整个学科，需要共同知晓。")
    baseline = select_reviewed(audit, review, Preferences(), now=NOW)
    last_ordinary = review["items"][-2]["candidate_id"]
    adjusted = select_reviewed(audit, review, Preferences(source_weights={"source-10": 2}), now=NOW)
    assert len(baseline.selected) == len(adjusted.selected) == 6  # 5 core + protected major
    assert {a.id for a in baseline.selected} != {a.id for a in adjusted.selected}
    assert last_ordinary in {a.id for a in adjusted.selected}
    major_id = review["items"][-1]["candidate_id"]
    downranked = select_reviewed(audit, review, Preferences(
        topic_weights={Topic.SCIENCE: 0.5}, source_weights={"source-11": 0.5}), now=NOW)
    assert major_id in {a.id for a in downranked.selected}
    review["items"][-1]["quality_score"] = 0.1
    unsafe = select_reviewed(audit, review, Preferences(), now=NOW)
    assert major_id not in {a.id for a in unsafe.selected}


@pytest.mark.parametrize("mutation", ["stale_body", "fake_quote", "unknown_id", "duplicate_review", "future_review"])
def test_refuses_unbound_or_fabricated_review_evidence(mutation):
    audit, review = inputs(1)
    if mutation == "stale_body":
        audit["sources"][0]["candidates"][0]["summary"] += " Changed content."
    elif mutation == "fake_quote":
        review["items"][0]["evidence_excerpt"] = "来源没有说过的文字"
    elif mutation == "unknown_id":
        review["items"][0]["candidate_id"] = "unknown"
    elif mutation == "duplicate_review":
        review["items"].append(deepcopy(review["items"][0]))
    else:
        review["reviewed_at"] = (NOW + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        select_reviewed(audit, review, Preferences(), now=NOW)


def test_single_reporting_source_and_unverified_claims_never_become_major():
    audit, review = inputs(2)
    audit["sources"][0]["evidence_role"] = "INDEPENDENT_REPORTING"
    review = {**prepare_review(audit), "reviewed_by": "test", "reviewed_at": NOW.isoformat(),
              "items": review["items"]}
    for item in review["items"]:
        item.update(importance_score=0.99, major_reason="reviewer claims major")
    review["items"][1]["fact_status"] = "unverified"
    result = select_reviewed(audit, review, Preferences(), now=NOW)
    assert not any(event.major for event in result.events)
    assert len(result.selected) == 1
    assert result.excluded[review["items"][1]["candidate_id"]] == "unverified_claim"


def test_history_suppresses_repeats_and_cannot_be_overridden_by_feedback():
    audit, review = inputs(2)
    first = select_reviewed(audit, review, Preferences(), now=NOW)
    second = select_reviewed(audit, review, Preferences(source_weights={"source-0": 2}),
                             now=NOW, historical_events=first.events)
    assert second.selected == []
    assert set(second.excluded.values()) == {"event_duplicate_no_delta"}


def test_shadow_sources_require_explicit_local_opt_in_and_old_capture_expires():
    audit, review = inputs(1)
    audit["sources"][0]["production_enabled"] = False
    review["snapshot_sha256"] = prepare_review(audit)["snapshot_sha256"]
    assert select_reviewed(audit, review, Preferences(), now=NOW).selected == []
    assert len(select_reviewed(audit, review, Preferences(), now=NOW, include_shadow=True).selected) == 1
    assert select_reviewed(audit, review, Preferences(), now=NOW + timedelta(days=2), include_shadow=True).selected == []


def test_fingerprint_binds_full_article_content_not_only_title():
    audit, _ = inputs(1)
    article = audit["sources"][0]["candidates"][0]
    assert article_fingerprint(article) != article_fingerprint({**article, "summary": "changed"})


def test_article_classification_can_correct_broad_feed_category():
    audit, review = inputs(1)
    review["items"][0].update(topic="business", coverage_domain="policy_public_affairs")
    selected = select_reviewed(audit, review, Preferences(), now=NOW)
    assert selected.selected[0].topic == Topic.BUSINESS
    assert selected.events[0].coverage_domain == "policy_public_affairs"


def test_preview_renders_actual_feedback_selection_and_keeps_reports_private(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview

    audit, review = inputs()
    public, state = tmp_path / "public", tmp_path / "private"
    result = build_reviewed_preview(audit, review, Preferences(), public, state,
                                    now=NOW, run_id="local-one")
    assert len(result["selected_ids"]) == 5  # no cross-topic backfill
    assert result["delivery_status"] == "not_confirmed"
    page = (public / "index.html").read_text()
    assert "本地审阅版" in page
    assert "local-feedback-state" not in page
    assert review["items"][0]["rationale"] not in page
    assert not (state / "deliveries").exists()
    assert list(state.glob("runs/*/*/*/*/reviewed-selection.json"))
    assert (public / "2026-09-28.html").read_bytes() == (public / "index.html").read_bytes()
    with pytest.raises(ValueError, match="run_id"):
        build_reviewed_preview(audit, review, Preferences(), public, state, now=NOW, run_id="local-one")


def test_preview_refuses_private_paths_within_public_before_creating_anything(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview

    audit, review = inputs(1)
    public = tmp_path / "public"
    with pytest.raises(ValueError, match="outside"):
        build_reviewed_preview(audit, review, Preferences(), public, public / "state",
                               now=NOW, run_id="invalid")
    assert not public.exists()


def test_real_local_feedback_import_changes_next_preview_and_undo_restores(tmp_path):
    import json

    from news_assistant.delivery_recovery import confirm_local_reading
    from news_assistant.feedback import public_event_ref
    from news_assistant.feedback_store import import_feedback_file
    from news_assistant.local_preview import build_reviewed_preview
    from news_assistant.state import StateStore

    audit, review = inputs()
    state = tmp_path / "state"
    first = build_reviewed_preview(audit, review, Preferences(), tmp_path / "first", state,
                                   now=NOW, run_id="first")
    store = StateStore(state)
    confirm_local_reading(store, "first", tmp_path / "first/2026-09-28.html", now=NOW)
    target_id = first["selected_ids"][0]
    event = next(e for e in store.load_recent_events(NOW, channel="local") if target_id in e.candidate_ids)
    payload = {"schema_version": "news-feedback-v2", "events": [{
        "event_ref": public_event_ref(event.event_id, run_id="first", update_id=event.latest_update_id),
        "vote": "reduce", "reason": "irrelevant", "tracking": False,
        "updated_at": (NOW + timedelta(minutes=1)).isoformat(),
    }]}
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(payload))
    import_feedback_file(path, state, now=NOW + timedelta(minutes=1), channel="local")
    newer, newer_review = inputs()
    # A new batch of unrelated events from the same sources, with equal quality.
    for source in newer["sources"]:
        source["candidates"][0]["title"] = "另一项独立研究：" + source["candidates"][0]["title"]
    scaffold = prepare_review(newer)
    for i, item in enumerate(newer_review["items"]):
        item.update(article_sha256=scaffold["items"][i]["article_sha256"],
                    evidence_excerpt=scaffold["items"][i]["evidence_excerpt"])
    newer_review["snapshot_sha256"] = scaffold["snapshot_sha256"]
    adjusted = build_reviewed_preview(newer, newer_review, Preferences(), tmp_path / "next", state,
                                      now=NOW + timedelta(minutes=2), run_id="next")
    assert adjusted["feedback_changed_selection"] is True
    assert target_id not in adjusted["selected_ids"]
    assert target_id in adjusted["without_feedback_ids"]
    payload["events"][0].update(vote=None, reason=None, updated_at=(NOW + timedelta(minutes=3)).isoformat())
    path.write_text(json.dumps(payload))
    import_feedback_file(path, state, now=NOW + timedelta(minutes=3), channel="local")
    undone = build_reviewed_preview(newer, newer_review, Preferences(), tmp_path / "undo", state,
                                    now=NOW + timedelta(minutes=4), run_id="undo")
    assert undone["feedback_changed_selection"] is False
    assert undone["selected_ids"] == adjusted["without_feedback_ids"]
    assert store.load_recent_events(NOW + timedelta(minutes=4)) == []


def test_repeated_feed_entries_have_one_review_but_conflicting_content_is_rejected():
    audit, _ = inputs(1)
    source = audit["sources"][0]
    source["candidates"].append(deepcopy(source["candidates"][0]))
    source.update(fresh_count=2, parsed_count=2)
    assert len(prepare_review(audit)["items"]) == 1
    source["candidates"][1]["summary"] = "同ID出现相互矛盾的另一版本"
    with pytest.raises(ValueError, match="conflicting"):
        prepare_review(audit)


def test_major_representative_is_supporting_reporting_not_high_scoring_opinion(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview

    audit, review = inputs(3)
    for i, source in enumerate(audit["sources"]):
        source["candidates"][0]["title"] = "机构确认发布公共影响事件"
        source["evidence_role"] = "PRIMARY_FACT" if i == 0 else "INDEPENDENT_REPORTING"
    scaffold = prepare_review(audit)
    review["snapshot_sha256"] = scaffold["snapshot_sha256"]
    for i, item in enumerate(review["items"]):
        item.update(article_sha256=scaffold["items"][i]["article_sha256"],
                    evidence_excerpt=scaffold["items"][i]["evidence_excerpt"],
                    quality_score=1 if i == 0 else 0.8, importance_score=0.9,
                    fact_status="opinion" if i == 0 else "reported", major_reason="跨领域重大影响",
                    origin_publisher_group=f"source-{i}")
    result = select_reviewed(audit, review, Preferences(), now=NOW)
    assert result.events[0].major is True
    assert result.selected[0].source != "source-0"
    # Opinion-only selections must remain visibly attributed as opinion.
    opinion_audit, opinion_review = inputs(1)
    opinion_review["items"][0]["fact_status"] = "opinion"
    build_reviewed_preview(opinion_audit, opinion_review, Preferences(), tmp_path / "reader",
                           tmp_path / "state", now=NOW, run_id="opinion")
    assert "观点分析" in (tmp_path / "reader/index.html").read_text()


def test_event_independent_count_collapses_feeds_of_same_publisher():
    audit, review = inputs(2)
    for source in audit["sources"]:
        source["publisher_group"] = "one-publisher"
        source["candidates"][0]["title"] = "机构发布统一事件报告"
    scaffold = prepare_review(audit)
    review["snapshot_sha256"] = scaffold["snapshot_sha256"]
    for i, item in enumerate(review["items"]):
        item.update(article_sha256=scaffold["items"][i]["article_sha256"],
                    evidence_excerpt=scaffold["items"][i]["evidence_excerpt"])
    result = select_reviewed(audit, review, Preferences(), now=NOW)
    assert result.events[0].independent_source_count == 1


def test_empty_local_edition_cannot_be_confirmed_repeatedly(tmp_path):
    from news_assistant.delivery_recovery import confirm_local_reading
    from news_assistant.local_preview import build_reviewed_preview
    from news_assistant.state import StateStore

    audit, _ = inputs(1)
    root = tmp_path / "public"
    state = tmp_path / "state"
    build_reviewed_preview(audit, prepare_review(audit), Preferences(), root, state, now=NOW, run_id="empty")
    for _ in range(2):
        with pytest.raises(ValueError, match="empty"):
            confirm_local_reading(StateStore(state), "empty", root / "2026-09-28.html", now=NOW)
    assert not (state / "deliveries").exists()


def test_model_cli_preflight_never_calls_provider_and_failed_attempt_cannot_retry(tmp_path, monkeypatch, capsys):
    import json

    from news_assistant import cli
    from news_assistant.editorial_model import EditorialModelClient

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(cli, "datetime", FrozenDatetime)
    audit, _ = inputs(1)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(audit))
    output = tmp_path / "review.json"
    base = ["news-assistant", "assess-review", "--snapshot", str(snapshot), "--output", str(output)]
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("test-secret-must-not-leak")

    monkeypatch.setattr(EditorialModelClient, "assess", fail)
    monkeypatch.setattr("sys.argv", base)
    assert cli.main() == 0
    assert calls == [] and not output.exists()
    assert json.loads(capsys.readouterr().out)["paid_requests"] == 0
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-must-not-leak")
    monkeypatch.setattr("sys.argv", [*base, "--execute"])
    with pytest.raises(SystemExit):
        cli.main()
    assert len(calls) == 1
    assert "test-secret-must-not-leak" not in capsys.readouterr().err
    assert json.loads((tmp_path / "review.json.attempt.json").read_text())["status"] == "failed"
    with pytest.raises(SystemExit):
        cli.main()
    assert len(calls) == 1


def test_local_preview_does_not_fill_missing_topics_with_business():
    audit, review = inputs(12)
    for item in review['items']:
        item.update(topic='business', coverage_domain='business_macro')
    result = select_reviewed(audit, review, Preferences(), now=NOW)
    assert len(result.selected) == 3
    assert all(item.topic == Topic.BUSINESS for item in result.selected)


def test_partial_preview_shows_review_coverage_and_topic_shortfalls(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview
    audit, review = inputs(6)
    for item in review['items'][:4]:
        item.update(topic='business', coverage_domain='business_macro')
    for item in review['items'][4:]:
        item.update(status='pending', quality_score=None, importance_score=None)
    report = build_reviewed_preview(audit, review, Preferences(), tmp_path/'reader',
                                   tmp_path/'state', now=NOW, run_id='partial-topics',
                                   email_base_url='http://localhost:8774')
    assert len(report['selected_ids']) == 3
    assert report['coverage']['pending'] == 2
    assert report['coverage']['selected_by_group'] == {'core': 0, 'business': 3, 'exploration': 0}
    page = (tmp_path/'reader'/'2026-09-28.html').read_text()
    email = next((tmp_path/'state').rglob('email-preview.html')).read_text()
    for rendered in (page, email):
        assert '待审 2' in rendered
        assert '科技／科学 0/5' in rendered
        assert '商业／市场 3/3' in rendered
        assert '探索 0/2' in rendered
        assert '不以其他主题补位' in rendered
