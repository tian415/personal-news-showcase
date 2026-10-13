import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from news_assistant import cli
from news_assistant.deepseek import fallback_enrichment
from news_assistant.models import Article, DeliveryRecord, Preferences
from news_assistant.publication import audit_public_directory
from news_assistant.state import StateStore

NOW = datetime(2026, 9, 12, 8, tzinfo=UTC)


class OfflineEnricher:
    enrich = staticmethod(fallback_enrichment)


def published(tmp_path):
    state = tmp_path / "state"
    article = Article(
        id="published",
        title="Official product launch",
        url="https://example.com/1",
        source="Source",
        language="en",
        published_at=NOW,
        topic="ai",
    )
    digest = cli.build_digest(
        [article],
        Preferences(),
        OfflineEnricher(),
        tmp_path / "public",
        selection_engine="layered_shadow_v1",
        state_dir=state,
        run_id="run-1",
        now=NOW,
    )
    StateStore(state).mark_candidates_delivered("run-1", ["published"], NOW)
    return state, digest


def test_rendered_alias_can_be_imported_only_after_delivery_and_replays_are_idempotent(tmp_path):
    state, digest = published(tmp_path)
    ref = digest.articles[0].event_ref
    assert ref and len(ref) == 24
    payload = {
        "schema_version": "news-feedback-v2",
        "events": [
            {
                "event_ref": ref,
                "vote": "helpful",
                "reason": None,
                "tracking": True,
                "updated_at": NOW.isoformat(),
            }
        ],
    }
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(payload))
    result = cli.import_feedback_file(path, state, now=NOW)
    assert result.preferences.topic_weights["ai"] == 1.1
    assert len(result.tracked_event_ids) == 1
    stored = state / "feedback-state.json"
    before = hashlib.sha256(stored.read_bytes()).hexdigest()
    cli.import_feedback_file(path, state, now=NOW)
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == before
    audit_public_directory(tmp_path / "public")
    assert "evt_" not in (tmp_path / "public" / "index.html").read_text()


def test_unknown_feedback_reference_cannot_create_or_modify_private_preferences(tmp_path):
    state, _ = published(tmp_path)
    path = tmp_path / "feedback.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "news-feedback-v2",
                "events": [
                    {
                        "event_ref": "0" * 24,
                        "vote": "helpful",
                        "reason": None,
                        "tracking": False,
                        "updated_at": NOW.isoformat(),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError):
        cli.import_feedback_file(path, state, now=NOW)
    assert not (state / "feedback-state.json").exists()


def _write_delivery(state, event_id: str, *, channel: str, delivered_at: datetime) -> None:
    target = state / "deliveries" / delivered_at.strftime("%Y/%m") / "deliveries.jsonl"
    StateStore._write_jsonl(
        target,
        [
            DeliveryRecord(
                event_id=event_id,
                channel=channel,
                status="delivered",
                delivered_at=delivered_at,
            ).model_dump(mode="json")
        ],
    )


@pytest.mark.parametrize("case", ["non_web", "future_delivery", "future_event"])
def test_feedback_target_requires_current_web_delivery_and_event(tmp_path, case):
    state = tmp_path / "state"
    article = Article(
        id="published",
        title="Official product launch",
        url="https://example.com/1",
        source="Source",
        language="en",
        published_at=NOW,
        topic="ai",
    )
    digest = cli.build_digest(
        [article],
        Preferences(),
        OfflineEnricher(),
        tmp_path / "public",
        selection_engine="layered_shadow_v1",
        state_dir=state,
        run_id="run-1",
        now=NOW,
    )
    ref = digest.articles[0].event_ref
    assert ref
    event_id = next(
        record["event_id"]
        for path in (state / "events").glob("*/*/*/events.jsonl")
        for record in StateStore(state)._read_jsonl(path)
        if "published" in record.get("candidate_ids", [])
    )
    if case == "non_web":
        _write_delivery(state, event_id, channel="email", delivered_at=NOW)
    elif case == "future_delivery":
        _write_delivery(state, event_id, channel="web", delivered_at=NOW + timedelta(hours=1))
    else:
        _write_delivery(state, event_id, channel="web", delivered_at=NOW)
        for path in (state / "events").glob("*/*/*/events.jsonl"):
            records = StateStore(state)._read_jsonl(path)
            for record in records:
                if record.get("event_id") == event_id:
                    record["latest_event_time"] = (NOW + timedelta(hours=1)).isoformat()
            StateStore._write_jsonl(path, records)

    payload = {
        "schema_version": "news-feedback-v2",
        "events": [
            {
                "event_ref": ref,
                "vote": "helpful",
                "reason": None,
                "tracking": False,
                "updated_at": NOW.isoformat(),
            }
        ],
    }
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown event_ref"):
        cli.import_feedback_file(path, state, now=NOW)
    assert not (state / "feedback-state.json").exists()


def test_import_feedback_cli_requires_private_state_outside_public(tmp_path, monkeypatch):
    import sys

    input_path = tmp_path / "input.json"
    input_path.write_text('{"schema_version":"news-feedback-v2","events":[]}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "news-assistant",
            "import-feedback",
            "--input",
            str(input_path),
            "--state-dir",
            str(tmp_path / "private"),
        ],
    )
    assert cli.main() == 0
    assert (tmp_path / "private" / "feedback-state.json").exists()


def test_feedback_stays_bound_to_published_run_even_if_candidate_is_reused(tmp_path):
    state, digest = published(tmp_path)
    ref = digest.articles[0].event_ref
    changed = Article(id="published", title="Unpublished changed title", url="https://example.com/2",
                      source="Unpublished source", language="en", topic="business", published_at=NOW)
    later = cli.build_digest([changed], Preferences(), OfflineEnricher(), tmp_path / "next-public",
                            selection_engine="layered_shadow_v1", state_dir=state,
                            run_id="run-2", now=NOW + timedelta(minutes=1))
    path = tmp_path / "feedback.json"
    def write(reference):
        path.write_text(json.dumps({"schema_version": "news-feedback-v2", "events": [{
            "event_ref": reference, "vote": "helpful", "reason": None, "tracking": False,
            "updated_at": NOW.isoformat(),
        }]}))
    write(ref)
    result = cli.import_feedback_file(path, state, now=NOW + timedelta(minutes=2))
    assert result.preferences.source_weights == {"Source": 1.1}
    assert result.preferences.topic_weights == {"ai": 1.1}
    before = (state / "feedback-state.json").read_bytes()
    write(later.articles[0].event_ref)
    with pytest.raises(ValueError, match="unknown event_ref"):
        cli.import_feedback_file(path, state, now=NOW + timedelta(minutes=2))
    assert (state / "feedback-state.json").read_bytes() == before


def test_repeated_legacy_story_keeps_feedback_reference_for_each_published_run(tmp_path):
    state, first = published(tmp_path)
    second = cli.build_digest([Article.model_validate(first.articles[0].model_dump())], Preferences(),
                              OfflineEnricher(), tmp_path / "second-public",
                              selection_engine="layered_shadow_v1", state_dir=state,
                              run_id="run-2", now=NOW + timedelta(minutes=1))
    assert second.articles[0].event_ref
    assert first.articles[0].event_ref != second.articles[0].event_ref
    StateStore(state).mark_candidates_delivered("run-2", ["published"], NOW + timedelta(minutes=1))
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps({"schema_version": "news-feedback-v2", "events": [{
        "event_ref": item.articles[0].event_ref, "vote": vote, "reason": None, "tracking": track,
        "updated_at": (NOW + timedelta(minutes=minutes)).isoformat(),
    } for item, vote, track, minutes in [(first, "helpful", True, 0), (second, None, False, 2)]]}))
    result = cli.import_feedback_file(path, state, now=NOW + timedelta(minutes=3))
    assert result.preferences.topic_weights == {}
    assert result.tracked_event_ids == []


def test_identical_feed_records_are_coalesced_before_layered_build(tmp_path):
    item = Article(id='dup', title='Identical feed record', url='https://example.com/dup',
                   source='source', language='en', topic='ai', published_at=NOW)
    digest = cli.build_digest([item, item.model_copy()], Preferences(), OfflineEnricher(),
                              tmp_path/'public', state_dir=tmp_path/'state',
                              selection_engine='layered_shadow_v1', run_id='unique', now=NOW)
    assert len(digest.articles) == 1
    rows = StateStore._read_jsonl(next((tmp_path/'state/runs').glob('*/*/*/*/candidates.jsonl')))
    assert len(rows) == 1


def test_conflicting_article_ids_fail_before_any_artifact(tmp_path):
    item = Article(id='dup', title='Original', url='https://example.com/dup', source='source',
                   language='en', topic='ai', published_at=NOW)
    with pytest.raises(ValueError, match='conflicting article id'):
        cli.build_digest([item, item.model_copy(update={'title':'Different'})], Preferences(),
                         OfflineEnricher(), tmp_path/'public', state_dir=tmp_path/'state',
                         selection_engine='layered_shadow_v1', run_id='unique', now=NOW)
    assert not (tmp_path/'public').exists()
    assert not (tmp_path/'state').exists()


def test_invalid_private_feedback_fails_before_model_calls_or_run_writes(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'feedback-state.json').write_text('{broken')
    class NeverEnrich:
        def enrich(self, _item):
            raise AssertionError('private state must be checked before model calls')
    item = Article(id='item', title='Valid story', url='https://example.com/a', source='source',
                   language='en', topic='ai', published_at=NOW)
    with pytest.raises(ValueError):
        cli.build_digest([item], Preferences(), NeverEnrich(), tmp_path/'public', state_dir=state,
                         selection_engine='layered_shadow_v1', run_id='corrupt', now=NOW)
    assert not (state/'runs').exists()
    assert not (tmp_path/'public').exists()
