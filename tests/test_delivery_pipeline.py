import hashlib
import json
from datetime import UTC, datetime, timedelta

from news_assistant.cli import build_digest
from news_assistant.models import Article, EnrichedArticle, Preferences, ReasonCode, Topic
from news_assistant.state import StateStore


class FakeEnricher:
    def enrich(self, item: Article) -> EnrichedArticle:
        return EnrichedArticle(
            **item.model_dump(),
            chinese_title="中文标题",
            chinese_summary="中文摘要",
        )


def test_build_saves_private_receipt_bound_to_rendered_page_and_selected_candidates(tmp_path):
    now = datetime(2026, 9, 21, 8, tzinfo=UTC)
    public = tmp_path / "public"
    state = tmp_path / "state"
    article = Article(
        id="receipt-selected", title="Selected article", url="https://example.com/receipt",
        source="Example", language="en", published_at=now, topic=Topic.AI,
    )
    build_digest(
        [article], Preferences(), FakeEnricher(), public,
        selection_engine="shadow_event_v1", state_dir=state, run_id="receipt-run", now=now,
    )
    receipt_path = state / "runs/2026/09/21/receipt-run/publication.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["run_id"] == "receipt-run"
    assert receipt["digest_date"] == "2026-09-21"
    assert receipt["candidate_ids"] == [article.id]
    assert receipt["page_sha256"] == hashlib.sha256(
        (public / "2026-09-21.html").read_bytes()
    ).hexdigest()
    assert not list(public.rglob("*.json"))
    assert StateStore(state).load_recent_events(now) == []


def test_mark_allows_legacy_candidate_deduplicated_out_of_second_shadow_run(tmp_path) -> None:
    first_time = datetime(2026, 9, 21, 8, tzinfo=UTC)
    second_time = first_time + timedelta(hours=1)
    article = Article(
        id="legacy-selected",
        title="Already delivered event",
        url="https://example.com/already-delivered",
        source="Example",
        language="en",
        published_at=first_time,
        topic=Topic.AI,
    )
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)
    first_manifest = tmp_path / "first-manifest.json"
    second_manifest = tmp_path / "second-manifest.json"

    build_digest(
        [article],
        Preferences(),
        FakeEnricher(),
        tmp_path / "first-public",
        selection_engine="shadow_event_v1",
        state_dir=state_dir,
        run_id="run-first",
        delivery_manifest=first_manifest,
        now=first_time,
    )
    store.mark_candidates_delivered(
        "run-first", json.loads(first_manifest.read_text()), first_time
    )
    build_digest(
        [article],
        Preferences(),
        FakeEnricher(),
        tmp_path / "second-public",
        selection_engine="shadow_event_v1",
        state_dir=state_dir,
        run_id="run-second",
        delivery_manifest=second_manifest,
        now=second_time,
    )

    second_run = state_dir / "runs" / "2026" / "09" / "21" / "run-second"
    assert [record["candidate_id"] for record in store._read_jsonl(second_run / "candidates.jsonl")] == [
        article.id
    ]
    # Legacy may publish this story again, even though shadow excludes it.
    # Its current-run publication snapshot is needed to bind browser feedback.
    assert len(store._read_jsonl(second_run / "events.jsonl")) == 1
    assert {
        record["primary_reason"]
        for record in store._read_jsonl(second_run / "decisions.jsonl")
    } == {ReasonCode.EVENT_DUPLICATE_NO_DELTA.value}

    store.mark_candidates_delivered(
        "run-second", json.loads(second_manifest.read_text()), second_time
    )

    deliveries = [
        record
        for path in (state_dir / "deliveries").glob("*/*/deliveries.jsonl")
        for record in store._read_jsonl(path)
    ]
    assert len(deliveries) == 2
    assert {record["run_id"] for record in deliveries} == {"run-first", "run-second"}
    assert deliveries[0]["event_id"] == deliveries[1]["event_id"]
