from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from news_assistant.delivery_recovery import reconcile_publications, record_publication
from news_assistant.models import CandidateArticle, CoverageDomain, Event, RunRecord, Topic
from news_assistant.state import StateStore

NOW = datetime(2026, 9, 21, 8, tzinfo=UTC)


def _write_run(
    store: StateStore,
    run_id: str,
    *,
    started_at: datetime = NOW,
    candidate_id: str | None = None,
) -> tuple[RunRecord, CandidateArticle]:
    candidate_id = candidate_id or f"candidate-{run_id}"
    run = RunRecord(
        run_id=run_id,
        started_at=started_at,
        selection_engine="shadow_event_v1",
    )
    candidate = CandidateArticle(
        id=candidate_id,
        candidate_id=candidate_id,
        title=f"Title for {run_id}",
        url=f"https://example.com/{candidate_id}",
        source="Example",
        source_id="example",
        language="en",
        published_at=started_at,
        topic=Topic.AI,
    )
    event = Event(
        event_id=f"event-{run_id}",
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        event_type="release",
        title=candidate.title,
        first_event_time=started_at,
        latest_event_time=started_at,
        candidate_ids=[candidate_id],
    )
    store.write_run(run, [candidate], [], [event])
    return run, candidate


def _rendered_page(root: Path, digest_date: date, contents: bytes) -> Path:
    page = root / f"{digest_date.isoformat()}.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_bytes(contents)
    return page


def test_record_publication_writes_exact_private_intent_to_owning_run(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-record")
    page = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"exact page bytes\n")

    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], page)

    receipt_path = store._run_directory(run) / "publication.json"
    assert json.loads(receipt_path.read_text()) == {
        "schema": "publication-intent-v1",
        "run_id": "run-record",
        "digest_date": "2026-09-21",
        "candidate_ids": [candidate.candidate_id],
        "page_sha256": hashlib.sha256(b"exact page bytes\n").hexdigest(),
    }


def test_record_publication_is_idempotent_but_rejects_changed_receipt(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-immutable-receipt")
    first = _rendered_page(tmp_path / "first", date(2026, 9, 21), b"first page")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], first)

    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], first)
    changed = _rendered_page(tmp_path / "changed", date(2026, 9, 21), b"changed page")
    with pytest.raises(ValueError, match="overwrite|different"):
        record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], changed)

    receipt = json.loads((store._run_directory(run) / "publication.json").read_text())
    assert receipt["page_sha256"] == hashlib.sha256(b"first page").hexdigest()


def test_record_publication_requires_exact_persisted_run_and_candidates(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-bound")
    page = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"page")

    mismatched_run = run.model_copy(update={"started_at": NOW + timedelta(minutes=1)})
    with pytest.raises(ValueError, match="run|binding"):
        record_publication(
            store,
            mismatched_run,
            date(2026, 9, 21),
            [candidate.candidate_id],
            page,
        )
    with pytest.raises(ValueError, match="candidate"):
        record_publication(store, run, date(2026, 9, 21), ["unknown"], page)

    assert not (store._run_directory(run) / "publication.json").exists()


def test_unpublished_intent_does_not_mark_delivery(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-unpublished")
    page = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"not published")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], page)
    published = tmp_path / "published"
    published.mkdir()

    assert reconcile_publications(store, published, now=NOW) == []
    assert store.load_recent_events(NOW) == []


def test_exact_published_bytes_recover_the_real_run_event(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-recover")
    rendered = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"published bytes")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], rendered)
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), b"published bytes")

    assert reconcile_publications(store, published, now=NOW) == [run.run_id]
    assert [event.event_id for event in store.load_recent_events(NOW)] == [
        f"event-{run.run_id}"
    ]


def test_wrong_page_hash_is_ignored(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-wrong-hash")
    rendered = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"intended bytes")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], rendered)
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), b"different bytes")

    assert reconcile_publications(store, published, now=NOW) == []
    assert store.load_recent_events(NOW) == []


def test_same_day_overwrite_only_recovers_receipt_matching_current_page(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    old_run, old_candidate = _write_run(store, "run-old")
    new_run, new_candidate = _write_run(
        store,
        "run-new",
        started_at=NOW + timedelta(minutes=5),
    )
    record_publication(
        store,
        old_run,
        date(2026, 9, 21),
        [old_candidate.candidate_id],
        _rendered_page(tmp_path / "old-render", date(2026, 9, 21), b"old page"),
    )
    record_publication(
        store,
        new_run,
        date(2026, 9, 21),
        [new_candidate.candidate_id],
        _rendered_page(tmp_path / "new-render", date(2026, 9, 21), b"new page"),
    )
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), b"new page")

    assert reconcile_publications(store, published, now=NOW + timedelta(minutes=10)) == [
        new_run.run_id
    ]
    assert [event.event_id for event in store.load_recent_events(NOW + timedelta(minutes=10))] == [
        f"event-{new_run.run_id}"
    ]


def test_recovery_is_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-idempotent")
    contents = b"same public page"
    record_publication(
        store,
        run,
        date(2026, 9, 21),
        [candidate.candidate_id],
        _rendered_page(tmp_path / "rendered", date(2026, 9, 21), contents),
    )
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), contents)

    assert reconcile_publications(store, published, now=NOW) == [run.run_id]
    assert reconcile_publications(store, published, now=NOW + timedelta(minutes=1)) == []
    delivery_records = [
        record
        for path in (store.root / "deliveries").glob("*/*/deliveries.jsonl")
        for record in store._read_jsonl(path)
    ]
    assert len(delivery_records) == 1


def test_already_delivered_run_skips_damaged_candidate_reparse(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-already-delivered")
    contents = b"published page"
    record_publication(
        store,
        run,
        date(2026, 9, 21),
        [candidate.candidate_id],
        _rendered_page(tmp_path / "rendered", date(2026, 9, 21), contents),
    )
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), contents)
    assert reconcile_publications(store, published, now=NOW) == [run.run_id]
    (store._run_directory(run) / "candidates.jsonl").write_text("not json\n")

    assert reconcile_publications(store, published, now=NOW + timedelta(minutes=1)) == []


def test_unknown_candidate_is_checked_only_after_page_hash_matches(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-unknown-candidate")
    intended = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"intended")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], intended)
    receipt_path = store._run_directory(run) / "publication.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["candidate_ids"] = ["unknown-candidate"]
    receipt_path.write_text(json.dumps(receipt))
    published = tmp_path / "published"
    page = _rendered_page(published, date(2026, 9, 21), b"different")

    assert reconcile_publications(store, published, now=NOW) == []

    page.write_bytes(b"intended")
    with pytest.raises(ValueError, match="unknown candidate"):
        reconcile_publications(store, published, now=NOW)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.pop("page_sha256"),
        lambda receipt: receipt.update(schema="unknown"),
        lambda receipt: receipt.update(run_id="different-run"),
        lambda receipt: receipt.update(digest_date="../../private"),
        lambda receipt: receipt.update(candidate_ids=["candidate", 7]),
        lambda receipt: receipt.update(page_sha256="not-a-sha256"),
        lambda receipt: receipt.update(extra="ambiguous"),
    ],
)
def test_malformed_or_mismatched_receipt_is_rejected(tmp_path: Path, mutate) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-invalid")
    page = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"page")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], page)
    receipt_path = store._run_directory(run) / "publication.json"
    receipt = json.loads(receipt_path.read_text())
    mutate(receipt)
    receipt_path.write_text(json.dumps(receipt))
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), b"page")

    with pytest.raises(ValueError):
        reconcile_publications(store, published, now=NOW)

    assert store.load_recent_events(NOW) == []


def test_receipt_in_wrong_run_partition_is_rejected(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-wrong-partition")
    page = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"page")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], page)
    original = store._run_directory(run)
    wrong = store.root / "runs" / "2026" / "09" / "22" / run.run_id
    wrong.parent.mkdir(parents=True)
    original.rename(wrong)
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), b"page")

    with pytest.raises(ValueError, match="partition|binding"):
        reconcile_publications(store, published, now=NOW)


@pytest.mark.parametrize("symlink_kind", ["receipt", "published_page"])
def test_symlink_inputs_are_rejected(tmp_path: Path, symlink_kind: str) -> None:
    store = StateStore(tmp_path / "state")
    run, candidate = _write_run(store, "run-symlink")
    page = _rendered_page(tmp_path / "rendered", date(2026, 9, 21), b"page")
    record_publication(store, run, date(2026, 9, 21), [candidate.candidate_id], page)
    published = tmp_path / "published"
    published.mkdir()
    if symlink_kind == "receipt":
        receipt = store._run_directory(run) / "publication.json"
        target = tmp_path / "receipt.json"
        receipt.replace(target)
        receipt.symlink_to(target)
        _rendered_page(published, date(2026, 9, 21), b"page")
    else:
        target = tmp_path / "page.html"
        target.write_bytes(b"page")
        (published / "2026-09-21.html").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        reconcile_publications(store, published, now=NOW)


def test_legacy_run_without_receipt_is_ignored(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    _write_run(store, "run-legacy")
    published = tmp_path / "published"
    _rendered_page(published, date(2026, 9, 21), b"some page")

    assert reconcile_publications(store, published, now=NOW) == []
    assert store.load_recent_events(NOW) == []
