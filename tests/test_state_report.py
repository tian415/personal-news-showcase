import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from news_assistant.models import (
    CandidateArticle,
    DecisionRecord,
    ProcessingStage,
    ReasonCode,
    RunRecord,
    SignalSnapshot,
    SourceHealthRecord,
    Topic,
)


def _evaluate_state_function():
    try:
        from news_assistant.state_report import evaluate_state
    except ModuleNotFoundError as error:
        pytest.fail(f"state report module is missing: {error}")
    return evaluate_state


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def _write_run(
    state_dir: Path,
    run_id: str,
    started_at: datetime,
    candidates: list[CandidateArticle],
    decisions: list[DecisionRecord],
) -> None:
    run = RunRecord(
        run_id=run_id,
        started_at=started_at,
        selection_engine="shadow_event_v1",
        source_candidate_counts={candidate.source_id: 1 for candidate in candidates},
    )
    run_dir = state_dir / "runs" / started_at.strftime("%Y/%m/%d") / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(run.model_dump(mode="json")))
    _write_jsonl(run_dir / "candidates.jsonl", [candidate.model_dump(mode="json") for candidate in candidates])
    _write_jsonl(run_dir / "decisions.jsonl", [decision.model_dump(mode="json") for decision in decisions])


def test_evaluate_state_reports_runs_selection_reasons_sources_signals_and_start_delay(
    tmp_path: Path,
) -> None:
    evaluate_state = _evaluate_state_function()

    state_dir = tmp_path / "state"
    first = datetime(2026, 9, 1, 23, 17, tzinfo=UTC)
    second = datetime(2026, 9, 2, 0, 20, tzinfo=UTC)
    candidates = [
        CandidateArticle(
            id="a1",
            candidate_id="a1",
            title="One",
            url="https://example.com/a1",
            source="Spai",
            source_id="sspai",
            language="zh",
            published_at=first,
            topic=Topic.AI,
        ),
        CandidateArticle(
            id="a2",
            candidate_id="a2",
            title="Two",
            url="https://example.com/a2",
            source="Spai",
            source_id="sspai",
            language="zh",
            published_at=first,
            topic=Topic.AI,
        ),
        CandidateArticle(
            id="a3",
            candidate_id="a3",
            title="Three",
            url="https://example.com/a3",
            source="V2EX",
            source_id="v2ex",
            language="zh",
            published_at=second,
            topic=Topic.TECH,
        ),
    ]
    _write_run(
        state_dir,
        "run-1",
        first,
        candidates[:2],
        [
            DecisionRecord(
                run_id="run-1",
                candidate_id="a1",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
                occurred_at=first,
            ),
            DecisionRecord(
                run_id="run-1",
                candidate_id="a2",
                stage=ProcessingStage.SCORED,
                selected=False,
                primary_reason=ReasonCode.LOW_EVIDENCE,
                occurred_at=first,
            ),
        ],
    )
    _write_run(
        state_dir,
        "run-2",
        second,
        candidates[2:],
        [
            DecisionRecord(
                run_id="run-2",
                candidate_id="a3",
                stage=ProcessingStage.SELECTED,
                selected=False,
                primary_reason=ReasonCode.SOURCE_CAP,
                occurred_at=second,
            )
        ],
    )
    _write_jsonl(
        state_dir / "signals/2026/09/02/snapshots.jsonl",
        [
            SignalSnapshot(
                source_id="v2ex",
                external_id="42",
                captured_at=second,
                rank=1,
            ).model_dump(mode="json"),
            SignalSnapshot(
                source_id="v2ex",
                external_id="43",
                captured_at=second,
                rank=2,
            ).model_dump(mode="json"),
        ],
    )
    _write_jsonl(
        state_dir / "signals/2026/09/02/source-health.jsonl",
        [
            SourceHealthRecord(
                source_id="v2ex",
                captured_at=second,
                status="ok",
                candidate_count=2,
            ).model_dump(mode="json"),
            SourceHealthRecord(
                source_id="sspai",
                captured_at=second,
                status="error",
                candidate_count=0,
                error_type="TimeoutError",
            ).model_dump(mode="json"),
            SourceHealthRecord(
                source_id="geekpark",
                captured_at=second,
                status="ok",
                candidate_count=1,
            ).model_dump(mode="json"),
        ],
    )

    report = evaluate_state(state_dir)

    assert report == {
        "schema_version": "state-evaluation-v1",
        "runs": {
            "count": 2,
            "first_started_at": "2026-09-01T23:17:00Z",
            "last_started_at": "2026-09-02T00:20:00Z",
        },
        "candidates": {"count": 3, "selected_count": 1},
        "reason_counts": {"LOW_EVIDENCE": 1, "MAJOR_GUARANTEE": 1, "SOURCE_CAP": 1},
        "source_candidate_counts": {"sspai": 2, "v2ex": 1},
        "signals": {
            "snapshot_count": 2,
            "source_health_count": 3,
            "health_status_counts": {"error": 1, "ok": 2},
        },
        "startup_delay_minutes": {"average": -11.5, "max": 20.0},
        "diagnostics": {
            "invalid_json_lines": 0,
            "invalid_run_files": 0,
            "invalid_run_fields": 0,
            "invalid_state_records": 0,
        },
    }


def test_evaluate_state_empty_state_returns_zero_values(tmp_path: Path) -> None:
    evaluate_state = _evaluate_state_function()

    assert evaluate_state(tmp_path / "missing-state") == {
        "schema_version": "state-evaluation-v1",
        "runs": {"count": 0, "first_started_at": None, "last_started_at": None},
        "candidates": {"count": 0, "selected_count": 0},
        "reason_counts": {},
        "source_candidate_counts": {},
        "signals": {
            "snapshot_count": 0,
            "source_health_count": 0,
            "health_status_counts": {},
        },
        "startup_delay_minutes": {"average": 0.0, "max": 0.0},
        "diagnostics": {
            "invalid_json_lines": 0,
            "invalid_run_files": 0,
            "invalid_run_fields": 0,
            "invalid_state_records": 0,
        },
    }


def test_evaluate_state_counts_each_selected_event_or_candidate_once(tmp_path: Path) -> None:
    evaluate_state = _evaluate_state_function()
    state_dir = tmp_path / "state"
    started_at = datetime(2026, 9, 1, 23, 17, tzinfo=UTC)
    candidate = CandidateArticle(
        id="a1",
        candidate_id="a1",
        title="One",
        url="https://example.com/a1",
        source="Spai",
        source_id="sspai",
        language="zh",
        published_at=started_at,
        topic=Topic.AI,
    )
    _write_run(
        state_dir,
        "same-run-id",
        started_at,
        [candidate],
        [
            DecisionRecord(
                event_id="evt-1",
                candidate_id="a1",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
            ),
            DecisionRecord(
                event_id="evt-1",
                candidate_id="a1",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
            ),
            DecisionRecord(
                candidate_id="a2",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
            ),
            DecisionRecord(
                candidate_id="a2",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
            ),
        ],
    )
    _write_run(
        state_dir,
        "same-run-id",
        datetime(2026, 9, 2, 23, 17, tzinfo=UTC),
        [candidate],
        [
            DecisionRecord(
                event_id="evt-1",
                candidate_id="a1",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
            )
        ],
    )

    assert evaluate_state(state_dir)["candidates"]["selected_count"] == 3


def test_evaluate_state_requires_boolean_selected_and_id_for_selection_count(
    tmp_path: Path,
) -> None:
    evaluate_state = _evaluate_state_function()
    state_dir = tmp_path / "state"
    started_at = datetime(2026, 9, 1, 23, 17, tzinfo=UTC)
    candidate = CandidateArticle(
        id="a1",
        candidate_id="a1",
        title="One",
        url="https://example.com/a1",
        source="Spai",
        source_id="sspai",
        language="zh",
        published_at=started_at,
        topic=Topic.AI,
    )
    _write_run(
        state_dir,
        "run-1",
        started_at,
        [candidate],
        [
            DecisionRecord(
                event_id="evt-1",
                candidate_id="a1",
                stage=ProcessingStage.SELECTED,
                selected=True,
                primary_reason=ReasonCode.MAJOR_GUARANTEE,
            )
        ],
    )
    decision_path = state_dir / "runs/2026/09/01/run-1/decisions.jsonl"
    decision_path.open("a").write(
        json.dumps(
            {
                "selected": "false",
                "event_id": "evt-2",
                "stage": "SELECTED",
                "primary_reason": "MAJOR_GUARANTEE",
            }
        )
        + "\n"
        + json.dumps(
            {
                "selected": True,
                "stage": "SELECTED",
                "primary_reason": "MAJOR_GUARANTEE",
            }
        )
        + "\n"
    )

    report = evaluate_state(state_dir)

    assert report["candidates"]["selected_count"] == 1
    assert report["diagnostics"]["invalid_json_lines"] == 0
    assert report["diagnostics"]["invalid_state_records"] == 2


def test_evaluate_state_skips_invalid_legacy_count_fields(tmp_path: Path) -> None:
    evaluate_state = _evaluate_state_function()
    state_dir = tmp_path / "state"
    started_at = datetime(2026, 9, 1, 23, 17, tzinfo=UTC)

    for run_id, source_counts, reason_counts in (
        ("invalid-dicts", ["not-a-dict"], "not-a-dict"),
        (
            "invalid-values",
            {"ok": "2", "bad": "oops", "negative": -1, "none": None},
            {"GOOD": "1", "bad": "oops", "negative": -2, "none": None},
        ),
    ):
        run_dir = state_dir / "runs/2026/09/01" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "started_at": started_at.isoformat().replace("+00:00", "Z"),
                    "source_candidate_counts": source_counts,
                    "reason_counts": reason_counts,
                }
            )
        )

    try:
        report = evaluate_state(state_dir)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as error:
        pytest.fail(f"invalid legacy count fields must be skipped, not raised: {error}")

    assert report["runs"]["count"] == 2
    assert report["candidates"]["count"] == 2
    assert report["source_candidate_counts"] == {"ok": 2}
    assert report["reason_counts"] == {"GOOD": 1}
    assert report["diagnostics"]["invalid_run_files"] == 0
    assert report["diagnostics"]["invalid_run_fields"] == 8
    assert report["diagnostics"]["invalid_state_records"] == 0


def test_evaluate_state_skips_corrupt_state_records_and_reports_diagnostics(tmp_path: Path) -> None:
    evaluate_state = _evaluate_state_function()
    state_dir = tmp_path / "state"
    started_at = datetime(2026, 9, 1, 23, 17, tzinfo=UTC)
    candidate = CandidateArticle(
        id="a1",
        candidate_id="a1",
        title="One",
        url="https://example.com/a1",
        source="Spai",
        source_id="sspai",
        language="zh",
        published_at=started_at,
        topic=Topic.AI,
    )
    decision = DecisionRecord(
        candidate_id="a1",
        stage=ProcessingStage.SELECTED,
        selected=True,
        primary_reason=ReasonCode.MAJOR_GUARANTEE,
    )
    _write_run(state_dir, "good", started_at, [candidate], [decision])
    bad_candidates = state_dir / "runs/2026/09/01/good/candidates.jsonl"
    bad_candidates.open("a").write("not-json\n")
    bad_decisions = state_dir / "runs/2026/09/01/good/decisions.jsonl"
    bad_decisions.open("a").write("{\"selected\": true\n")
    snapshot_path = state_dir / "signals/2026/09/02/snapshots.jsonl"
    _write_jsonl(
        snapshot_path,
        [
            SignalSnapshot(
                source_id="v2ex",
                external_id="42",
                captured_at=started_at,
                rank=1,
            ).model_dump(mode="json")
        ],
    )
    snapshot_path.open("a").write("not-json\n")
    health_path = state_dir / "signals/2026/09/02/source-health.jsonl"
    _write_jsonl(
        health_path,
        [
            SourceHealthRecord(
                source_id="v2ex",
                captured_at=started_at,
                status="ok",
                candidate_count=1,
            ).model_dump(mode="json")
        ],
    )
    health_path.open("a").write("not-json\n")
    bad_json_run = state_dir / "runs/2026/09/01/bad-json/run.json"
    bad_json_run.parent.mkdir(parents=True)
    bad_json_run.write_text("not-json\n")
    bad_started_run = state_dir / "runs/2026/09/01/bad-started/run.json"
    bad_started_run.parent.mkdir(parents=True)
    bad_started_run.write_text(json.dumps({"run_id": "bad-started"}))

    try:
        report = evaluate_state(state_dir)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as error:
        pytest.fail(f"corrupt state must be skipped, not raised: {error}")

    assert report["runs"]["count"] == 1
    assert report["candidates"]["count"] == 1
    assert report["candidates"]["selected_count"] == 1
    assert report["signals"] == {
        "snapshot_count": 1,
        "source_health_count": 1,
        "health_status_counts": {"ok": 1},
    }
    assert report["diagnostics"] == {
        "invalid_json_lines": 4,
        "invalid_run_files": 2,
        "invalid_run_fields": 0,
        "invalid_state_records": 0,
    }
