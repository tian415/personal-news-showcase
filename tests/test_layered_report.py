import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from news_assistant.layered import ContentLayer, LayeredItem
from news_assistant.layered_report import evaluate_layers, write_layered_evaluation
from news_assistant.models import CoverageDomain, Event, ScoreCard

NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


def _event(event_id: str, *candidate_ids: str, domain: CoverageDomain = CoverageDomain.AI_TECHNOLOGY) -> Event:
    return Event(
        event_id=event_id,
        coverage_domain=domain,
        event_type="release",
        title=f"Event {event_id}",
        first_event_time=NOW,
        latest_event_time=NOW,
        candidate_ids=list(candidate_ids),
        scores=ScoreCard(
            freshness=90,
            importance=80,
            relevance=70,
            exploration=60,
            evidence_quality=75,
        ),
    )


def _report(items: list[LayeredItem], **overrides: object) -> dict:
    value = {
        "schema_version": "layered-shadow-v1",
        "items": [item.model_dump(mode="json") for item in items],
        "excluded": {},
        "layer_counts": {"decision_brief": 999, "signal_radar": 999, "exploration": 999},
        "legacy_only": ["legacy-only"],
        "layered_only": ["layered-only"],
        "overlap": ["shared"],
    }
    value.update(overrides)
    return value


def _write_run(state_dir, run_id: str, started_at: datetime, report: dict | None = None) -> None:
    run_dir = state_dir / "runs" / started_at.strftime("%Y/%m/%d") / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "selection_engine": "layered_shadow_v1",
            }
        )
    )
    if report is not None:
        (run_dir / "layered-shadow.json").write_text(json.dumps(report))


def test_evaluate_layers_empty_state_is_valid_but_not_ready(tmp_path) -> None:
    report = evaluate_layers(tmp_path / "state", now=NOW)

    assert report["schema_version"] == "layered-evaluation-v1"
    assert report["window"] == {
        "from": "2026-09-05T12:00:00Z",
        "to": "2026-09-12T12:00:00Z",
        "days": 7,
    }
    assert report["runs"] == {"count": 0, "observed_days": 0}
    assert report["diagnostics"] == {
        "invalid_run_files": 0,
        "invalid_reports": 0,
        "duplicate_events": 0,
        "missing_reports": 0,
        "missing_source_metadata": 0,
    }
    assert report["release_readiness"]["status"] == "not_ready"
    assert len(report["release_readiness"]["missing_evidence"]) == 4


def test_evaluate_layers_filters_future_and_expired_runs_and_counts_actual_items(tmp_path) -> None:
    state = tmp_path / "state"
    valid = _report(
        [
            LayeredItem(
                event=_event("event-1", "source-a", "source-b"),
                layer=ContentLayer.DECISION_BRIEF,
                reasons=["MAJOR_GUARANTEE"],
                fact_status="evidence_supported",
            ),
            LayeredItem(
                event=_event("event-1", "source-a", "source-b"),
                layer=ContentLayer.SIGNAL_RADAR,
                reasons=["DUPLICATE_SHOULD_NOT_COUNT"],
                fact_status="unconfirmed_signal",
            ),
            LayeredItem(
                event=_event("event-2", "source-c", domain=CoverageDomain.EXPLORATION),
                layer=ContentLayer.EXPLORATION,
                reasons=["EVIDENCED_CROSS_DOMAIN_CONNECTION"],
                fact_status="evidence_supported",
            ),
        ]
    )
    valid["items"][0]["source_ids"] = ["same-source", "same-source"]
    valid["items"][1]["source_ids"] = ["other-source"]
    valid["items"][2]["source_ids"] = ["exploration-source"]
    _write_run(state, "valid", NOW - timedelta(days=1), valid)
    _write_run(state, "future", NOW + timedelta(hours=1), valid)
    _write_run(state, "expired", NOW - timedelta(days=8), valid)

    result = evaluate_layers(state, now=NOW)

    assert result["runs"] == {"count": 1, "observed_days": 1}
    assert result["layer_counts"] == {"decision_brief": 1, "signal_radar": 0, "exploration": 1}
    assert result["reason_counts"] == {
        "EVIDENCED_CROSS_DOMAIN_CONNECTION": 1,
        "MAJOR_GUARANTEE": 1,
    }
    assert result["domain_counts"] == {"ai_technology": 1, "exploration": 1}
    assert result["source_counts"] == {"exploration-source": 1, "same-source": 1}
    assert result["comparison"] == {
        "legacy_only": ["legacy-only"],
        "layered_only": ["layered-only"],
        "overlap": ["shared"],
    }
    assert result["diagnostics"]["duplicate_events"] == 1


def test_old_report_without_source_metadata_does_not_treat_candidate_ids_as_sources(tmp_path) -> None:
    state = tmp_path / "state"
    report = _report(
        [
            LayeredItem(
                event=_event("event-1", "article-a", "article-b"),
                layer=ContentLayer.DECISION_BRIEF,
                reasons=["MAJOR_GUARANTEE"],
                fact_status="evidence_supported",
            )
        ]
    )
    _write_run(state, "old", NOW - timedelta(hours=1), report)

    result = evaluate_layers(state, now=NOW)

    assert result["source_counts"] == {}
    assert result["diagnostics"]["missing_source_metadata"] == 1


def test_evaluate_layers_diagnoses_missing_and_invalid_reports(tmp_path) -> None:
    state = tmp_path / "state"
    _write_run(state, "missing", NOW - timedelta(hours=1))
    _write_run(state, "bad-json", NOW - timedelta(hours=2), {})
    bad_dir = state / "runs/2026/09/12/bad-json"
    (bad_dir / "layered-shadow.json").write_text("not-json\n")
    _write_run(
        state,
        "bad-model",
        NOW - timedelta(hours=3),
        {"schema_version": "layered-shadow-v1", "items": [{"layer": "bad"}]},
    )

    report = evaluate_layers(state, now=NOW)

    assert report["runs"]["count"] == 3
    assert report["diagnostics"] == {
        "invalid_run_files": 0,
        "invalid_reports": 2,
        "duplicate_events": 0,
        "missing_reports": 1,
        "missing_source_metadata": 0,
    }


def test_write_layered_evaluation_is_atomic_and_rejects_state_output(tmp_path) -> None:
    state = tmp_path / "state"
    _write_run(state, "run-1", NOW - timedelta(hours=1), _report([]))
    state_file = state / "runs/2026/09/12/run-1/run.json"
    before = hashlib.sha256(state_file.read_bytes()).hexdigest()
    output = tmp_path / "reports/evaluation.json"

    written = write_layered_evaluation(state, output, now=NOW)

    assert json.loads(output.read_text()) == written
    assert hashlib.sha256(state_file.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="state"):
        write_layered_evaluation(state, state / "evaluation.json", now=NOW)
    assert not (state / "evaluation.json").exists()
