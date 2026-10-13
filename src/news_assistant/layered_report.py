"""Read-only evaluation of private layered shadow reports."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import ValidationError

from .layered import ContentLayer, LayeredSelectionResult
from .models import RunRecord

_MISSING_EVIDENCE = [
    "major_event_recall_not_proven",
    "fact_error_manual_acceptance_not_completed",
    "continuous_2_to_4_week_observation_not_proven",
    "8_week_reference_set_not_proven",
]


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value, "datetime").isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_runs(state_dir: Path) -> tuple[list[tuple[Path, RunRecord, datetime]], int]:
    runs: list[tuple[Path, RunRecord, datetime]] = []
    invalid = 0
    for path in sorted((state_dir / "runs").glob("*/*/*/*/run.json")):
        try:
            raw = _load_json(path)
            if not isinstance(raw, dict):
                raise TypeError("run.json must contain an object")
            started_at = _utc(
                datetime.fromisoformat(str(raw.get("started_at"))),
                "started_at",
            )
            run = RunRecord.model_validate(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, ValidationError):
            invalid += 1
            continue
        runs.append((path.parent, run, started_at))
    return runs, invalid


def _validate_comparison_fields(raw: dict[str, object]) -> None:
    for field in ("legacy_only", "layered_only", "overlap"):
        values = raw.get(field)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{field} must be a list of strings")


def _validate_item_shape(raw: dict[str, object]) -> None:
    items = raw.get("items")
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    for item in items:
        if not isinstance(item, dict) or not {"event", "layer", "reasons", "fact_status"}.issubset(item):
            raise ValueError("layered item is missing required fields")
        if "source_ids" in item and (
            not isinstance(item["source_ids"], list)
            or not all(isinstance(source_id, str) for source_id in item["source_ids"])
        ):
            raise ValueError("source_ids must be a list of strings")
        event = item.get("event")
        if not isinstance(event, dict) or not {"event_id", "coverage_domain", "candidate_ids"}.issubset(event):
            raise ValueError("layered item event is missing required fields")


def _load_layered_report(path: Path) -> tuple[LayeredSelectionResult, dict[str, object]]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise TypeError("layered report must contain an object")
    if raw.get("schema_version") != "layered-shadow-v1":
        raise ValueError("unsupported layered report schema")
    _validate_comparison_fields(raw)
    _validate_item_shape(raw)
    return LayeredSelectionResult.model_validate(raw), raw


def evaluate_layers(state_dir: Path, *, now: datetime, days: int = 7) -> dict[str, object]:
    """Aggregate valid layered reports in a UTC run-start window without writing state."""

    reference = _utc(now, "now")
    if days <= 0:
        raise ValueError("days must be positive")
    cutoff = reference - timedelta(days=days)
    root = Path(state_dir)
    run_records, invalid_run_files = _iter_runs(root)
    layer_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    comparison_sets = {"legacy_only": set(), "layered_only": set(), "overlap": set()}
    observed_days: set[datetime.date] = set()
    diagnostics = {
        "invalid_run_files": invalid_run_files,
        "invalid_reports": 0,
        "duplicate_events": 0,
        "missing_reports": 0,
        "missing_source_metadata": 0,
    }
    valid_run_count = 0

    for run_dir, _run, started_at in run_records:
        if not cutoff <= started_at <= reference:
            continue
        valid_run_count += 1
        observed_days.add(started_at.date())
        report_path = run_dir / "layered-shadow.json"
        if not report_path.exists():
            diagnostics["missing_reports"] += 1
            continue
        try:
            result, raw = _load_layered_report(report_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, ValidationError):
            diagnostics["invalid_reports"] += 1
            continue

        for field, values in comparison_sets.items():
            values.update(raw[field])
        seen_events: set[str] = set()
        for index, item in enumerate(result.items):
            event_id = item.event.event_id
            if event_id in seen_events:
                diagnostics["duplicate_events"] += 1
                continue
            seen_events.add(event_id)
            layer_counts[item.layer.value] += 1
            reason_counts.update(item.reasons)
            domain_counts[item.event.coverage_domain.value] += 1
            source_ids = raw["items"][index].get("source_ids", ())
            if not source_ids:
                diagnostics["missing_source_metadata"] += 1
                continue
            source_counts.update(set(source_ids))

    return {
        "schema_version": "layered-evaluation-v1",
        "window": {"from": _iso(cutoff), "to": _iso(reference), "days": days},
        "runs": {"count": valid_run_count, "observed_days": len(observed_days)},
        "layer_counts": {layer.value: layer_counts[layer.value] for layer in ContentLayer},
        "reason_counts": dict(sorted(reason_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "comparison": {
            field: sorted(values) for field, values in comparison_sets.items()
        },
        "diagnostics": diagnostics,
        "release_readiness": {
            "status": "not_ready",
            "missing_evidence": list(_MISSING_EVIDENCE),
        },
    }


def write_layered_evaluation(
    state_dir: Path, output: Path, *, now: datetime, days: int = 7
) -> dict[str, object]:
    """Write an evaluation atomically while keeping the private state read-only."""

    root = Path(state_dir).resolve()
    target = Path(output)
    lexical_target = target.absolute()
    if lexical_target == root or root in lexical_target.parents:
        raise ValueError("evaluation output must be outside state")
    try:
        target.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("evaluation output must be outside state")

    report = evaluate_layers(state_dir, now=now, days=days)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
