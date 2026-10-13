from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime, time
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _record_is_complete(record: dict[str, object], record_kind: str | None) -> bool:
    if record_kind == "candidate":
        return bool(record.get("candidate_id")) and bool(
            record.get("source_id") or record.get("source")
        )
    if record_kind == "decision":
        return (
            isinstance(record.get("selected"), bool)
            and bool(record.get("stage"))
            and bool(record.get("primary_reason"))
            and bool(record.get("event_id") or record.get("candidate_id"))
        )
    if record_kind == "snapshot":
        return all(record.get(key) for key in ("source_id", "external_id", "captured_at", "rank"))
    if record_kind == "health":
        return all(
            record.get(key) is not None
            for key in ("source_id", "captured_at", "status", "candidate_count")
        )
    return True


def _json_records(
    path: Path, diagnostics: Counter[str], record_kind: str | None = None
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError:
        diagnostics["invalid_json_lines"] += 1
        return []
    records: list[dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            diagnostics["invalid_json_lines"] += 1
            continue
        if not isinstance(value, dict):
            diagnostics["invalid_json_lines"] += 1
            continue
        if not _record_is_complete(value, record_kind):
            diagnostics["invalid_state_records"] += 1
            continue
        records.append(value)
    return records


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("state timestamp must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("state timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _iter_run_records(
    state_dir: Path, diagnostics: Counter[str]
) -> list[tuple[Path, dict[str, object], datetime]]:
    records: list[tuple[Path, dict[str, object], datetime]] = []
    for path in sorted((state_dir / "runs").glob("*/*/*/*/run.json")):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise TypeError("run.json must contain an object")
            started_at = _parse_datetime(value.get("started_at"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            diagnostics["invalid_run_files"] += 1
            continue
        records.append((path.parent, value, started_at))
    return records


def _startup_delay_minutes(started_at: datetime) -> float:
    local = started_at.astimezone(_SHANGHAI)
    target = datetime.combine(local.date(), time(8), tzinfo=_SHANGHAI)
    return (local - target).total_seconds() / 60


def _run_identity(state_dir: Path, run_dir: Path) -> str:
    """Use the canonical relative runs directory, not only the persisted run_id."""

    return run_dir.relative_to(state_dir / "runs").as_posix()


def _valid_count_fields(
    value: object, diagnostics: Counter[str]
) -> dict[str, int]:
    if not isinstance(value, dict):
        diagnostics["invalid_run_fields"] += 1
        return {}
    valid: dict[str, int] = {}
    for key, raw_value in value.items():
        if isinstance(raw_value, bool):
            diagnostics["invalid_run_fields"] += 1
            continue
        try:
            count = int(raw_value)
            if count < 0 or (isinstance(raw_value, float) and raw_value != count):
                raise ValueError("count must be a non-negative integer")
        except (TypeError, ValueError, OverflowError):
            diagnostics["invalid_run_fields"] += 1
            continue
        valid[str(key)] = count
    return valid


def evaluate_state(state_dir: Path) -> dict[str, object]:
    """Read private state and return a machine-readable aggregate evaluation."""

    root = Path(state_dir)
    diagnostics: Counter[str] = Counter()
    runs = _iter_run_records(root, diagnostics)
    started_at = sorted(started for _, _, started in runs)
    candidate_count = 0
    selected_units: set[str] = set()
    reason_counts: Counter[str] = Counter()
    source_candidate_counts: Counter[str] = Counter()
    startup_delays: list[float] = []

    for run_dir, run, run_started_at in runs:
        startup_delays.append(_startup_delay_minutes(run_started_at))
        candidate_path = run_dir / "candidates.jsonl"
        candidates = _json_records(candidate_path, diagnostics, "candidate")
        if candidate_path.exists():
            candidate_count += len(candidates)
            source_candidate_counts.update(
                str(candidate.get("source_id") or candidate.get("source") or "unknown")
                for candidate in candidates
            )
        else:
            # Older state may only have the per-run source counter in run.json.
            source_counts = _valid_count_fields(run.get("source_candidate_counts", {}), diagnostics)
            candidate_count += sum(source_counts.values())
            source_candidate_counts.update(source_counts)

        decision_path = run_dir / "decisions.jsonl"
        decisions = _json_records(decision_path, diagnostics, "decision")
        if decision_path.exists():
            # A selected event may have multiple decision records; count each
            # unique event per run, falling back to the candidate when event_id is absent.
            run_identity = _run_identity(root, run_dir)
            for decision in decisions:
                if decision.get("selected") is True:
                    key = decision.get("event_id") or decision.get("candidate_id")
                    if key:
                        selected_units.add(f"{run_identity}:{key}")
                    else:
                        diagnostics["invalid_state_records"] += 1
            reason_counts.update(
                str(decision["primary_reason"])
                for decision in decisions
                if decision.get("primary_reason")
            )
        else:
            # Preserve metrics for state written before decision JSONL was introduced.
            run_reasons = _valid_count_fields(run.get("reason_counts", {}), diagnostics)
            reason_counts.update(run_reasons)

    snapshot_count = sum(
        len(_json_records(path, diagnostics, "snapshot"))
        for path in sorted((root / "signals").glob("*/*/*/snapshots.jsonl"))
    )
    health_records = [
        record
        for path in sorted((root / "signals").glob("*/*/*/source-health.jsonl"))
        for record in _json_records(path, diagnostics, "health")
    ]
    health_status_counts = Counter(str(record["status"]) for record in health_records if record.get("status"))

    average_delay = sum(startup_delays) / len(startup_delays) if startup_delays else 0.0
    maximum_delay = max(startup_delays) if startup_delays else 0.0
    return {
        "schema_version": "state-evaluation-v1",
        "runs": {
            "count": len(runs),
            "first_started_at": _format_datetime(started_at[0] if started_at else None),
            "last_started_at": _format_datetime(started_at[-1] if started_at else None),
        },
        "candidates": {"count": candidate_count, "selected_count": len(selected_units)},
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_candidate_counts": dict(sorted(source_candidate_counts.items())),
        "signals": {
            "snapshot_count": snapshot_count,
            "source_health_count": len(health_records),
            "health_status_counts": dict(sorted(health_status_counts.items())),
        },
        "startup_delay_minutes": {
            "average": round(average_delay, 2),
            "max": round(maximum_delay, 2),
        },
        "diagnostics": {
            "invalid_json_lines": diagnostics["invalid_json_lines"],
            "invalid_run_files": diagnostics["invalid_run_files"],
            "invalid_run_fields": diagnostics["invalid_run_fields"],
            "invalid_state_records": diagnostics["invalid_state_records"],
        },
    }


def write_state_evaluation(state_dir: Path, output: Path) -> dict[str, object]:
    """Write an evaluation with an atomic replacement and return its contents."""

    report = evaluate_state(state_dir)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, target)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return report
