from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import (
    CandidateArticle,
    DecisionRecord,
    DeliveryRecord,
    Event,
    RunRecord,
    SignalSnapshot,
    SourceHealthRecord,
)


class StateStore:
    """Private, append-auditable state stored as date-partitioned JSONL."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_run(
        self,
        run: RunRecord,
        candidates: Iterable[CandidateArticle],
        decisions: Iterable[DecisionRecord],
        events: Iterable[Event],
    ) -> None:
        candidates = list(candidates)
        decisions = [item.model_copy(update={"run_id": run.run_id}) for item in decisions]
        events = list(events)
        run_value = run.model_dump(mode="json")
        candidate_values = [item.model_dump(mode="json") for item in candidates]
        decision_values = [item.model_dump(mode="json") for item in decisions]
        event_values = [item.model_dump(mode="json") for item in events]
        run_dir = self._run_directory(run)
        if self._run_has_delivery(run.run_id):
            delivered_run_dir = self._find_run_directory(run.run_id)
            if self._run_contents_match(
                delivered_run_dir,
                run_value,
                candidate_values,
                decision_values,
                event_values,
            ):
                return
            raise ValueError(f"cannot overwrite delivered run: {run.run_id}")
        self._write_json(run_dir / "run.json", run_value)
        self._write_jsonl(
            run_dir / "candidates.jsonl", candidate_values
        )
        self._write_jsonl(run_dir / "decisions.jsonl", decision_values)
        self._write_jsonl(run_dir / "events.jsonl", event_values)
        for event in events:
            event_time = event.latest_event_time.astimezone(UTC)
            target = self.root / "events" / event_time.strftime("%Y/%m/%d") / "events.jsonl"
            existing = self._read_jsonl(target)
            self._write_jsonl(target, [*existing, event.model_dump(mode="json")])

    def load_recent_events(
        self,
        now: datetime,
        days: int = 90,
        *,
        channel: str = "web",
    ) -> list[Event]:
        self._validate_delivery_channel(channel)
        self._require_aware_datetime(now, "now")
        reference = now.astimezone(UTC)
        cutoff = reference - timedelta(days=days)
        events: dict[str, tuple[datetime, datetime, Event]] = {}
        for path in sorted((self.root / "deliveries").glob("*/*/deliveries.jsonl")):
            for value in self._read_jsonl(path):
                delivery = DeliveryRecord.model_validate(value)
                if (
                    delivery.run_id is None
                    or delivery.event is None
                    or delivery.channel != channel
                    or delivery.status != "delivered"
                    or delivery.delivered_at is None
                    or delivery.delivered_at.astimezone(UTC) > reference
                ):
                    continue
                event = delivery.event
                latest_event_time = event.latest_event_time.astimezone(UTC)
                if not cutoff <= latest_event_time <= reference:
                    continue
                delivered_at = delivery.delivered_at.astimezone(UTC)
                version = (latest_event_time, delivered_at, event)
                previous = events.get(event.event_id)
                if previous is None or version[:2] > previous[:2]:
                    events[event.event_id] = version
        return [version[2] for version in events.values()]

    def mark_candidates_delivered(
        self,
        run_id: str,
        candidate_ids: Iterable[str],
        delivered_at: datetime,
        *,
        channel: str = "web",
    ) -> None:
        self._validate_delivery_channel(channel)
        self._require_aware_datetime(delivered_at, "delivered_at")
        delivered_candidates = set(candidate_ids)
        run_dir = self._find_run_directory(run_id)
        run = RunRecord.model_validate_json((run_dir / "run.json").read_text())
        self._require_aware_datetime(run.started_at, "run started_at")
        if channel == "local" and run.selection_engine != "reviewed_local_v1":
            raise ValueError("local delivery requires reviewed_local_v1")
        if channel == "web" and run.selection_engine == "reviewed_local_v1":
            raise ValueError("reviewed_local_v1 cannot be marked as web delivery")
        delivered_at_utc = delivered_at.astimezone(UTC)
        if run.started_at.astimezone(UTC) > delivered_at_utc:
            raise ValueError("cannot deliver a future run")
        event_snapshot = run_dir / "events.jsonl"
        if not event_snapshot.exists():
            raise ValueError(f"run has no event snapshot: {run_id}")
        events = [
            Event.model_validate(event)
            for event in self._read_jsonl(event_snapshot)
        ]
        for event in events:
            self._require_aware_datetime(event.latest_event_time, "event latest_event_time")
        if any(event.latest_event_time.astimezone(UTC) > delivered_at_utc for event in events):
            raise ValueError("cannot deliver a future event")
        known_candidates = {
            candidate.candidate_id
            for candidate in (
                CandidateArticle.model_validate(value)
                for value in self._read_jsonl(run_dir / "candidates.jsonl")
            )
        }
        event_candidates = {
            candidate_id for event in events for candidate_id in event.candidate_ids
        }
        inconsistent_candidates = event_candidates - known_candidates
        if inconsistent_candidates:
            inconsistent = ", ".join(sorted(inconsistent_candidates))
            raise ValueError(
                f"event snapshot references unknown candidate ids for run {run_id}: "
                f"{inconsistent}"
            )
        unknown_candidates = delivered_candidates - known_candidates
        if unknown_candidates:
            unknown = ", ".join(sorted(unknown_candidates))
            raise ValueError(f"unknown candidate ids for run {run_id}: {unknown}")
        delivered_events = {
            (event.event_id, event.latest_update_id): event
            for event in events
            if delivered_candidates.intersection(event.candidate_ids)
        }
        target = self.root / "deliveries" / delivered_at_utc.strftime("%Y/%m") / "deliveries.jsonl"
        existing = self._read_jsonl(target)
        existing_versions = {
            (
                record.get("channel"),
                record.get("run_id"),
                record.get("event_id"),
                record.get("update_id"),
            )
            for path in (self.root / "deliveries").glob("*/*/deliveries.jsonl")
            for record in self._read_jsonl(path)
            if record.get("channel") == channel and record.get("status") == "delivered"
        }
        new_records = [
            DeliveryRecord(
                run_id=run_id,
                event_id=event_id,
                update_id=update_id,
                event=event,
                candidate_ids=sorted(delivered_candidates.intersection(event.candidate_ids)),
                channel=channel,
                status="delivered",
                delivered_at=delivered_at,
            ).model_dump(mode="json")
            for (event_id, update_id), event in sorted(
                delivered_events.items(), key=lambda item: (item[0][0], item[0][1] or "")
            )
            if (channel, run_id, event_id, update_id) not in existing_versions
        ]
        if new_records:
            self._write_jsonl(target, [*existing, *new_records])

    def write_signal_collection(
        self,
        captured_at: datetime,
        snapshots: Iterable[SignalSnapshot],
        health_records: Iterable[SourceHealthRecord],
        candidates: Iterable[CandidateArticle] = (),
    ) -> None:
        timestamp = captured_at.astimezone(UTC)
        partition = self.root / "signals" / timestamp.strftime("%Y/%m/%d")
        snapshot_target = partition / "snapshots.jsonl"
        health_target = partition / "source-health.jsonl"
        candidate_target = partition / "candidates.jsonl"
        existing_snapshots = self._read_jsonl(snapshot_target)
        existing_health = self._read_jsonl(health_target)
        existing_candidates = self._read_jsonl(candidate_target)
        self._write_jsonl(
            snapshot_target,
            [*existing_snapshots, *(item.model_dump(mode="json") for item in snapshots)],
        )
        self._write_jsonl(
            health_target,
            [*existing_health, *(item.model_dump(mode="json") for item in health_records)],
        )
        self._write_jsonl(
            candidate_target,
            [*existing_candidates, *(item.model_dump(mode="json") for item in candidates)],
        )


    def load_recent_snapshots(
        self, source_id: str, external_id: str, now: datetime, hours: int = 24
    ) -> list[SignalSnapshot]:
        reference = now.astimezone(UTC)
        cutoff = reference - timedelta(hours=hours)
        snapshots = [
            SignalSnapshot.model_validate(record)
            for path in sorted((self.root / "signals").glob("*/*/*/snapshots.jsonl"))
            for record in self._read_jsonl(path)
            if record.get("source_id") == source_id and record.get("external_id") == external_id
        ]
        return sorted(
            (
                snapshot
                for snapshot in snapshots
                if cutoff <= snapshot.captured_at.astimezone(UTC) <= reference
            ),
            key=lambda snapshot: snapshot.captured_at,
        )

    def _run_directory(self, run: RunRecord) -> Path:
        timestamp = run.started_at.astimezone(UTC)
        return self.root / "runs" / timestamp.strftime("%Y/%m/%d") / run.run_id

    def _run_has_delivery(self, run_id: str) -> bool:
        return any(
            record.get("run_id") == run_id and record.get("status") == "delivered"
            for path in (self.root / "deliveries").glob("*/*/deliveries.jsonl")
            for record in self._read_jsonl(path)
        )

    def _run_contents_match(
        self,
        run_dir: Path,
        run: dict[str, object],
        candidates: list[dict[str, object]],
        decisions: list[dict[str, object]],
        events: list[dict[str, object]],
    ) -> bool:
        run_path = run_dir / "run.json"
        return (
            run_path.exists()
            and json.loads(run_path.read_text()) == run
            and self._read_jsonl(run_dir / "candidates.jsonl") == candidates
            and self._read_jsonl(run_dir / "decisions.jsonl") == decisions
            and self._read_jsonl(run_dir / "events.jsonl") == events
        )

    def _find_run_directory(self, run_id: str, *, required: bool = True) -> Path | None:
        self._validate_run_id(run_id)
        matches = sorted(
            path
            for path in (self.root / "runs").glob("*/*/*/*")
            if path.is_dir() and path.name == run_id
        )
        if len(matches) == 1:
            return matches[0]
        if not matches and not required:
            return None
        if not matches:
            raise FileNotFoundError(f"run not found: {run_id}")
        raise ValueError(f"run_id is not unique: {run_id}")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty path segment")

    @staticmethod
    def _validate_delivery_channel(channel: str) -> None:
        if channel not in {"web", "local"}:
            raise ValueError("delivery channel must be web or local")

    @staticmethod
    def _require_aware_datetime(value: datetime, label: str) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{label} must be timezone-aware")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _write_json(target: Path, value: dict[str, object]) -> None:
        StateStore._atomic_write(target, json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_jsonl(target: Path, records: Iterable[dict[str, object]]) -> None:
        contents = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        )
        StateStore._atomic_write(target, contents)

    @staticmethod
    def _atomic_write(target: Path, contents: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            stream.write(contents)
            temporary = Path(stream.name)
        temporary.replace(target)

    def load_signal_snapshots(self, now: datetime, hours: int = 24) -> list[SignalSnapshot]:
        reference = self._as_utc(now, "now")
        cutoff = reference - timedelta(hours=hours)
        snapshots = [
            SignalSnapshot.model_validate(record)
            for path in sorted((self.root / "signals").glob("*/*/*/snapshots.jsonl"))
            for record in self._read_jsonl(path)
        ]
        return sorted(
            (
                snapshot
                for snapshot in snapshots
                if cutoff <= self._as_utc(snapshot.captured_at, "captured_at") <= reference
            ),
            key=lambda snapshot: self._as_utc(snapshot.captured_at, "captured_at"),
        )


    def load_recent_signal_candidates(
        self, now: datetime, hours: int = 36
    ) -> list[CandidateArticle]:
        reference = self._as_utc(now, "now")
        cutoff = reference - timedelta(hours=hours)
        latest: dict[tuple[str, str], CandidateArticle] = {}
        for path in sorted((self.root / "signals").glob("*/*/*/candidates.jsonl")):
            for record in self._read_jsonl(path):
                candidate = CandidateArticle.model_validate(record)
                published_at = self._as_utc(candidate.published_at, "published_at")
                discovered_at = self._as_utc(candidate.discovered_at, "discovered_at")
                if not (
                    cutoff <= published_at <= reference
                    and discovered_at <= reference
                ):
                    continue
                key = (candidate.source_id, candidate.candidate_id)
                previous = latest.get(key)
                if previous is None or discovered_at >= self._as_utc(
                    previous.discovered_at, "discovered_at"
                ):
                    latest[key] = candidate
        return sorted(
            latest.values(),
            key=lambda candidate: (
                self._as_utc(candidate.published_at, "published_at"),
                self._as_utc(candidate.discovered_at, "discovered_at"),
                candidate.source_id,
                candidate.candidate_id,
            ),
        )


    def write_private_report(self, run: RunRecord, name: str, report: dict) -> None:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", "..", ".json"}
            or ".." in name
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or not name.endswith(".json")
        ):
            raise ValueError("report name must be a simple .json filename")
        run_dir = self._run_directory(run)
        target = run_dir / name
        if target.parent != run_dir:
            raise ValueError("report name must not escape the run directory")
        try:
            target.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("report path must remain inside the state root") from exc
        self._write_json(target, report)


    @staticmethod
    def _as_utc(value: datetime, field: str) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")
        return value.astimezone(UTC)
