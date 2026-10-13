from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .models import CandidateArticle, DeliveryRecord, RunRecord
from .state import StateStore

_SCHEMA = "publication-intent-v1"
_RECEIPT_FIELDS = {
    "schema",
    "run_id",
    "digest_date",
    "candidate_ids",
    "page_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def record_publication(
    store: StateStore,
    run: RunRecord,
    digest_date: date,
    candidate_ids: list[str],
    rendered_page: Path,
) -> None:
    """Record the exact private publication intent for one persisted run."""
    run_dir = store._run_directory(run)
    persisted_run = _load_bound_run(store, run_dir)
    if persisted_run.model_dump(mode="json") != run.model_dump(mode="json"):
        raise ValueError("publication run does not match persisted run binding")
    _validate_candidate_ids(store, run_dir, candidate_ids)
    _require_regular_file(rendered_page, "rendered page")

    receipt = {
        "schema": _SCHEMA,
        "run_id": run.run_id,
        "digest_date": digest_date.isoformat(),
        "candidate_ids": candidate_ids,
        "page_sha256": _file_sha256(rendered_page),
    }
    receipt_path = run_dir / "publication.json"
    if receipt_path.is_symlink():
        raise ValueError("publication receipt must not be a symlink")
    if receipt_path.exists():
        existing = _load_receipt(receipt_path)
        if existing == receipt:
            return
        raise ValueError(f"cannot overwrite publication receipt with different content: {run.run_id}")
    StateStore._atomic_write(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n",
    )


def reconcile_publications(
    store: StateStore,
    published_dir: Path,
    *,
    now: datetime,
) -> list[str]:
    """Recover unmatched delivery facts and return newly confirmed run IDs."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    _require_directory(published_dir, "published directory")

    recovered: list[str] = []
    runs_root = store.root / "runs"
    if not runs_root.exists():
        return recovered
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise ValueError("private runs root must be a regular directory")

    # A successful mark writes all selected event snapshots atomically. Load
    # those confirmations once; do not parse every old candidate batch or
    # rewrite delivery partitions on each daily reconciliation.
    confirmed_runs = {
        delivery.run_id
        for path in (store.root / "deliveries").glob("*/*/deliveries.jsonl")
        for value in store._read_jsonl(path)
        for delivery in [DeliveryRecord.model_validate(value)]
        if delivery.channel == "web"
        and delivery.status == "delivered"
        and delivery.run_id is not None
        and delivery.event is not None
        and delivery.delivered_at is not None
        and delivery.delivered_at <= now
    }

    for receipt_path in sorted(runs_root.glob("*/*/*/*/publication.json")):
        _reject_symlink_components(receipt_path, runs_root)
        receipt = _load_receipt(receipt_path)
        run_dir = receipt_path.parent
        run = _load_bound_run(store, run_dir)
        if receipt["run_id"] != run.run_id:
            raise ValueError("publication receipt run_id does not match run binding")
        if run.selection_engine == "reviewed_local_v1":
            continue
        if run.run_id in confirmed_runs:
            continue
        candidate_ids = receipt["candidate_ids"]

        published_page = published_dir / f"{receipt['digest_date']}.html"
        if published_page.is_symlink():
            raise ValueError("published page must not be a symlink")
        if not published_page.exists():
            continue
        _require_regular_file(published_page, "published page")
        if _file_sha256(published_page) != receipt["page_sha256"]:
            continue

        _validate_candidate_ids(store, run_dir, candidate_ids)
        store.mark_candidates_delivered(run.run_id, candidate_ids, now)
        recovered.append(run.run_id)
    return recovered


def confirm_local_reading(
    store: StateStore,
    run_id: str,
    rendered_page: Path,
    *,
    now: datetime,
) -> bool:
    """Confirm that an exact reviewed local page was read for its owning run."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    run_dir = store._find_run_directory(run_id)
    run = _load_bound_run(store, run_dir)
    if run.selection_engine != "reviewed_local_v1":
        raise ValueError("local reading requires reviewed_local_v1")

    receipt = _load_receipt(run_dir / "publication.json")
    if receipt["run_id"] != run.run_id:
        raise ValueError("publication receipt run_id does not match run binding")
    expected_name = f"{receipt['digest_date']}.html"
    if rendered_page.name != expected_name:
        raise ValueError("rendered page basename does not match receipt date")
    _require_regular_file(rendered_page, "rendered page")
    if _file_sha256(rendered_page) != receipt["page_sha256"]:
        raise ValueError("rendered page hash does not match publication receipt")
    candidate_ids = receipt["candidate_ids"]
    if not candidate_ids:
        raise ValueError("cannot confirm an empty local edition")
    _validate_candidate_ids(store, run_dir, candidate_ids)

    already_confirmed = any(
        record.get("run_id") == run.run_id
        and record.get("channel") == "local"
        and record.get("status") == "delivered"
        for path in (store.root / "deliveries").glob("*/*/deliveries.jsonl")
        for record in store._read_jsonl(path)
    )
    store.mark_candidates_delivered(
        run.run_id,
        candidate_ids,
        now,
        channel="local",
    )
    return not already_confirmed


def _load_bound_run(store: StateStore, run_dir: Path) -> RunRecord:
    runs_root = store.root / "runs"
    _reject_symlink_components(run_dir, runs_root)
    if not run_dir.is_dir():
        raise ValueError("publication run directory is missing")
    run_path = run_dir / "run.json"
    _require_regular_file(run_path, "run record")
    try:
        raw = json.loads(run_path.read_text())
        run = RunRecord.model_validate(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid publication run record") from error
    if store._run_directory(run).resolve(strict=False) != run_dir.resolve(strict=False):
        raise ValueError("publication run partition does not match run binding")
    return run


def _validate_candidate_ids(
    store: StateStore,
    run_dir: Path,
    candidate_ids: Any,
) -> None:
    if (
        not isinstance(candidate_ids, list)
        or any(type(value) is not str or not value for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("publication candidate_ids must be unique non-empty strings")
    candidates_path = run_dir / "candidates.jsonl"
    _require_regular_file(candidates_path, "candidate snapshot")
    try:
        known = {
            CandidateArticle.model_validate(value).candidate_id
            for value in store._read_jsonl(candidates_path)
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid candidate snapshot") from error
    unknown = set(candidate_ids) - known
    if unknown:
        raise ValueError(f"publication receipt references unknown candidate ids: {sorted(unknown)}")


def _load_receipt(path: Path) -> dict[str, Any]:
    _require_regular_file(path, "publication receipt")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid publication receipt JSON") from error
    if not isinstance(value, dict) or set(value) != _RECEIPT_FIELDS:
        raise ValueError("publication receipt has invalid fields")
    if value["schema"] != _SCHEMA:
        raise ValueError("publication receipt has unknown schema")
    if type(value["run_id"]) is not str or not value["run_id"]:
        raise ValueError("publication receipt has invalid run_id")
    StateStore._validate_run_id(value["run_id"])
    if type(value["digest_date"]) is not str:
        raise ValueError("publication receipt has invalid digest_date")
    try:
        parsed_date = date.fromisoformat(value["digest_date"])
    except ValueError as error:
        raise ValueError("publication receipt has invalid digest_date") from error
    if parsed_date.isoformat() != value["digest_date"]:
        raise ValueError("publication receipt has invalid digest_date")
    _validate_candidate_ids_shape(value["candidate_ids"])
    if type(value["page_sha256"]) is not str or not _SHA256.fullmatch(value["page_sha256"]):
        raise ValueError("publication receipt has invalid page_sha256")
    return value


def _validate_candidate_ids_shape(candidate_ids: Any) -> None:
    if (
        not isinstance(candidate_ids, list)
        or any(type(value) is not str or not value for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("publication receipt has invalid candidate_ids")


def _reject_symlink_components(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("publication path is outside private runs root") from error
    current = root
    if current.is_symlink():
        raise ValueError("private runs root must not be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("publication path must not contain symlinks")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")


def _require_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
