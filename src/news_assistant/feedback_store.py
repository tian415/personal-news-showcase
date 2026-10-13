"""Import feedback only against immutable deliveries confirmed for one channel."""

import json
from datetime import datetime
from pathlib import Path

from .feedback import FeedbackState, FeedbackTarget, merge_feedback, public_event_ref
from .models import CandidateArticle, DeliveryRecord
from .state import StateStore


def import_feedback_file(
    input_path: Path,
    state_dir: Path,
    *,
    now: datetime,
    channel: str = "web",
) -> FeedbackState:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    StateStore._validate_delivery_channel(channel)
    if input_path.stat().st_size > 5_000_000:
        raise ValueError("feedback file exceeds 5 MB")
    store = StateStore(state_dir)
    targets: dict[str, FeedbackTarget] = {}
    for path in sorted((state_dir / "deliveries").glob("*/*/deliveries.jsonl")):
        for raw in store._read_jsonl(path):
            delivery = DeliveryRecord.model_validate(raw)
            if (
                delivery.channel != channel or delivery.status != "delivered"
                or delivery.run_id is None or delivery.event is None
                or delivery.delivered_at is None or delivery.delivered_at > now
                or delivery.event.latest_event_time > now
            ):
                continue
            event = delivery.event
            run_dir = store._find_run_directory(delivery.run_id)
            # Read only this published run. A newer, unpublished batch must
            # never replace the topic/source attached to an old feedback link.
            candidates = {
                item.candidate_id: item
                for value in store._read_jsonl(run_dir / "candidates.jsonl")
                for item in [CandidateArticle.model_validate(value)]
            }
            ids = delivery.candidate_ids or event.candidate_ids
            candidate = next((candidates[i] for i in ids if i in candidates), None)
            if candidate is None:
                raise ValueError("published delivery has no candidate snapshot")
            ref = public_event_ref(event.event_id, run_id=delivery.run_id,
                                   update_id=delivery.update_id)
            targets[ref] = FeedbackTarget(event_id=event.event_id,
                                          topic=candidate.topic, source=candidate.source)
    output = state_dir / (
        "feedback-state.json" if channel == "web" else "local-feedback-state.json"
    )
    previous = FeedbackState.model_validate_json(output.read_text()) if output.exists() else None
    state = merge_feedback(json.loads(input_path.read_text()), targets, previous=previous, now=now)
    store._write_json(output, state.model_dump(mode="json"))
    return state
