from __future__ import annotations

from pydantic import BaseModel, Field

from .models import Event, ScoreCard, SelectionPolicy


class EvidenceAssessment(BaseModel):
    quality: float = Field(ge=0, le=100)
    has_primary_fact: bool = False
    independent_reporting_count: int = Field(default=0, ge=0)


def is_major(
    event: Event,
    scores: ScoreCard,
    evidence: EvidenceAssessment,
    policy: SelectionPolicy,
) -> bool:
    del policy
    return (
        scores.importance >= 85
        and evidence.quality >= 70
        and (
            evidence.has_primary_fact
            or evidence.independent_reporting_count >= 2
        )
        and event.event_type != "news"
    )
