"""The experiment must not count two duplicate slots as two important-event hits."""
import pytest

from experiments.multisource_validation.evaluate import (
    hosting_distribution,
    measure,
    validate_reference,
)


def test_event_hit_denominator_and_duplicate_slots_are_separate():
    arm = {
        "clusters": [["a"], ["b"]], "excluded": {}, "metrics": {},
        "selected": [{"member_ids": [i], "headline": i, "confidence": {}}
                     for i in ("a", "b")],
    }
    reference = {
        "pairs": [{"left_id": "a", "right_id": "b", "same_event": True}],
        "important_events": [
            {"member_ids": ["a", "b"], "importance": "must", "eligible_in_window": "yes"},
            {"member_ids": ["c"], "importance": "must", "eligible_in_window": "uncertain"},
        ],
    }
    reviews = {i: {"claims": [], "time_status": "unknown", "conflict": False,
                   "story_shape": "background"} for i in ("a", "b")}
    result = measure(arm, reference, reviews)
    assert result["important_reference"]["eligible_count"] == 1
    assert result["important_reference"]["hit_count"] == 1
    assert result["important_reference"]["duplicate_slots_in_reference_events"] == 1
    assert result["pair_evaluation"]["correctly_merged"] == 0
    assert result["selection_audit"]["without_bound_core_support"] == 2


def test_reference_rejects_quotation_from_a_different_body():
    reference = {
        "pairs": [{"left_id": "a", "right_id": "b", "same_event": True,
                   "left_quote": "actual", "right_quote": "imagined"}],
        "important_events": [],
    }
    with pytest.raises(ValueError, match="quotation mismatch"):
        validate_reference(reference, {"a": {"body": "actual"}, "b": {"body": "actual"}})


def test_two_channels_of_one_publisher_do_not_gain_two_distribution_votes():
    articles = {"a": {"publisher_group": "same_group", "language": "en"},
                "b": {"publisher_group": "same_group", "language": "en"},
                "c": {"publisher_group": "another_group", "language": "zh"}}
    result = hosting_distribution([{"member_ids": ["a", "b", "c"]}], articles)
    assert result["publisher_group"]["distribution"] == {"another_group": .5, "same_group": .5}
    assert result["publisher_group"]["hhi"] == .5


def test_one_prediction_spanning_two_reference_events_cannot_count_as_two_hits():
    arm = {"clusters": [["a", "b"]], "excluded": {}, "metrics": {},
           "selected": [{"member_ids": ["a", "b"], "headline": "overbroad", "confidence": {}}]}
    reference = {"pairs": [], "important_events": [
        {"event_id": i, "member_ids": [i], "importance": "must", "eligible_in_window": "yes"}
        for i in ("a", "b")]}
    reviews = {i: {"claims": [], "time_status": "confirmed", "conflict": False,
                   "story_shape": "single_event"} for i in ("a", "b")}
    result = measure(arm, reference, reviews)
    assert result["important_reference"]["hit_count"] == 1
    assert result["important_reference"]["article_overlap_hit_count"] == 2
    assert len(result["important_reference"]["cross_reference_merges"]) == 1
