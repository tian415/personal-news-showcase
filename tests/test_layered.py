from datetime import UTC, datetime, timedelta

from news_assistant.events import assign_events
from news_assistant.layered import ContentLayer, run_layered_selection
from news_assistant.models import (
    CandidateArticle,
    CoverageDomain,
    EvidenceRole,
    Preferences,
    SignalSnapshot,
    Topic,
)

NOW = datetime(2026, 9, 12, 7, tzinfo=UTC)


def item(identifier, role=EvidenceRole.COMMUNITY_SIGNAL, **changes):
    fields = {
        "id": identifier,
        "candidate_id": identifier,
        "source_id": identifier,
        "publisher_group": identifier,
        "signal_external_id": "42",
        "title": "New device launches",
        "url": f"https://example.com/{identifier}",
        "source": identifier,
        "language": "en",
        "published_at": NOW,
        "discovered_at": NOW,
        "topic": Topic.AI,
        "coverage_domain": CoverageDomain.AI_TECHNOLOGY,
        "evidence_role": role,
        "quality_score": 0.8,
        "importance_score": 0.7,
    }
    return CandidateArticle(**(fields | changes))


def rising(source):
    return [
        SignalSnapshot(
            source_id=source,
            external_id="42",
            captured_at=NOW - timedelta(hours=6),
            rank=5,
            reply_count=10,
        ),
        SignalSnapshot(source_id=source, external_id="42", captured_at=NOW, rank=2, reply_count=40),
    ]


def test_two_independent_rising_communities_enter_radar_without_becoming_facts():
    result = run_layered_selection(
        [item("forum-a"), item("forum-b")],
        [],
        Preferences(),
        snapshots=rising("forum-a") + rising("forum-b"),
        now=NOW,
    )
    assert len(result.items) == 1
    assert result.items[0].layer is ContentLayer.SIGNAL_RADAR
    assert result.items[0].event.major is False
    assert result.items[0].fact_status == "unconfirmed_signal"
    assert result.items[0].independent_platforms == 2
    assert all(trend.attention_acceleration is None for trend in result.items[0].trends.values())


def test_two_feeds_of_one_publisher_do_not_fake_cross_platform_evidence():
    result = run_layered_selection(
        [item("hot", publisher_group="forum"), item("latest", publisher_group="forum")],
        [],
        Preferences(),
        snapshots=rising("hot") + rising("latest"),
        now=NOW,
    )
    assert result.items == []
    assert "INSUFFICIENT_TREND_EVIDENCE" in next(iter(result.excluded.values()))


def test_single_snapshots_do_not_enter_radar_even_if_reply_counts_are_large():
    result = run_layered_selection(
        [item("a"), item("b")],
        [],
        Preferences(),
        snapshots=[rising("a")[-1], rising("b")[-1]],
        now=NOW,
    )
    assert result.items == []


def test_major_fact_survives_negative_preference_and_appears_only_once():
    official = item("official", EvidenceRole.PRIMARY_FACT, importance_score=0.95)
    result = run_layered_selection(
        [official, item("a"), item("b")],
        [],
        Preferences(topic_weights={Topic.AI: 0.5}),
        snapshots=rising("a") + rising("b"),
        now=NOW,
    )
    assert len(result.items) == 1
    assert result.items[0].layer is ContentLayer.DECISION_BRIEF
    assert "MAJOR_GUARANTEE" in result.items[0].reasons


def test_exploration_requires_explicit_connection_and_evidence():
    science = item(
        "science",
        EvidenceRole.PRIMARY_FACT,
        importance_score=0.5,
        topic=Topic.SCIENCE,
        coverage_domain=CoverageDomain.SCIENCE_HEALTH,
        expansion_connection="This material reduces chip fabrication energy.",
    )
    result = run_layered_selection([science], [], Preferences(), snapshots=[], now=NOW)
    assert result.items[0].layer is ContentLayer.EXPLORATION
    without_connection = science.model_copy(update={"expansion_connection": None})
    assert (
        run_layered_selection([without_connection], [], Preferences(), snapshots=[], now=NOW).items
        == []
    )


def test_major_events_can_exceed_soft_limit_and_future_events_are_rejected():
    facts = [
        item(
            str(i), EvidenceRole.PRIMARY_FACT, title=f"Product {i} launches", importance_score=0.95
        )
        for i in range(6)
    ]
    future = item(
        "future",
        EvidenceRole.PRIMARY_FACT,
        title="Future release",
        published_at=NOW + timedelta(days=1),
    )
    result = run_layered_selection(facts + [future], [], Preferences(), snapshots=[], now=NOW)
    assert len(result.items) == 6
    assert all(entry.event.major for entry in result.items)


def test_candidates_older_than_the_feed_window_are_rejected():
    old = item(
        "old",
        EvidenceRole.PRIMARY_FACT,
        published_at=NOW - timedelta(hours=36, seconds=1),
        importance_score=0.95,
    )

    result = run_layered_selection([old], [], Preferences(), snapshots=[], now=NOW)

    assert result.items == []


def test_duplicate_candidate_ids_are_rejected_instead_of_silently_overwriting():
    first = item("duplicate", EvidenceRole.PRIMARY_FACT, title="Alpha launches")
    second = item("duplicate", EvidenceRole.PRIMARY_FACT, title="Beta launches")

    try:
        run_layered_selection([first, second], [], Preferences(), snapshots=[], now=NOW)
    except ValueError as error:
        assert "duplicate candidate_id" in str(error)
    else:
        raise AssertionError("duplicate candidate_id must be rejected")


def test_private_report_explains_legacy_difference_and_empty_layers():
    result = run_layered_selection(
        [item("fact", EvidenceRole.PRIMARY_FACT)], [], Preferences(), snapshots=[], now=NOW
    )
    report = result.comparison(["legacy-only"])
    assert report["schema_version"] == "layered-shadow-v1"
    assert report["legacy_only"] == ["legacy-only"]
    assert report["layered_only"] == ["fact"]
    assert report["layer_counts"] == {"decision_brief": 1, "signal_radar": 0, "exploration": 0}


def test_tracked_material_update_with_sufficient_evidence_enters_decision_once():
    rumor = item(
        "rumor",
        EvidenceRole.INDEPENDENT_REPORTING,
        title="Acme acquisition rumor",
        importance_score=0.5,
    )
    historical = assign_events([rumor], []).events
    confirmation = item(
        "confirmed",
        EvidenceRole.PRIMARY_FACT,
        source_id="acme",
        publisher_group="acme",
        title="Acme signs acquisition agreement",
        importance_score=0.5,
    )

    result = run_layered_selection(
        [confirmation],
        historical,
        Preferences(),
        snapshots=[],
        now=NOW,
        tracked_event_ids=[historical[0].event_id],
    )

    assert len(result.items) == 1
    selected = result.items[0]
    assert selected.layer is ContentLayer.DECISION_BRIEF
    assert selected.event.event_id == historical[0].event_id
    assert len({item.event.event_id for item in result.items}) == 1
    assert "TRACKED_MATERIAL_UPDATE" in selected.reasons
    assert len(selected.event.update_history) == 1


def test_tracking_does_not_resurrect_non_material_duplicate_or_low_quality_update():
    rumor = item(
        "rumor",
        EvidenceRole.INDEPENDENT_REPORTING,
        title="Acme acquisition rumor",
        importance_score=0.5,
    )
    historical = assign_events([rumor], []).events
    repeated = item(
        "repeated",
        EvidenceRole.PRIMARY_FACT,
        title="Acme acquisition rumor",
        importance_score=0.5,
    )
    duplicate = run_layered_selection(
        [repeated],
        historical,
        Preferences(),
        snapshots=[],
        now=NOW,
        tracked_event_ids=[historical[0].event_id],
    )
    assert duplicate.items == []

    low_quality = item(
        "confirmed-low-quality",
        EvidenceRole.PRIMARY_FACT,
        title="Acme signs acquisition agreement",
        importance_score=0.5,
        quality_score=0.5,
    )
    insufficient = run_layered_selection(
        [low_quality],
        historical,
        Preferences(),
        snapshots=[],
        now=NOW,
        tracked_event_ids=[historical[0].event_id],
    )
    assert insufficient.items == []
    assert "BELOW_QUALITY_THRESHOLD" in insufficient.excluded[historical[0].event_id]
