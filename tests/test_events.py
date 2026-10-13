from datetime import UTC, datetime

from news_assistant.events import assign_events
from news_assistant.models import CandidateArticle, CoverageDomain, ReasonCode, SourceTier, Topic


def candidate(identifier: str, title: str, source: str, tier: SourceTier = SourceTier.T1) -> CandidateArticle:
    now = datetime(2026, 8, 13, 8, tzinfo=UTC)
    return CandidateArticle(
        id=identifier,
        candidate_id=identifier,
        title=title,
        url=f"https://{source}.example.com/{identifier}",
        source=source,
        source_id=source,
        source_tier=tier,
        coverage_domain=CoverageDomain.AI_TECHNOLOGY,
        language="en",
        published_at=now,
        topic=Topic.AI,
    )


def test_five_reposts_about_one_announcement_form_one_event_with_all_sources() -> None:
    articles = [candidate(str(index), "OpenAI launches GPT-Next", f"source-{index}") for index in range(5)]

    result = assign_events(articles, [])

    assert len(result.events) == 1
    assert result.events[0].candidate_ids == ["0", "1", "2", "3", "4"]
    assert result.events[0].independent_source_count == 5


def test_commentary_about_historical_event_is_marked_no_delta() -> None:
    first = candidate("first", "OpenAI launches GPT-Next", "official", SourceTier.T0)
    historical = assign_events([first], []).events

    result = assign_events([candidate("comment", "OpenAI launches GPT-Next", "commentary")], historical)

    assert result.events == []
    assert result.decisions[0].primary_reason is ReasonCode.EVENT_DUPLICATE_NO_DELTA


def test_official_acquisition_confirmation_reuses_event_as_material_update() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events

    result = assign_events(
        [candidate("confirmed", "Acme signs acquisition agreement", "acme", SourceTier.T0)],
        historical,
    )

    assert result.events[0].event_id == historical[0].event_id
    assert result.events[0].state == "official_confirmation"
    assert result.events[0].latest_update_id != historical[0].latest_update_id


def test_non_primary_acquisition_confirmation_cannot_upgrade_event() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events

    result = assign_events(
        [candidate("commentary", "Acme signs acquisition agreement", "commentary", SourceTier.T1)],
        historical,
    )

    assert result.events[0].event_id != historical[0].event_id
    assert result.events[0].state != "official_confirmation"


def test_shared_generic_acquisition_terms_do_not_merge_distinct_entities() -> None:
    rumor = candidate("acme-rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events

    result = assign_events(
        [candidate("apex-confirmed", "Apex signs acquisition agreement", "apex", SourceTier.T0)],
        historical,
    )

    assert result.events[0].event_id != historical[0].event_id
    assert result.events[0].state != "official_confirmation"


def test_same_buyer_different_acquisition_target_does_not_merge() -> None:
    rumor = candidate("rumor", "Acme acquisition Beta rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events

    result = assign_events(
        [candidate("confirmed", "Acme signs acquisition Gamma agreement", "acme", SourceTier.T0)],
        historical,
    )

    assert result.events[0].event_id != historical[0].event_id
    assert result.events[0].state != "official_confirmation"


def test_same_manufacturer_different_model_does_not_merge() -> None:
    rumor = candidate("rumor", "Acme launches Alpha model rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events

    result = assign_events(
        [candidate("confirmed", "Acme officially launches Beta model", "acme", SourceTier.T0)],
        historical,
    )

    assert result.events[0].event_id != historical[0].event_id
    assert result.events[0].state != "official_confirmation"


def test_confirmation_earlier_than_historical_latest_does_not_merge() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events
    earlier = candidate("confirmed", "Acme signs acquisition agreement", "acme", SourceTier.T0).model_copy(
        update={"event_time": datetime(2026, 8, 13, 7, tzinfo=UTC)}
    )

    result = assign_events([earlier], historical)

    assert result.events[0].event_id != historical[0].event_id
    assert result.events[0].state != "official_confirmation"


def test_legacy_event_without_identity_terms_can_receive_confirmation() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events[0].model_copy(update={"identity_terms": []})
    confirmation = candidate("confirmed", "Acme signs acquisition agreement", "acme", SourceTier.T0)

    result = assign_events([confirmation], [historical])

    assert result.events[0].event_id == historical.event_id
    assert result.events[0].state == "official_confirmation"


def test_material_update_id_is_stable_when_the_same_input_is_replayed() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events
    confirmation = candidate("confirmed", "Acme signs acquisition agreement", "acme", SourceTier.T0)

    first = assign_events([confirmation], historical).events[0]
    replay = assign_events([confirmation], historical).events[0]

    assert first.latest_update_id == replay.latest_update_id
    assert first.update_history == replay.update_history


def test_repeated_confirmation_does_not_create_a_second_update() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    historical = assign_events([rumor], []).events
    confirmation = candidate("confirmed", "Acme signs acquisition agreement", "acme", SourceTier.T0)
    confirmed = assign_events([confirmation], historical).events[0]

    replay = assign_events([confirmation], [confirmed])

    assert replay.events == []
    assert replay.decisions[0].primary_reason is ReasonCode.EVENT_DUPLICATE_NO_DELTA


def test_common_chinese_acquisition_confirmation_matches_same_entity() -> None:
    rumor = candidate("rumor", "阿里收购传闻", "wire", SourceTier.T1).model_copy(
        update={"language": "zh"}
    )
    historical = assign_events([rumor], []).events
    confirmation = candidate("confirmed", "阿里签署收购协议", "alibaba", SourceTier.T0).model_copy(
        update={"language": "zh"}
    )

    result = assign_events([confirmation], historical)

    assert result.events[0].event_id == historical[0].event_id
    assert result.events[0].state == "official_confirmation"


def test_exact_title_duplicate_points_to_latest_historical_event() -> None:
    first = assign_events(
        [candidate("first", "OpenAI launches GPT-Next", "official", SourceTier.T0)], []
    ).events[0]
    newer = first.model_copy(
        update={
            "event_id": "evt_newer",
            "latest_event_time": datetime(2026, 8, 13, 9, tzinfo=UTC),
        }
    )

    result = assign_events(
        [candidate("duplicate", "OpenAI launches GPT-Next", "commentary")],
        [first, newer],
    )

    assert result.events == []
    assert result.decisions[0].event_id == "evt_newer"
    assert result.decisions[0].primary_reason is ReasonCode.EVENT_DUPLICATE_NO_DELTA


def test_material_update_continues_from_latest_historical_version() -> None:
    rumor = candidate("rumor", "Acme acquisition rumor", "wire", SourceTier.T1)
    original = assign_events([rumor], []).events[0]
    confirmation = candidate(
        "confirmed", "Acme signs acquisition agreement", "acme", SourceTier.T0
    )
    confirmed = assign_events([confirmation], [original]).events[0]
    completion = candidate(
        "completed", "Acme acquisition completed", "acme", SourceTier.T0
    ).model_copy(update={"event_time": datetime(2026, 8, 13, 9, tzinfo=UTC)})

    result = assign_events([completion], [original, confirmed])

    updated = result.events[0]
    assert updated.event_id == original.event_id
    assert len(updated.update_history) == 2
    assert updated.update_history[-1].previous_update_id == confirmed.latest_update_id
