from copy import deepcopy
from datetime import timedelta

import pytest
from test_reviewed_selection import NOW, inputs

from news_assistant.coverage_evaluation import _fingerprint
from news_assistant.models import Preferences
from news_assistant.reviewed_selection import prepare_review, select_reviewed


def value_inputs(count=12):
    audit, review = inputs(count)
    value = {
        'schema_version': 'news-value-review-v1',
        'snapshot_sha256': _fingerprint(audit), 'review_sha256': _fingerprint(review),
        'reviewed_by': 'test editor', 'reviewed_at': NOW.isoformat(),
        'items': [{
            'candidate_id': item['candidate_id'], 'timing': 'confirmed',
            'event_at': (NOW - timedelta(hours=2)).isoformat(),
            'time_basis': 'material_update', 'time_quote': item['evidence_excerpt'],
            'why_today': '今天披露新的实验结果，仍需复核。',
            'impact': '研究人员获得新的实验路线，尚未进入实际应用。',
            'impact_quote': audit['sources'][i]['candidates'][0]['summary'],
            'importance': 0.65,
        } for i, item in enumerate(review['items'])],
    }
    return audit, review, value


def test_value_selection_is_not_limited_by_topic_slots():
    audit, review, value = value_inputs()
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert len(result.selected) == 12
    assert all(result.value_details[a.id]['why_today'] for a in result.selected)


def test_today_republication_does_not_make_old_event_fresh_and_unknown_is_excluded():
    audit, review, value = value_inputs(3)
    value['items'][0]['event_at'] = (NOW - timedelta(days=365)).isoformat()
    value['items'][1].update(timing='unknown', event_at=None)
    value['items'][2]['importance'] = 0.1
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert not result.selected
    assert set(result.excluded.values()) == {'old_event', 'event_time_unknown', 'low_news_value'}


def test_old_review_without_value_cannot_fill_strict_edition():
    audit, review, value = value_inputs(2)
    value['items'] = []
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert not result.selected
    assert set(result.excluded.values()) == {'pending_news_value'}


@pytest.mark.parametrize('change', ['quote', 'snapshot', 'review', 'duplicate', 'future'])
def test_value_decisions_are_bound_and_attributed(change):
    audit, review, value = value_inputs(1)
    if change == 'quote':
        value['items'][0]['impact_quote'] = 'invented evidence'
    elif change == 'snapshot':
        value['snapshot_sha256'] = '0' * 64
    elif change == 'review':
        review['items'][0]['importance_score'] = 0.9
    elif change == 'duplicate':
        value['items'] *= 2
    else:
        value['reviewed_at'] = (NOW + timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError):
        select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)


def test_material_importance_precedes_preferences_and_hotness_is_unknown():
    audit, review, value = value_inputs(2)
    value['items'][0]['importance'] = 0.9
    value['items'][1]['importance'] = 0.6
    result = select_reviewed(audit, review, Preferences(source_weights={'source-1': 2}),
                             now=NOW, value_review=value, limit=1)
    assert result.selected[0].id == value['items'][0]['candidate_id']
    assert result.value_details[result.selected[0].id]['attention_status'] == 'no_data'


def test_traceable_rank_can_admit_low_impact_topic_without_claiming_growth():
    audit, review, value = value_inputs(1)
    value['items'][0]['importance'] = 0.2
    value['items'][0]['attention'] = {
        'url': 'https://example.org/ranking/123',
        'snapshots': [{'source_id': 'ranking', 'external_id': '123',
                       'captured_at': NOW.isoformat(), 'rank': 2, 'reply_count': 500}],
    }
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert len(result.selected) == 1
    detail = result.value_details[result.selected[0].id]
    assert detail['attention_status'] == 'snapshot_only'
    assert detail['attention_velocity'] is None


def test_preview_value_explanations_and_shortfall_are_in_web_and_email(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview

    audit, review, value = value_inputs(2)
    result = build_reviewed_preview(audit, review, Preferences(), tmp_path/'web', tmp_path/'state',
        now=NOW, run_id='value', value_review=value, email_base_url='http://127.0.0.1:8775')
    assert result['coverage']['shortfall'] == 8
    assert result['discovery_plan']['needed'] == 8
    email = next((tmp_path/'state').rglob('email-preview.html')).read_text()
    page = (tmp_path/'web/index.html').read_text()
    for content in [email, page]:
        assert '为什么今天值得看' in content
        assert '影响是什么' in content
        assert '热点未知' in content
        assert '10–15' in content
        assert '补充发现' in content
        assert '/5' not in content
    assert result['delivery_status'] == 'not_confirmed'


def test_value_cap_and_feedback_only_break_similar_value_ties():
    audit, review, value = value_inputs()
    baseline = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value, limit=10)
    target = next(i for i in value['items'] if i['candidate_id'] not in {a.id for a in baseline.selected})
    source = target['candidate_id'].split(':')[0]
    adjusted = select_reviewed(audit, review, Preferences(source_weights={source: 2}),
                              now=NOW, value_review=value, limit=10)
    assert target['candidate_id'] in {a.id for a in adjusted.selected}
    assert len(adjusted.selected) == 10


def test_cross_language_event_identity_merges_and_replays_without_hotness_from_copies():
    audit, review, value = value_inputs(2)
    for item in value['items']:
        item['event_title'] = '同一研究的新结果'
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert len(result.selected) == 1
    assert len(result.events) == 1
    assert result.events[0].latest_event_time == NOW - timedelta(hours=2)
    second = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value,
                             historical_events=result.events)
    assert not second.selected


def test_event_value_is_separate_from_the_most_reliable_representative():
    audit, review, value = value_inputs(3)
    audit['sources'][1]['evidence_role'] = 'INDEPENDENT_REPORTING'
    review['snapshot_sha256'] = _fingerprint(audit)
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    value['items'][0].update(event_title='同一事件', importance=0.55)
    value['items'][1].update(event_title='同一事件', importance=0.99)
    value['items'][2]['importance'] = 0.65
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value, limit=1)
    assert result.selected[0].id == value['items'][0]['candidate_id']
    assert result.value_details[result.selected[0].id]['importance'] == 0.99


def test_reviewed_material_delta_keeps_event_identity_and_updates_only_once():
    audit, review, value = value_inputs(1)
    value['items'][0]['event_title'] = '稳定事件标题'
    first = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    # Real newly captured content, not just a different sidecar timestamp/quote.
    audit['sources'][0]['candidates'][0]['summary'] += ' 新结果显示已进入扩大验证阶段。'
    rebound = prepare_review(audit)
    review['snapshot_sha256'] = rebound['snapshot_sha256']
    review['items'][0]['article_sha256'] = rebound['items'][0]['article_sha256']
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    updated = deepcopy(value)
    updated['items'][0].update(event_at=NOW.isoformat(), time_basis='material_update',
        previous_event_id=first.events[0].event_id, previous_update_id=None,
        development='从初步结果进入扩大验证阶段',
        development_quote=audit['sources'][0]['candidates'][0]['summary'])
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=updated,
                             historical_events=first.events)
    assert len(result.selected) == 1
    event = result.events[0]
    assert event.event_id == first.events[0].event_id
    assert event.latest_update_id
    assert event.update_history[-1].material
    replay = select_reviewed(audit, review, Preferences(), now=NOW, value_review=updated,
                             historical_events=result.events)
    assert not replay.selected
    # A timestamp-only republish is still a duplicate.
    updated['items'][0].pop('development')
    updated['items'][0].pop('development_quote')
    no_delta = select_reviewed(audit, review, Preferences(), now=NOW, value_review=updated,
                               historical_events=first.events)
    assert not no_delta.selected


def test_supplement_discovers_missing_domains_without_changing_source_enablement():
    from news_assistant.news_value import discover_supplement
    from news_assistant.source_registry import SourceRegistry

    registry = SourceRegistry.model_validate({'version': 'source-registry-v1', 'sources': [
        {'id': name, 'name': name, 'tier': 'T0', 'adapter': 'rss',
         'url': f'https://example.org/{name}', 'topics': ['science'],
         'coverage_domains': [domain], 'publisher_group': name, 'production_enabled': enabled}
        for name, domain, enabled in [('business', 'business_macro', True),
                                     ('science', 'science_health', True),
                                     ('shadow', 'science_health', False)]
    ]})
    visited = []
    def load(url):
        visited.append(url)
        return b'<rss version="2.0"><channel><title>empty</title></channel></rss>'
    result = discover_supplement(registry, {'needed': 3, 'priority_domains': ['science_health']},
                                now=NOW, max_sources=1, loader=load)
    assert visited == ['https://example.org/science']
    assert result['candidate_snapshot_available']
    assert result['discovery_context']['shortfall_before'] == 3
    assert registry.sources[-1].production_enabled is False


def test_reworded_event_title_cannot_bypass_explicit_version_or_unchanged_evidence():
    audit, review, value = value_inputs(1)
    first = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    item = value['items'][0]
    item.update(event_title='改写标题', previous_event_id=first.events[0].event_id,
                previous_update_id='wrong-version', development='编造进展',
                development_quote=audit['sources'][0]['candidates'][0]['summary'], event_at=NOW.isoformat())
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value,
                             historical_events=first.events)
    assert not result.selected
    item.update(previous_update_id=None, event_title=None)
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value,
                             historical_events=first.events)
    assert not result.selected


def test_distinct_developments_cannot_pool_primary_evidence():
    audit, review, value = value_inputs(2)
    for item in value['items']:
        item['event_title'] = '同一事件'
    first = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    audit['sources'][1]['evidence_role'] = 'INDEPENDENT_REPORTING'
    for i, source in enumerate(audit['sources']):
        source['candidates'][0]['summary'] += f' 新进展编号{i}。'
    rebound = prepare_review(audit)
    review['snapshot_sha256'] = rebound['snapshot_sha256']
    for target, source in zip(review['items'], rebound['items']):
        target['article_sha256'] = source['article_sha256']
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    for i, item in enumerate(value['items']):
        item.update(event_at=(NOW-timedelta(minutes=10-i)).isoformat(),
                    previous_event_id=first.events[0].event_id, previous_update_id=None,
                    development=f'实质进展{i}',
                    development_quote=audit['sources'][i]['candidates'][0]['summary'])
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value,
                             historical_events=first.events)
    assert result.events[0].candidate_ids == [value['items'][0]['candidate_id']]
    assert result.events[0].editorial_development == '实质进展0'
    assert result.excluded[value['items'][1]['candidate_id']] == 'unsupported_event_development'


def test_value_worksheet_cli_is_unjudged_private_and_non_overwriting(tmp_path, monkeypatch):
    import json

    from news_assistant.cli import main

    audit, review, _ = value_inputs(1)
    snapshot, decisions, output = [tmp_path/name for name in ['snapshot.json', 'review.json', 'value.json']]
    snapshot.write_text(json.dumps(audit))
    decisions.write_text(json.dumps(review))
    args = ['prepare-value-review', '--snapshot', str(snapshot), '--review', str(decisions),
            '--output', str(output), '--public-dir', str(tmp_path/'public')]
    monkeypatch.setattr("sys.argv", ["news-assistant", *args])
    assert main() == 0
    worksheet = json.loads(output.read_text())
    assert worksheet['reviewed_by'] is None
    assert worksheet['items'][0]['importance'] == 0
    with pytest.raises(SystemExit):
        monkeypatch.setattr("sys.argv", ["news-assistant", *args])
        main()
    args[args.index('--output')+1] = str(tmp_path/'public/value.json')
    with pytest.raises(SystemExit):
        monkeypatch.setattr("sys.argv", ["news-assistant", *args])
        main()
    assert not (tmp_path/'public/value.json').exists()


def test_value_edition_feedback_import_reorders_next_batch_and_undo_restores(tmp_path):
    import json

    from news_assistant.delivery_recovery import confirm_local_reading
    from news_assistant.feedback import public_event_ref
    from news_assistant.feedback_store import import_feedback_file
    from news_assistant.local_preview import build_reviewed_preview
    from news_assistant.state import StateStore

    audit, review, value = value_inputs(12)
    state = tmp_path/'state'
    first = build_reviewed_preview(audit, review, Preferences(), tmp_path/'first', state,
                                   now=NOW, run_id='first', value_review=value)
    store = StateStore(state)
    confirm_local_reading(store, 'first', tmp_path/'first/2026-09-28.html', now=NOW)
    target = first['selected_ids'][0]
    event = next(e for e in store.load_recent_events(NOW, channel='local') if target in e.candidate_ids)
    payload = {'schema_version': 'news-feedback-v2', 'events': [{
        'event_ref': public_event_ref(event.event_id, run_id='first', update_id=event.latest_update_id),
        'vote': 'reduce', 'reason': 'irrelevant', 'tracking': False,
        'updated_at': (NOW+timedelta(minutes=1)).isoformat()}]}
    path = tmp_path/'feedback.json'
    path.write_text(json.dumps(payload))
    import_feedback_file(path, state, now=NOW+timedelta(minutes=1), channel='local')
    for source in audit['sources']:
        article = source['candidates'][0]
        article['title'] = '另一项独立研究：' + article['title']
        article['url'] += '/next'
    rebound = prepare_review(audit)
    review['snapshot_sha256'] = rebound['snapshot_sha256']
    for item, replacement in zip(review['items'], rebound['items']):
        item['article_sha256'] = replacement['article_sha256']
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    adjusted = build_reviewed_preview(audit, review, Preferences(), tmp_path/'next', state,
        now=NOW+timedelta(minutes=2), run_id='next', value_review=value)
    assert adjusted['feedback_changed_selection']
    assert adjusted['selected_ids'].index(target) > adjusted['without_feedback_ids'].index(target)
    payload['events'][0].update(vote=None, reason=None, updated_at=(NOW+timedelta(minutes=3)).isoformat())
    path.write_text(json.dumps(payload))
    import_feedback_file(path, state, now=NOW+timedelta(minutes=3), channel='local')
    undone = build_reviewed_preview(audit, review, Preferences(), tmp_path/'undo', state,
        now=NOW+timedelta(minutes=4), run_id='undo', value_review=value)
    assert undone['selected_ids'] == adjusted['without_feedback_ids']
    assert store.load_recent_events(NOW+timedelta(minutes=4)) == []


def _retitle_single(audit, review, value, title):
    from news_assistant.reviewed_selection import article_fingerprint

    article = audit['sources'][0]['candidates'][0]
    article['title'] = title
    review['items'][0].update(article_sha256=article_fingerprint(article), evidence_excerpt=title)
    review['snapshot_sha256'] = _fingerprint(audit)
    value['items'][0].update(time_quote=title)
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))


@pytest.mark.parametrize('title', [
    '早报｜苹果发布新品/三星减产/保时捷裁员',
    '派早报：英伟达发布新品、Anthropic 发布模型等',
    '【钛晨报】央行调整政策；车企召回；电影获奖',
    'The Download: AI’s refusal problem and weight-loss drug side effects',
    '安踏收购；星巴克进新疆｜消研所周报',
])
def test_accepted_branded_roundup_cannot_count_as_one_event(title):
    audit, review, value = value_inputs(1)
    _retitle_single(audit, review, value, title)
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert not result.selected
    assert result.excluded['source-0:0'] == 'roundup_requires_event_split'


def test_editor_can_mark_unbranded_multi_event_article_as_roundup():
    audit, review, value = value_inputs(1)
    _retitle_single(audit, review, value, '云服务发布模型；机器人融资；汽车业务上线')
    review['items'][0]['story_shape'] = 'roundup'
    value['review_sha256'] = _fingerprint(review)
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert not result.events
    assert result.excluded['source-0:0'] == 'roundup_requires_event_split'


def test_single_event_with_semicolons_or_newspaper_name_is_not_a_roundup():
    audit, review, value = value_inputs(1)
    _retitle_single(audit, review, value, '南华早报披露新调查；研究流程公开；结论仍需核查')
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert len(result.selected) == 1


def test_reprints_of_one_origin_do_not_create_independent_attention():
    audit, review, value = value_inputs(2)
    for source in audit['sources']:
        source['evidence_role'] = 'INDEPENDENT_REPORTING'
    for item in review['items']:
        item['origin_publisher_group'] = 'one_wire_service'
    review['snapshot_sha256'] = _fingerprint(audit)
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    for item in value['items']:
        item.update(event_title='同一事件的两篇转载', importance=0.2)
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert not result.selected
    assert set(result.excluded.values()) == {'low_news_value'}
    assert all(d['independent_reporting_count'] == 0 for d in result.value_details.values())


def test_unknown_reporting_origins_cannot_unlock_low_value_or_major_protection():
    audit, review, value = value_inputs(2)
    for source in audit['sources']:
        source['evidence_role'] = 'INDEPENDENT_REPORTING'
    for item in review['items']:
        item.update(major_reason='需要独立证据支持的重要事件', importance_score=0.95)
    for item in value['items']:
        item.update(event_title='多站报道但原始出处待核对', importance=0.2)
    review['snapshot_sha256'] = _fingerprint(audit)
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert not result.selected
    assert all(d['independent_reporting_count'] == 0 for d in result.value_details.values())
    for item in value['items']:
        item['importance'] = 0.9
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert len(result.selected) == 1
    assert result.events[0].major is False
    assert result.events[0].independent_source_count == 0
    for index, item in enumerate(review['items']):
        item['origin_publisher_group'] = f'verified-origin-{index}'
    value['review_sha256'] = _fingerprint(review)
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    assert result.events[0].major is True
    assert result.events[0].independent_source_count == 2


def test_disclosure_timestamp_is_labelled_as_disclosure_in_reader_and_email(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview

    audit, review, value = value_inputs(1)
    value['items'][0]['time_basis'] = 'first_disclosure'
    build_reviewed_preview(audit, review, Preferences(), tmp_path/'web', tmp_path/'state',
        now=NOW, run_id='disclosure', value_review=value, email_base_url='http://127.0.0.1:8775')
    page = (tmp_path/'web/index.html').read_text()
    email = next((tmp_path/'state').rglob('email-preview.html')).read_text()
    for content in [page, email]:
        assert '本次披露日期：' in content
        assert '事件／进展日期：' not in content


def test_unknown_reporting_origins_cannot_authorize_historical_material_update():
    audit, review, value = value_inputs(2)
    for source in audit['sources']:
        source['evidence_role'] = 'INDEPENDENT_REPORTING'
    review['snapshot_sha256'] = _fingerprint(audit)
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    for item in value['items']:
        item['event_title'] = '持续报道同一事件'
    first = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value)
    for source in audit['sources']:
        source['candidates'][0]['summary'] += ' 实验开始扩大验证。'
    rebound = prepare_review(audit)
    review['snapshot_sha256'] = rebound['snapshot_sha256']
    for target, source in zip(review['items'], rebound['items']):
        target['article_sha256'] = source['article_sha256']
    value.update(snapshot_sha256=_fingerprint(audit), review_sha256=_fingerprint(review))
    for item in value['items']:
        item.update(event_at=NOW.isoformat(), previous_event_id=first.events[0].event_id,
                    previous_update_id=None, development='实验开始扩大验证',
                    development_quote='实验开始扩大验证。')
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value,
                             historical_events=first.events)
    assert not result.selected
    for index, item in enumerate(review['items']):
        item['origin_publisher_group'] = f'verified-origin-{index}'
    value['review_sha256'] = _fingerprint(review)
    result = select_reviewed(audit, review, Preferences(), now=NOW, value_review=value,
                             historical_events=first.events)
    assert len(result.selected) == 1
    assert result.events[0].latest_update_id
