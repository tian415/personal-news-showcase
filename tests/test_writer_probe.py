import copy

import pytest

from experiments.daily_writer import probe


def sample():
    return {'items': [{'id': 'a', 'title': '新进展', 'summary': '事实尚不确定。',
                       'paragraphs': [{'text': '事实尚不确定。', 'source_ids': ['s']}]}]}


def test_process_scan_covers_unseen_phrasing():
    value = sample()
    value['items'][0]['paragraphs'][0]['text'] += '本轮未能取得完整正文，原池还有其他标题。'
    result = probe.screen(value)
    assert 'editorial_process' in {f['code'] for f in result['findings']}
    assert result['automatic_pass'] is False


def test_clean_text_is_pending_not_semantic_pass():
    assert probe.screen(sample())['result'] == 'pending_semantic_review'
    assert probe.screen(sample())['automatic_pass'] is False


def test_mutated_reading_requires_new_semantic_review():
    value = sample()
    evidence = probe.review_record(value, {'factual': 'fail'})
    changed = copy.deepcopy(value)
    changed['items'][0]['summary'] = '新摘要'
    with pytest.raises(ValueError, match='binding'):
        probe.verify_review(changed, evidence)


def test_excerpt_gate_not_confused_with_writing_quality():
    value = sample()
    value['items'][0]['limits_in_body'] = {'limit': '不在正文的摘录'}
    result = probe.screen(value)
    assert any(f['code'] == 'bad_excerpt' for f in result['findings'])


def test_prepare_does_not_overwrite(tmp_path):
    packet = {'items': []}
    probe.prepare(tmp_path, packet)
    with pytest.raises(FileExistsError):
        probe.prepare(tmp_path, packet)


def test_new_author_contract_rejects_unknown_sources_and_review_fields():
    request = {'request_hash': 'req', 'packet': {'items': [{'id': 'a', 'sources': {'s': {}}}]}}
    value = sample()
    value['request_hash'] = 'req'
    probe.validate_author(request, value)
    value['items'][0]['paragraphs'][0]['source_ids'] = ['invented']
    with pytest.raises(ValueError):
        probe.validate_author(request, value)
    value['items'][0]['paragraphs'][0]['source_ids'] = ['s']
    value['items'][0]['editorial_notes'] = []
    with pytest.raises(ValueError):
        probe.validate_author(request, value)
