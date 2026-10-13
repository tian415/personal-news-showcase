"""Catch stale evidence, false coverage, unsafe rendering and duplicate paid attempts."""
import copy
import json
from pathlib import Path

import pytest

from experiments.daily_writer import pipeline as writer


def packet():
    return {
        'reference_utc': '2026-10-12T02:06:47.252871+00:00',
        'standard': '背景清楚；关键不确定性留在正文。',
        'items': [{'id': 'a', 'sources': {'s': {'url': 'https://example.org/a'}},
                   'required_limits': {'uncertain': '伤亡不确定'}, 'materials': '部门称发生袭击'}],
    }


def response(request):
    return {'request_hash': request['request_hash'], 'introduction': '早上好。', 'items': [{
        'id': 'a', 'title': '<script>alert(1)</script>',
        'paragraphs': [{'text': '部门称发生袭击，伤亡仍不确定。', 'source_ids': ['s']}],
        'limits_in_body': {'uncertain': '伤亡仍不确定'},
        'editorial_notes': ['只取得通报。'],
    }]}


def test_changed_packet_rejects_old_response(tmp_path):
    request = writer.prepare(tmp_path, packet())
    changed = copy.deepcopy(request)
    changed['packet']['items'][0]['materials'] = '新的事实'
    (tmp_path / 'request.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='binding'):
        writer.ingest(tmp_path, response(request))
    assert not (tmp_path / 'preview.html').exists()


@pytest.mark.parametrize('change', ['missing_item', 'unknown_source', 'hidden_limit', 'fake_span'])
def test_bad_response_never_renders(tmp_path, change):
    request = writer.prepare(tmp_path, packet())
    value = response(request)
    if change == 'missing_item':
        value['items'] = []
    elif change == 'unknown_source':
        value['items'][0]['paragraphs'][0]['source_ids'] = ['invented']
    elif change == 'hidden_limit':
        value['items'][0]['limits_in_body'] = {}
    else:
        value['items'][0]['limits_in_body']['uncertain'] = '正文中没有的限定'
    with pytest.raises(ValueError):
        writer.ingest(tmp_path, value)
    assert not (tmp_path / 'preview.html').exists()


def test_valid_draft_is_escaped_pending_and_preserves_evidence(tmp_path):
    request = writer.prepare(tmp_path, packet())
    result = writer.ingest(tmp_path, response(request))
    page = (tmp_path / 'preview.html').read_text()
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page
    assert '<details>' in page and '只取得通报' in page
    assert '伤亡仍不确定' in page
    assert result['semantic_review'] == 'pending'
    assert result['approved'] is False
    assert not (tmp_path / 'approval.json').exists()


def test_attempt_is_single_use_even_after_uncertain_failure(tmp_path):
    writer.prepare(tmp_path, packet())
    writer.reserve_attempt(tmp_path)
    with pytest.raises(ValueError, match='already'):
        writer.reserve_attempt(tmp_path)


def test_prepare_never_overwrites_existing_work(tmp_path):
    writer.prepare(tmp_path, packet())
    with pytest.raises(ValueError, match='exists'):
        writer.prepare(tmp_path, packet())


def test_failed_reingest_invalidates_stale_preview(tmp_path):
    request = writer.prepare(tmp_path, packet())
    writer.ingest(tmp_path, response(request))
    bad = response(request)
    bad['request_hash'] = 'old'
    with pytest.raises(ValueError):
        writer.ingest(tmp_path, bad)
    assert not (tmp_path / 'preview.html').exists()


def test_payload_contains_standard_and_materials_without_gold_draft(tmp_path):
    request = writer.prepare(tmp_path, packet())
    payload = writer.payload(request)
    assert payload['model'] == 'deepseek-flash'
    assert payload['max_tokens'] == 8000
    text = json.dumps(payload, ensure_ascii=False)
    assert '关键不确定性留在正文' in text and '部门称发生袭击' in text
    assert 'limits_in_body' in text
    assert 'approved' not in json.loads(payload['messages'][1]['content'])


def test_frozen_three_excludes_accepted_prose():
    root = Path(__file__).resolve().parents[1]
    if not (root / 'work/full-editorial-2026-10-12/proposal.json').exists():
        pytest.skip('local frozen corpus unavailable')
    pack = writer.frozen_packet(root)
    assert [i['id'] for i in pack['items']] == ['pakistan', 'heart', 'riyadh']
    assert all(i['required_limits'] for i in pack['items'])
    assert 'paragraphs' not in json.dumps(pack)
    assert 'copy = ' not in json.dumps(pack)


def test_provider_failure_never_retries_and_is_not_a_draft(tmp_path):
    import httpx
    request = writer.prepare(tmp_path, packet())
    calls = []

    def fail(req):
        calls.append(req)
        return httpx.Response(503, json={'error': 'temporary'})

    with pytest.raises(ValueError, match='provider'):
        writer.generate(tmp_path, authorized_hash=request['request_hash'], api_key='test-key',
                        transport=httpx.MockTransport(fail))
    with pytest.raises(ValueError, match='already'):
        writer.generate(tmp_path, authorized_hash=request['request_hash'], api_key='test-key',
                        transport=httpx.MockTransport(fail))
    assert len(calls) == 1
    assert not (tmp_path / 'preview.html').exists()
    assert 'test-key' not in ''.join(p.read_text() for p in tmp_path.iterdir())


def test_provider_generation_preserves_raw_result_and_unknown_usage(tmp_path):
    import httpx
    request = writer.prepare(tmp_path, packet())
    def succeed(req):
        body = json.loads(req.content)
        assert body['thinking'] == {'type': 'disabled'}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                    'message': {'content': json.dumps(response(request))}}]})
    writer.generate(tmp_path, authorized_hash=request['request_hash'], api_key='test-key',
                    transport=httpx.MockTransport(succeed))
    retained = json.loads((tmp_path / 'model-response.json').read_text())
    assert retained['usage'] is None
    assert json.loads(retained['content']) == response(request)
    assert (tmp_path / 'preview.html').exists()


def test_authorization_mismatch_does_not_call_provider(tmp_path):
    import httpx
    writer.prepare(tmp_path, packet())
    calls = []
    def fail(req):
        calls.append(req)
        return httpx.Response(500)
    with pytest.raises(ValueError, match='authorization'):
        writer.generate(tmp_path, authorized_hash='wrong', api_key='test-key',
                        transport=httpx.MockTransport(fail))
    assert calls == []
    assert not (tmp_path / 'attempt.json').exists()


def test_reader_exposes_original_and_version_bound_feedback_outside_disclosures(tmp_path):
    from html.parser import HTMLParser

    class ReaderParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_details = 0
            self.links = []
            self.buttons = []
            self.radios = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'details':
                self.in_details += 1
            if tag == 'a' and not self.in_details:
                self.links.append(attrs)
            if tag == 'input' and attrs.get('type') == 'radio' and not self.in_details:
                self.radios.append(attrs)
            if tag == 'button' and not self.in_details:
                self.buttons.append(attrs)
        def handle_endtag(self, tag):
            if tag == 'details':
                self.in_details -= 1

    request = writer.prepare(tmp_path, packet())
    value = response(request)
    page = writer.render(request, value)
    parsed = ReaderParser()
    parsed.feed(page)
    assert any(a.get('href') == 'https://example.org/a' for a in parsed.links)
    actions = {b.get('data-action') for b in parsed.buttons}
    assert 'read' in actions
    assert [r['value'] for r in parsed.radios] == ['1', '2', '3', '4', '5']
    assert all(r['data-item'] == 'a' for r in parsed.radios)
    assert f'data-draft-hash="{writer.digest(value)}"' in page
    assert f'data-request-hash="{request["request_hash"]}"' in page
    changed = copy.deepcopy(value)
    changed['items'][0]['paragraphs'][0]['text'] += '新版本。'
    assert f'data-draft-hash="{writer.digest(changed)}"' in writer.render(request, changed)


def test_visible_original_link_rejects_unsafe_scheme(tmp_path):
    pack = packet()
    pack['items'][0]['sources']['s']['url'] = 'javascript:alert(1)'
    request = writer.prepare(tmp_path, pack)
    with pytest.raises(ValueError, match='unsafe'):
        writer.render(request, response(request))


def test_feedback_store_rejects_wrong_draft_and_keeps_rating_separate_from_read(tmp_path):
    from experiments.daily_writer.feedback_store import FeedbackStore
    store = FeedbackStore(tmp_path, 'req', 'draft', ['a'])
    value = {'schema': 'daily-writer-draft-feedback-v2', 'request_hash': 'req',
             'draft_hash': 'draft', 'scope': 'local_draft_only', 'approval': False,
             'rating_dimension': 'reading_value', 'scale': {'min': 1, 'max': 5},
             'items': [{'item_id': 'a', 'read': False, 'rating': 1, 'reason': 'unclear',
                        'updated_at': '2026-10-12T06:00:00+00:00'}]}
    store.save(value)
    assert store.load()['items'][0]['rating'] == 1
    assert store.load()['items'][0]['read'] is False
    bad = copy.deepcopy(value)
    bad['draft_hash'] = 'other'
    with pytest.raises(ValueError):
        store.save(bad)
    assert store.load()['items'] == value['items']
    value['items'][0]['rating'] = 5
    value['items'][0]['reason'] = None
    store.save(value)
    assert len(store.load()['items']) == 1
    assert store.load()['items'][0]['rating'] == 5
    # Clearing a rating isn't a low rating; no inference from an absent row.
    value['items'] = []
    store.save(value)
    assert store.load()['items'] == []
    assert not (tmp_path / 'approval.json').exists()


@pytest.mark.parametrize('field,value', [('rating', 0), ('rating', True),
                                       ('rating', 6), ('reason', 'invented')])
def test_feedback_invalid_values_never_append(tmp_path, field, value):
    from experiments.daily_writer.feedback_store import FeedbackStore
    store = FeedbackStore(tmp_path, 'req', 'draft', ['a'])
    row = {'item_id': 'a', 'read': True, 'rating': None, 'reason': None,
           'updated_at': '2026-10-12T06:00:00Z'}
    row[field] = value
    payload = store.empty()
    payload['items'] = [row]
    with pytest.raises(ValueError):
        store.save(payload)
    assert not (tmp_path / 'reader-feedback.jsonl').exists()


def test_feedback_endpoint_rejects_cross_origin_before_writing(tmp_path):
    from io import BytesIO

    from experiments.daily_writer.server import handler

    request = writer.prepare(tmp_path, packet())
    value = response(request)
    (tmp_path / 'model-response.json').write_text(json.dumps({'content': json.dumps(value)}))
    payload = {'schema': 'daily-writer-draft-feedback-v2', 'request_hash': request['request_hash'],
               'draft_hash': writer.digest(value), 'scope': 'local_draft_only', 'approval': False,
               'rating_dimension': 'reading_value', 'scale': {'min': 1, 'max': 5},
               'items': [{'item_id': 'a', 'read': True, 'rating': 4, 'reason': None,
                          'updated_at': '2026-10-12T06:00:00Z'}]}
    body = json.dumps(payload).encode()
    class Socket:
        def __init__(self, origin):
            self.input = BytesIO((f'POST /feedback HTTP/1.1\r\nHost: 127.0.0.1:8774\r\n'
                                 f'Origin: {origin}\r\nContent-Type: application/json\r\n'
                                 f'Content-Length: {len(body)}\r\n\r\n').encode() + body)
            self.output = bytearray()
        def makefile(self, *args):
            return self.input
        def sendall(self, data):
            self.output.extend(data)
        def settimeout(self, value):
            pass
    cls = handler(tmp_path, 8774, 'preview.html')
    denied = Socket('https://evil.example')
    cls(denied, ('127.0.0.1', 1), None)
    assert b'403' in denied.output
    assert not (tmp_path / 'reader-feedback.jsonl').exists()
    accepted = Socket('http://127.0.0.1:8774')
    cls(accepted, ('127.0.0.1', 1), None)
    assert b'200 OK' in accepted.output
    assert json.loads((tmp_path / 'reader-feedback.jsonl').read_text())['feedback'] == payload


@pytest.mark.parametrize("compact", [False, True])
def test_future_writer_reads_explicit_feedback_without_changing_materials(tmp_path, compact):
    from experiments.daily_writer.feedback_store import FeedbackStore, review_context
    old = tmp_path / 'old'
    request = writer.prepare(old, packet())
    value = response(request)
    (old / 'model-response.json').write_text(json.dumps({'content': json.dumps(value)}))
    if compact:
        value['items'][0]['summary'] = '简介，伤亡仍不确定。'
        (old / 'reader-response.json').write_text(json.dumps(value))
    store = FeedbackStore(old, request['request_hash'], writer.digest(value), ['a'])
    feedback = store.empty()
    feedback['items'] = [{'item_id': 'a', 'read': True, 'rating': 2, 'reason': 'unclear',
                          'updated_at': '2026-10-12T06:00:00Z'}]
    store.save(feedback)
    new_packet = packet()
    new_packet['reader_feedback_context'] = review_context([old])
    new = writer.prepare(tmp_path / 'new', new_packet)
    sent = json.loads(writer.payload(new)['messages'][1]['content'])
    assert sent['items'] == packet()['items']
    assert sent['reader_feedback_context']['items'][0]['rating'] == 2
    assert sent['reader_feedback_context']['items'][0]['reason'] == 'unclear'
    assert sent['reader_feedback_context']['items'][0]['draft_hash'] == writer.digest(value)


def test_compact_summary_keeps_details_closed_and_actions_visible(tmp_path):
    request = writer.prepare(tmp_path, packet())
    value = response(request)
    value['items'][0]['summary'] = '部门称发生袭击，伤亡仍不确定。'
    writer.validate(request, value)
    page = writer.render(request, value)
    assert '<details class="full-story"><summary>' in page
    assert '<p class="story-summary">部门称发生袭击，伤亡仍不确定。</p>' in page
    start = page.index('<details class="full-story">')
    end = page.index('</details>', start)
    assert '部门称发生袭击' in page[start:end]
    assert page.index('class="original"') > end
    assert page.index('class="rating"') > end
    changed = copy.deepcopy(value)
    changed['items'][0]['summary'] += '新信息。'
    assert writer.digest(changed) != writer.digest(value)


def compact_packet():
    value = packet()
    value['output_contract'] = 'compact_v2'
    return value


def compact_response(request):
    value = response(request)
    value['items'][0].update(summary='部门称发生袭击，伤亡仍不确定。',
                             summary_source_ids=['s'],
                             limits_in_summary={'uncertain': '伤亡仍不确定'})
    return value


@pytest.mark.parametrize('failure', ['missing_summary', 'missing_refs', 'fake_limit', 'process'])
def test_automatic_compact_draft_rejects_incomplete_or_editorial_summary(tmp_path, failure):
    request = writer.prepare(tmp_path, compact_packet())
    value = compact_response(request)
    item = value['items'][0]
    if failure == 'missing_summary':
        del item['summary']
    elif failure == 'missing_refs':
        item['summary_source_ids'] = ['unknown']
    elif failure == 'fake_limit':
        item['limits_in_summary']['uncertain'] = '摘要不存在'
    else:
        item['summary'] += '本期采用部门通报。'
    with pytest.raises(ValueError):
        writer.ingest(tmp_path, value)
    assert not (tmp_path / 'preview.html').exists()


def test_automatic_compact_contract_reaches_provider_and_preview(tmp_path):
    request = writer.prepare(tmp_path, compact_packet())
    assert json.loads(writer.payload(request)['messages'][1]['content'])['output_contract'] == 'compact_v2'
    result = writer.ingest(tmp_path, compact_response(request))
    assert result['approved'] is False
    assert 'summary_entailment' in result['review_required']
    assert '<details class="full-story">' in (tmp_path / 'preview.html').read_text()
