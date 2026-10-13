import importlib.util
from datetime import UTC, datetime
from pathlib import Path


def collector():
    path = Path(__file__).parents[1] / 'experiments/multisource_validation/collect.py'
    assert path.exists(), 'independent collector is implemented'
    spec = importlib.util.spec_from_file_location('collection_trial', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_window_is_bounded_and_date_conflicts_are_preserved():
    c = collector()
    reference = datetime(2026, 10, 10, 12, tzinfo=UTC)
    rows = [{'url': 'https://example.com/a', 'language': 'zh', 'published_at': '2026-10-10T10:00:00+00:00', 'source_id': 'a'},
            {'url': 'https://example.com/a', 'language': 'zh', 'published_at': '2026-10-10T11:00:00+00:00', 'source_id': 'b'},
            {'url': 'https://example.com/old', 'language': 'en', 'published_at': '2026-10-08T00:00:00+00:00', 'source_id': 'a'},
            {'url': 'https://example.com/future', 'language': 'en', 'published_at': '2026-10-11T00:00:00+00:00', 'source_id': 'a'}]
    articles, exclusions, conflicts = c.merge_window(rows, reference)
    assert len(articles) == 1
    assert articles[0]['appearances'] == ['a', 'b']
    assert articles[0]['published_at'] == rows[0]['published_at']
    assert conflicts[0]['field'] == 'published_at'
    assert {r['reason'] for r in exclusions} == {'outside_window', 'future_publication'}


def test_generic_html_requires_explicit_article_publication_time():
    c = collector()
    source = {'id':'yicai', 'name':'第一财经', 'language':'zh', 'url':'https://www.yicai.com/', 'kind':'html'}
    html = b'<html><h1>Article</h1><p>Copyright 2026-10-10</p></html>'
    try:
        c.parse_generic_article(html, 'https://www.yicai.com/news/1.html', source)
    except ValueError as error:
        assert 'publication' in str(error)
    else:
        assert False, 'copyright date must not become publication date'


def test_body_challenge_and_short_nonbody_are_not_success():
    c = collector()
    assert c.body_result(b'<html>Just a moment... cf-chl-</html>', 'Please wait')[:2] == ('restricted', 'challenge')
    assert c.body_result(b'<html><p>Home</p></html>', 'Home')[:2] == ('failed', 'nonbody')
    text = 'A meaningful article paragraph. ' * 30
    assert c.body_result(b'<html></html>', text) == ('ok', None, text)


def test_publisher_article_timestamp_containers():
    c = collector()
    for source_id, markup, expected in [
        ('yicai', '<div class="title f-pr"><h1>News</h1><p><em class="f-fl">2026-10-09 22:09:33</em></p></div>', '2026-10-09T14:09:33+00:00'),
        ('chuapp', '<h1>News</h1><span class="fn-right friendly_time">2026年10月09日 18时23分</span>', '2026-10-09T10:23:00+00:00')]:
        row = c.parse_generic_article(markup.encode(), 'https://example.com/article', {'id': source_id, 'language': 'zh'})
        assert row['published_at'] == expected


def test_replay_reads_saved_raw_without_network(tmp_path):
    import json
    c = collector()
    raw = tmp_path / 'raw.response'
    raw.write_bytes(b'public bytes')
    (tmp_path / 'requests.json').write_text(json.dumps([{'url':'https://example.com/a', 'retrieved_at':'2026-10-10T01:00:00+00:00', 'hops':[{'status':200,'raw_path':str(raw),'url':'https://example.com/a'}]}]))
    acquisition = c.Acquisition(tmp_path, replay=True)
    assert acquisition.fetch('https://example.com/a')[0] == b'public bytes'
    try:
        acquisition.fetch('https://example.com/missing')
    except ValueError as error:
        assert 'replay' in str(error)
    else:
        assert False


def test_captcha_library_does_not_make_real_body_restricted():
    c = collector()
    text = 'Article body paragraph. ' * 30
    raw = b'<h1>News</h1><script src="https://o.alicdn.com/captcha-frontend/aliyunCaptcha.js"></script>'
    assert c.body_result(raw, text)[0] == 'ok'


def test_public_fetch_enforces_response_size_and_redirect_budget(tmp_path, monkeypatch):
    import httpx
    c = collector()
    monkeypatch.setattr(c, '_public_address', lambda *args, **kwargs: '93.184.216.34')
    original_client = httpx.Client
    calls = []
    def redirect(request):
        calls.append(request)
        return httpx.Response(302, headers={'location': 'https://example.com/next'})
    monkeypatch.setattr(c.httpx, 'Client', lambda **kwargs: original_client(transport=httpx.MockTransport(redirect), **kwargs))
    try:
        c.Acquisition(tmp_path / 'redirect').fetch('https://example.com/a')
    except c.ContentFetchError as error:
        assert error.reason == 'redirect_limit'
    assert len(calls) == 4
    assert all(r.headers['Host'] == 'example.com' and r.extensions['sni_hostname'] == 'example.com' for r in calls)
    monkeypatch.setattr(c.httpx, 'Client', lambda **kwargs: original_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b'x' * (c.MAX_BYTES + 1))), **kwargs))
    try:
        c.Acquisition(tmp_path / 'large').fetch('https://example.com/a')
    except c.ContentFetchError as error:
        assert error.reason == 'response_too_large'
    else:
        assert False


def test_different_languages_are_not_merged_and_identity_collision_is_recorded():
    c = collector()
    rows = [{'url':'https://example.com/a','language':lang,'published_at':'2026-10-10T01:00:00+00:00','source_id':lang,'id':'same'} for lang in ('zh','en')]
    articles, _, conflicts = c.merge_window(rows, datetime(2026,10,10,12,tzinfo=UTC))
    assert len(articles) == 2
    assert any(conflict['field']=='language_identity_collision' for conflict in conflicts)


def test_diagnostic_selection_does_not_replace_missing_ids_or_expand_budget():
    c = collector()
    corpus = {'articles':[{'id':'a','body_status':'pending'}, {'id':'b','body_status':'ok'}],
              'sampling':{'body_attempt_count':74}}
    selected, decisions = c.diagnostic_selection(corpus, {'missing':'missing check','a':'event comparison','b':'already sampled'})
    assert [article['id'] for article in selected] == ['a']
    assert [d['status'] for d in decisions] == ['skip_missing', 'selected', 'skip_existing_body']
    assert all(d['reason'] for d in decisions)


def test_run_refuses_frozen_corpus_before_acquisition(tmp_path, monkeypatch):
    import json

    import pytest
    c = collector()
    path = tmp_path / 'corpus' / 'corpus.json'
    path.parent.mkdir()
    original = json.dumps({'sampling': {'phase': 'frozen'}})
    path.write_text(original)
    def unexpected_acquisition(*args, **kwargs):
        pytest.fail('Acquisition must not be called for a frozen corpus')
    monkeypatch.setattr(c, 'Acquisition', unexpected_acquisition)
    with pytest.raises(ValueError, match='new output directory'):
        c.run(tmp_path, datetime(2026, 10, 10, 12, tzinfo=UTC), [])
    assert path.read_text() == original
