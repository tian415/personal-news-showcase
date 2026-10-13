"""Native-worker isolation: failures propagate, never become empty successful text."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from experiments.multisource_validation import collect as collection


def test_body_extraction_uses_child_process_and_preserves_text(monkeypatch):
    calls = []
    real = subprocess.run

    def tracked(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', tracked)
    text = 'This is a substantive public news paragraph with concrete facts. ' * 20
    raw = f'<html><body><article><h1>News</h1><p>{text}</p></article></body></html>'.encode()
    result = collection.extract_body(raw)
    assert 'substantive public news' in result
    assert len(calls) == 1
    assert calls[0][0] == sys.executable
    assert '-X' in calls[0] and 'faulthandler' in calls[0]


def test_native_crash_is_reported_without_retry(monkeypatch):
    calls = []
    real = subprocess.run

    def crash(command, **kwargs):
        calls.append(command)
        return real([sys.executable, '-c',
                     'import os,signal;os.kill(os.getpid(),signal.SIGSEGV)'], **kwargs)

    monkeypatch.setattr(subprocess, 'run', crash)
    with pytest.raises(collection.ExtractionFailure, match='-11'):
        collection.extract_body(b'<html></html>')
    assert len(calls) == 1


def test_collection_calls_isolated_boundary_and_propagates_failure(tmp_path, monkeypatch):
    source = {'id': 'fixture', 'name': 'Fixture', 'language': 'en',
              'publisher_group': 'fixture', 'domain_hint': 'science_health',
              'kind': 'rss', 'url': 'https://example.org/feed'}
    rss = b'''<rss version="2.0"><channel><title>Fixture</title><item><title>Fresh</title>
    <link>https://example.org/article</link><pubDate>Mon, 12 Oct 2026 01:00:00 GMT</pubDate>
    <description>Public story.</description></item></channel></rss>'''
    monkeypatch.setattr(collection.Acquisition, 'fetch',
                        lambda self, url: (rss if url.endswith('feed') else b'<html/>', url,
                                           '2026-10-12T02:00:00+00:00'))
    def failed(raw):
        raise collection.ExtractionFailure('test native worker failure')
    monkeypatch.setattr(collection, 'extract_body', failed)
    from datetime import datetime
    with pytest.raises(collection.ExtractionFailure):
        collection.run(tmp_path, datetime.fromisoformat('2026-10-12T02:00:00+00:00'), [source])
    saved = json.loads((tmp_path/'corpus/corpus.json').read_text())
    assert saved['sampling']['phase'] == 'metadata_complete'
    assert list((tmp_path/'extraction-failures').glob('*.json'))


def test_separate_workers_do_not_share_parser_process(monkeypatch):
    real = subprocess.run
    def pids(command, **kwargs):
        return real([sys.executable, '-c', 'import os,json;print(json.dumps(str(os.getpid())))'],
                    **kwargs)
    monkeypatch.setattr(subprocess, 'run', pids)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(collection.extract_body, [b'x']*4))
    assert len(set(ids)) == 4
    assert str(os.getpid()) not in ids


def test_worker_timeout_and_invalid_output_block(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(collection.ExtractionFailure, match='timeout'):
        collection.extract_body(b'x')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess([],0,b'null',b''))
    with pytest.raises(collection.ExtractionFailure, match='output'):
        collection.extract_body(b'x')
