"""Local chain gates, using synthetic review approvals only."""
import json

import pytest
from test_multisource_v3 import inputs

from experiments.local_pipeline import approve, import_feedback, prepare, run


def setup_run(tmp_path):
    root = tmp_path / 'private-run'
    root.mkdir()
    paths = {}
    for name, data in zip(('corpus', 'reviews', 'facts', 'packets'), inputs(), strict=True):
        path = root / (name + '.json')
        path.write_text(json.dumps(data))
        paths[name] = path
    return root, paths


def test_gate_and_exact_hash_approval(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    assert not (root / 'reader.html').exists()
    assert '待人工审核' in (root / 'review-preview.html').read_text()
    with pytest.raises(ValueError, match='hash'):
        approve(root, reviewer='test human', expected_hash='wrong')
    approve(root, reviewer='synthetic test approval', expected_hash=result['edition_hash'])
    assert (root / 'reader.html').exists()
    paths['reviews'].write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        approve(root, reviewer='test human', expected_hash=result['edition_hash'])
    assert not (root / 'reader.html').exists()


def test_pending_review_fail_closed_removes_old_reader(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    data = json.loads(paths['reviews'].read_text())
    data['items'][0]['status'] = 'pending'
    paths['reviews'].write_text(json.dumps(data))
    with pytest.raises(ValueError, match='pending'):
        run(root, **paths)
    assert not (root / 'reader.html').exists()


def test_feedback_is_bound_deduplicated_and_consumed_by_prepare(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    payload = {'schema': 'local-v4-feedback-v1', 'edition_hash': result['edition_hash'],
               'items': [{'event_id': result['result']['selected'][0]['id'], 'action': 'useful'}]}
    feedback = root / 'export.json'
    feedback.write_text(json.dumps(payload))
    assert import_feedback(root, feedback)['imported'] == 1
    assert import_feedback(root, feedback)['imported'] == 0
    context = prepare(root, paths['corpus'])
    assert context['feedback'][0]['action'] == 'useful'
    payload['items'][0]['event_id'] = 'unknown'
    feedback.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='unknown event'):
        import_feedback(root, feedback)


def test_public_root_rejected(tmp_path):
    with pytest.raises(ValueError, match='private'):
        prepare(tmp_path / 'public', tmp_path / 'missing.json')


def test_render_keeps_progress_evidence_separate_and_escapes(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    page = (root / 'review-preview.html').read_text()
    assert page.count('class="event"') == 1
    assert page.count('class="development"') == 2
    assert '独立证据链' in page and '时间依据' in page
    assert result['result']['selected'][0]['developments'][0]['confidence']['independent_chains'] == 1


def test_next_run_context_consumes_previous_feedback_without_scoring(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    payload = {'schema': 'local-v4-feedback-v1', 'edition_hash': result['edition_hash'],
               'items': [{'event_id': result['result']['selected'][0]['id'], 'action': 'not_useful'}]}
    feedback = root / 'export.json'
    feedback.write_text(json.dumps(payload))
    import_feedback(root, feedback)
    context = prepare(tmp_path / 'next-run', paths['corpus'], feedback_roots=[root])
    assert context['feedback'][0]['action'] == 'not_useful'
    assert 'ranking unchanged' in context['feedback_policy']
    assert run(root, **paths)['result'] == result['result']


def test_tampered_result_cannot_be_approved(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    changed = json.loads((root / 'selection.json').read_text())
    changed['result']['selected'][0]['headline'] = 'Injected title'
    (root / 'selection.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='result changed'):
        approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    assert not (root / 'reader.html').exists()


def test_invalid_quote_and_missing_input_fail_without_stale_edition(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    reviews = json.loads(paths['reviews'].read_text())
    reviews['items'][0]['impact_quote'] = 'not present in source'
    paths['reviews'].write_text(json.dumps(reviews))
    with pytest.raises(ValueError, match='quotation'):
        run(root, **paths)
    assert not (root / 'reader.html').exists()
    assert not (root / 'selection.json').exists()
    paths['reviews'].unlink()
    with pytest.raises(FileNotFoundError):
        run(root, **paths)


def test_html_escapes_injected_packet_title(tmp_path):
    root, paths = setup_run(tmp_path)
    packets = json.loads(paths['packets'].read_text())
    packets['packets'][0]['title'] = '<script>alert(1)</script>'
    paths['packets'].write_text(json.dumps(packets))
    run(root, **paths)
    page = (root / 'review-preview.html').read_text()
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page


def test_prepare_invalidates_approved_reader_after_corpus_change(tmp_path):
    root, paths = setup_run(tmp_path)
    prepare(root, paths['corpus'])
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    corpus = json.loads(paths['corpus'].read_text())
    corpus['articles'][0]['title'] = 'Changed source title'
    paths['corpus'].write_text(json.dumps(corpus))
    prepare(root, paths['corpus'])
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()


def test_prepare_invalidates_reader_if_engine_changes(tmp_path, monkeypatch):
    import experiments.local_pipeline as pipeline
    root, paths = setup_run(tmp_path)
    prepare(root, paths['corpus'])
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    monkeypatch.setattr(pipeline, '_engine_hashes', lambda: {'engine_v4.py': 'changed'})
    prepare(root, paths['corpus'])
    assert not (root / 'reader.html').exists()


def test_coverage_reports_reviewed_articles_without_support_facts(tmp_path):
    root, paths = setup_run(tmp_path)
    facts = json.loads(paths['facts'].read_text())
    facts['items'][1]['facts'] = []
    facts['items'][1]['reason'] = 'No supported fact'
    paths['facts'].write_text(json.dumps(facts))
    packets = json.loads(paths['packets'].read_text())
    from experiments.multisource_validation.engine import corpus_hash
    packets['facts_sha256'] = corpus_hash(facts)
    packets['items'] = packets['items'][:1]
    paths['packets'].write_text(json.dumps(packets))
    run(root, **paths)
    page = (root / 'review-preview.html').read_text()
    assert '2 条元数据，2 篇正文，0 篇未取得正文，1 篇未提出支持事实' in page
    assert '未提出支持事实' in page


def test_prepare_missing_corpus_removes_old_reader_and_approval(tmp_path):
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    paths['corpus'].unlink()
    with pytest.raises(FileNotFoundError):
        prepare(root, paths['corpus'])
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()


def test_approval_disk_failure_leaves_no_approved_reader(tmp_path, monkeypatch):
    import experiments.local_pipeline as pipeline
    root, paths = setup_run(tmp_path)
    result = run(root, **paths)
    original = pipeline._write

    def fail_receipt(path, value):
        if path.name == '.approval.json.pending':
            raise OSError('simulated disk write failure')
        original(path, value)

    monkeypatch.setattr(pipeline, '_write', fail_receipt)
    with pytest.raises(OSError, match='disk write'):
        approve(root, reviewer='synthetic', expected_hash=result['edition_hash'])
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()


def test_replay_cli_defaults_to_saved_reference(tmp_path, monkeypatch):
    import sys

    import experiments.local_pipeline.__main__ as cli
    root, paths = setup_run(tmp_path)
    (root / 'corpus').mkdir()
    (root / 'corpus/corpus.json').write_text(paths['corpus'].read_text())
    expected = json.loads(paths['corpus'].read_text())['reference_utc']
    observed = []

    def fake_collect(output, reference, **kwargs):
        observed.append((output, reference.isoformat(), kwargs['replay']))
        return {'test': True}

    monkeypatch.setattr(cli, 'collect', fake_collect)
    monkeypatch.setattr(sys, 'argv', ['local_pipeline', 'collect', '--root', str(root), '--replay'])
    cli.main()
    assert observed == [(root, expected, True)]


@pytest.mark.parametrize('broken', ['context', 'feedback'])
def test_prepare_late_failure_revokes_edition(tmp_path, broken):
    root, paths = setup_run(tmp_path)
    prepare(root, paths['corpus'])
    edition = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=edition['edition_hash'])
    target = root / ('review/context.json' if broken == 'context' else 'feedback.jsonl')
    target.write_text('{bad json')
    with pytest.raises(ValueError):
        prepare(root, paths['corpus'])
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()
    assert not (root / 'selection.json').exists()


def test_render_failure_leaves_no_approvable_edition(tmp_path, monkeypatch):
    import experiments.local_pipeline as pipeline
    root, paths = setup_run(tmp_path)
    def fail_render(*args, **kwargs):
        raise RuntimeError('render failed')
    monkeypatch.setattr(pipeline, '_render', fail_render)
    with pytest.raises(RuntimeError, match='render failed'):
        run(root, **paths)
    assert not (root / 'selection.json').exists()
    assert not (root / 'reader.html').exists()


def test_stale_feedback_revokes_reader_but_invalid_feedback_preserves_valid_reader(tmp_path):
    root, paths = setup_run(tmp_path)
    edition = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=edition['edition_hash'])
    payload = root / 'feedback-input.json'
    payload.write_text(json.dumps({'edition_hash': 'wrong', 'schema': 'bad', 'items': []}))
    with pytest.raises(ValueError):
        import_feedback(root, payload)
    assert (root / 'reader.html').exists()
    paths['reviews'].write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        import_feedback(root, payload)
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()


@pytest.mark.parametrize('broken', ['reference', 'sources'])
def test_collect_cli_parse_failure_revokes_reader(tmp_path, monkeypatch, broken):
    import sys

    import experiments.local_pipeline.__main__ as cli
    root, paths = setup_run(tmp_path)
    edition = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=edition['edition_hash'])
    args = ['local_pipeline', 'collect', '--root', str(root), '--fetch']
    if broken == 'reference':
        args.extend(['--reference-utc', 'not-a-date'])
    else:
        source = root / 'bad-sources.json'
        source.write_text('{bad json')
        args.extend(['--sources', str(source)])
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit):
        cli.main()
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()


@pytest.mark.parametrize('phase', ['page', 'receipt'])
def test_process_exit_during_approval_staging_has_no_reader(tmp_path, phase):
    import subprocess
    import sys
    root, paths = setup_run(tmp_path)
    edition = run(root, **paths)
    script = '''
import os, sys
from pathlib import Path
import experiments.local_pipeline as pipeline
root, expected, phase = sys.argv[1:]
original_write = pipeline._write
original_text = Path.write_text
def interrupted_write(path, value):
    result = original_write(path, value)
    if 'approval' in path.name and phase == 'receipt':
        os._exit(73)
    return result
def interrupted_text(path, *args, **kwargs):
    result = original_text(path, *args, **kwargs)
    if 'reader' in path.name and phase == 'page':
        os._exit(73)
    return result
pipeline._write = interrupted_write
Path.write_text = interrupted_text
pipeline.approve(Path(root), reviewer='synthetic subprocess', expected_hash=expected)
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), edition['edition_hash'], phase],
                            check=False)
    assert result.returncode == 73
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()



def test_reader_commit_failure_removes_receipt_and_staging(tmp_path, monkeypatch):
    from pathlib import Path

    root, paths = setup_run(tmp_path)
    edition = run(root, **paths)
    original = Path.replace
    def failed_commit(path, target):
        if path.name == '.reader.html.pending':
            raise OSError('reader commit failed')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', failed_commit)
    with pytest.raises(OSError, match='commit failed'):
        approve(root, reviewer='synthetic', expected_hash=edition['edition_hash'])
    for name in ('reader.html', 'approval.json', '.reader.html.pending', '.approval.json.pending'):
        assert not (root / name).exists()


def test_collect_runtime_failure_cli_revokes_and_reports_blocked(tmp_path, monkeypatch, capsys):
    import sys

    import experiments.local_pipeline.__main__ as cli
    root, paths = setup_run(tmp_path)
    edition = run(root, **paths)
    approve(root, reviewer='synthetic', expected_hash=edition['edition_hash'])
    def failed_collect(*args, **kwargs):
        raise RuntimeError('native extraction worker failed')
    monkeypatch.setattr(cli, 'collect', failed_collect)
    monkeypatch.setattr(sys, 'argv', ['local_pipeline', 'collect', '--root', str(root), '--fetch'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert 'Blocked: native extraction worker failed' in capsys.readouterr().err
    assert not (root / 'reader.html').exists()
    assert not (root / 'approval.json').exists()
