"""Checkout-only, private v4 orchestration; no model, mail or publication calls."""
from __future__ import annotations

import hashlib
import html
import json
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit

from experiments.multisource_validation.engine import corpus_hash
from experiments.multisource_validation.engine_v4 import run_v4

CORE = ('engine.py', 'engine_v2.py', 'engine_v3.py', 'engine_v4.py')


def _root(root):
    root = Path(root).resolve()
    if {'public', 'site', 'dist', '_site', 'docs'} & set(root.parts):
        raise ValueError('output requires a private run directory')
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _engine_hashes():
    folder = Path(__file__).resolve().parents[1] / 'multisource_validation'
    return {name: _sha(folder / name) for name in CORE}


def _invalidate(root):
    for name in ('reader.html', 'approval.json', '.reader.html.pending',
                 '.approval.json.pending', '.approval.json.pending.tmp'):
        (root / name).unlink(missing_ok=True)


def _failed_edition(root):
    _invalidate(root)
    for name in ('selection.json', 'review-preview.html'):
        (root / name).unlink(missing_ok=True)


def _fail_closed(operation):
    @wraps(operation)
    def guarded(root, *args, **kwargs):
        root = _root(root)
        try:
            return operation(root, *args, **kwargs)
        except BaseException:
            # Also revoke approvable drafts after late rendering/context failures.
            _failed_edition(root)
            raise
    return guarded


@_fail_closed
def collect(root, reference, *, sources=None, replay=False):
    """Public bounded collector and raw-evidence freeze, explicitly invoked only."""
    from experiments.multisource_validation import collect as acquisition
    root = _root(root)
    _invalidate(root)
    if reference.tzinfo is None:
        raise ValueError('reference requires timezone')
    if sources is None:
        sources = _read(Path(acquisition.__file__).with_name('sources.json'))
    acquisition.run(root, reference.astimezone(UTC), sources, replay=replay)
    acquisition.supplement(root, {}, fetch=False)
    return prepare(root, root / 'corpus/corpus.json')


@_fail_closed
def prepare(root, corpus, *, feedback_roots=()):
    """Generate pending templates without inferring dates, facts or review scores."""
    root = _root(root)
    try:
        data = _read(corpus)
        bound = corpus_hash(data)
    except (ValueError, TypeError, OSError):
        _invalidate(root)
        raise
    context_path = root / 'review/context.json'
    if context_path.exists():
        old = _read(context_path)
        if old.get('corpus_sha256') != bound or old.get('engine_sha256') != _engine_hashes():
            _invalidate(root)
    if (root / 'selection.json').exists():
        try:
            edition = _read(root / 'selection.json')
            _checked(root, edition['edition_hash'])
            if _sha(corpus) != edition['binding']['inputs']['corpus']['sha256']:
                _invalidate(root)
        except (ValueError, KeyError, TypeError, OSError):
            _invalidate(root)
    captured = [a for a in data['articles'] if a.get('body_status') in {'ok', 'extracted', 'captured'}]
    feedback = []
    for directory in (root, *map(Path, feedback_roots)):
        path = directory / 'feedback.jsonl'
        if path.exists():
            feedback.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    context = {'schema': 'local-v4-review-context-v1', 'corpus_sha256': bound,
               'engine_sha256': _engine_hashes(), 'feedback': feedback,
               'feedback_policy': 'Human review context only; frozen v4 scores and ranking unchanged.',
               'reference_utc': data['reference_utc'],
               'captured': len(captured), 'metadata_count': len(data['articles']),
               'instructions': 'Fill exact body quotations, entity/fact boundaries, dates and judgments. '
                               'Run creates a pending draft. An explicit approve command is required.'}
    _write(root / 'review/context.json', context)
    templates = {
        'reviews': {'corpus_sha256': bound, 'reviewer': '', 'items': [
            {'id': a['id'], 'body_sha256': a['body_sha256'], 'status': 'pending'} for a in captured]},
        'facts': {'schema': 'event-facts-v2', 'reviewer': '', 'corpus_sha256': bound,
                  'aliases': [], 'items': [
                      {'id': a['id'], 'body_sha256': a['body_sha256'], 'reason': '', 'facts': []}
                      for a in captured]},
        'packets': {'schema': 'event-packets-v3', 'reviewer': '', 'corpus_sha256': bound,
                    'facts_sha256': None, 'value_basis': 'within frozen corpus; no automatic feedback scores',
                    'packets': [], 'items': []},
    }
    for name, template in templates.items():
        # Regeneration must never overwrite editorial work.
        path = root / 'review/templates' / (name + '.json')
        if not path.exists():
            _write(path, template)
    _write(root / 'review/materials.json', {'corpus_sha256': bound, 'articles': captured})
    return context


@_fail_closed
def run(root, *, corpus, reviews, facts, packets, supplements=None, limit=12):
    """Validate bound editorial inputs and render a draft that awaits human approval."""
    root = _root(root)
    _invalidate(root)
    # Remove stale drafts/results too: a failed new execution has no approvable edition.
    for name in ('selection.json', 'review-preview.html'):
        (root / name).unlink(missing_ok=True)
    paths = {'corpus': corpus, 'reviews': reviews, 'facts': facts, 'packets': packets}
    if supplements is not None:
        paths['supplements'] = supplements
    values = {name: _read(path) for name, path in paths.items()}
    captured = {a['id'] for a in values['corpus']['articles'] if a['body_status'] in {'ok', 'extracted', 'captured'}}
    reviewed = {r['id'] for r in values['reviews']['items'] if r['status'] == 'reviewed'}
    if captured - reviewed:
        raise ValueError('pending or missing captured article reviews')
    result = run_v4(values['corpus'], values['reviews'], values['facts'], values['packets'],
                    supplements=values.get('supplements'), limit=limit)
    if not result['selected']:
        raise ValueError('no eligible events; reader blocked')
    inputs = {name: {'path': str(Path(path).resolve()), 'sha256': _sha(path)}
              for name, path in paths.items()}
    binding = {'inputs': inputs, 'engine_sha256': _engine_hashes(), 'limit': limit,
               'result_sha256': corpus_hash(result)}
    edition = {'schema': 'local-v4-edition-v1', 'status': 'pending_human_approval',
               'edition_hash': corpus_hash(binding), 'binding': binding, 'result': result,
               'created_at': datetime.now(UTC).isoformat()}
    _write(root / 'selection.json', edition)
    (root / 'review-preview.html').write_text(_render(edition, values['corpus'], values['reviews'], approved=False))
    return edition


def _checked(root, expected_hash):
    edition = _read(root / 'selection.json')
    if edition['edition_hash'] != expected_hash or corpus_hash(edition['binding']) != expected_hash:
        raise ValueError('edition hash mismatch')
    if corpus_hash(edition['result']) != edition['binding']['result_sha256']:
        raise ValueError('result changed')
    if _engine_hashes() != edition['binding']['engine_sha256']:
        raise ValueError('engine changed')
    for entry in edition['binding']['inputs'].values():
        if not Path(entry['path']).exists() or _sha(entry['path']) != entry['sha256']:
            raise ValueError('input changed or missing')
    return edition


def approve(root, *, reviewer, expected_hash):
    """Human-operated explicit approval, bound to exact inputs and result."""
    root = _root(root)
    _invalidate(root)
    if not reviewer.strip():
        raise ValueError('human reviewer attribution required')
    edition = _checked(root, expected_hash)
    corpus = _read(edition['binding']['inputs']['corpus']['path'])
    reviews = _read(edition['binding']['inputs']['reviews']['path'])
    page = _render(edition, corpus, reviews, approved=True)
    receipt = {'schema': 'local-v4-approval-v1', 'reviewer': reviewer.strip(),
               'edition_hash': expected_hash, 'approved_at': datetime.now(UTC).isoformat(),
               'scope': 'local reader only', 'reader_sha256': hashlib.sha256(page.encode()).hexdigest()}
    try:
        staged_page = root / '.reader.html.pending'
        staged_receipt = root / '.approval.json.pending'
        staged_page.write_text(page)
        _write(staged_receipt, receipt)
        staged_receipt.replace(root / 'approval.json')
        # The reader is the final commit marker: never expose a partial page.
        staged_page.replace(root / 'reader.html')
    except BaseException:
        _invalidate(root)
        raise
    return receipt


def import_feedback(root, input_path):
    root = _root(root)
    # Validate the current edition first, independently of a malformed export.
    # Bad feedback does not revoke an otherwise valid reader; stale inputs do.
    try:
        current = _read(root / 'selection.json')
        edition = _checked(root, current['edition_hash'])
        approval = _read(root / 'approval.json')
        if (approval['edition_hash'] != edition['edition_hash']
                or _sha(root / 'reader.html') != approval['reader_sha256']):
            raise ValueError('feedback requires unchanged approved edition')
    except (ValueError, KeyError, TypeError, OSError):
        _invalidate(root)
        raise
    payload = _read(input_path)
    if payload['edition_hash'] != edition['edition_hash']:
        raise ValueError('feedback edition hash mismatch')
    if payload.get('schema') != 'local-v4-feedback-v1':
        raise ValueError('invalid feedback schema')
    events = {event['id']: event for event in edition['result']['selected']}
    rows = []
    for item in payload['items']:
        if item['event_id'] not in events:
            raise ValueError('unknown event feedback target')
        if item['action'] not in {'useful', 'not_useful', 'read'}:
            raise ValueError('invalid feedback action')
        event = events[item['event_id']]
        rows.append({'edition_hash': edition['edition_hash'], 'event_id': item['event_id'],
                     'headline': event['headline'], 'domain': event['domain'],
                     'source_ids': event['source_ids'], 'event_scope': event['scope'],
                     'edition_scope': {'reference_utc': _read(edition['binding']['inputs']['corpus']['path'])['reference_utc'],
                                       'window_hours': 36, 'channel': 'local'},
                     'action': item['action'], 'feedback_id': corpus_hash({
                         'edition_hash': edition['edition_hash'], 'event_id': item['event_id'],
                         'action': item['action']})})
    path = root / 'feedback.jsonl'
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    ids = {row['feedback_id'] for row in existing}
    new = []
    for row in rows:
        if row['feedback_id'] not in ids:
            row['imported_at'] = datetime.now(UTC).isoformat()
            new.append(row)
            ids.add(row['feedback_id'])
    if new:
        with path.open('a') as handle:
            for row in new:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'imported': len(new), 'policy': 'next human review context; no v4 score changes'}


def _render(edition, corpus, reviews, *, approved):
    esc = lambda value: html.escape(str(value), quote=True)
    articles = {a['id']: a for a in corpus['articles']}
    judgments = {r['id']: r for r in reviews['items']}
    ledger = edition['result']['temporal_ledger']
    sections = []
    for event in edition['result']['selected']:
        developments = []
        for child in event['developments']:
            citations = []
            for quote in child['confidence']['support_quotes']:
                article = articles[quote['article_id']]
                url = article['url']
                if urlsplit(url).scheme not in {'http', 'https'}:
                    raise ValueError('unsafe source URL')
                citations.append(f'<li><a href="{esc(url)}">{esc(article["source"])} · '
                                 f'{esc(article["title"])}</a><blockquote>{esc(quote["quote"])}</blockquote></li>')
            time = {fid: ledger[fid] for fid in child['fact_ids']}
            confidence = child['confidence']
            editorial = judgments[child['member_ids'][0]]
            developments.append(f'<section class="development"><h3>{esc(child["headline"])}</h3>'
                                f'<p>为什么今天值得看：{esc(editorial["why_today"])}</p>'
                                f'<p>影响：{esc(editorial["impact"])}</p>'
                                f'<p>新增信息价值 {esc(child["incremental_value"])}；'
                                f'独立证据链 {esc(confidence["independent_chains"])}</p>'
                                f'<details><summary>时间依据</summary><pre>{esc(json.dumps(time, ensure_ascii=False, indent=2))}</pre></details>'
                                f'<details><summary>置信度依据及限制</summary><pre>{esc(json.dumps(confidence, ensure_ascii=False, indent=2))}</pre></details>'
                                f'<ul>{"".join(citations)}</ul></section>')
        buttons = (f'<button data-event="{esc(event["id"])}" data-action="useful">有帮助</button>'
                   f'<button data-event="{esc(event["id"])}" data-action="not_useful">帮助不大</button>'
                   f'<button data-event="{esc(event["id"])}" data-action="read">已读</button>') if approved else ''
        sections.append(f'<article class="event"><h2>{esc(event["headline"])}</h2>'
                        f'<p>{esc(event["scope"])} · {esc(event["domain"])} · {esc(event["region"])}</p>'
                        f'<p>事件重要程度 {esc(event["event_importance"])}</p>{"".join(developments)}{buttons}</article>')
    script = ''
    if approved:
        # JSON strings use escaping for script termination; all labels above HTML-escaped.
        bound = json.dumps(edition['edition_hash'])
        script = '''<button id="export">导出反馈 JSON</button><p id="feedback-status"></p><script>
const feedback = {schema:'local-v4-feedback-v1', edition_hash:EDITION, items:[]};
document.querySelectorAll('[data-action]').forEach(button => button.onclick = () => {
 const row = {event_id:button.dataset.event, action:button.dataset.action};
 if(!feedback.items.some(x=>x.event_id===row.event_id && x.action===row.action)) feedback.items.push(row);
 document.getElementById('feedback-status').textContent = '已记录 '+feedback.items.length+' 条，导出后用 import-feedback 导入。';
});
document.getElementById('export').onclick = () => {
 const blob = new Blob([JSON.stringify(feedback,null,2)],{type:'application/json'});
 const a = document.createElement('a'); a.href=URL.createObjectURL(blob);a.download='local-v4-feedback.json';a.click();URL.revokeObjectURL(a.href);
};</script>'''.replace('EDITION', bound)
    facts = _read(edition['binding']['inputs']['facts']['path'])
    fact_counts = {item['id']: len(item['facts']) for item in facts['items']}
    captured = [a for a in corpus['articles'] if a['body_status'] in {'ok', 'extracted', 'captured'}]
    empty = [a for a in captured if fact_counts.get(a['id'], 0) == 0]
    selected_members = {identifier for event in edition['result']['selected'] for identifier in event['member_ids']}
    exclusions = []
    for article in captured:
        if article['id'] not in selected_members:
            judgment = judgments[article['id']]
            reason = ('未提出支持事实；' if article in empty else '未入选；') + judgment['reason']
            exclusions.append(f'<li>{esc(article["title"])}：{esc(reason)}</li>')
    coverage = (f'<details><summary>整池审核覆盖：{len(corpus["articles"])} 条元数据，'
                f'{len(captured)} 篇正文，{len(corpus["articles"])-len(captured)} 篇未取得正文，'
                f'{len(empty)} 篇未提出支持事实</summary><p>采集基准：{esc(corpus["reference_utc"])}；36 小时窗口。'
                f'有正文文章均已完成编辑评估；最终人工批准仍是独立门禁。</p>'
                f'<ul>{"".join(exclusions)}</ul><p>事实排除原因：</p><pre>'
                f'{esc(json.dumps(edition["result"]["excluded_facts"], ensure_ascii=False, indent=2))}</pre></details>')
    title = '本地阅读版（人工已批准）' if approved else '待人工审核 · 本地草稿'
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>' + title + '</title><style>body{max-width:900px;margin:30px auto;padding:0 18px;'
            'font:16px/1.7 system-ui;background:#faf9f5;color:#24302a}article{margin:24px 0;padding:24px;'
            'background:white;border:1px solid #ccc;border-radius:12px}section{border-top:1px solid #ddd;'
            'padding:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{border-left:3px solid #ccc;'
            'padding-left:12px}button{padding:8px;margin:5px}a{color:#265c47}</style><body><h1>' + title +
            '</h1><p>算法：冻结 v4。每个进展保留自己的时间、证据与置信度。反馈用于下一次人工审核。</p>'
            '<p>审核绑定：<code>' + esc(edition['edition_hash']) + '</code></p>' + coverage + ''.join(sections) + script + '</body></html>')
