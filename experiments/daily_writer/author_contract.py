"""Offline gate for evidence-cited drafts. Never repairs or approves model prose.

Risk rules are reviewer-owned, evidence-bound regression tripwires, not a factual
truth classifier. Clean output still needs the frozen six-dimension review.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from .probe import PROCESS as OLD_PROCESS

PROCESS = re.compile(
    OLD_PROCESS.pattern + r'|本刊|冻结报道|不能写成|不推定她|须分清|范围须|'
    r'(?:必须|应|须)分别归属|归属(?:清楚|明确)|批评归属|只使用已给材料|'
    r'当期进展是|不据摘要|citing\s|本日新发生'
)


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate key: {key}')
        value[key] = item
    return value


def _shape(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f'expected fields: {keys}')


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('nonempty text required')


def _validate(request, draft):
    _shape(draft, 'request_hash items input_gaps')
    if draft['request_hash'] != request['request_hash']:
        raise ValueError('request binding mismatch')
    specs = {i['id']: i for i in request['packet']['items']}
    items = draft['items']
    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
        raise ValueError('items must be objects')
    if [i.get('id') for i in items] != list(specs):
        raise ValueError('exact item coverage/order required')
    for item in items:
        _shape(item, 'id title summary paragraphs')
        _text(item['title'])
        _text(item['summary'])
        if len(item['summary']) > 160:
            raise ValueError('summary exceeds 160 characters')
        if not isinstance(item['paragraphs'], list) or not item['paragraphs']:
            raise ValueError('nonempty paragraphs required')
        known = {p['id'] for p in specs[item['id']]['evidence_passages']}
        for paragraph in item['paragraphs']:
            _shape(paragraph, 'text evidence_ids')
            _text(paragraph['text'])
            refs = paragraph['evidence_ids']
            if not isinstance(refs, list) or not refs or any(
                not isinstance(r, str) or r not in known for r in refs
            ) or len(refs) != len(set(refs)):
                raise ValueError('invalid evidence reference')
    if not isinstance(draft['input_gaps'], list):
        raise TypeError('input_gaps must be a list at top level only')
    for gap in draft['input_gaps']:
        _shape(gap, 'id missing')
        if not isinstance(gap['id'], str) or gap['id'] not in specs:
            raise ValueError('unknown gap item')
        _text(gap['missing'])


def check(request, raw, risk_rules=()):
    """Inspect raw bytes as text; no normalization/repair and no provider access."""
    specs = {i['id']: i for i in request['packet']['items']}
    for rule in risk_rules:
        known = {p['id'] for p in specs.get(rule['item_id'], {}).get('evidence_passages', [])}
        if not rule['evidence_ids'] or not set(rule['evidence_ids']) <= known:
            raise ValueError('risk rule evidence binding mismatch')
    result = {'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
              'findings': [], 'automatic_pass': False, 'result': 'pending_semantic_review'}
    findings = result['findings']
    try:
        draft = json.loads(raw, object_pairs_hook=_object,
                           parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, TypeError) as exc:
        findings.append({'code': 'invalid_json', 'reason': str(exc)})
        result['result'] = 'rejected'
        return result
    try:
        _validate(request, draft)
    except (ValueError, TypeError) as exc:
        findings.append({'code': 'contract', 'reason': str(exc)})
    # Continue diagnostics even on shape failure, without accepting an adapted draft.
    items = draft.get('items', []) if isinstance(draft, dict) else []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        fields = [('title', item.get('title')), ('summary', item.get('summary'))]
        paragraphs = item.get('paragraphs')
        if isinstance(paragraphs, list):
            fields += [(f'paragraphs[{n}].text', p.get('text'))
                       for n, p in enumerate(paragraphs) if isinstance(p, dict)]
        for field, text in fields:
            if not isinstance(text, str):
                continue
            for match in PROCESS.finditer(text):
                findings.append({'code': 'editorial_process', 'id': item.get('id'),
                                 'field': field, 'excerpt': text,
                                 'matched': match.group()})
            for rule in risk_rules:
                if item.get('id') == rule['item_id'] and re.search(rule['pattern'], text):
                    findings.append({'code': 'semantic_risk', 'id': item.get('id'),
                                     'field': field, 'excerpt': text, 'rule': rule['name'],
                                     'evidence_ids': rule['evidence_ids'],
                                     'status': 'requires_source_comparison_not_automatic_fact_verdict'})
    if findings:
        result['result'] = 'rejected'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--risk-rules', type=Path)
    args = parser.parse_args()
    rules = json.loads(args.risk_rules.read_text()) if args.risk_rules else []
    result = check(json.loads(args.request.read_text()), args.raw.read_text(), rules)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Zero means schema screening only; caller must still obtain semantic review.
    return 2 if result['findings'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
