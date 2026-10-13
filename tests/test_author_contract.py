import copy
import json

import pytest

from experiments.daily_writer.author_contract import check


def case():
    request = {'request_hash': 'frozen', 'packet': {'items': [
        {'id': 'a', 'evidence_passages': [{'id': 'e', 'quote': 'source'}]}]}}
    draft = {'request_hash': 'frozen', 'items': [{'id': 'a', 'title': '新闻',
             'summary': '损失尚未确定。', 'paragraphs': [
                 {'text': '这是公司主张，不是性能实测。', 'evidence_ids': ['e']}]}],
             'input_gaps': []}
    return request, draft


def test_clean_contract_never_implies_semantic_acceptance():
    request, draft = case()
    result = check(request, json.dumps(draft))
    assert result['result'] == 'pending_semantic_review'
    assert result['automatic_pass'] is False


@pytest.mark.parametrize('change,code', [
    (lambda d: d['items'][0].update(input_gaps=[]), 'contract'),
    (lambda d: d.update(request_hash='other'), 'contract'),
    (lambda d: d['items'].append(copy.deepcopy(d['items'][0])), 'contract'),
    (lambda d: d['items'][0]['paragraphs'][0].update(evidence_ids=['other']), 'contract'),
    (lambda d: d['items'][0].update(summary='长' * 161), 'contract'),
    (lambda d: d.update(input_gaps=[{'id': 'unknown', 'missing': '材料'}]), 'contract'),
    (lambda d: d['items'][0].update(paragraphs=None), 'contract'),
])
def test_contract_rejects_invalid_output_without_repair(change, code):
    request, draft = case()
    change(draft)
    assert code in {f['code'] for f in check(request, json.dumps(draft))['findings']}


def test_duplicate_keys_are_rejected():
    request, draft = case()
    raw = json.dumps(draft).replace('"request_hash": "frozen"',
                                    '"request_hash":"wrong","request_hash":"frozen"')
    assert check(request, raw)['result'] == 'rejected'


@pytest.mark.parametrize('text', ['本刊不自行裁决。', '另一家冻结报道显示。',
    '这些数量不能写成已经进口。', '财政部原指控与当事人否认必须分别归属。',
    '当期进展是公司回应，不是解雇本日新发生。'])
def test_process_language_is_detected_even_when_contract_also_fails(text):
    request, draft = case()
    draft['items'][0]['input_gaps'] = []
    draft['items'][0]['paragraphs'][0]['text'] = text
    codes = {f['code'] for f in check(request, json.dumps(draft))['findings']}
    assert {'contract', 'editorial_process'} <= codes


def test_evidence_bound_risk_rule_flags_original_error_not_separate_cases():
    request, draft = case()
    rules = [{'name': 'case_conflation', 'pattern': r'7月[^。；\n]{0,30}对拜登政府提起诉讼',
              'item_id': 'a', 'evidence_ids': ['e']}]
    draft['items'][0]['paragraphs'][0]['text'] = '她7月对拜登政府提起诉讼。'
    result = check(request, json.dumps(draft), rules)
    assert any(f['code'] == 'semantic_risk' for f in result['findings'])
    draft['items'][0]['paragraphs'][0]['text'] = '她此前对拜登政府提起诉讼。7月她又提出第二次申诉。'
    assert check(request, json.dumps(draft), rules)['result'] == 'pending_semantic_review'
    with pytest.raises(ValueError, match='evidence'):
        check(request, json.dumps(draft), [{**rules[0], 'evidence_ids': ['invented']}])


def test_letter_channel_tripwire_and_legitimate_uncertainty():
    request, draft = case()
    rules = [{'name': 'channel_conflation', 'item_id': 'a', 'evidence_ids': ['e'],
              'pattern': r'公开信[^。；\n]{0,100}安全置于[^。；\n]{0,45}近期利益'}]
    draft['items'][0]['paragraphs'][0]['text'] = '三人在公开信中说，他们把安全置于公司的近期利益之上。'
    result = check(request, json.dumps(draft), rules)
    assert any(f['code'] == 'semantic_risk' for f in result['findings'])
    draft['items'][0]['paragraphs'][0]['text'] = '巴莱斯尼在X上说，他们把安全置于公司近期利益之上。公开信另谈文化变化。动机尚未获独立证实。'
    assert check(request, json.dumps(draft), rules)['result'] == 'pending_semantic_review'


@pytest.mark.parametrize('value', [None, [], {'items': None}, {'items': [None]},
                                     {'items': [{'paragraphs': [None, 3]}]}])
def test_malformed_output_is_reported_not_crashed(value):
    request, _ = case()
    assert check(request, json.dumps(value))['result'] == 'rejected'
