import hashlib
import importlib
import socket
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from news_assistant.editorial_model import EditorialModelClient
from news_assistant.models import Preferences
from news_assistant.reviewed_selection import prepare_review, select_reviewed

NOW = datetime(2026, 9, 29, 8, tzinfo=UTC)


def batching():
    return importlib.import_module('news_assistant.review_batches')


def snapshot(*, bodies=True, count=4, duplicate=False):
    sources = []
    contents = {}
    for source_id in ('b', 'a'):
        articles = []
        for index in range(count // 2):
            identifier = f'{source_id}-{index}'
            url = f'https://example.org/{identifier}'
            article = {
                'id': identifier, 'title': f'实验报告 {identifier}',
                'summary': '仅在实验室观察到效果，需要后续验证。',
                'source': source_id, 'url': url, 'language': 'zh',
                'published_at': (NOW - timedelta(hours=1)).isoformat(), 'topic': 'science',
            }
            articles.insert(0, article)
            text = '正文中的具体实验事实，保留局限。' * 100
            contents[f'{source_id}:{identifier}'] = {
                'status': 'extracted', 'requested_url': url, 'final_url': url,
                'retrieved_at': NOW.isoformat(), 'extractor_version': 'test',
                'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'html_sha256': '0' * 64, 'truncated': False,
            }
        if duplicate:
            articles.append(deepcopy(articles[0]))
        sources.append({
            'id': source_id, 'name': source_id, 'publisher_group': source_id,
            'production_enabled': False, 'status': 'ok', 'evidence_role': 'PRIMARY_FACT',
            'coverage_domains': ['science_health'], 'feed_url': f'https://example.org/{source_id}.xml',
            'parsed_count': len(articles) + 9, 'fresh_count': len(articles),
            'stale_count': 7, 'future_count': 2, 'candidates': articles,
        })
    audit = {'schema_version': 'source-audit-v1', 'candidate_snapshot_available': True,
             'captured_at': NOW.isoformat(), 'max_age_hours': 36, 'sources': sources,
             'run_label': 'preserve metadata'}
    if bodies:
        audit['article_content'] = {'schema_version': 'article-content-v1', 'items': contents}
    return audit


def reviews_for(batches):
    reviews = []
    for index, batch in enumerate(batches):
        review = prepare_review(batch)
        review.update(reviewed_by=f'reviewer-{index}',
                      reviewed_at=(NOW + timedelta(minutes=index)).isoformat())
        for item in review['items']:
            item.update(status='rejected', rationale='需要更多证据。',
                        quality_score=0.3, importance_score=0.4,
                        evidence_excerpt='正文中的具体实验事实，保留局限。' if 'article_content' in batch
                        else item['evidence_excerpt'])
        reviews.append(review)
    return reviews


def test_split_respects_actual_prompt_budget_and_preserves_complete_body_binding():
    audit = snapshot(duplicate=True)
    original = deepcopy(audit)
    batches = batching().split_review_batches(audit, now=NOW, max_chars=3500)
    assert len(batches) == 4
    identifiers = []
    for batch in batches:
        prepared = EditorialModelClient('')._prepare_request(batch, now=NOW, max_candidates=200)
        assert len(prepared.user_prompt) <= 3500
        ids = [item['candidate_id'] for item in prepared.template['items']]
        identifiers.extend(ids)
        assert set(batch['article_content']['items']) == set(ids)
        for identifier in ids:
            assert batch['article_content']['items'][identifier] == audit['article_content']['items'][identifier]
        assert batch['batch_origin'] == {
            'snapshot_sha256': prepare_review(audit)['snapshot_sha256'], 'counts_scope': 'batch'}
        for source in batch['sources']:
            assert source['parsed_count'] == source['fresh_count'] == len(source['candidates'])
            assert source['stale_count'] == source['future_count'] == 0
            assert source['feed_url'].endswith('.xml')
    assert identifiers == ['a:a-0', 'a:a-1', 'b:b-0', 'b:b-1']
    assert audit == original


def test_split_supports_old_rss_snapshot_and_runs_without_network_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('offline batching must not access network')
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    audit = snapshot(bodies=False)
    batches = batching().split_review_batches(audit, now=NOW)
    assert len(batches) == 1
    assert 'article_content' not in batches[0]
    merged = batching().merge_review_batches(audit, batches, reviews_for(batches), now=NOW)
    assert len(merged['items']) == 4


@pytest.mark.parametrize('budget', [0, -1, 60001, True, 3500.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        batching().split_review_batches(snapshot(), now=NOW, max_chars=budget)


def test_single_oversized_candidate_fails_explicitly_without_truncation():
    with pytest.raises(ValueError, match='candidate.*fit|candidate.*exceed'):
        batching().split_review_batches(snapshot(), now=NOW, max_chars=1000)


def test_merge_rebinds_only_original_hash_and_retains_every_judgment():
    audit = snapshot()
    batches = batching().split_review_batches(audit, now=NOW, max_chars=3500)
    reviews = reviews_for(batches)
    reviews[0]['items'][0].update(status='pending', quality_score=None, importance_score=None)
    originals = deepcopy((audit, batches, reviews))
    merged = batching().merge_review_batches(audit, batches, reviews, now=NOW + timedelta(hours=1))
    assert merged['snapshot_sha256'] == prepare_review(audit)['snapshot_sha256']
    expected = {item['candidate_id']: item for review in reviews for item in review['items']}
    assert {item['candidate_id']: item for item in merged['items']} == expected
    for index in range(4):
        assert f'reviewer-{index}' in merged['reviewed_by']
    assert merged['reviewed_at'] == reviews[-1]['reviewed_at']
    # All decisions are pending or rejected and all sources shadow: validation must not require selection.
    assert select_reviewed(audit, merged, Preferences(), now=NOW + timedelta(hours=1)).selected == []
    assert (audit, batches, reviews) == originals


@pytest.mark.parametrize('mutation', [
    'missing_batch', 'duplicate_batch', 'missing_review', 'wrong_origin', 'body',
    'article', 'source_metadata', 'top_metadata', 'counts', 'unknown_candidate',
    'review_hash', 'review_fingerprint', 'quote', 'missing_item', 'duplicate_item',
    'future_review', 'early_review', 'unattributed_review',
])
def test_merge_rejects_incomplete_or_tampered_batches_and_reviews(mutation):
    audit = snapshot()
    batches = batching().split_review_batches(audit, now=NOW, max_chars=3500)
    reviews = reviews_for(batches)
    if mutation == 'missing_batch':
        batches.pop(); reviews.pop()
    elif mutation == 'duplicate_batch':
        batches.append(deepcopy(batches[0])); reviews.append(deepcopy(reviews[0]))
    elif mutation == 'missing_review':
        reviews.pop()
    elif mutation == 'wrong_origin':
        batches[0]['batch_origin']['snapshot_sha256'] = '0' * 64
    elif mutation == 'body':
        body = next(iter(batches[0]['article_content']['items'].values()))
        body['text'] += '假内容'
        body['text_sha256'] = hashlib.sha256(body['text'].encode()).hexdigest()
        reviews[0] = reviews_for([batches[0]])[0]
    elif mutation == 'article':
        batches[0]['sources'][0]['candidates'][0]['summary'] += '假内容'
        reviews[0] = reviews_for([batches[0]])[0]
    elif mutation == 'source_metadata':
        batches[0]['sources'][0]['publisher_group'] = 'other publisher'
        reviews[0] = reviews_for([batches[0]])[0]
    elif mutation == 'top_metadata':
        batches[0]['run_label'] = 'tampered'
        reviews[0] = reviews_for([batches[0]])[0]
    elif mutation == 'counts':
        batches[0]['sources'][0]['stale_count'] = 1
        batches[0]['sources'][0]['parsed_count'] += 1
        reviews[0] = reviews_for([batches[0]])[0]
    elif mutation == 'unknown_candidate':
        batches[0]['sources'][0]['candidates'][0]['id'] = 'unknown'
    elif mutation == 'review_hash':
        reviews[0]['snapshot_sha256'] = '0' * 64
    elif mutation == 'review_fingerprint':
        reviews[0]['items'][0]['article_sha256'] = '0' * 64
    elif mutation == 'quote':
        reviews[0]['items'][0]['evidence_excerpt'] = '原文没有的引句'
    elif mutation == 'missing_item':
        reviews[0]['items'].clear()
    elif mutation == 'duplicate_item':
        reviews[0]['items'].append(deepcopy(reviews[0]['items'][0]))
    elif mutation == 'future_review':
        reviews[0]['reviewed_at'] = (NOW + timedelta(days=1)).isoformat()
    elif mutation == 'early_review':
        reviews[0]['reviewed_at'] = (NOW - timedelta(minutes=1)).isoformat()
    else:
        reviews[0]['reviewed_by'] = None
    with pytest.raises(ValueError):
        batching().merge_review_batches(audit, batches, reviews, now=NOW + timedelta(hours=1))


def test_split_handles_real_60000_character_failure_without_losing_text():
    audit = snapshot()
    for body in audit['article_content']['items'].values():
        body['text'] = '保留每一段原文。' * 2500
        body['text_sha256'] = hashlib.sha256(body['text'].encode()).hexdigest()
    with pytest.raises(ValueError, match='exceeds 60000'):
        EditorialModelClient('')._prepare_request(audit, now=NOW, max_candidates=200)
    batches = batching().split_review_batches(audit, now=NOW)
    assert len(batches) == 2
    assert sum(len(batch['article_content']['items']) for batch in batches) == 4
    for batch in batches:
        assert len(EditorialModelClient('')._prepare_request(
            batch, now=NOW, max_candidates=200).user_prompt) <= 60000
        for identifier, content in batch['article_content']['items'].items():
            assert content['text'] == audit['article_content']['items'][identifier]['text']


def test_stable_sort_is_by_source_identity_then_candidate_not_joined_id():
    audit = snapshot(bodies=False)
    source = audit['sources'][0]
    source.update(id='a-', name='a-')
    for article in source['candidates']:
        article['source'] = 'a-'
    batches = batching().split_review_batches(audit, now=NOW, max_chars=1000)
    identifiers = [item['candidate_id'] for batch in batches for item in prepare_review(batch)['items']]
    assert identifiers == ['a:a-0', 'a:a-1', 'a-:b-0', 'a-:b-1']


def test_merge_preserves_accepted_chinese_copy_and_pending_null_scores():
    audit = snapshot()
    batches = batching().split_review_batches(audit, now=NOW)
    reviews = reviews_for(batches)
    reviews[0]['items'][0].update(status='accepted', quality_score=0.8, importance_score=0.6,
                                chinese_title='实验观察仍需验证', chinese_summary='据报告，结果仅限实验室。')
    merged = batching().merge_review_batches(audit, batches, reviews, now=NOW)
    item = next(item for item in merged['items'] if item['status'] == 'accepted')
    assert item['chinese_title'] == '实验观察仍需验证'
    assert item['chinese_summary'] == '据报告，结果仅限实验室。'
    assert item['quality_score'] == 0.8


@pytest.mark.parametrize('mutation', ['future_capture', 'future_body', 'conflicting_duplicate'])
def test_split_rejects_invalid_original_before_building_batches(mutation):
    audit = snapshot(duplicate=True)
    if mutation == 'future_capture':
        audit['captured_at'] = (NOW + timedelta(minutes=1)).isoformat()
    elif mutation == 'future_body':
        next(iter(audit['article_content']['items'].values()))['retrieved_at'] = (
            NOW + timedelta(minutes=1)).isoformat()
    else:
        audit['sources'][0]['candidates'][-1]['summary'] += '冲突内容'
    with pytest.raises(ValueError):
        batching().split_review_batches(audit, now=NOW)


def test_merge_rejects_type_changed_metadata_even_if_python_values_compare_equal():
    audit = snapshot()
    audit['run_number'] = 1
    batches = batching().split_review_batches(audit, now=NOW)
    batches[0]['run_number'] = True
    reviews = reviews_for(batches)
    with pytest.raises(ValueError, match='original snapshot subset'):
        batching().merge_review_batches(audit, batches, reviews, now=NOW)


def test_batch_schedule_interleaves_topics_before_exhausting_first_source():
    audit = snapshot(count=8)
    for source in audit['sources']:
        for article in source['candidates']:
            article['topic'] = 'business' if source['id'] == 'a' else 'science'
    batches = batching().split_review_batches(audit, now=NOW, max_chars=3500)
    early_topics = [batch['sources'][0]['candidates'][0]['topic'] for batch in batches[:2]]
    assert set(early_topics) == {'business', 'science'}
    assert len(batches) == 8
    assert batches == batching().split_review_batches(audit, now=NOW, max_chars=3500)
