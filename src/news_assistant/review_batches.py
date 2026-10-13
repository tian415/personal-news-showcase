"""Offline, lossless batching and content-bound merging of editorial reviews."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime

from .coverage_evaluation import _aware, _fingerprint
from .editorial_model import _MAX_SOURCE_CHARS, EditorialModelClient
from .models import Preferences
from .reviewed_selection import EditorialReview, prepare_review, select_reviewed


def _validated_template(audit: dict, *, now: datetime) -> dict:
    template = prepare_review(audit)
    # Pending judgments validate the input bindings without selecting or scoring it.
    select_reviewed(audit, template, Preferences(), now=now)
    return template


def _candidate_articles(audit: dict) -> dict[str, dict]:
    articles = {}
    for source in sorted(audit['sources'], key=lambda item: item['id']):
        for article in sorted(source['candidates'], key=lambda item: item['id']):
            articles.setdefault(f"{source['id']}:{article['id']}", article)
    return articles


def _batch_snapshot(audit: dict, identifiers: set[str], *, snapshot_hash: str) -> dict:
    """Construct the sole allowed representation of a subset of this snapshot."""
    batch = deepcopy(audit)
    articles = _candidate_articles(audit)
    batch['sources'] = []
    for source in sorted(audit['sources'], key=lambda item: item['id']):
        source_ids = {f"{source['id']}:{article['id']}" for article in source['candidates']}
        selected = sorted(identifiers & source_ids)
        if not selected:
            continue
        subset = deepcopy(source)
        subset.update(candidates=[deepcopy(articles[identifier]) for identifier in selected],
                      parsed_count=len(selected), fresh_count=len(selected),
                      stale_count=0, future_count=0, status='ok')
        batch['sources'].append(subset)
    if 'article_content' in batch:
        batch['article_content']['items'] = {
            identifier: deepcopy(audit['article_content']['items'][identifier])
            for identifier in sorted(identifiers)
        }
    batch['batch_origin'] = {'snapshot_sha256': snapshot_hash, 'counts_scope': 'batch'}
    return batch


def split_review_batches(audit: dict, *, now: datetime, max_chars: int = 60000) -> list[dict]:
    """Greedily partition all unique candidates using the actual model prompt size.

    Each result is a separately reviewable source-audit-v1 snapshot. No key lookup,
    provider request, scoring, or truncation occurs. One unfit candidate is an error.
    """
    if type(max_chars) is not int or not 1 <= max_chars <= _MAX_SOURCE_CHARS:
        raise ValueError('max_chars must be an integer from 1 to 60000')
    now = _aware(now)
    template = _validated_template(audit, now=now)
    # Alternate captured topic groups so interruption does not leave every later
    # subject unreviewed merely because its source id sorts last. No scoring here.
    by_topic = defaultdict(deque)
    for identifier, article in _candidate_articles(audit).items():
        by_topic[article["topic"]].append(identifier)
    identifiers = []
    while any(by_topic.values()):
        for topic in sorted(by_topic):
            if by_topic[topic]:
                identifiers.append(by_topic[topic].popleft())
    if not identifiers:
        raise ValueError('no captured candidates to assess')
    client = EditorialModelClient('')  # _prepare_request is local and never uses api_key.

    def build(selected: list[str]) -> dict:
        return _batch_snapshot(audit, set(selected), snapshot_hash=template['snapshot_sha256'])

    def fits(batch: dict, count: int) -> bool:
        if count > 200:
            return False
        try:
            request = client._prepare_request(batch, now=now, max_candidates=200)
        except ValueError as error:
            if str(error) != 'source material user prompt exceeds 60000 characters':
                raise
            return False
        return len(request.user_prompt) <= max_chars

    batches = []
    current: list[str] = []
    for identifier in identifiers:
        trial = [*current, identifier]
        if fits(build(trial), len(trial)):
            current = trial
            continue
        if current:
            batches.append(build(current))
        current = [identifier]
        if not fits(build(current), 1):
            raise ValueError(f'candidate {identifier} cannot fit within {max_chars} prompt characters')
    batches.append(build(current))
    return batches


def merge_review_batches(
    audit: dict, batches: list[dict], reviews: list[dict], *, now: datetime,
) -> dict:
    """Validate exact original subsets and each review before rebinding to the origin.

    The batch_origin claim is never sufficient: every batch must equal a subset
    reconstructed from the supplied original. Pending items retain their null scores.
    """
    now = _aware(now)
    template = _validated_template(audit, now=now)
    expected = {item['candidate_id'] for item in template['items']}
    if not batches or len(batches) != len(reviews):
        raise ValueError('one review is required for each nonempty batch')
    seen: set[str] = set()
    merged_items = {}
    reviewers = set()
    review_times = []
    for batch, review in zip(batches, reviews, strict=True):
        batch_template = prepare_review(batch)
        identifiers = {item['candidate_id'] for item in batch_template['items']}
        if not identifiers or not identifiers <= expected:
            raise ValueError('batch contains no candidates or unknown candidates')
        if seen & identifiers:
            raise ValueError('batch candidates must be covered exactly once')
        reconstructed = _batch_snapshot(
            audit, identifiers, snapshot_hash=template['snapshot_sha256'])
        if _fingerprint(batch) != _fingerprint(reconstructed):
            raise ValueError('batch does not match the original snapshot subset')
        assessed = EditorialReview.model_validate(review)
        if not assessed.reviewed_by or not assessed.reviewed_by.strip() or assessed.reviewed_at is None:
            raise ValueError('each batch review requires a reviewer and review time')
        # Validation applies even to rejected/pending/expired/shadow-only batches;
        # whether a candidate is selected is irrelevant to merging authority.
        select_reviewed(batch, review, Preferences(), now=now)
        seen.update(identifiers)
        merged_items.update({item['candidate_id']: deepcopy(item) for item in review['items']})
        reviewers.add(assessed.reviewed_by)
        review_times.append(assessed.reviewed_at)
    if seen != expected:
        raise ValueError('batches must cover every original candidate exactly once')
    merged = {
        'schema_version': 'editorial-review-v1',
        'snapshot_sha256': template['snapshot_sha256'],
        'reviewed_by': 'batch reviewers: ' + json.dumps(sorted(reviewers), ensure_ascii=False),
        'reviewed_at': max(review_times).isoformat(),
        'items': [merged_items[identifier] for identifier in _candidate_articles(audit)],
    }
    select_reviewed(audit, merged, Preferences(), now=now)
    return merged
