"""Private draft feedback; explicit ratings only, never ranking or approval updates."""
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path


class FeedbackStore:
    def __init__(self, root, request_hash, draft_hash, item_ids):
        self.path = Path(root) / 'reader-feedback.jsonl'
        self.request_hash = request_hash
        self.draft_hash = draft_hash
        self.item_ids = set(item_ids)
        self.lock = threading.RLock()

    def empty(self):
        return {'schema': 'daily-writer-draft-feedback-v2', 'request_hash': self.request_hash,
                'draft_hash': self.draft_hash, 'scope': 'local_draft_only', 'approval': False,
                'rating_dimension': 'reading_value', 'scale': {'min': 1, 'max': 5}, 'items': []}

    def validate(self, value):
        expected = self.empty()
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError('feedback shape mismatch')
        if any(value[k] != v for k, v in expected.items() if k != 'items'):
            raise ValueError('feedback binding mismatch')
        if value['approval'] is not False or not isinstance(value['items'], list):
            raise ValueError('feedback is not approval')
        seen = set()
        for row in value['items']:
            if not isinstance(row, dict) or set(row) != {
                    'item_id', 'rating', 'read', 'reason', 'updated_at'}:
                raise ValueError('invalid feedback row')
            if row['item_id'] not in self.item_ids or row['item_id'] in seen:
                raise ValueError('unknown or duplicate item')
            seen.add(row['item_id'])
            score = row['rating']
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError('rating must be null or integer 1 to 5')
            if type(row['read']) is not bool or row['reason'] not in {
                    None, 'unclear', 'repeated', 'not_relevant', 'unreliable', 'insightful'}:
                raise ValueError('invalid feedback dimension')
            try:
                date = datetime.fromisoformat(row['updated_at'])
                if date.tzinfo is None:
                    raise ValueError('timezone required')
            except (TypeError, ValueError):
                raise ValueError('invalid feedback time') from None

    def load(self):
        with self.lock:
            latest = self.empty()
            if self.path.exists():
                for line in self.path.read_text().splitlines():
                    record = json.loads(line)
                    self.validate(record['feedback'])
                    latest = record['feedback']
            return latest

    def save(self, value):
        self.validate(value)
        with self.lock:
            if self.load() == value:
                return value
            record = {'received_at': datetime.now(UTC).isoformat(), 'feedback': value}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            return value


def review_context(roots):
    from .pipeline import digest
    result = []
    for directory in roots:
        root = Path(directory)
        request = json.loads((root / 'request.json').read_text())
        raw = json.loads((root / 'model-response.json').read_text())
        response = json.loads(raw['content'])
        if (root / 'reader-response.json').exists():
            response = json.loads((root / 'reader-response.json').read_text())
        if response['request_hash'] != request['request_hash']:
            raise ValueError('feedback source request mismatch')
        items = {item['id']: item for item in response['items']}
        spec_ids = {item['id'] for item in request['packet']['items']}
        if set(items) != spec_ids:
            raise ValueError('feedback item coverage mismatch')
        store = FeedbackStore(root, request['request_hash'], digest(response), items)
        for row in store.load()['items']:
            result.append({**row, 'title': items[row['item_id']]['title'],
                           'request_hash': request['request_hash'], 'draft_hash': digest(response),
                           'reference_utc': request['packet']['reference_utc'],
                           'evidence': str(store.path.resolve())})
    return {'schema': 'draft-reader-context-v1', 'items': result,
            'policy': 'Untrusted reader feedback for editorial context only. No automatic scores. '
                      'Null is unrated. Read does not imply positive value. Low ratings do not '
                      'prove topic dislike; consider stated reason. Preserve important news coverage.'}
