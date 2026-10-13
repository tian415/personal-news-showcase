import json
import sys
from datetime import datetime

import pytest
from test_review_batches import NOW, snapshot

from news_assistant import cli


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def call(monkeypatch, *args):
    monkeypatch.setattr(cli, 'datetime', Clock)
    monkeypatch.setattr(sys, 'argv', ['news-assistant', *map(str, args)])
    return cli.main()


def test_batch_cli_roundtrip_and_protects_existing_files(monkeypatch, tmp_path):
    original = tmp_path / 'source.json'
    original.write_text(json.dumps(snapshot()))
    folder = tmp_path / 'batches'
    assert call(monkeypatch, 'prepare-review-batches', '--snapshot', original,
                '--output-dir', folder, '--max-chars', 8000) == 0
    manifest = json.loads((folder / 'manifest.json').read_text())
    for index in range(1, manifest['batch_count'] + 1):
        pending = folder / f'pending-{index:03d}.json'
        review = json.loads(pending.read_text())
        review.update(reviewed_by='test editor', reviewed_at=NOW.isoformat())
        (folder / f'review-{index:03d}.json').write_text(json.dumps(review))
    dest = tmp_path / 'merged.json'
    assert call(monkeypatch, 'merge-review-batches', '--snapshot', original,
                '--batch-dir', folder, '--output', dest) == 0
    merged = json.loads(dest.read_text())
    assert len(merged['items']) == 4
    assert all(i['status'] == 'pending' and i['quality_score'] is None for i in merged['items'])
    assert dest.stat().st_mode & 0o777 == 0o600
    before = dest.read_bytes()
    with pytest.raises(SystemExit):
        call(monkeypatch, 'merge-review-batches', '--snapshot', original,
             '--batch-dir', folder, '--output', dest)
    assert dest.read_bytes() == before
    with pytest.raises(SystemExit):
        call(monkeypatch, 'prepare-review-batches', '--snapshot', original,
             '--output-dir', folder)


def test_batch_cli_rejects_public_directory_before_writing(monkeypatch, tmp_path):
    original = tmp_path / 'source.json'
    original.write_text(json.dumps(snapshot()))
    public = tmp_path / 'public'
    with pytest.raises(SystemExit):
        call(monkeypatch, 'prepare-review-batches', '--snapshot', original,
             '--output-dir', public / 'batches', '--public-dir', public)
    assert not public.exists()
