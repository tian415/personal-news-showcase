from copy import deepcopy
from datetime import timedelta
from hashlib import sha256

import pytest
from test_news_value import value_inputs
from test_reviewed_selection import NOW

from news_assistant.local_preview import build_reviewed_preview
from news_assistant.models import Preferences


def reading_manifest():
    material = 'Residents raised concerns about water use. The company described its response.'
    return {
        'schema_version': 'background-readings-v1',
        'reviewed_by': 'test editor', 'reviewed_at': NOW.isoformat(),
        'items': [{
            'title': '资源分配背后的不同处境', 'url': 'https://example.net/background',
            'source': 'Example', 'published_on': '2026-09-01', 'kind': 'analysis',
            'why_read': '从具体采访理解资源分配的冲突。',
            'limitation': '这是单个地方的报道，不能外推所有项目。', 'language': 'en',
            'material': material, 'material_sha256': sha256(material.encode()).hexdigest(),
            'evidence_excerpt': 'Residents raised concerns about water use.',
            'review_scope': 'excerpt',
        }],
    }


def build(tmp_path, manifest, run='reading', now=NOW):
    audit, review, value = value_inputs(1)
    return build_reviewed_preview(
        audit, review, Preferences(), tmp_path/run, tmp_path/'state',
        now=now, run_id=run, value_review=value, background_readings=manifest,
        email_base_url='https://example.org/news',
    )


def test_reading_is_separate_from_news_feedback_and_email(tmp_path):
    result = build(tmp_path, reading_manifest())
    html = (tmp_path/'reading/index.html').read_text()
    assert len(result['selected_ids']) == 1
    assert result['background_reading_count'] == 1
    assert html.count('data-article=') == 1
    assert '资源分配背后的不同处境' in html
    assert '2026-09-01' in html
    assert '公开片段' in html
    assert 'Residents raised concerns' not in html
    assert 'test editor' not in html
    email = next((tmp_path/'state').rglob('email-preview.html')).read_text()
    assert '资源分配背后的不同处境' not in email
    assert result['delivery_status'] == 'not_confirmed'


@pytest.mark.parametrize('mutation', ['hash', 'quote', 'future', 'naive', 'missing_author',
                                     'future_article', 'duplicate', 'over_limit', 'unsafe_url'])
def test_invalid_readings_are_rejected_before_writes(tmp_path, mutation):
    manifest = reading_manifest()
    item = manifest['items'][0]
    if mutation == 'hash':
        item['material'] += ' changed'
    elif mutation == 'quote':
        item['evidence_excerpt'] = 'Invented evidence'
    elif mutation == 'future':
        manifest['reviewed_at'] = (NOW + timedelta(hours=1)).isoformat()
    elif mutation == 'naive':
        manifest['reviewed_at'] = NOW.replace(tzinfo=None).isoformat()
    elif mutation == 'missing_author':
        manifest['reviewed_by'] = ' '
    elif mutation == 'future_article':
        item['published_on'] = '2027-01-01'
    elif mutation == 'duplicate':
        manifest['items'].append(deepcopy(item))
    elif mutation == 'over_limit':
        manifest['items'] *= 3
    else:
        item['url'] = 'javascript:alert(1)'
    with pytest.raises(ValueError):
        build(tmp_path, manifest)
    assert not (tmp_path/'state').exists()
    assert not (tmp_path/'reading').exists()


def test_same_url_as_news_is_omitted(tmp_path):
    manifest = reading_manifest()
    manifest['items'][0]['url'] = 'https://example.org/0?utm_source=test'
    result = build(tmp_path, manifest)
    assert result['background_reading_count'] == 0


def test_only_confirmed_local_editions_suppress_repeat_readings(tmp_path):
    from news_assistant.delivery_recovery import confirm_local_reading
    from news_assistant.state import StateStore

    first = build(tmp_path, reading_manifest(), run='first')
    assert first['background_reading_count'] == 1
    assert build(tmp_path, reading_manifest(), run='unconfirmed')['background_reading_count'] == 1
    confirm_local_reading(StateStore(tmp_path/'state'), 'first',
                          tmp_path/'first/2026-09-28.html', now=NOW)
    assert build(tmp_path, reading_manifest(), run='after')['background_reading_count'] == 0


def test_reading_text_is_escaped(tmp_path):
    manifest = reading_manifest()
    manifest['items'][0]['title'] = '<img src=x onerror=alert(1)>'
    build(tmp_path, manifest)
    html = (tmp_path/'reading/index.html').read_text()
    assert '<img src=x onerror=alert(1)>' not in html
    assert '&lt;img' in html


def test_cli_accepts_private_readings_and_reports_separate_count(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from datetime import datetime

    from news_assistant import cli

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    audit, review, value = value_inputs(1)
    inputs = {'snapshot': audit, 'review': review, 'value-review': value,
              'background-readings': reading_manifest()}
    args = ['news-assistant', 'preview-reviewed']
    for name, data in inputs.items():
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(data))
        args.extend([f'--{name}', str(path)])
    args.extend(['--output', str(tmp_path/'public'), '--state-dir', str(tmp_path/'state'),
                 '--run-id', 'cli-reading'])
    monkeypatch.setattr(cli, 'datetime', Clock)
    monkeypatch.setattr(sys, 'argv', args)
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['selected_count'] == 1
    assert result['background_reading_count'] == 1
    assert result['delivery_status'] == 'not_confirmed'


def test_cli_rejects_reading_manifest_inside_public_output_before_reading(monkeypatch, tmp_path):
    import sys

    from news_assistant import cli

    public = tmp_path / 'public'
    monkeypatch.setattr(sys, 'argv', [
        'news-assistant', 'preview-reviewed', '--snapshot', str(tmp_path/'snapshot.json'),
        '--review', str(tmp_path/'review.json'), '--background-readings', str(public/'private.json'),
        '--output', str(public), '--state-dir', str(tmp_path/'state'), '--run-id', 'unsafe',
    ])
    with pytest.raises(SystemExit):
        cli.main()
    assert not public.exists()
    assert not (tmp_path/'state').exists()
