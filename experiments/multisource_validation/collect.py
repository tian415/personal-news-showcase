"""Explicit, bounded public collection for a private frozen comparison corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import httpx
from lxml import html

from experiments.multisource_validation.native_extract import ExtractionFailure, extract_body
from news_assistant.article_content import ContentFetchError, _public_address
from news_assistant.editorial_sources import (
    jiemian_article_urls,
    parse_jiemian_article,
    parse_sixthtone_listing,
)
from news_assistant.feeds import FeedDiagnostics, FeedSource, canonicalize_url, parse_feed

MAX_BYTES = 2 * 1024 * 1024
LOCK = threading.Lock()


def iso_now():
    return datetime.now(UTC).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def merge_window(rows, reference):
    merged, exclusions, conflicts = {}, [], []
    for row in rows:
        published = datetime.fromisoformat(row['published_at'])
        reason = ('future_publication' if published > reference else
                  'outside_window' if published < reference - timedelta(hours=36) else None)
        if reason:
            exclusions.append({**row, 'reason': reason})
            continue
        key = (canonicalize_url(row['url']), row['language'])
        if key not in merged:
            for other_key, other in merged.items():
                if other_key[0] == key[0] and other_key[1] != key[1]:
                    conflicts.append({'url': key[0], 'field': 'language_identity_collision',
                                      'values': [other_key[1], key[1]],
                                      'source_ids': [other['source_id'], row['source_id']]})
            merged[key] = {**row, 'appearances': [row['source_id']]}
        else:
            old = merged[key]
            for field in ('published_at', 'title', 'id'):
                if field in row and old.get(field) != row[field]:
                    conflicts.append({'url': key[0], 'field': field, 'values': [old.get(field), row[field]],
                                      'source_ids': [old['source_id'], row['source_id']]})
            if row['source_id'] not in old['appearances']:
                old['appearances'].append(row['source_id'])
    return list(merged.values()), exclusions, conflicts


def body_result(raw, text):
    sample = raw.decode('utf-8', errors='replace').lower()
    if any(x in sample for x in ('cf-chl-', 'just a moment...', 'verify you are human')):
        return 'restricted', 'challenge', ''
    if any(x in sample for x in ('subscribe to continue reading', 'sign in to continue reading',
                                  'subscription required', '订阅后阅读全文', '付费阅读')):
        return 'restricted', 'paywall', ''
    if not text or len(text.strip()) < 250:
        return 'failed', 'nonbody', ''
    return 'ok', None, text


def parse_generic_article(raw, url, source):
    doc = html.fromstring(raw.decode('utf-8', errors='replace'))
    headings = doc.xpath('//h1')
    title = ' '.join(''.join(headings[0].itertext()).split()) if headings else ''
    dates = doc.xpath('//meta[@property="article:published_time"]/@content | '
                      '//meta[@name="pubdate" or @name="publishdate" or @name="publishDate"]/@content | '
                      '//time[@datetime]/@datetime')
    # JSON-LD datePublished is an explicitly attributed publication field.
    dates += re.findall(r'"datePublished"\s*:\s*"([^\"]+)"', raw.decode('utf-8', errors='replace'))
    # Publisher-specific article timestamp containers only; never scan arbitrary text dates.
    selectors = {'yicai': '//div[contains(concat(" ",normalize-space(@class)," ")," title ")]//em',
                 'thepaper': '//*[contains(@class,"index_time") or @class="news_about"]',
                 'chuapp': '//*[contains(concat(" ",normalize-space(@class)," ")," friendly_time ")]'}
    for node in doc.xpath(selectors.get(source['id'], '//time')):
        node_text = ''.join(node.itertext())
        node_text = re.sub(r'(\d{4})年(\d{2})月(\d{2})日\s+(\d{2})时(\d{2})分', r'\1-\2-\3 \4:\5', node_text)
        dates += re.findall(r'\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(?::\d{2})?', node_text)
    if not dates:
        raise ValueError('missing explicit publication timestamp')
    times = []
    for value in dates:
        try:
            stamp = datetime.fromisoformat(value.strip().replace('/', '-'))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=ZoneInfo('Asia/Shanghai'))
            times.append(stamp.astimezone(UTC).isoformat())
        except ValueError:
            continue
    if not times or not title:
        raise ValueError('missing title or valid publication timestamp')
    if len(set(times)) != 1:
        raise ValueError('conflicting publication timestamps')
    descriptions = doc.xpath('//meta[@name="description"]/@content')
    return {'title': title, 'url': url, 'published_at': times[0], 'language': source['language'],
                'summary': descriptions[0] if descriptions else ''}


def generic_links(raw, source):
    patterns = {'yicai': r'^/news/\d+\.html$', 'thepaper': r'^/newsDetail_forward_\d+$',
                'chuapp': r'^/article/\d+\.html$'}
    result = []
    for anchor in html.fromstring(raw).xpath('//a[@href]'):
        url = urljoin(source['url'], anchor.get('href'))
        parts = urlsplit(url)
        if (parts.hostname == urlsplit(source['url']).hostname
                and re.fullmatch(patterns[source['id']], parts.path) and url not in result):
            result.append(url)
        if len(result) == 12:
            break
    return result


class Acquisition:
    def __init__(self, root, *, replay=False):
        self.root = root
        self.requests = []
        self.cache = {}
        self.replay = replay
        self.failures = {}
        if (root / 'requests.json').exists():
            self.requests = json.loads((root / 'requests.json').read_text())
            for row in self.requests:
                hops = row.get('hops', [])
                if not row.get('error') and hops and hops[-1]['status'] == 200:
                    self.cache[row['url']] = (Path(hops[-1]['raw_path']).read_bytes(), hops[-1]['url'], row['retrieved_at'])
                else:
                    self.failures[row['url']] = row

    def fetch(self, url):
        if url in self.cache:
            return self.cache[url]
        if self.replay:
            prior = self.failures.get(url, {})
            raise ContentFetchError(prior.get('error', 'replay_missing_response'), status_code=prior.get('http_status'))
        row = {'url': url, 'retrieved_at': iso_now(), 'hops': []}
        try:
            with httpx.Client(timeout=15, follow_redirects=False, trust_env=False) as client:
                current = httpx.URL(url)
                for hop in range(4):
                    address = _public_address(current, client=client, dns_mode='cloudflare')
                    with client.stream('GET', current.copy_with(host=address), follow_redirects=False,
                                       headers={'Host': current.netloc.decode(), 'User-Agent': 'PersonalNewsAssistant/0.1',
                                                'Accept': '*/*', 'Accept-Encoding': 'identity'},
                                       extensions={'sni_hostname': current.host}) as response:
                        evidence = {'url': str(current), 'status': response.status_code,
                                    'content_type': response.headers.get('content-type', '')}
                        raw = bytearray()
                        for chunk in response.iter_bytes(65536):
                            if len(raw) + len(chunk) > MAX_BYTES:
                                raise ContentFetchError('response_too_large')
                            raw.extend(chunk)
                        raw = bytes(raw)
                        digest = hashlib.sha256(raw).hexdigest()
                        path = self.root / 'raw' / (digest + '.response')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(raw)
                        evidence.update(raw_path=str(path), raw_sha256=digest, size_bytes=len(raw))
                        row['hops'].append(evidence)
                        if response.status_code in {301, 302, 303, 307, 308}:
                            if hop == 3 or 'location' not in response.headers:
                                raise ContentFetchError('redirect_limit')
                            current = current.join(response.headers['location'])
                            continue
                        if response.status_code != 200:
                            raise ContentFetchError('http_error', status_code=response.status_code)
                        if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                            raise ContentFetchError('compressed_response')
                        self.cache[url] = (raw, str(current), row['retrieved_at'])
                        return self.cache[url]
        except (ValueError, httpx.HTTPError, TypeError, OverflowError) as error:
            row['error'] = f'{type(error).__name__}: {error}'
            if getattr(error, 'status_code', None):
                row['http_status'] = error.status_code
            raise
        finally:
            with LOCK:
                self.requests.append(row)
                dump(self.root / 'requests.json', self.requests)
        raise ContentFetchError('redirect_limit')


def source_rows(source, acquisition):
    report = {**source, 'status': 'failed', 'error': None, 'entry_count': 0,
              'in_window_count': 0, 'body_success_count': 0, 'rejections': []}
    rows = []
    if source['kind'] == 'restricted':
        report.update(status='restricted', error=source['restriction_evidence'])
        return report, rows
    try:
        raw, final, retrieved = acquisition.fetch(source['url'])
        feed_source = FeedSource(name=source['name'], url=source['url'], language=source['language'],
                                 topic='tech', naive_timezone=source.get('naive_timezone'))
        if source['kind'] == 'rss':
            diagnostics = FeedDiagnostics()
            entries = parse_feed(raw, feed_source, diagnostics=diagnostics)
            report.update(entry_count=diagnostics.entry_count, parser_rejections=diagnostics.rejection_reasons)
            entries = [x.model_dump(mode='json') for x in entries]
        elif source['id'] == 'sixthtone':
            entries = [x.model_dump(mode='json') for x in parse_sixthtone_listing(raw, feed_source)][:12]
            report['entry_count'] = len(entries)
        else:
            urls = (jiemian_article_urls(raw) if source['id'] == 'jiemian' else generic_links(raw, source))
            report['entry_count'] = len(urls)
            entries = []
            for url in urls:
                try:
                    article_raw, _, _ = acquisition.fetch(url)
                    entry = (parse_jiemian_article(article_raw, url, feed_source).model_dump(mode='json')
                             if source['id'] == 'jiemian' else parse_generic_article(article_raw, url, source))
                    entries.append(entry)
                except (ValueError, httpx.HTTPError, TypeError, OverflowError) as error:
                    report['rejections'].append({'url': url, 'reason': str(error)})
        for entry in entries:
            url = canonicalize_url(entry['url'])
            rows.append({k: entry[k] for k in ('title', 'published_at', 'summary', 'language')} | {
                'url': url, 'id': hashlib.sha256(url.encode()).hexdigest()[:16], 'source_id': source['id'],
                'source': source['name'], 'publisher_group': source['publisher_group'],
                'domain_hint': source['domain_hint'], 'body': '', 'body_sha256': None,
                'body_status': 'pending', 'body_reason': 'sampling_budget', 'retrieved_at': retrieved})
        report['status'] = 'ok' if entries else 'empty'
        report['final_url'] = final
    except (ValueError, httpx.HTTPError, TypeError, OverflowError) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    return report, rows


def run(root, reference, sources, *, replay=False):
    existing = root / 'corpus/corpus.json'
    if existing.exists() and json.loads(existing.read_text()).get('sampling', {}).get('phase') == 'frozen':
        raise ValueError('corpus is frozen; use a new output directory')
    acquisition = Acquisition(root / 'acquisition', replay=replay)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda source: source_rows(source, acquisition), sources))
    reports = [x[0] for x in results]
    articles, exclusions, conflicts = merge_window([row for _, rows in results for row in rows], reference)
    for report in reports:
        report['in_window_count'] = sum(report['id'] in a['appearances'] for a in articles)
    corpus = {'schema': 'multisource-corpus-v1', 'reference_utc': reference.isoformat(), 'window_hours': 36,
                  'sources': reports, 'articles': articles, 'exclusions': exclusions, 'conflicts': conflicts,
                  'sampling': {'per_source_body_limit': 4, 'per_group_body_limit': 8, 'total_body_limit': 75,
                            'html_listing_limit': 12, 'phase': 'metadata_complete'}}
    dump(root / 'corpus/corpus.json', corpus)
    print(json.dumps({'phase':'metadata_complete', 'articles':len(articles), 'sources':reports}, ensure_ascii=False), flush=True)
    selected, group_counts = [], {}
    for source in sources:
        candidates = sorted([a for a in articles if a['source_id'] == source['id']],
                            key=lambda a: a['published_at'], reverse=True)
        for article in candidates[:4]:
            group = article['publisher_group']
            if len(selected) >= 75 or group_counts.get(group, 0) >= 8:
                break
            selected.append(article)
            group_counts[group] = group_counts.get(group, 0) + 1

    def capture(article):
        try:
            raw, final, retrieved = acquisition.fetch(article['url'])
            text = extract_body(raw)
            status, reason, body = body_result(raw, text)
            article.update(body_status=status, body_reason=reason, body=body, retrieved_at=retrieved,
                           body_sha256=hashlib.sha256(body.encode()).hexdigest() if body else None,
                           body_final_url=final)
        except ExtractionFailure as error:
            dump(root / 'extraction-failures' / (article['id'] + '.json'),
                 {'article_id': article['id'], 'url': article['url'],
                  'raw_sha256': hashlib.sha256(raw).hexdigest(), 'error': str(error),
                  'policy': 'batch aborted; no retry or reader'})
            raise
        except (ValueError, httpx.HTTPError, TypeError, OverflowError) as error:
            article.update(body_status='restricted' if getattr(error, 'status_code', None) in {401,403,429} else 'failed',
                           body_reason=f'{type(error).__name__}: {error}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(capture, selected))
    for report in reports:
        report['body_success_count'] = sum(a['body_status'] == 'ok' and report['id'] in a['appearances'] for a in articles)
    corpus['sampling'].update(phase='body_complete', body_attempt_count=len(selected),
                              actual_group_attempts=group_counts)
    dump(root / 'corpus/corpus.json', corpus)
    print(json.dumps({'phase':'body_complete','articles':len(articles), 'body_attempts':len(selected),
                      'body_ok':sum(a['body_status']=='ok' for a in articles)}, ensure_ascii=False), flush=True)


def diagnostic_selection(corpus, requested):
    by_id = {a['id']: a for a in corpus['articles']}
    selected, decisions = [], []
    remaining = max(0, 75 - corpus['sampling']['body_attempt_count'])
    for identifier, reason in requested.items():
        article = by_id.get(identifier)
        status = ('skip_missing' if article is None else
                  'skip_existing_body' if article['body_status'] != 'pending' else
                  'skip_budget' if len(selected) >= min(23, remaining) else 'selected')
        decisions.append({'id': identifier, 'reason': reason, 'status': status})
        if status == 'selected':
            selected.append(article)
    return selected, decisions


def attach_raw_evidence(article, acquisition):
    rows = [r for r in acquisition.requests if r['url'] == article['url']
            and not r.get('error') and r.get('hops') and r['hops'][-1]['status'] == 200]
    if rows:
        evidence = rows[-1]['hops'][-1]
        raw = Path(evidence['raw_path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != evidence['raw_sha256']:
            raise ValueError('raw evidence hash mismatch')
        article.update(raw_path=evidence['raw_path'], html_sha256=digest,
                       body_final_url=evidence['url'], retrieved_at=rows[-1]['retrieved_at'],
                       http_status=200, extractor='trafilatura')


def supplement(root, requested, *, fetch=False):
    from importlib.metadata import version
    path = root / 'corpus/corpus.json'
    corpus = json.loads(path.read_text())
    if corpus['sampling'].get('phase') == 'frozen':
        raise ValueError('corpus is frozen')
    acquisition = Acquisition(root / 'acquisition', replay=not fetch)
    selected, decisions = diagnostic_selection(corpus, requested)
    for article in corpus['articles']:
        article['sample_stratum'] = 'balanced_latest' if article['body_status'] != 'pending' else 'metadata_only'
        if article['body_status'] == 'ok':
            attach_raw_evidence(article, acquisition)
            article['extractor_version'] = version('trafilatura')

    def capture(article):
        article['sample_stratum'] = 'targeted_diagnostic'
        article['sampling_reason'] = requested[article['id']]
        try:
            raw, final, retrieved = acquisition.fetch(article['url'])
            text = extract_body(raw)
            status, reason, body = body_result(raw, text)
            article.update(body_status=status, body_reason=reason, body=body, retrieved_at=retrieved,
                           body_sha256=hashlib.sha256(body.encode()).hexdigest() if body else None,
                           body_final_url=final, extractor_version=version('trafilatura'))
            attach_raw_evidence(article, acquisition)
        except ExtractionFailure as error:
            dump(root / 'extraction-failures' / (article['id'] + '.json'),
                 {'article_id': article['id'], 'url': article['url'],
                  'raw_sha256': hashlib.sha256(raw).hexdigest(), 'error': str(error),
                  'policy': 'batch aborted; no retry or reader'})
            raise
        except (ValueError, httpx.HTTPError, TypeError, OverflowError) as error:
            article.update(body_status='restricted' if getattr(error, 'status_code', None) in {401,403,429} else 'failed',
                           body_reason=f'{type(error).__name__}: {error}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(capture, selected))
    for report in corpus['sources']:
        report['body_success_count'] = sum(a['body_status'] == 'ok' and report['id'] in a['appearances'] for a in corpus['articles'])
    corpus['sampling'].update(base_body_attempt_count=corpus['sampling']['body_attempt_count'],
                              diagnostic_body_attempt_count=len(selected),
                              body_attempt_count=corpus['sampling']['body_attempt_count'] + len(selected),
                              diagnostic_decisions=decisions, phase='frozen', frozen_at=iso_now(),
                              distribution_warning='Targeted diagnostic additions are not natural traffic or balanced sampling.')
    dump(path, corpus)
    print(json.dumps({'phase':'frozen', 'targeted_attempts':len(selected), 'body_status_counts':
                     {status:sum(a['body_status']==status for a in corpus['articles']) for status in ('ok','pending','failed','restricted')}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='explicitly enable public network collection')
    parser.add_argument('--replay', action='store_true', help='offline replay of saved raw responses')
    parser.add_argument('--diagnostic-json', type=Path, help='ID-to-reason mapping; offline unless --fetch; freezes existing corpus')
    parser.add_argument('--reference-utc', default=None)
    parser.add_argument('--output', type=Path, default=Path('work/multisource-validation-2026-10-10'))
    args = parser.parse_args()
    if args.diagnostic_json:
        supplement(args.output, json.loads(args.diagnostic_json.read_text()), fetch=args.fetch)
        return
    if not args.fetch and not args.replay:
        print('Offline default: no requests. Pass --fetch to collect public material.')
        return
    if args.replay and not args.reference_utc:
        args.reference_utc = json.loads((args.output / 'corpus/corpus.json').read_text())['reference_utc']
    reference = datetime.fromisoformat(args.reference_utc) if args.reference_utc else datetime.now(UTC)
    if reference.tzinfo is None:
        parser.error('reference must include timezone')
    sources = json.loads(Path(__file__).with_name('sources.json').read_text())
    run(args.output, reference.astimezone(UTC), sources, replay=args.replay)


if __name__ == '__main__':
    main()
