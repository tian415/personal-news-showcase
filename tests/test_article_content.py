from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime

import httpx
import pytest

from news_assistant.article_content import (
    ContentFetchError,
    PageCapture,
    extract_snapshot_content,
    fetch_public_html,
)
from news_assistant.editorial_model import EditorialModelClient
from news_assistant.models import Preferences
from news_assistant.reviewed_selection import prepare_review, select_reviewed

NOW = datetime(2026, 9, 29, 8, tzinfo=UTC)
LIMITATION = "这项研究仅在小鼠中开展，尚未进行人体试验。"
BODY = "研究团队公布了一项视网膜修复研究。" + LIMITATION + "仍需要进一步研究验证其有效性。"


def snapshot(count=1):
    return {
        "schema_version": "source-audit-v1", "candidate_snapshot_available": True,
        "captured_at": NOW.isoformat(), "max_age_hours": 36,
        "sources": [{
            "id": "science", "name": "科学来源", "publisher_group": "science",
            "production_enabled": True, "status": "ok", "parsed_count": count,
            "fresh_count": count, "stale_count": 0, "future_count": 0,
            "evidence_role": "INDEPENDENT_REPORTING", "coverage_domains": ["science_health"],
            "candidates": [{"id": f"a{i}", "title": f"研究进展 {i}",
                            "url": f"https://news.example/article/{i}", "source": "科学来源",
                            "language": "zh", "published_at": NOW.isoformat(),
                            "topic": "science", "summary": "RSS 仅有一条短摘录。"}
                           for i in range(count)],
        }],
    }


def page(url):
    return PageCapture(url, ("<html><body><article><h1>研究进展</h1><p>" + BODY * 4
                            + "</p><p>" + BODY * 3 + "</p></article></body></html>").encode())


def test_real_extractor_preserves_rss_and_explicit_pending_review():
    audit = snapshot()
    original = copy.deepcopy(audit)
    enriched = extract_snapshot_content(audit, loader=page, now=NOW)
    assert audit == original and enriched["sources"] == audit["sources"]
    item = enriched["article_content"]["items"]["science:a0"]
    assert item["status"] == "extracted" and LIMITATION in item["text"]
    assert item["text_sha256"] == hashlib.sha256(item["text"].encode()).hexdigest()
    assert item["extractor"] == "trafilatura" and item["extractor_version"]
    review = prepare_review(enriched)
    assert review["items"][0]["status"] == "pending"
    assert review["items"][0]["quality_score"] is None
    assert review["items"][0]["article_sha256"] != prepare_review(audit)["items"][0]["article_sha256"]
    review["items"][0]["evidence_excerpt"] = LIMITATION
    assert select_reviewed(enriched, review, Preferences(), now=NOW).excluded == {
        "science:a0": "pending_review"}


def test_body_changes_invalidate_both_snapshot_and_article_review_bindings():
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    old_review = prepare_review(enriched)
    changed = copy.deepcopy(enriched)
    content = changed["article_content"]["items"]["science:a0"]
    content["text"] += "新增重要限定"
    content["text_sha256"] = hashlib.sha256(content["text"].encode()).hexdigest()
    with pytest.raises(ValueError, match="different candidate snapshot"):
        select_reviewed(changed, old_review, Preferences(), now=NOW)
    rebound = copy.deepcopy(old_review)
    rebound["snapshot_sha256"] = prepare_review(changed)["snapshot_sha256"]
    with pytest.raises(ValueError, match="article fingerprint mismatch"):
        select_reviewed(changed, rebound, Preferences(), now=NOW)


def test_failures_and_request_budget_are_visible_and_do_not_drop_candidates():
    calls = []

    def failing(url):
        calls.append(url)
        raise ContentFetchError("http_error", status_code=403)

    enriched = extract_snapshot_content(snapshot(3), loader=failing, now=NOW, max_articles=2)
    items = enriched["article_content"]["items"]
    assert len(calls) == 2 and len(items) == 3
    assert items["science:a0"]["status"] == "rss_fallback"
    assert items["science:a0"]["http_status"] == 403
    assert items["science:a2"]["reason"] == "budget_limit"
    assert all(not i["text"] for i in items.values())
    assert len(prepare_review(enriched)["items"]) == 3


def test_truncation_is_recorded_and_not_silently_used_as_complete_body():
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW,
                                        extractor=lambda _: BODY * 1000)
    item = enriched["article_content"]["items"]["science:a0"]
    assert len(item["text"]) == 20_000 and item["truncated"] is True


@pytest.mark.parametrize("mutation", ["hash", "url", "missing", "unknown"])
def test_rejects_invalid_content_provenance(mutation):
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    items = enriched["article_content"]["items"]
    if mutation == "hash":
        items["science:a0"]["text"] += "changed"
    elif mutation == "url":
        items["science:a0"]["requested_url"] = "https://other.example/"
    elif mutation == "missing":
        items.clear()
    else:
        items["unknown"] = copy.deepcopy(items["science:a0"])
    with pytest.raises(ValueError):
        prepare_review(enriched)


def test_preflight_includes_body_status_and_limits_without_network():
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    client = EditorialModelClient(api_key="", transport=httpx.MockTransport(
        lambda r: pytest.fail("preflight must not request a provider")))
    prepared = client._prepare_request(enriched, now=NOW, max_candidates=100)
    payload = json.loads(prepared.request_body["messages"][1]["content"])
    evidence = payload["candidates"][0]["article_content"]
    assert LIMITATION in evidence["text"]
    assert evidence["status"] == "extracted" and evidence["truncated"] is False


def test_network_boundary_blocks_private_redirect_and_credentials(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ContentFetchError, match="blocked_url"):
            fetch_public_html("https://public.example/article", client=client)
        assert len(calls) == 1
        with pytest.raises(ContentFetchError, match="blocked_url"):
            fetch_public_html("https://user:password@public.example/", client=client)
        assert len(calls) == 1


@pytest.mark.parametrize("response,reason", [
    (httpx.Response(403), "http_error"),
    (httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"PDF"), "non_html"),
    (httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 101),
     "response_too_large"),
])
def test_http_limits_before_extraction(monkeypatch, response, reason):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    with (httpx.Client(transport=httpx.MockTransport(lambda r: response)) as client,
          pytest.raises(ContentFetchError, match=reason)):
        fetch_public_html("https://public.example/article", client=client, max_bytes=100)


def test_cli_validates_private_destination_before_fetch(monkeypatch, tmp_path):
    import news_assistant.article_content as module
    from news_assistant import cli
    source = tmp_path / "snapshot.json"
    source.write_text(json.dumps(snapshot()))
    public = tmp_path / "public"
    monkeypatch.setattr(module, "fetch_public_html", lambda *a: pytest.fail("network attempted"))
    for dest in (public / "content.json", source):
        monkeypatch.setattr(sys, "argv", ["news-assistant", "extract-content", "--snapshot",
                                         str(source), "--output", str(dest), "--public-dir",
                                         str(public)])
        with pytest.raises(SystemExit):
            cli.main()


def test_local_reader_shows_bound_qualification_but_never_full_body(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview

    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    review = prepare_review(enriched)
    review.update(reviewed_by="test-reviewer", reviewed_at=NOW.isoformat())
    review["items"][0].update(status="accepted", quality_score=0.8, importance_score=0.7,
                              fact_status="reported", evidence_excerpt=LIMITATION,
                              rationale="测试引用展示，非质量验收")
    reader = tmp_path / "reader"
    build_reviewed_preview(enriched, review, Preferences(), reader, tmp_path / "state",
                           now=NOW, run_id="content-quote")
    output = (reader / "2026-09-29.html").read_text()
    assert LIMITATION in output
    assert "审阅所据原文" in output
    assert enriched["article_content"]["items"]["science:a0"]["text"] not in output
    assert "html_sha256" not in output and "trafilatura" not in output


def test_compressed_response_rejected_before_decoding():
    class UnreadStream(httpx.SyncByteStream):
        def __iter__(self):
            pytest.fail("compressed stream must never be read or decoded")
            yield b""

    def respond(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"content-type": "text/html",
                                            "content-encoding": "gzip"}, stream=UnreadStream())

    with (httpx.Client(transport=httpx.MockTransport(respond)) as client,
          pytest.raises(ContentFetchError, match="compressed_response")):
        fetch_public_html("https://93.184.216.34/article", client=client)


def test_cli_does_not_overwrite_file_created_during_fetch(monkeypatch, tmp_path):
    import news_assistant.article_content as module
    from news_assistant import cli

    source, dest = tmp_path / "rss.json", tmp_path / "content.json"
    source.write_text(json.dumps(snapshot()))

    def loader(url):
        dest.write_text("other process material")
        return page(url)

    monkeypatch.setattr(module, "fetch_public_html", loader)
    monkeypatch.setattr(module, "extract_snapshot_content",
                        lambda audit, **kwargs: extract_snapshot_content(audit, now=NOW, **kwargs))
    monkeypatch.setattr(sys, "argv", ["news-assistant", "extract-content", "--snapshot",
                                     str(source), "--output", str(dest)])
    with pytest.raises(SystemExit):
        cli.main()
    assert dest.read_text() == "other process material"


def test_explicit_public_dns_pins_ip_preserving_tls_and_revalidates_redirect():
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.host == "cloudflare-dns.com":
            assert request.url.params["name"] == "news.example"
            return httpx.Response(200, json={"Status": 0, "TC": False,
                "Answer": [{"type": 1, "data": "93.184.216.34"}]})
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "news.example"
        assert request.extensions["sni_hostname"] == "news.example"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    with (httpx.Client(transport=httpx.MockTransport(respond)) as client,
          pytest.raises(ContentFetchError, match="blocked_url")):
        fetch_public_html("https://news.example/article", client=client, dns_mode="cloudflare")
    assert len(calls) == 2


@pytest.mark.parametrize("answer", [
    {"Status": 0, "Answer": [{"type": 1, "data": "127.0.0.1"}]},
    {"Status": 0, "Answer": [{"type": 1, "data": "198.18.0.1"}]},
    {"Status": 0, "Answer": [{"type": 1, "data": "not-an-ip"}]},
    {"Status": 2}, {"Status": 0, "TC": True}, [],
])
def test_public_dns_rejects_invalid_or_nonpublic_answers(answer):
    with (httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=answer))) as client,
          pytest.raises(ContentFetchError)):
        fetch_public_html("https://news.example/article", client=client, dns_mode="cloudflare")


def test_reader_uses_attributed_chinese_copy_and_escapes_it(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    review = prepare_review(enriched)
    review.update(reviewed_by="editor", reviewed_at=NOW.isoformat())
    review["items"][0].update(status="accepted", quality_score=0.8, importance_score=0.7,
        fact_status="reported", evidence_excerpt=LIMITATION, rationale="保留实验阶段",
        topic="business", coverage_domain="business_macro",
        chinese_title="研究仍处于动物实验阶段", chinese_summary="<script>不执行</script>尚无人体验证。")
    build_reviewed_preview(enriched, review, Preferences(), tmp_path / "reader",
                           tmp_path / "state", now=NOW, run_id="chinese-copy")
    output = (tmp_path / "reader/2026-09-29.html").read_text()
    assert "研究仍处于动物实验阶段" in output
    assert "&lt;script&gt;不执行&lt;/script&gt;尚无人体验证。" in output
    assert "<script>不执行</script>" not in output
    assert "审阅整理" in output
    assert "自动降级内容" not in output


def test_review_copy_requires_both_title_and_summary():
    from news_assistant.reviewed_selection import EditorialReview
    review = prepare_review(snapshot())
    review["items"][0]["chinese_title"] = "仅有标题"
    with pytest.raises(ValueError, match="title and summary"):
        EditorialReview.model_validate(review)


def test_body_model_response_requires_and_preserves_chinese_copy():
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    client = EditorialModelClient('')
    prepared = client._prepare_request(enriched, now=NOW, max_candidates=100)
    item = {key: value for key, value in prepared.template['items'][0].items()
            if key not in {'article_sha256', 'quotation_basis'}}
    item.update(status='accepted', quality_score=.8, importance_score=.7,
                fact_status='reported', evidence_excerpt=LIMITATION, rationale='保留限制')
    with pytest.raises(ValueError, match='Chinese copy'):
        client._validate_response(enriched, prepared, json.dumps({'items': [item]}),
                                  'stop', now=NOW)
    item.update(chinese_title='动物实验结果', chinese_summary='仅在小鼠中观察到，尚未开展人体试验。')
    review = client._validate_response(enriched, prepared, json.dumps({'items': [item]}),
                                       'stop', now=NOW)
    assert review['items'][0]['chinese_summary'] == item['chinese_summary']


def test_reviewed_email_matches_reader_and_remains_private(tmp_path):
    from news_assistant.local_preview import build_reviewed_preview
    enriched = extract_snapshot_content(snapshot(), loader=page, now=NOW)
    review = prepare_review(enriched)
    review.update(reviewed_by='editor', reviewed_at=NOW.isoformat())
    review['items'][0].update(status='accepted', quality_score=.8, importance_score=.7,
        fact_status='opinion', rationale='限制明确', evidence_excerpt=LIMITATION,
        chinese_title='同一份中文标题', chinese_summary='同一份中文导语，尚未验证人体效果。')
    reader, state = tmp_path / 'reader', tmp_path / 'state'
    build_reviewed_preview(enriched, review, Preferences(), reader, state,
        now=NOW, run_id='email-test', email_base_url='http://localhost:8771')
    email = next(state.rglob('email-preview.html')).read_text()
    assert '同一份中文标题' in email and '同一份中文导语' in email
    assert '本地审阅版' in email and '观点分析' in email
    assert 'http://localhost:8771/2026-09-29.html' in email
    assert 'article-science%3Aa0' in email
    assert not list(reader.rglob('email-preview.html'))
