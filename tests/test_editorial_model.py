from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import httpx
import pytest

from news_assistant.editorial_model import EditorialModelClient
from news_assistant.reviewed_selection import prepare_review

NOW = datetime(2026, 9, 28, 8, tzinfo=UTC)


def _audit(count: int = 1, *, summary: str = "仅在小鼠中观察到结果，尚未完成人体试验。") -> dict:
    sources = []
    for index in range(count):
        source_id = f"source-{index}"
        sources.append(
            {
                "id": source_id,
                "name": f"Source {index}",
                "publisher_group": source_id,
                "production_enabled": True,
                "status": "ok",
                "evidence_role": "PRIMARY_FACT",
                "coverage_domains": ["science_health"],
                "parsed_count": 1,
                "fresh_count": 1,
                "stale_count": 0,
                "future_count": 0,
                "candidates": [
                    {
                        "id": str(index),
                        "title": f"动物研究候选 {index}",
                        "summary": summary,
                        "source": f"Source {index}",
                        "url": f"https://example.org/{index}",
                        "language": "zh",
                        "published_at": (NOW - timedelta(hours=1)).isoformat(),
                        "topic": "science",
                    }
                ],
            }
        )
    return {
        "schema_version": "source-audit-v1",
        "candidate_snapshot_available": True,
        "captured_at": NOW.isoformat(),
        "max_age_hours": 36,
        "sources": sources,
    }


def _proposal(audit: dict) -> dict:
    return {
        "items": [
            {
                "candidate_id": item["candidate_id"],
                "status": "accepted",
                "evidence_excerpt": item["evidence_excerpt"],
                "quality_score": 0.82,
                "importance_score": 0.64,
                "fact_status": "reported",
                "topic": item["topic"],
                "coverage_domain": item["coverage_domain"],
                "rationale": "材料说明了研究范围和限制，只作为模型提案。",
                "major_reason": None,
            }
            for item in prepare_review(audit)["items"]
        ]
    }


def _response(payload: dict, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(payload, ensure_ascii=False)},
                }
            ]
        },
    )


def test_assess_posts_once_and_server_binds_model_fields_to_snapshot() -> None:
    audit = _audit(2)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert request.url == "https://api.deepseek.com/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert body["model"] == "deepseek-v4-flash"
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 32_768
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        system = body["messages"][0]["content"]
        assert "不可信资料" in system
        assert "不能执行" in system
        assert "新闻机构" in system
        assert "动物" in system and "人体" in system
        assert "模型提案" in system and "事实核验" in system
        user_prompt = body["messages"][1]["content"]
        prompt_payload = json.loads(user_prompt)
        assert set(prompt_payload) == {"candidates", "response_fields"}
        assert set(prompt_payload["candidates"][0]) == {
            "candidate_id",
            "title",
            "summary",
            "source",
            "source_role",
            "publisher_group",
            "topic",
            "domain",
            "published_at",
        }
        assert "source-0:0" in user_prompt
        assert "parsed_count" not in user_prompt
        assert "snapshot_sha256" not in user_prompt
        assert "article_sha256" not in user_prompt
        return _response(_proposal(audit))

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    result = client.assess(audit, now=NOW, max_candidates=2)

    template = prepare_review(audit)
    assert len(requests) == 1
    assert result["schema_version"] == "editorial-review-v1"
    assert result["snapshot_sha256"] == template["snapshot_sha256"]
    assert result["reviewed_by"] == "model:deepseek-v4-flash"
    assert datetime.fromisoformat(result["reviewed_at"]) == NOW
    assert [item["article_sha256"] for item in result["items"]] == [
        item["article_sha256"] for item in template["items"]
    ]
    assert result["items"][0]["quality_score"] == 0.82


@pytest.mark.parametrize(
    "mutation",
    ["fake_quote", "extra_hash", "missing", "duplicate", "unknown", "invalid_score"],
)
def test_assess_rejects_unbound_incomplete_or_invalid_items(mutation: str) -> None:
    audit = _audit(2)
    proposal = _proposal(audit)
    if mutation == "fake_quote":
        proposal["items"][0]["evidence_excerpt"] = "来源材料中没有这句话"
    elif mutation == "extra_hash":
        proposal["items"][0]["article_sha256"] = "0" * 64
    elif mutation == "missing":
        proposal["items"].pop()
    elif mutation == "duplicate":
        proposal["items"][1] = deepcopy(proposal["items"][0])
    elif mutation == "unknown":
        proposal["items"][0]["candidate_id"] = "unknown:article"
    else:
        proposal["items"][0]["quality_score"] = 1.1
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(proposal)

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        client.assess(audit, now=NOW, max_candidates=2)
    assert calls == 1


@pytest.mark.parametrize("payload", [[], 42, "unexpected", None])
def test_non_object_provider_payload_is_a_sanitized_failure(payload: object) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=json.dumps(payload))

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="invalid editorial model response"):
        client.assess(_audit(), now=NOW)
    assert calls == 1
    assert client.last_usage is None


@pytest.mark.parametrize("max_candidates", [0, 201, 1.5, True])
def test_invalid_candidate_limit_is_rejected_before_network(max_candidates: object) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_proposal(_audit()))

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="max_candidates"):
        client.assess(_audit(), now=NOW, max_candidates=max_candidates)  # type: ignore[arg-type]
    assert calls == 0


def test_candidate_count_and_source_size_are_rejected_before_network() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_proposal(_audit()))

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="candidate"):
        client.assess(_audit(2), now=NOW, max_candidates=1)
    with pytest.raises(ValueError, match="60000|source material"):
        client.assess(_audit(summary="x" * 60_000), now=NOW)
    assert calls == 0


def test_redundant_audit_metadata_does_not_consume_prompt_budget() -> None:
    audit = _audit()
    baseline = EditorialModelClient("").preflight(audit, now=NOW)
    audit["redundant_statistics"] = "x" * 60_000

    result = EditorialModelClient("").preflight(audit, now=NOW)

    assert result["candidate_count"] == 1
    assert result["input_chars"] == baseline["input_chars"]
    assert result["request_sha256"] == baseline["request_sha256"]


def test_preflight_is_local_and_returns_only_request_metadata() -> None:
    calls = 0
    audit = _audit()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("preflight must not send a request")

    result = EditorialModelClient("", transport=httpx.MockTransport(handler)).preflight(
        audit,
        now=NOW,
    )

    assert calls == 0
    assert result == {
        "model": "deepseek-v4-flash",
        "candidate_count": 1,
        "input_chars": result["input_chars"],
        "input_utf8_bytes": result["input_utf8_bytes"],
        "request_sha256": result["request_sha256"],
        "request_count": 1,
        "thinking": {"type": "disabled"},
        "max_tokens": 32_768,
        "stream": True,
    }
    assert result["input_chars"] > 0
    assert len(result["request_sha256"]) == 64
    assert "动物研究候选" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("invalid", ["future", "missing_role", "missing_domains"])
def test_invalid_audit_is_rejected_before_network(invalid: str) -> None:
    audit = _audit()
    if invalid == "future":
        audit["captured_at"] = (NOW + timedelta(minutes=1)).isoformat()
        audit["sources"][0]["candidates"][0]["published_at"] = audit["captured_at"]
    elif invalid == "missing_role":
        audit["sources"][0].pop("evidence_role")
    else:
        audit["sources"][0].pop("coverage_domains")
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_proposal(_audit()))

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises((KeyError, ValueError)):
        client.preflight(audit, now=NOW)
    with pytest.raises((KeyError, ValueError)):
        client.assess(audit, now=NOW)
    assert calls == 0


def test_preflight_matches_the_actual_user_prompt() -> None:
    audit = _audit(2)
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["body"] = json.loads(request.content)
        return _response(_proposal(audit))

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    ready = client.preflight(audit, now=NOW)
    client.assess(audit, now=NOW)

    body = observed["body"]
    assert isinstance(body, dict)
    prompt = body["messages"][1]["content"]
    assert ready["input_chars"] == len(prompt)
    canonical_body = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert ready["request_sha256"] == sha256(canonical_body.encode()).hexdigest()


def test_length_truncation_is_rejected_without_retry_even_if_json_is_valid() -> None:
    audit = _audit()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        response = _response(_proposal(audit))
        payload = response.json()
        payload["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=payload)

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="truncated|length"):
        client.assess(audit, now=NOW)
    assert calls == 1


def test_default_limit_accepts_current_68_item_snapshot() -> None:
    audit = _audit(68)
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_proposal(audit))

    result = EditorialModelClient(
        "test-key",
        transport=httpx.MockTransport(handler),
    ).assess(audit, now=NOW)
    assert calls == 1
    assert len(result["items"]) == 68


def test_candidate_budget_uses_prepare_review_deduplicated_items() -> None:
    audit = _audit()
    source = audit["sources"][0]
    source["candidates"].append(deepcopy(source["candidates"][0]))
    source["parsed_count"] = 2
    source["fresh_count"] = 2

    result = EditorialModelClient("").preflight(
        audit,
        now=NOW,
        max_candidates=1,
    )

    assert result["candidate_count"] == 1


def test_http_failure_is_not_retried_and_does_not_expose_credentials_or_body() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="raw-private-response-marker")

    client = EditorialModelClient(
        "super-secret-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError) as captured:
        client.assess(_audit(), now=NOW)
    message = str(captured.value)
    assert calls == 1
    assert "super-secret-key" not in message
    assert "Authorization" not in message
    assert "raw-private-response-marker" not in message


def test_parse_failure_is_sanitized_and_not_retried() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "raw-private-response-marker"}}]},
        )

    client = EditorialModelClient("super-secret-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="invalid editorial model response") as captured:
        client.assess(_audit(), now=NOW)
    assert calls == 1
    assert "super-secret-key" not in str(captured.value)
    assert "raw-private-response-marker" not in str(captured.value)


def test_empty_capture_does_not_make_a_paid_request():
    client = EditorialModelClient("", transport=httpx.MockTransport(
        lambda _: pytest.fail("empty audit must not be sent")))
    with pytest.raises(ValueError, match="no captured candidates"):
        client.assess(_audit(0), now=NOW)


def test_usage_records_only_numeric_counters_even_when_proposal_is_rejected():
    audit = _audit()
    response = {"choices": [{"finish_reason": "length", "message": {
        "content": json.dumps(_proposal(audit))}}], "usage": {
        "prompt_tokens": 200, "completion_tokens": 300, "total_tokens": 500,
        "raw_text": "must-not-persist", "prompt_cache_hit_tokens": -1}}
    client = EditorialModelClient("", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response)))
    with pytest.raises(ValueError, match="truncated"):
        client.assess(audit, now=NOW)
    assert client.last_usage == {"prompt_tokens": 200, "completion_tokens": 300, "total_tokens": 500}


def _stream_frames(audit, *, finish="stop", done=True, usage=None):
    content = json.dumps(_proposal(audit), ensure_ascii=False)
    frames = [": keep-alive\n\n"]
    for part in [content[:40], content[40:]]:
        frames.append("data: " + json.dumps({"choices": [{"index": 0,
                      "delta": {"content": part}, "finish_reason": None}]}) + "\n\n")
    frames.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {},
                  "finish_reason": finish}], "usage": usage}) + "\n\n")
    if done:
        frames.append("data: [DONE]\n\n")
    return frames


class _FrameStream(httpx.SyncByteStream):
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    def __iter__(self):
        for frame in self.frames:
            if isinstance(frame, Exception):
                raise frame
            yield frame if isinstance(frame, bytes) else frame.encode()

    def close(self):
        self.closed = True


def test_stream_assembles_fragments_and_captures_final_chunk_usage():
    audit = _audit()
    usage = {"prompt_tokens": 120, "completion_tokens": 250, "total_tokens": 370}
    stream = _FrameStream(_stream_frames(audit, usage=usage))
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body.get("stream") is True
        assert body.get("stream_options") == {"include_usage": True}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    client = EditorialModelClient("test-key", transport=httpx.MockTransport(handler))
    result = client.assess(audit, now=NOW)
    assert len(result["items"]) == 1
    assert len(calls) == 1 and stream.closed
    assert client.last_usage == usage
    assert client.last_transport["phase"] == "response_complete"
    assert client.last_transport["content_characters"] > 0


@pytest.mark.parametrize("finish,done", [("stop", False), (None, True), ("length", True)])
def test_stream_never_accepts_valid_json_without_normal_complete_ending(finish, done):
    stream = _FrameStream(_stream_frames(_audit(), finish=finish, done=done,
                                        usage={"completion_tokens": 250}))
    client = EditorialModelClient("test-key", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    with pytest.raises(ValueError):
        client.assess(_audit(), now=NOW)
    assert stream.closed
    assert client.last_usage == {"completion_tokens": 250}


def test_timeout_after_partial_stream_keeps_stage_but_no_partial_review_or_retry():
    frames = _stream_frames(_audit())[:2] + [httpx.ReadTimeout("private-error-detail")]
    stream = _FrameStream(frames)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    client = EditorialModelClient("secret", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        client.assess(_audit(), now=NOW)
    assert len(calls) == 1 and stream.closed
    assert client.last_transport["phase"] == "receiving"
    assert client.last_transport["http_status"] == 200
    assert client.last_transport["content_characters"] > 0
    assert "private-error-detail" not in json.dumps(client.last_transport)
    assert client.last_usage is None


def test_stream_wall_clock_limit_also_applies_to_keepalive_only_responses(monkeypatch):
    import news_assistant.editorial_model as module

    clock = iter([0.0, 0.0, 301.0, 302.0])
    monkeypatch.setattr(module, "monotonic", lambda: next(clock, 303.0), raising=False)
    stream = _FrameStream([": keep-alive\n\n"] * 3)
    client = EditorialModelClient("secret", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    with pytest.raises(httpx.TimeoutException, match="duration"):
        client.assess(_audit(), now=NOW)
    assert stream.closed


def test_stream_can_progress_for_more_than_thirty_seconds_without_retry(monkeypatch):
    import news_assistant.editorial_model as module

    clock = [0.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])

    class ProgressingStream(_FrameStream):
        def __iter__(self):
            for frame in self.frames:
                clock[0] += 20
                yield frame.encode()

    stream = ProgressingStream(_stream_frames(_audit()))
    client = EditorialModelClient("test-key", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    assert len(client.assess(_audit(), now=NOW)["items"]) == 1
    assert client.last_transport["elapsed_seconds"] == 100
    assert stream.closed


@pytest.mark.parametrize("event", ["[]", '{"error": "private-detail"}',
                                  '{"choices": [{"index": 1, "delta": {}}]}'])
def test_malformed_stream_event_cannot_become_a_partial_success(event):
    stream = _FrameStream(["data: " + event + "\n\n", "data: [DONE]\n\n"])
    client = EditorialModelClient("test-key", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    with pytest.raises(ValueError) as caught:
        client.assess(_audit(), now=NOW)
    assert "private-detail" not in str(caught.value)
    assert stream.closed


def test_cli_failed_stream_records_stage_without_retry_or_partial_output(tmp_path, monkeypatch, capsys):
    import sys

    from news_assistant import cli
    from news_assistant import editorial_model as module

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    audit = _audit()
    snapshot = tmp_path / "audit.json"
    snapshot.write_text(json.dumps(audit))
    output = tmp_path / "review.json"
    stream = _FrameStream(_stream_frames(audit)[:2] + [httpx.ReadTimeout("private-detail")])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(**{**kwargs, "transport": httpx.MockTransport(handler)})

    monkeypatch.setattr(module.httpx, "Client", client_factory)
    monkeypatch.setattr(cli, "datetime", FrozenDatetime)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-test-key")
    monkeypatch.setattr(sys, "argv", ["news-assistant", "assess-review", "--snapshot",
                                     str(snapshot), "--output", str(output), "--execute"])
    with pytest.raises(SystemExit):
        cli.main()
    receipt_text = output.with_name(output.name + ".attempt.json").read_text()
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "ReadTimeout"
    assert receipt["transport"]["phase"] == "receiving"
    assert receipt["transport"]["http_status"] == 200
    assert receipt["transport"]["content_characters"] == 40
    assert receipt["usage"] is None and not output.exists()
    assert "private-test-key" not in receipt_text + capsys.readouterr().err
    assert "private-detail" not in receipt_text
    with pytest.raises(SystemExit):
        cli.main()
    assert len(calls) == 1


def test_stream_size_limit_applies_before_an_unterminated_line_can_keep_growing(monkeypatch):
    import news_assistant.editorial_model as module

    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 100)
    stream = _FrameStream(["data: " + " " * 150, AssertionError("must stop before next chunk")])
    client = EditorialModelClient("test-key", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    with pytest.raises(ValueError, match="size limit"):
        client.assess(_audit(), now=NOW)
    assert stream.closed


def test_stream_handles_network_splits_inside_utf8_and_sse_lines():
    audit = _audit()
    content = json.dumps(_proposal(audit), ensure_ascii=False)
    event = "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content},
                                               "finish_reason": "stop"}]}, ensure_ascii=False)
    data = (event + "\r\n\r\ndata: [DONE]\r\n\r\n").encode()
    stream = _FrameStream([data[i:i + 7] for i in range(0, len(data), 7)])
    client = EditorialModelClient("test-key", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    assert client.assess(audit, now=NOW)["items"][0]["rationale"] == _proposal(audit)["items"][0]["rationale"]
    assert stream.closed


@pytest.mark.parametrize('failure,stage', [
    ('json', 'json'), ('schema', 'schema'), ('missing', 'candidate_ids'),
    ('quote', 'content_binding'), ('length', 'completion'),
])
def test_invalid_completed_proposals_retain_replay_data_and_safe_failure_stage(failure, stage):
    audit = _audit(2)
    proposed = _proposal(audit)
    if failure == 'schema':
        proposed['items'][0]['quality_score'] = 9
        proposed['items'][0]['private-unknown-field'] = 'private-field-value'
    elif failure == 'missing':
        proposed['items'].pop()
    elif failure == 'quote':
        proposed['items'][0]['evidence_excerpt'] = '没有出处的引文'
    content = 'private-invalid-json' if failure == 'json' else json.dumps(proposed)
    payload = {'choices': [{'finish_reason': 'length' if failure == 'length' else 'stop',
                            'message': {'content': content}}]}
    client = EditorialModelClient('test-key', transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload)))
    with pytest.raises(ValueError):
        client.assess(audit, now=NOW)
    assert client.last_validation['stage'] == stage
    assert client.last_response['content'] == content
    diagnostics = json.dumps(client.last_validation)
    assert all(value not in diagnostics for value in (
        'test-key', 'private-invalid-json', 'private-unknown-field', 'private-field-value'))
    replay = EditorialModelClient('', transport=httpx.MockTransport(
        lambda _: pytest.fail('offline replay must not contact provider')))
    with pytest.raises(ValueError):
        replay.replay(audit, client.last_response, now=NOW)
    assert replay.last_validation['stage'] == stage


def test_successful_replay_preserves_decisions_and_requires_matching_input_and_content():
    audit = _audit()
    client = EditorialModelClient('test-key', transport=httpx.MockTransport(
        lambda _: _response(_proposal(audit))))
    result = client.assess(audit, now=NOW)
    saved = client.last_response
    replay = EditorialModelClient('', transport=httpx.MockTransport(
        lambda _: pytest.fail('offline replay must not contact provider')))
    restored = replay.replay(audit, saved, now=NOW + timedelta(hours=1))
    assert restored['items'] == result['items']
    assert restored['reviewed_at'] == result['reviewed_at']
    assert restored['reviewed_by'].endswith(':offline-replay')
    for field in ('snapshot_sha256', 'request_sha256', 'content_sha256'):
        changed = {**saved, field: '0' * 64}
        with pytest.raises(ValueError):
            replay.replay(audit, changed, now=NOW)
    changed = {**saved, 'requested_at': (NOW + timedelta(hours=1)).isoformat()}
    with pytest.raises(ValueError):
        replay.replay(audit, changed, now=NOW)
    changed_audit = deepcopy(audit)
    changed_audit['sources'][0]['candidates'][0]['summary'] = '修改后的材料'
    with pytest.raises(ValueError):
        replay.replay(changed_audit, saved, now=NOW)


def test_provider_echoing_credential_is_rejected_without_retaining_it():
    audit = _audit()
    proposed = _proposal(audit)
    proposed['items'][0]['rationale'] = 'do not persist private-test-key'
    client = EditorialModelClient('private-test-key', transport=httpx.MockTransport(
        lambda _: _response(proposed)))
    with pytest.raises(ValueError):
        client.assess(audit, now=NOW)
    assert client.last_response is None
    assert client.last_validation['code'] == 'credential_echo'
    assert 'private-test-key' not in json.dumps(client.last_validation)


def test_cli_retains_rejected_response_privately_and_replays_without_credentials(tmp_path, monkeypatch, capsys):
    import sys

    from news_assistant import cli
    from news_assistant import editorial_model as module

    audit = _audit()
    snapshot = tmp_path / 'audit.json'
    snapshot.write_text(json.dumps(audit))
    output = tmp_path / 'review.json'
    proposed = _proposal(audit)
    proposed['items'][0]['evidence_excerpt'] = '没有出处的引文'
    calls = []

    def handler(request):
        calls.append(request)
        return _response(proposed)

    real_client = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: real_client(
        **{**kwargs, 'transport': httpx.MockTransport(handler)}))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'private-test-key')
    base = ['news-assistant', 'assess-review', '--snapshot', str(snapshot), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', base + ['--execute'])
    with pytest.raises(SystemExit):
        cli.main()
    receipt_path = output.with_name('review.json.attempt.json')
    receipt_text = receipt_path.read_text()
    receipt = json.loads(receipt_text)
    assert receipt['validation']['stage'] == 'content_binding'
    response = output.with_name('review.json.response.json')
    assert receipt['response_retained'] is True and response.exists()
    assert response.stat().st_mode & 0o777 == 0o600
    assert '没有出处的引文' in response.read_text()
    assert 'private-test-key' not in response.read_text() + receipt_text + capsys.readouterr().err
    assert not output.exists()
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    replay_output = tmp_path / 'replayed.json'
    monkeypatch.setattr(sys, 'argv', ['news-assistant', 'assess-review', '--snapshot', str(snapshot),
        '--output', str(replay_output), '--replay-response', str(response)])
    with pytest.raises(SystemExit):
        cli.main()
    diagnostic = json.loads(replay_output.with_name('replayed.json.validation.json').read_text())
    assert diagnostic['paid_requests'] == 0
    assert diagnostic['validation']['stage'] == 'content_binding'
    assert receipt_path.read_text() == receipt_text
    assert not replay_output.exists() and len(calls) == 1


def test_cli_refuses_replay_inside_public_output_and_conflicting_modes(tmp_path, monkeypatch):
    import sys

    from news_assistant import cli

    snapshot = tmp_path / 'audit.json'
    snapshot.write_text(json.dumps(_audit()))
    public = tmp_path / 'public'
    public.mkdir()
    response = public / 'response.json'
    response.write_text('{}')
    output = tmp_path / 'replayed.json'
    base = ['news-assistant', 'assess-review', '--snapshot', str(snapshot), '--output', str(output),
            '--public-dir', str(public), '--replay-response', str(response)]
    monkeypatch.setattr(sys, 'argv', base)
    with pytest.raises(SystemExit):
        cli.main()
    assert not output.exists()
    monkeypatch.setattr(sys, 'argv', base + ['--execute'])
    with pytest.raises(SystemExit):
        cli.main()
    assert not output.with_name('replayed.json.attempt.json').exists()


def test_cli_successful_replay_is_free_and_does_not_replace_existing_artifacts(tmp_path, monkeypatch, capsys):
    import sys

    from news_assistant import cli

    audit = _audit()
    capture = EditorialModelClient('', transport=httpx.MockTransport(
        lambda _: _response(_proposal(audit))))
    capture.assess(audit, now=NOW)
    saved = tmp_path / 'saved.json'
    saved.write_text(json.dumps(capture.last_response))
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(audit))
    output = tmp_path / 'review.json'
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: pytest.fail('replay must never use HTTP'))
    args = ['news-assistant', 'assess-review', '--snapshot', str(snapshot), '--output', str(output),
            '--replay-response', str(saved)]
    monkeypatch.setattr(sys, 'argv', args)
    assert cli.main() == 0
    result = json.loads(output.read_text())
    assert result['items'][0]['status'] == 'accepted'
    assert result['reviewed_by'].endswith(':offline-replay')
    assert json.loads(capsys.readouterr().out)['paid_requests'] == 0
    assert not output.with_name('review.json.attempt.json').exists()
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        cli.main()
    assert output.read_bytes() == original


@pytest.mark.parametrize('suffix', ['.response.json', '.validation.json'])
def test_cli_existing_diagnostics_block_paid_calls_before_credentials(tmp_path, monkeypatch, suffix):
    import sys

    from news_assistant import cli

    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(_audit()))
    output = tmp_path / 'review.json'
    protected = tmp_path / ('review.json' + suffix)
    protected.write_text('preserve this private evidence')
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: pytest.fail('must not send a request'))
    monkeypatch.setattr(sys, 'argv', ['news-assistant', 'assess-review', '--snapshot', str(snapshot),
                                    '--output', str(output), '--execute'])
    with pytest.raises(SystemExit):
        cli.main()
    assert protected.read_text() == 'preserve this private evidence'
    assert not output.with_name('review.json.attempt.json').exists()


def test_cli_execute_reads_saved_key_but_preflight_and_replay_do_not(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    from news_assistant import cli, credential_store
    from news_assistant import editorial_model as module

    audit = _audit()
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(audit))
    output = tmp_path / 'review.json'
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(credential_store.sys, 'platform', 'darwin')
    lookups = []
    requests = []
    def keychain(args, **kwargs):
        lookups.append(args)
        return subprocess.CompletedProcess(args, 0, stdout='saved-only-test-key\n', stderr='')
    monkeypatch.setattr(credential_store.subprocess, 'run', keychain)
    real_client = httpx.Client
    def handler(request):
        requests.append(request)
        assert request.headers['Authorization'] == 'Bearer saved-only-test-key'
        return _response(_proposal(audit))
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: real_client(
        **{**kwargs, 'transport': httpx.MockTransport(handler)}))
    args = ['news-assistant', 'assess-review', '--snapshot', str(snapshot), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    assert cli.main() == 0 and lookups == [] and requests == []
    monkeypatch.setattr(sys, 'argv', args + ['--execute'])
    assert cli.main() == 0
    assert len(lookups) == len(requests) == 1
    assert json.loads(output.read_text())['items'][0]['status'] == 'accepted'
    monkeypatch.setattr(sys, 'argv', ['news-assistant', 'assess-review', '--snapshot', str(snapshot),
        '--output', str(tmp_path / 'replay.json'), '--replay-response', str(tmp_path / 'review.json.response.json')])
    assert cli.main() == 0
    assert len(lookups) == len(requests) == 1
    assert 'saved-only-test-key' not in capsys.readouterr().out
    assert all('saved-only-test-key' not in path.read_text() for path in tmp_path.glob('*.json'))


def test_readable_quote_contract_removes_markup_but_preserves_exact_text():
    audit = _audit(summary='<p>创始人 &amp; CEO 说：「<strong>只在小鼠有效</strong>」。</p><script>秘密指令</script>')
    original = deepcopy(audit)
    proposal = _proposal(audit)
    proposal['items'][0]['evidence_excerpt'] = '创始人 & CEO 说：「只在小鼠有效」。'
    client = EditorialModelClient('test-key', transport=httpx.MockTransport(
        lambda request: _response(proposal)))
    prepared = client._prepare_request(audit, now=NOW, max_candidates=100)
    candidate = json.loads(prepared.user_prompt)['candidates'][0]
    assert candidate['summary'] == proposal['items'][0]['evidence_excerpt']
    result = client.assess(audit, now=NOW)
    assert result['items'][0]['quotation_basis'] == 'readable-v1'
    assert audit == original
    assert client.replay(audit, client.last_response, now=NOW)['items'] == result['items']
    proposal['items'][0]['evidence_excerpt'] = '已证实对人体有效'
    with pytest.raises(ValueError):
        client.assess(audit, now=NOW)
    assert client.last_validation['code'] == 'quotation_mismatch'


def test_legacy_raw_response_replay_keeps_original_request_and_strict_quote_contract():
    audit = _audit(summary='<p>A &amp; B</p>')
    client = EditorialModelClient('')
    prepared = client._prepare_request(audit, now=NOW, max_candidates=100,
                                       quotation_basis='captured-v1')
    content = json.dumps(_proposal(audit))
    saved = {'schema_version': 'editorial-model-response-v1', 'model': client.model,
             'requested_at': NOW.isoformat(), 'snapshot_sha256': prepared.template['snapshot_sha256'],
             'request_sha256': prepared.metadata['request_sha256'],
             'content_sha256': sha256(content.encode()).hexdigest(),
             'finish_reason': 'stop', 'content': content}
    assert client.replay(audit, saved, now=NOW)['items'][0]['quotation_basis'] == 'captured-v1'
    saved['quotation_basis'] = 'readable-v1'
    with pytest.raises(ValueError, match='does not match'):
        client.replay(audit, saved, now=NOW)


def test_model_proposal_preserves_story_shape_and_attributed_origin():
    audit = _audit()
    proposal = _proposal(audit)
    proposal['items'][0].update(status='rejected', story_shape='roundup',
                                origin_publisher_group='one-wire-service',
                                rationale='多事件转载合集，需拆分。')
    client = EditorialModelClient('test-key', transport=httpx.MockTransport(
        lambda request: _response(proposal)))
    result = client.assess(audit, now=NOW, max_candidates=1)
    assert result['items'][0]['story_shape'] == 'roundup'
    assert result['items'][0]['origin_publisher_group'] == 'one-wire-service'
