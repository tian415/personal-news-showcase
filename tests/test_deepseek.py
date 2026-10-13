import json
from datetime import UTC, datetime

import httpx

from news_assistant.deepseek import DeepSeekClient, fallback_enrichment
from news_assistant.models import Article, Topic


def sample(topic: Topic = Topic.AI) -> Article:
    return Article(id="1", title="AI launch", url="https://example.com/1", source="Example", language="en", published_at=datetime.now(UTC), topic=topic, summary="Details")


def test_enrich_parses_structured_chinese_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"chinese_title":"新模型发布","chinese_summary":"一句话摘要","key_points":["要点"],"tags":["AI"],"beginner_context":null,"claim_type":"fact"}'}}]})
    client = DeepSeekClient("secret", transport=httpx.MockTransport(handler))
    result = client.enrich(sample())
    assert result.chinese_title == "新模型发布"
    assert result.claim_type == "fact"


def test_enrich_falls_back_after_malformed_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})
    result = DeepSeekClient("secret", transport=httpx.MockTransport(handler), retries=1).enrich(sample())
    assert result.degraded is True
    assert result.chinese_title == "AI launch"


def test_enrich_retries_after_invalid_utf8_response_and_preserves_article() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                content=b'{"choices":[{"message":{"content":"\xff"}}]}',
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"chinese_title":"新标题","chinese_summary":"新摘要",'
                            '"key_points":[],"tags":[],"beginner_context":null,"claim_type":"fact"}',
                        }
                    }
                ]
            },
        )

    article = sample()
    result = DeepSeekClient("secret", transport=httpx.MockTransport(handler), retries=2).enrich(article)

    assert calls == 2
    assert result.chinese_title == "新标题"
    assert result.chinese_summary == "新摘要"
    assert result.model_dump(include=set(Article.model_fields)) == article.model_dump()


def test_enrich_retries_after_blank_summary_fields_and_preserves_article() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = (
            '{"chinese_title":"   ","chinese_summary":"\\t",'
            '"key_points":[],"tags":[],"beginner_context":null,"claim_type":"fact"}'
            if calls == 1
            else '{"chinese_title":"有效标题","chinese_summary":"有效摘要",'
            '"key_points":[],"tags":[],"beginner_context":null,"claim_type":"fact"}'
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    article = sample()
    result = DeepSeekClient("secret", transport=httpx.MockTransport(handler), retries=2).enrich(article)

    assert calls == 2
    assert result.chinese_title == "有效标题"
    assert result.chinese_summary == "有效摘要"
    assert result.model_dump(include=set(Article.model_fields)) == article.model_dump()


def test_business_fallback_contains_beginner_context() -> None:
    result = fallback_enrichment(sample(Topic.BUSINESS))
    assert result.beginner_context
    assert "买卖" in result.beginner_context


def test_oversized_generated_summary_is_rejected_within_existing_retry_budget() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        payload = {
            "chinese_title": "公司发布新产品", "chinese_summary": "重复背景" * 80,
            "key_points": [], "tags": [], "beginner_context": None, "claim_type": "fact",
        }
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(payload),
        }}]})

    result = DeepSeekClient("secret", transport=httpx.MockTransport(handler), retries=2).enrich(sample())
    assert result.degraded
    assert len(result.chinese_summary) <= 160
    assert len(calls) == 2
    prompt = calls[0]["messages"][0]["content"]
    assert "160" in prompt
    assert "发布时间不是事件发生时间" in prompt
    assert "宣称" in prompt and "否定" in prompt


def test_fallback_summary_extracts_html_text_entities_and_paragraphs() -> None:
    source_summary = (
        "<p>第一段 &amp; 事实</p><p>第二段 &lt;比较符&gt;</p>"
        "<script>不应显示的脚本文本</script>"
        "<style>不应显示的样式文本</style>"
    )
    article = sample().model_copy(
        update={"summary": source_summary}
    )

    result = fallback_enrichment(article)

    assert result.chinese_summary == "第一段 & 事实\n第二段 <比较符>"
    assert result.degraded is True
    assert article.summary == source_summary
    assert "来源摘要节选" not in result.chinese_summary


def test_fallback_summary_preserves_plaintext_less_than_comparison() -> None:
    article = sample().model_copy(update={"summary": "事实：2 < 3，4 > 1"})

    result = fallback_enrichment(article)

    assert result.chinese_summary == "事实：2 < 3，4 > 1"


def test_fallback_summary_marks_only_truncated_text_as_source_excerpt() -> None:
    summary = "<p>重要事实。</p>" + ("后续事实。" * 200) + "<p>新的结果。</p>"
    article = sample().model_copy(update={"summary": summary})

    result = fallback_enrichment(article)

    assert len(result.chinese_summary) <= 160
    assert result.chinese_summary.startswith("来源摘要节选：")
    assert result.chinese_summary.endswith("…")
    assert result.degraded is True
    assert article.summary == summary


def test_fallback_summary_removes_credits_promotion_and_duplicate_sentences() -> None:
    source_summary = (
        "<p>作者｜Li Yuan</p>"
        "<p>Figure 1. 实验显示芯片在 80°C 下稳定运行。</p>"
        "<p>重复铺垫：这项研究关注芯片稳定性。</p>"
        "<p>重复铺垫：这项研究关注芯片稳定性。</p>"
        "<p>研究团队确认结果可重复。</p>"
        "<p>欢迎关注极客公园公众号获取更多科技内容。</p>"
    )
    article = sample().model_copy(update={"summary": source_summary})

    result = fallback_enrichment(article)

    assert len(result.chinese_summary) <= 160
    assert "作者" not in result.chinese_summary
    assert "Li Yuan" not in result.chinese_summary
    assert "欢迎关注" not in result.chinese_summary
    assert result.chinese_summary.count("重复铺垫：这项研究关注芯片稳定性。") == 1
    assert result.chinese_summary.count("Figure 1.") == 1
    assert result.chinese_summary.endswith("…")


def test_fallback_summary_does_not_cut_an_unfittable_sentence() -> None:
    long_sentence = "事实" + ("，事实" * 100) + "。"
    article = sample().model_copy(update={"summary": f"<p>{long_sentence}</p><p>后续事实。</p>"})

    result = fallback_enrichment(article)

    assert result.chinese_summary == "暂时无法生成中文摘要，请查看原文。"
    assert result.degraded is True


def test_default_model_uses_current_flash_model() -> None:
    assert DeepSeekClient("secret").model == "deepseek-v4-flash"
