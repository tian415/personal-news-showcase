from __future__ import annotations

import json
import re

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from .models import Article, EnrichedArticle, Topic
from .source_text import readable_source_text

# Keep source-only fallback content useful while preventing RSS boilerplate and
# very long summaries from overwhelming the daily page.
_FALLBACK_SUMMARY_LIMIT = 160
_FALLBACK_EXCERPT_MARKER = "来源摘要节选："
_FALLBACK_UNAVAILABLE = "暂时无法生成中文摘要，请查看原文。"
_CREDIT_LINE = re.compile(r"^(?:作者|编辑|撰文|供稿|责编|责任编辑)\s*[|｜:：]\s*.+$")
_PROMOTION = re.compile(r"欢迎关注[^。！？!?；;\n]{0,120}(?:公众号|微信|账号)")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")

def _fallback_summary(summary: str) -> str:
    text = readable_source_text(summary)
    if not text:
        return _FALLBACK_UNAVAILABLE

    sentences: list[tuple[str, int]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or _CREDIT_LINE.fullmatch(line):
            continue
        for sentence in _SENTENCE_BOUNDARY.split(line):
            sentence = " ".join(sentence.split())
            if not sentence or _PROMOTION.search(sentence) or sentence in seen:
                continue
            seen.add(sentence)
            sentences.append((sentence, line_number))
    if not sentences:
        return _FALLBACK_UNAVAILABLE

    def render(items: list[tuple[str, int]]) -> str:
        result = items[0][0]
        previous_line = items[0][1]
        for sentence, line_number in items[1:]:
            result += ("\n" if line_number != previous_line else " ") + sentence
            previous_line = line_number
        return result

    maximum_sentences = min(2, len(sentences))
    complete = render(sentences[:maximum_sentences])
    if len(sentences) <= 2 and len(complete) <= _FALLBACK_SUMMARY_LIMIT:
        return complete

    ellipsis = "…"
    for sentence_count in range(maximum_sentences, 0, -1):
        candidate = render(sentences[:sentence_count])
        excerpt = f"{_FALLBACK_EXCERPT_MARKER}{candidate}{ellipsis}"
        if len(excerpt) <= _FALLBACK_SUMMARY_LIMIT:
            return excerpt
    return _FALLBACK_UNAVAILABLE


class EnrichmentPayload(BaseModel):
    chinese_title: str = Field(max_length=64)
    chinese_summary: str = Field(max_length=160)
    key_points: list[str]
    tags: list[str]
    beginner_context: str | None = Field(default=None, max_length=80)
    claim_type: str

    @field_validator("chinese_title", "chinese_summary")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("enrichment text must not be blank")
        return value


def fallback_enrichment(article: Article) -> EnrichedArticle:
    business = article.topic in {Topic.BUSINESS, Topic.MARKETS}
    context = "这是自动降级内容，仅作背景了解，不构成任何买卖建议。" if business else None
    return EnrichedArticle(
        **article.model_dump(),
        chinese_title=article.title,
        chinese_summary=_fallback_summary(article.summary),
        beginner_context=context,
        degraded=True,
    )


class DeepSeekClient:
    def __init__(self, api_key: str, *, base_url: str = "https://api.deepseek.com", model: str = "deepseek-v4-flash", transport: httpx.BaseTransport | None = None, retries: int = 2) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport
        self.retries = max(1, retries)

    def enrich(self, article: Article) -> EnrichedArticle:
        prompt = (
            "把以下来源材料整理为简短、准确的中文新闻导语。只使用提供的事实，禁止补造信息。"
            "标题中性且不超过64字符。摘要1至2句、最多160字符，先说明主体做了什么和关键结果；"
            "时间、地点仅在材料明确给出且影响理解时写入。发布时间不是事件发生时间，不能偷换。"
            "保留关键数字、否定、条件以及‘宣称/据报道/尚未验证’等归因，不能把宣传或观点写成事实。"
            "删除作者署名、推广、引语铺垫、重复背景；不重复标题，不罗列公司履历。"
            "背景仅在理解新闻确有必要时补一句，最多80字符，不要重复摘要。"
            "返回 JSON，字段为 chinese_title、chinese_summary、key_points、tags、"
            "beginner_context、claim_type。claim_type 只能是 fact/opinion/speculation。"
            "商业新闻应给初学者背景且不提供买卖建议。\n"
            f"来源：{article.source}\n来源发布时间：{article.published_at.isoformat()}\n"
            f"标题：{article.title}\n摘要：{article.summary}\n主题：{article.topic.value}"
        )
        for _ in range(self.retries):
            try:
                with httpx.Client(transport=self.transport, timeout=30) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": self.model, "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]},
                    )
                    response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                payload = EnrichmentPayload.model_validate(json.loads(content))
                return EnrichedArticle(**article.model_dump(), **payload.model_dump())
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValidationError,
            ):
                continue
        return fallback_enrichment(article)
