from datetime import UTC, datetime

import pytest

from news_assistant.models import Article, Topic


def article(title: str, summary: str = "") -> Article:
    return Article(
        id="1",
        title=title,
        url="https://example.com/1",
        source="Example",
        language="zh",
        published_at=datetime(2026, 9, 21, tzinfo=UTC),
        topic=Topic.TECH,
        summary=summary,
    )


@pytest.mark.parametrize(
    "title,summary",
    [
        ("广告：某产品体验", ""),
        ("商业推广：新品介绍", ""),
        ("品牌合作 | 秋季新品", ""),
        ("Sponsored Content: New device", ""),
        ("Paid Post: Product guide", ""),
        ("新品体验计划", "本文为赞助内容。"),
    ],
)
def test_promotional_reason_detects_explicit_markers(title: str, summary: str) -> None:
    from news_assistant.content_quality import promotional_reason

    assert promotional_reason(article(title, summary)) == "explicit_promotional_marker"


def test_promotional_reason_detects_first_person_experience_recruitment_cta() -> None:
    from news_assistant.content_quality import promotional_reason

    item = article(
        "体验过华为 Pura X View 后，他们最想留下的「阔体验」是……",
        "不久前，我们发起了「华为 Pura X View 新品探索计划」，请大家去线下门店亲自感受，也向我们分享真实的上手体验。",
    )

    assert promotional_reason(item) == "experience_recruitment_cta"


@pytest.mark.parametrize(
    "title,summary",
    [
        ("广告监管新规落地，平台将加强审核", "监管部门公布了执法要求。"),
        ("某公司发布新产品，介绍技术规格", "这是公司公开发布的产品信息。"),
        ("订阅模式调整说明", "我们解释新的订阅价格与服务范围。"),
        ("本周值得阅读的科技新闻", "欢迎关注公众号获取后续更新。"),
        ("我们发起用户调查", "欢迎报名填写问卷，结果将公开。"),
    ],
)
def test_promotional_reason_does_not_overfilter_reporting_or_fixed_footer(
    title: str, summary: str
) -> None:
    from news_assistant.content_quality import promotional_reason

    assert promotional_reason(article(title, summary)) is None
