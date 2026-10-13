"""Conservative, explainable exclusion of disclosed promotional content."""

import re

from .models import Article

_DISCLOSURE = re.compile(
    r"(?:^|[\s。！？.!?|｜])(?:广告\s*[:：]|商业推广|品牌合作|"
    r"sponsored\s+(?:content|post)|paid\s+post)|"
    r"(?:本文|本篇|本内容)(?:为|是|属于).{0,8}(?:赞助|推广|广告)",
    re.IGNORECASE,
)

# Deliberately narrow edition labels observed in captured feeds. Semicolons,
# multiple subjects, or a newspaper name alone are not evidence of a roundup.
_ROUNDUP_TITLE = re.compile(
    r"^(?:【钛晨报】|(?:派早报|早报)\s*[:：|｜]|The Download\s*:|"
    r"Edge AI Daily\s+早报)|[|｜]\s*消研所周报\s*$",
    re.IGNORECASE,
)


def roundup_reason(article: Article) -> str | None:
    """A known multi-event edition needs individually sourced events before selection."""
    if _ROUNDUP_TITLE.search(article.title.strip()):
        return "explicit_roundup_title"
    return None


def promotional_reason(article: Article) -> str | None:
    text = f"{article.title}\n{article.summary}"
    if _DISCLOSURE.search(text):
        return "explicit_promotional_marker"
    # Recruitment for branded hands-on campaigns needs all three signals;
    # a product launch, independent review or a newsletter footer alone is valid.
    if (
        re.search(r"我们.{0,12}(?:发起|邀请|招募)", text)
        and re.search(r"(?:新品|产品|品牌).{0,12}(?:计划|体验|探索)", text)
        and re.search(r"(?:线下门店|到店|报名体验|亲自感受)", text)
    ):
        return "experience_recruitment_cta"
    return None
