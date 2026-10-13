from datetime import UTC, date, datetime

import pytest
from markupsafe import escape

from news_assistant.models import (
    BackgroundReading,
    Digest,
    EnrichedArticle,
    NewsValueDisplay,
    Topic,
)
from news_assistant.render import daily_digest_url, render_digest, render_email_digest


def sample_digest() -> Digest:
    article = EnrichedArticle(
        id="a1",
        title="English title",
        url="https://example.com/story",
        source="Example",
        language="en",
        published_at=datetime.now(UTC),
        topic=Topic.BUSINESS,
        chinese_title="商业新闻",
        chinese_summary="一句话摘要",
        beginner_context="面向初学者的解释，不构成买卖建议。",
        tags=["商业"],
    )
    return Digest(digest_date=date(2026, 8, 12), articles=[article])


def test_render_digest_has_links_feedback_and_no_private_data(tmp_path) -> None:
    path = render_digest(sample_digest(), tmp_path)
    html = path.read_text()
    assert "商业新闻" in html
    assert "Google 翻译" in html
    assert "data-feedback" in html
    assert "localStorage" in (tmp_path / "assets" / "feedback.js").read_text()
    assert "GMAIL_APP_PASSWORD" not in html
    assert "example-private-account" not in html


def test_render_email_digest_is_self_contained_and_links_to_dated_article(
    tmp_path,
) -> None:
    path = render_email_digest(
        sample_digest(),
        tmp_path / "mail" / "digest.html",
        "https://example-user.github.io/personal-news-digest/",
    )
    html = path.read_text()
    assert "https://example-user.github.io/personal-news-digest/2026-08-12.html" in html
    assert "2026-08-12.html#article-a1" in html
    assert "阅读原文" in html
    assert "中文翻译" in html
    assert "去网页反馈" in html
    assert "<script" not in html
    assert "stylesheet" not in html
    assert "assets/" not in html
    assert "<button" not in html


def test_render_web_digest_has_stable_article_anchor(tmp_path) -> None:
    path = render_digest(sample_digest(), tmp_path)
    assert 'id="article-a1"' in path.read_text()


@pytest.mark.parametrize("degraded", [False, True])
def test_source_only_fallback_is_visible_in_web_and_email(tmp_path, degraded) -> None:
    digest = sample_digest()
    digest.articles[0].degraded = degraded
    web = render_digest(digest, tmp_path / "public").read_text()
    email = render_email_digest(
        digest, tmp_path / "email.html", "https://example.com/news/"
    ).read_text()
    for html in (web, email):
        assert ("来源摘要 · 自动整理暂不可用" in html) is degraded


def test_web_feedback_explains_local_storage_and_export(tmp_path) -> None:
    path = render_digest(sample_digest(), tmp_path)
    html = path.read_text()
    script = (tmp_path / "assets" / "feedback.js").read_text()
    styles = (tmp_path / "assets" / "styles.css").read_text()
    assert "导出偏好文件" in html
    assert "导入私有配置后用于推荐评估，尚不自动改变邮件" in html
    assert "已保存在此浏览器" in html
    assert "aria-pressed" in html
    assert "data[ref] = entry" in script
    assert "news-feedback-v2" in script
    assert "scrollIntoView" in script
    assert "target-highlight" in styles


def test_render_escapes_untrusted_article_values_in_web_and_email(tmp_path) -> None:
    hostile = '"><script>alert(1)</script><img src=x onerror=alert(1)>'
    article = sample_digest().articles[0].model_copy(
        update={
            "id": hostile,
            "title": hostile,
            "source": hostile,
            "chinese_title": hostile,
            "chinese_summary": hostile,
            "beginner_context": hostile,
        }
    )
    digest = sample_digest().model_copy(update={"articles": [article]})

    web_html = render_digest(digest, tmp_path).read_text()
    email_html = render_email_digest(
        digest,
        tmp_path / "mail" / "digest.html",
        "https://example.com/news/",
    ).read_text()
    escaped = str(escape(hostile))

    assert "<img src=x onerror=alert(1)>" not in web_html
    assert "<img src=x onerror=alert(1)>" not in email_html
    assert '<script>alert(1)</script>' not in web_html
    assert '<script>alert(1)</script>' not in email_html
    assert escaped in web_html
    assert escaped in email_html


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.com/news?next=/",
        "https://example.com/news?",
        "https://example.com/news#today",
        "https://example.com/news#",
        "javascript:alert(1)",
        "//example.com/news",
        "mailto:reader@example.com",
    ],
)
def test_daily_digest_url_rejects_unsafe_or_non_base_urls(base_url) -> None:
    with pytest.raises(ValueError):
        daily_digest_url(base_url, date(2026, 8, 12))


def test_daily_digest_url_preserves_valid_base_path() -> None:
    assert (
        daily_digest_url("https://example.com/news/", date(2026, 8, 12))
        == "https://example.com/news/2026-08-12.html"
    )


def test_archive_lists_only_real_dated_pages_in_reverse_order(tmp_path):
    from datetime import date

    from news_assistant.render import render_archive

    for filename in ['2026-09-27.html', '2026-09-28.html', '2026-09-29.html', '2026-02-30.html']:
        (tmp_path / filename).write_text('page')
    (tmp_path / 'private.json').write_text('{}')
    (tmp_path / '2026-09-26.html').symlink_to(tmp_path / '2026-09-27.html')
    html = render_archive(tmp_path, through=date(2026, 9, 28)).read_text()
    assert html.index('2026-09-28.html') < html.index('2026-09-27.html')
    for forbidden in ['2026-09-29', '2026-02-30', '2026-09-26', 'private.json']:
        assert forbidden not in html


def test_reading_contents_link_to_each_unique_feedback_target(tmp_path) -> None:
    import re

    digest = sample_digest()
    second = digest.articles[0].model_copy(update={"id": "b/2", "event_ref": "b" * 24})
    digest.articles[0].event_ref = "a" * 24
    digest.articles.append(second)
    html = render_digest(digest, tmp_path).read_text()
    contents = re.search(r'<nav[^>]*aria-label="本期目录"[^>]*>(.*?)</nav>', html, re.DOTALL)
    assert contents is not None
    for anchor in ("article-a1", "article-b%2F2"):
        assert f'href="#{anchor}"' in contents.group(1)
        assert html.count(f'id="{anchor}"') == 1
    cards = re.findall(r'<article\b(.*?)</article>', html, re.DOTALL)
    assert len(cards) == 2
    for card in cards:
        assert 'data-event-ref=' in card
        assert 'class="feedback-status"' in card
        assert 'class="feedback-reason"' in card
        for action in ("helpful", "reduce", "track"):
            assert card.count(f'data-feedback="{action}"') == 1


def test_background_reading_is_local_context_without_news_feedback(tmp_path) -> None:
    hostile = '<script>alert(1)</script>'
    reading = BackgroundReading(
        id="background-1", title=hostile, url="https://example.com/background",
        source="Example", published_on=date(2024, 1, 2), kind="analysis",
        why_read="理解长期背景", limitation="这不是今日的新进展", language="en",
    )
    digest = sample_digest().model_copy(update={
        "local_preview": True, "background_readings": [reading],
    })
    html = render_digest(digest, tmp_path / "local").read_text()
    assert 'id="background-readings"' in html
    background = html.split('id="background-readings"', 1)[1].split('</section>', 1)[0]
    assert "不计今日新闻条数" in background
    assert "理解长期背景" in background
    assert "2024-01-02" in background
    assert "这不是今日的新进展" in background
    assert "中文翻译" in background
    assert str(escape(hostile)) in background
    assert hostile not in background
    for forbidden in ("<article", "data-event-ref", "data-feedback"):
        assert forbidden not in background
    assert "1 条精选" in html
    header = html.split('</header>', 1)[0]
    assert 'aria-label="阅读栏目"' in header
    assert 'href="#background-readings"' in header
    public_html = render_digest(
        digest.model_copy(update={"local_preview": False}), tmp_path / "public"
    ).read_text()
    assert 'id="background-readings"' not in public_html
    assert "理解长期背景" not in public_html


def test_empty_digest_omits_contents_and_background_sections(tmp_path) -> None:
    digest = sample_digest().model_copy(update={"articles": [], "local_preview": True})
    html = render_digest(digest, tmp_path).read_text()
    assert "今天暂时没有符合质量标准的新闻" in html
    assert 'aria-label="本期目录"' not in html
    assert 'id="background-readings"' not in html


def test_value_explanation_stays_visible_and_evidence_is_collapsible(tmp_path) -> None:
    import re

    digest = sample_digest()
    digest.local_preview = True
    digest.coverage_notes = ["本期科学领域素材不足"]
    digest.articles[0].news_value = NewsValueDisplay(
        why_today="今天出现新的制度变化", impact="将影响消费者的选择",
        event_time=datetime(2026, 8, 12, tzinfo=UTC),
        attention_note="单次快照不能证明热度增长",
        value_basis_url="https://example.com/evidence", value_basis_source="来源文件",
    )
    digest.articles[0].key_points = ["材料中的原文依据"]
    html = render_digest(digest, tmp_path).read_text()
    evidence = re.search(r'<details class="evidence-details"(.*?)</details>', html, re.DOTALL)
    assert evidence is not None
    assert "open" not in evidence.group(1).split(">", 1)[0]
    assert "单次快照不能证明热度增长" in evidence.group(1)
    assert "材料中的原文依据" in evidence.group(1)
    assert "https://example.com/evidence" in evidence.group(1)
    assert "今天出现新的制度变化" not in evidence.group(1)
    assert "将影响消费者的选择" not in evidence.group(1)
    assert "今天出现新的制度变化" in html
    assert "将影响消费者的选择" in html
    assert "尚未线上发布" in html
    assert "本期科学领域素材不足" in html
