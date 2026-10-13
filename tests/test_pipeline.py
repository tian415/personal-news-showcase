from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from news_assistant.cli import build_digest
from news_assistant.models import Article, EnrichedArticle, Preferences, Topic


class FakeEnricher:
    def enrich(self, item: Article) -> EnrichedArticle:
        return EnrichedArticle(**item.model_dump(), chinese_title="中文标题", chinese_summary="中文摘要")


def test_build_digest_writes_public_page_without_network(tmp_path: Path) -> None:
    articles = [Article(id="1", title="Story", url="https://example.com/1", source="Example", language="en", published_at=datetime.now(UTC), topic=Topic.AI)]
    digest = build_digest(articles, Preferences(), FakeEnricher(), tmp_path, digest_date=date(2026, 8, 12))
    assert digest.articles[0].chinese_title == "中文标题"
    assert (tmp_path / "index.html").exists()


def test_build_digest_writes_public_and_private_email_outputs(tmp_path: Path) -> None:
    articles = [Article(id="1", title="Story", url="https://example.com/1", source="Example", language="en", published_at=datetime.now(UTC), topic=Topic.AI)]
    public = tmp_path / "public"
    email = tmp_path / "work" / "digest-email.html"
    build_digest(articles, Preferences(), FakeEnricher(), public, digest_date=date(2026, 8, 12), email_output=email, digest_base_url="https://example.com/news/")
    assert (public / "index.html").exists()
    assert email.exists()
    assert "2026-08-12.html#article-1" in email.read_text()
    assert not (public / "digest-email.html").exists()


def test_shadow_mode_records_private_proposal_but_renders_legacy_selection(tmp_path: Path) -> None:
    articles = [
        Article(
            id="legacy-article",
            title="Story",
            url="https://example.com/1",
            source="Example",
            language="en",
            published_at=datetime.now(UTC),
            topic=Topic.AI,
        )
    ]

    digest = build_digest(
        articles,
        Preferences(),
        FakeEnricher(),
        tmp_path / "public",
        digest_date=date(2026, 8, 12),
        selection_engine="shadow_event_v1",
        state_dir=tmp_path / "state",
    )

    assert [article.id for article in digest.articles] == ["legacy-article"]
    assert list((tmp_path / "state").rglob("decisions.jsonl"))


@pytest.mark.parametrize(
    "options",
    [
        {"selection_engine": "unknown"},
        {"selection_engine": "shadow_event_v1"},
        {"email_output": "public/email.html", "digest_base_url": "https://example.com/"},
        {"delivery_manifest": "public/selected.json"},
        {"state_dir": "public/state", "selection_engine": "shadow_event_v1"},
        {"state_dir": ".", "selection_engine": "shadow_event_v1"},
        {"state_dir": "state", "selection_engine": "shadow_event_v1", "run_id": "../escape"},
        {"email_output": "email.html", "digest_base_url": "javascript:alert(1)"},
        {"email_output": "shared.html", "delivery_manifest": "shared.html",
         "digest_base_url": "https://example.com/"},
    ],
)
def test_build_rejects_invalid_or_public_private_paths_before_side_effects(
    tmp_path: Path, options: dict,
) -> None:
    for key in ("email_output", "delivery_manifest", "state_dir"):
        if key in options:
            options[key] = tmp_path / options[key]
    public = tmp_path / "public"
    public.mkdir()
    (public / "index.html").write_text("previous digest")
    item = Article(id="1", title="Story", url="https://example.com/1", source="Example",
                   language="en", published_at=datetime.now(UTC), topic=Topic.AI)

    class NoCalls:
        def enrich(self, item):
            pytest.fail("invalid configuration must fail before enrichment")

    with pytest.raises(ValueError):
        build_digest([item], Preferences(), NoCalls(), public, **options)
    assert (public / "index.html").read_text() == "previous digest"
    assert list(public.iterdir()) == [public / "index.html"]


def test_build_resolves_symlinks_before_checking_private_paths(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(public, target_is_directory=True)
    with pytest.raises(ValueError):
        build_digest([], Preferences(), FakeEnricher(), public,
                     delivery_manifest=alias / "selected.json")
    assert list(public.iterdir()) == []


def test_enrichment_failure_preserves_existing_delivery_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.json"
    manifest.write_text('["previous"]')
    item = Article(id="1", title="Story", url="https://example.com/1", source="Example",
                   language="en", published_at=datetime.now(UTC), topic=Topic.AI)

    class FailingEnricher:
        def enrich(self, item):
            raise RuntimeError("enrichment failed")

    with pytest.raises(RuntimeError, match="enrichment failed"):
        build_digest([item], Preferences(), FailingEnricher(), tmp_path / "public",
                     delivery_manifest=manifest)
    assert manifest.read_text() == '["previous"]'
    assert not (tmp_path / "public").exists()


def test_build_uses_one_timestamp_for_selection_date_and_private_run(tmp_path: Path) -> None:
    import json

    build_time = datetime(2026, 9, 20, 16, 1, tzinfo=UTC)
    digest = build_digest([], Preferences(), FakeEnricher(), tmp_path / "public",
                          now=build_time, selection_engine="shadow_event_v1",
                          state_dir=tmp_path / "state", run_id="midnight")
    assert digest.digest_date == date(2026, 9, 21)
    run_path = tmp_path / "state/runs/2026/09/20/midnight/run.json"
    assert json.loads(run_path.read_text())["started_at"] == "2026-09-20T16:01:00Z"


def test_build_rejects_naive_clock_before_rendering(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        build_digest([], Preferences(), FakeEnricher(), tmp_path / "public",
                     now=datetime(2026, 9, 21))  # noqa: DTZ001 - invalid input
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize('link_location', ['root', 'assets', 'index.html', 'archive.html'])
def test_build_rejects_public_symlinks_before_enrichment(tmp_path, link_location):
    outside = tmp_path / 'outside'
    outside.mkdir()
    sentinel = outside / 'private-note.html'
    sentinel.write_text('keep')
    output = tmp_path / 'public'
    if link_location == 'root':
        output.symlink_to(outside, target_is_directory=True)
    else:
        output.mkdir()
        (output / link_location).symlink_to(
            outside if link_location == 'assets' else sentinel,
            target_is_directory=link_location == 'assets',
        )
    class NeverEnrich:
        def enrich(self, item):
            raise AssertionError('must fail before enrichment')
    with pytest.raises(ValueError, match='symlink'):
        build_digest([], Preferences(), NeverEnrich(), output)
    assert list(outside.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'keep'
