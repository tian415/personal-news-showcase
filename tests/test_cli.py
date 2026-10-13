import json
import sys
from pathlib import Path

import pytest

from news_assistant import cli
from news_assistant.discovery.v2ex import V2exFetchResult
from news_assistant.models import SignalSnapshot
from news_assistant.publication import UnsafePublicArtifactError


def test_init_command_creates_independent_workspace_without_fetching(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "fetch_articles", lambda *_: pytest.fail("init must stay offline"))
    target = tmp_path / "reader"
    monkeypatch.setattr(sys, "argv", ["news-assistant", "init", "--directory", str(target)])
    assert cli.main() == 0
    assert (target / "config" / "source_registry.yaml").is_file()
    assert (target / "config" / "preferences.json").is_file()


def test_evaluate_coverage_command_writes_reproducible_private_report(monkeypatch, tmp_path):
    from datetime import UTC, datetime

    from news_assistant.source_audit import audit_sources
    from news_assistant.source_registry import SourceRegistry

    captured = tmp_path / "snapshot.json"
    refs = tmp_path / "references.json"
    output = tmp_path / "evaluation.json"
    captured.write_text(json.dumps(audit_sources(
        SourceRegistry(version="source-registry-v1", sources=[]),
        now=datetime(2026, 9, 28, tzinfo=UTC), capture_candidates=True,
    )))
    refs.write_text(json.dumps({
        "schema_version": "coverage-reference-v1", "synthetic": True,
        "purpose": "evaluation", "reviewed_by": "fixture", "reviewed_at": "2026-09-28T00:00:00Z",
        "reference_origin": "synthetic CLI fixture", "events": [],
    }))
    monkeypatch.setattr(cli, "fetch_articles", lambda *_: pytest.fail("replay must stay offline"))
    monkeypatch.setattr(sys, "argv", ["news-assistant", "evaluate-coverage", "--snapshot",
                                     str(captured), "--reference", str(refs), "--output", str(output)])
    assert cli.main() == 0
    report = json.loads(output.read_text())
    assert report["summary"]["production_major_discovery_recall"] is None
    assert len(report["input_fingerprints"]["audit_sha256"]) == 64
    before = output.read_bytes()
    assert cli.main() == 0
    assert output.read_bytes() == before


@pytest.mark.parametrize("case", ["public", "input_collision", "public_symlink"])
def test_evaluate_coverage_rejects_unsafe_output_before_reading_input(monkeypatch, tmp_path, case):
    public = tmp_path / "public"
    public.mkdir()
    captured = tmp_path / "snapshot.json"
    captured.write_text("preserve me")
    if case == "public":
        output = public / "private.json"
    elif case == "input_collision":
        output = captured
    else:
        link = tmp_path / "link"
        link.symlink_to(public, target_is_directory=True)
        output = link / "private.json"
    monkeypatch.setattr(sys, "argv", ["news-assistant", "evaluate-coverage", "--snapshot",
                                     str(captured), "--reference", str(tmp_path / "missing.json"),
                                     "--output", str(output), "--public-dir", str(public)])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert captured.read_text() == "preserve me"
    assert not (public / "private.json").exists()


def test_send_email_uses_existing_html_without_fetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "index.html").write_text("<h1>Already built</h1>")
    sent: dict[str, object] = {}
    monkeypatch.setenv("GMAIL_ADDRESS", "sender@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "secret")
    monkeypatch.setenv("NEWS_RECIPIENT", "reader@gmail.com")
    monkeypatch.setattr(
        cli,
        "fetch_articles",
        lambda *_: pytest.fail("send-email must not fetch feeds"),
    )
    monkeypatch.setattr(
        cli,
        "send_html",
        lambda html, plain_text, digest_date, settings: sent.update(
            html=html,
            plain_text=plain_text,
            digest_date=digest_date,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "news-assistant",
            "send-email",
            "--input",
            str(output / "index.html"),
            "--digest-date",
            "2026-08-13",
            "--digest-url",
            "https://example-user.github.io/personal-news-digest/",
        ],
    )

    assert cli.main() == 0
    assert sent == {
        "html": "<h1>Already built</h1>",
        "plain_text": "今日新闻合集：https://example-user.github.io/personal-news-digest/2026-08-13.html",
        "digest_date": cli.date(2026, 8, 13),
    }


def test_audit_public_command_calls_auditor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[Path] = []
    monkeypatch.setattr(
        cli,
        "audit_public_directory",
        lambda root: called.append(root) or [],
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["news-assistant", "audit-public", "--output", str(tmp_path)],
    )

    assert cli.main() == 0
    assert called == [tmp_path]


def test_build_rejects_invalid_configuration_before_fetching(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "fetch_articles", lambda *_: pytest.fail("must validate first"))
    monkeypatch.setattr(sys, "argv", ["news-assistant", "build", "--selection-engine",
                                      "shadow_event_v1", "--output", str(tmp_path / "public")])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert not (tmp_path / "public").exists()


def test_explicit_missing_preferences_is_not_silently_ignored(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cli._load_preferences(tmp_path / "missing.json")


def test_preserve_history_command_audits_combined_output(monkeypatch, tmp_path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "public"
    published.mkdir()
    output.mkdir()
    (output / "index.html").write_text("current digest")
    (published / "2026-09-20.html").write_text("DEEPSEEK_API_KEY")
    monkeypatch.setattr(sys, "argv", ["news-assistant", "preserve-history", "--published",
                                      str(published), "--output", str(output),
                                      "--digest-date", "2026-09-21"])
    with pytest.raises(UnsafePublicArtifactError, match="credential name"):
        cli.main()


def test_preserve_history_command_retains_dated_page_without_copying_private_files(
    monkeypatch, tmp_path,
) -> None:
    published = tmp_path / "published"
    output = tmp_path / "public"
    published.mkdir()
    output.mkdir()
    (output / "index.html").write_text("current digest")
    (published / "2026-09-20.html").write_bytes(b"old digest\r\n")
    (published / "preferences.json").write_text("private")
    monkeypatch.setattr(sys, "argv", ["news-assistant", "preserve-history", "--published",
                                      str(published), "--output", str(output),
                                      "--digest-date", "2026-09-21"])
    assert cli.main() == 0
    assert (output / "2026-09-20.html").read_bytes() == b"old digest\r\n"
    assert not (output / "preferences.json").exists()


def test_audit_sources_command_writes_private_report_without_model_calls(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    registry_file = tmp_path / "sources.yaml"
    registry_file.write_text("version: source-registry-v1\nsources: []\n")
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["news-assistant", "audit-sources", "--sources",
                                      str(registry_file), "--output", str(report)])
    assert cli.main() == 0
    saved = json.loads(report.read_text())
    assert saved["schema_version"] == "source-audit-v1"
    assert saved["summary"]["production"]["source_count"] == 0
    assert saved["sources"] == []


def test_audit_sources_rejects_public_report_path_before_network(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_source_registry", lambda *_: pytest.fail("must validate first"))
    monkeypatch.setattr(sys, "argv", ["news-assistant", "audit-sources", "--output",
                                      str(tmp_path / "public" / "report.json")])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


def test_source_audit_cannot_overwrite_registry_before_loading_or_fetching(monkeypatch, tmp_path):
    sources = tmp_path / "sources.yaml"
    sources.write_text("preserve original configuration")
    monkeypatch.setattr(cli, "audit_sources", lambda *_args, **_kwargs: pytest.fail("must reject first"))
    monkeypatch.setattr(cli, "load_source_registry", lambda *_: pytest.fail("must reject first"))
    monkeypatch.setattr(sys, "argv", ["news-assistant", "audit-sources", "--sources", str(sources),
                                     "--output", str(sources)])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert sources.read_text() == "preserve original configuration"


def test_collect_signals_records_success_and_continues_after_source_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        """version: source-registry-v2
sources:
  - id: good
    name: Good V2EX
    tier: T3
    adapter: v2ex
    url: https://example.com/good
    topics: [ai]
    coverage_domains: [ai_technology]
    publisher_group: good
    language: zh
    terms_reviewed_at: 2026-08-17
    access_stability: public_api
  - id: bad
    name: Bad V2EX
    tier: T3
    adapter: v2ex
    url: https://example.com/bad
    topics: [ai]
    coverage_domains: [ai_technology]
    publisher_group: bad
    language: zh
    terms_reviewed_at: 2026-08-17
    access_stability: public_api
"""
    )
    state_dir = tmp_path / "state"

    def fetcher(source, *, loader, now):
        del loader
        if source.id == "bad":
            raise OSError("unavailable")
        return V2exFetchResult(
            snapshots=[
                SignalSnapshot(
                    source_id="good",
                    external_id="42",
                    captured_at=now,
                    rank=1,
                    reply_count=12,
                )
            ]
        )

    monkeypatch.setattr(cli, "fetch_v2ex_topics", fetcher, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "news-assistant",
            "collect-signals",
            "--sources",
            str(sources),
            "--state-dir",
            str(state_dir),
        ],
    )

    assert cli.main() == 0

    health_path = next((state_dir / "signals").glob("*/*/*/source-health.jsonl"))
    statuses = {item["source_id"]: item["status"] for item in map(json.loads, health_path.read_text().splitlines())}
    assert statuses == {"good": "ok", "bad": "error"}
    assert list((state_dir / "signals").glob("*/*/*/snapshots.jsonl"))


def test_collect_signals_observes_shadow_rss_without_enabling_public_selection(
    monkeypatch, tmp_path,
) -> None:
    from datetime import UTC, datetime

    from news_assistant.source_registry import RegisteredSource, SourceRegistry

    now = datetime(2026, 9, 21, 12, tzinfo=UTC)
    source = RegisteredSource(id="shadow", name="Shadow", tier="T1", adapter="rss",
                              url="https://example.com/rss", topics=["science"],
                              coverage_domains=["science_health"], publisher_group="example",
                              production_enabled=False)
    production = source.model_copy(update={"id": "production", "name": "Production",
                                           "production_enabled": True})
    broken = source.model_copy(update={"id": "broken", "name": "Broken",
                                       "url": "https://example.com/broken"})
    registry = SourceRegistry(version="source-registry-v1", sources=[source, production, broken])
    payload = b'''<rss version="2.0"><channel><title>News</title>
    <item><title>New science</title><link>https://example.com/science</link>
      <pubDate>Mon, 21 Sep 2026 11:00:00 GMT</pubDate></item>
    <item><title>Old science</title><link>https://example.com/old</link>
      <pubDate>Mon, 01 Sep 2026 11:00:00 GMT</pubDate></item></channel></rss>'''

    def load(url):
        return b"<html>Error page</html>" if url.endswith("broken") else payload

    monkeypatch.setattr(cli, "_http_loader", load)
    cli.collect_signals(registry, tmp_path / "state", now=now)
    partition = tmp_path / "state/signals/2026/09/21"
    health = [json.loads(row) for row in (partition / "source-health.jsonl").read_text().splitlines()]
    assert {row["source_id"]: (row["status"], row["candidate_count"]) for row in health} == {
        "shadow": ("ok", 1), "broken": ("error", 0),
    }
    snapshots = [json.loads(row) for row in (partition / "snapshots.jsonl").read_text().splitlines()]
    assert len(snapshots) == 1
    assert snapshots[0]["source_id"] == "shadow"
    assert registry.enabled_feed_sources() == [production]


def test_source_only_build_never_constructs_paid_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.credential_store, 'load_deepseek_key', lambda: pytest.fail('source-only must not access credentials'))
    import sys

    monkeypatch.setattr(sys, 'argv', ['news-assistant', 'build', '--source-only',
                                    '--output', str(tmp_path / 'public')])
    monkeypatch.setattr(cli, 'fetch_articles', lambda *_args, **_kwargs: [])
    def forbidden(*_args, **_kwargs):
        raise AssertionError('paid provider must not be constructed')
    monkeypatch.setattr(cli, 'DeepSeekClient', forbidden)
    assert cli.main() == 0
    assert (tmp_path / 'public' / 'index.html').exists()


def test_evaluate_state_command_keeps_existing_report_contract(tmp_path, monkeypatch):
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", ["news-assistant", "evaluate-state", "--state-dir",
                                    str(tmp_path / "state"), "--output", str(output)])
    assert cli.main() == 0
    assert json.loads(output.read_text())["schema_version"] == "state-evaluation-v1"


def test_evaluate_state_rejects_report_inside_public_output(tmp_path, monkeypatch):
    output = tmp_path / "public" / "evaluation.json"
    monkeypatch.setattr(sys, "argv", ["news-assistant", "evaluate-state", "--state-dir",
                                    str(tmp_path / "state"), "--output", str(output),
                                    "--public-dir", str(tmp_path / "public")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not output.exists()


def test_layered_cli_refreshes_signals_at_the_build_observation_time(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["news-assistant", "build", "--source-only",
                                    "--selection-engine", "layered_shadow_v1", "--state-dir",
                                    str(tmp_path / "state"), "--output", str(tmp_path / "public")])
    calls = []
    monkeypatch.setattr(cli, "collect_signals", lambda _registry, _state, *, now:
                        calls.append(("signals", now)))
    def fetch(_sources, *, now):
        calls.append(("fetch", now))
        return []
    monkeypatch.setattr(cli, "fetch_articles", fetch)
    monkeypatch.setattr(cli, "build_digest", lambda *_args, **kwargs:
                        calls.append(("build", kwargs["now"])))
    assert cli.main() == 0
    assert [name for name, _ in calls] == ["signals", "fetch", "build"]
    assert len({now for _, now in calls}) == 1
