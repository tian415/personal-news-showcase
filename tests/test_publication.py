from pathlib import Path

import pytest

from news_assistant.publication import UnsafePublicArtifactError, audit_public_directory


def test_audit_returns_only_supported_public_files(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<h1>News</h1>")
    (tmp_path / "assets" / "styles.css").write_text("body{}")
    (tmp_path / "assets" / "feedback.js").write_text(
        "localStorage.setItem('x', 'y')"
    )

    assert [
        path.relative_to(tmp_path).as_posix()
        for path in audit_public_directory(tmp_path)
    ] == [
        "assets/feedback.js",
        "assets/styles.css",
        "index.html",
    ]


@pytest.mark.parametrize(
    "content",
    [
        "DEEPSEEK_API_KEY",
        "GMAIL_ADDRESS",
        "GMAIL_APP_PASSWORD",
        "NEWS_RECIPIENT",
        "DIGEST_REPO_TOKEN",
        "/Users/example/work/project",
        "/home/runner/work/project",
        '"rawEvents": []',
        '"raw_events": []',
    ],
)
def test_audit_rejects_private_markers(tmp_path: Path, content: str) -> None:
    (tmp_path / "index.html").write_text(content)

    with pytest.raises(UnsafePublicArtifactError):
        audit_public_directory(tmp_path)


def test_audit_rejects_unsupported_file_type(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>News</h1>")
    (tmp_path / "debug.json").write_text('{"email": "private@example.com"}')

    with pytest.raises(UnsafePublicArtifactError, match="unsupported file type"):
        audit_public_directory(tmp_path)


def test_audit_rejects_symlinks(tmp_path: Path) -> None:
    private = tmp_path.parent / "private.txt"
    private.write_text("secret")
    (tmp_path / "index.html").symlink_to(private)

    with pytest.raises(UnsafePublicArtifactError, match="symlink"):
        audit_public_directory(tmp_path)


def test_audit_requires_index_page(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "styles.css").write_text("body{}")

    with pytest.raises(UnsafePublicArtifactError, match="index.html is missing"):
        audit_public_directory(tmp_path)


def test_audit_rejects_private_event_selection_fields(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text('{"estimated_cost": 0.02}')

    with pytest.raises(UnsafePublicArtifactError, match="event selection data"):
        audit_public_directory(tmp_path)
