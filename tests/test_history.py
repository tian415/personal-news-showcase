from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from news_assistant.history import preserve_history


def test_copies_missing_dated_pages_byte_for_byte(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    content = b"\x00old\xff\n"
    source = published / "2026-09-10.html"
    source.write_bytes(content)
    source_hash = hashlib.sha256(content).hexdigest()

    copied = preserve_history(published, output, before_date=date(2026, 9, 21))

    assert copied == [output / "2026-09-10.html"]
    assert (output / "2026-09-10.html").read_bytes() == content
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_existing_output_page_is_never_overwritten(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    (published / "2026-09-12.html").write_bytes(b"old published page")
    current = output / "2026-09-12.html"
    current.write_bytes(b"new page from this run")

    assert preserve_history(published, output, before_date=date(2026, 9, 21)) == []
    assert current.read_bytes() == b"new page from this run"


def test_skips_today_and_future_pages_without_overwriting_today(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    (published / "2026-09-20.html").write_bytes(b"yesterday")
    (published / "2026-09-21.html").write_bytes(b"published today")
    (published / "2026-09-22.html").write_bytes(b"tomorrow")
    today = output / "2026-09-21.html"
    today.write_bytes(b"new page from this run")

    copied = preserve_history(published, output, before_date=date(2026, 9, 21))

    assert copied == [output / "2026-09-20.html"]
    assert (output / "2026-09-20.html").read_bytes() == b"yesterday"
    assert today.read_bytes() == b"new page from this run"
    assert not (output / "2026-09-22.html").exists()


def test_only_valid_dated_pages_are_copied(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    (published / "2026-02-30.html").write_bytes(b"invalid date")
    (published / "2026-1-2.html").write_bytes(b"not padded")
    (published / "index.html").write_bytes(b"index")
    (published / "private.json").write_bytes(b"private")
    (published / "assets").mkdir()

    assert preserve_history(published, output, before_date=date(2026, 9, 21)) == []
    assert list(output.iterdir()) == []


def test_source_page_symlink_is_rejected(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    target = tmp_path / "real.html"
    target.write_bytes(b"page")
    (published / "2026-09-10.html").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        preserve_history(published, output, before_date=date(2026, 9, 21))


def test_target_page_symlink_is_rejected(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    (published / "2026-09-10.html").write_bytes(b"page")
    target = tmp_path / "target.html"
    target.write_bytes(b"do not overwrite")
    (output / "2026-09-10.html").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        preserve_history(published, output, before_date=date(2026, 9, 21))

    assert target.read_bytes() == b"do not overwrite"


@pytest.mark.parametrize("root_name", ["published", "output"])
def test_root_symlink_is_rejected(tmp_path: Path, root_name: str) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    root = published if root_name == "published" else output
    root.rmdir()
    root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        preserve_history(published, output, before_date=date(2026, 9, 21))


def test_missing_published_directory_is_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="published_dir is missing"):
        preserve_history(
            tmp_path / "missing", tmp_path / "output", before_date=date(2026, 9, 21)
        )


@pytest.mark.parametrize(
    "relationship", ["same", "published_inside_output", "output_inside_published"]
)
def test_overlapping_roots_are_rejected(tmp_path: Path, relationship: str) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    if relationship == "same":
        published.mkdir()
        output = published
    elif relationship == "published_inside_output":
        output.mkdir()
        published = output / "published"
        published.mkdir()
    else:
        published.mkdir()
        output = published / "output"
        output.mkdir()

    with pytest.raises(ValueError, match="overlap"):
        preserve_history(published, output, before_date=date(2026, 9, 21))


def test_all_sources_are_validated_before_any_copy(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    (published / "2026-09-10.html").write_bytes(b"valid page")
    (published / "2026-09-11.html").mkdir()

    with pytest.raises(ValueError, match="regular file"):
        preserve_history(published, output, before_date=date(2026, 9, 21))

    assert list(output.iterdir()) == []


def test_no_valid_candidates_returns_empty_list(tmp_path: Path) -> None:
    published = tmp_path / "published"
    output = tmp_path / "output"
    published.mkdir()
    output.mkdir()
    (published / "notes.txt").write_bytes(b"ignore")

    assert preserve_history(published, output, before_date=date(2026, 9, 21)) == []
