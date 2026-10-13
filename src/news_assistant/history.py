from __future__ import annotations

import re
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_DATED_PAGE = re.compile(r"^\d{4}-\d{2}-\d{2}\.html$")
_DIGEST_TIME_ZONE = ZoneInfo("Asia/Shanghai")


def preserve_history(
    published_dir: Path, output_dir: Path, *, before_date: date | None = None
) -> list[Path]:
    """Copy missing dated HTML pages from a prior public release.

    Only immediate files named with a valid ISO date are candidates. All source
    and destination paths are validated before the first copy so a malformed
    candidate cannot leave a partially restored output directory. Pages dated
    on or after ``before_date`` are skipped; when omitted, the current date in
    the digest timezone is used.
    """
    cutoff = before_date or datetime.now(_DIGEST_TIME_ZONE).date()
    _validate_root(published_dir, name="published_dir", required=True)
    _validate_root(output_dir, name="output_dir", required=False)

    published_root = published_dir.resolve()
    output_root = output_dir.resolve(strict=False)
    if (
        published_root == output_root
        or published_root.is_relative_to(output_root)
        or output_root.is_relative_to(published_root)
    ):
        raise ValueError("published_dir and output_dir must not overlap")

    dated_candidates = [
        (path, date.fromisoformat(path.stem))
        for path in sorted(published_dir.iterdir())
        if _DATED_PAGE.fullmatch(path.name) and _valid_date(path.stem)
    ]
    for source, _ in dated_candidates:
        if source.is_symlink():
            raise ValueError(f"{source}: symlink source is not allowed")
        if not source.is_file():
            raise ValueError(f"{source}: source must be a regular file")
    candidates = [path for path, page_date in dated_candidates if page_date < cutoff]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    to_copy: list[tuple[Path, Path]] = []
    for source in candidates:
        target = output_dir / source.name
        if target.is_symlink():
            raise ValueError(f"{target}: symlink target is not allowed")
        if target.exists() and not target.is_file():
            raise ValueError(f"{target}: target must be a regular file")
        if not target.exists():
            to_copy.append((source, target))

    for source, target in to_copy:
        shutil.copyfile(source, target)
    return [target for _, target in to_copy]


def _validate_root(path: Path, *, name: str, required: bool) -> None:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    if not path.exists():
        if required:
            raise ValueError(f"{name} is missing")
        return
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory")


def _valid_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True
