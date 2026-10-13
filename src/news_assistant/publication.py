from __future__ import annotations

import re
from pathlib import Path


class UnsafePublicArtifactError(ValueError):
    """Raised when generated output is unsafe to publish."""


ALLOWED_SUFFIXES = {
    ".css",
    ".html",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".png",
    ".svg",
    ".webp",
}

TEXT_SUFFIXES = {".css", ".html", ".js", ".svg"}

FORBIDDEN_PATTERNS = {
    "credential name": re.compile(
        r"DEEPSEEK_API_KEY|GMAIL_ADDRESS|GMAIL_APP_PASSWORD|NEWS_RECIPIENT|"
        r"DIGEST_REPO_TOKEN"
    ),
    "raw reading events": re.compile(r"rawEvents|raw_events"),
    "event selection data": re.compile(
        r"estimated_cost|candidate_id|event_id|decisions\.jsonl|candidates\.jsonl|news-state"
    ),
    "local user path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
}


def validate_public_output(root: Path) -> None:
    """Reject existing symlinks before rendering can follow one on write."""
    if root.is_symlink():
        raise UnsafePublicArtifactError("public root must not be a symlink")
    if root.exists() and not root.is_dir():
        raise UnsafePublicArtifactError("public root must be a directory")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise UnsafePublicArtifactError(f"{path.name}: symlink is not publishable")


def audit_public_directory(root: Path) -> list[Path]:
    """Return sorted publishable files or reject unsafe generated output."""
    validate_public_output(root)
    if not root.is_dir():
        raise UnsafePublicArtifactError("public directory is missing")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise UnsafePublicArtifactError(f"{relative}: symlink is not publishable")
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise UnsafePublicArtifactError(f"{relative}: unsupported file type")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(errors="replace")
            for rule, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(text):
                    raise UnsafePublicArtifactError(f"{relative}: blocked by {rule}")
        files.append(path)

    if not any(path.relative_to(root).as_posix() == "index.html" for path in files):
        raise UnsafePublicArtifactError("index.html is missing")

    return files
