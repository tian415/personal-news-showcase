from datetime import UTC, date, datetime

from news_assistant.models import Digest, EnrichedArticle, Topic
from news_assistant.render import render_digest


def test_public_artifacts_exclude_private_fields(tmp_path) -> None:
    item = EnrichedArticle(id="safe", title="Story", url="https://example.com", source="Source", language="en", published_at=datetime.now(UTC), topic=Topic.AI, chinese_title="标题", chinese_summary="摘要")
    render_digest(Digest(digest_date=date(2026, 8, 12), articles=[item]), tmp_path)
    public_text = "\n".join(path.read_text() for path in tmp_path.rglob("*") if path.is_file())
    for forbidden in ("GMAIL_APP_PASSWORD", "DEEPSEEK_API_KEY", "gmail_address", "rawEvents"):
        assert forbidden not in public_text


def test_public_audit_rejects_root_symlink(tmp_path):
    import pytest

    from news_assistant.publication import audit_public_directory

    private = tmp_path / 'private'
    private.mkdir()
    (private / 'index.html').write_text('private')
    output = tmp_path / 'public'
    output.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        audit_public_directory(output)
