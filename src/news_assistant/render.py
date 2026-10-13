from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo

from jinja2 import Environment, PackageLoader, select_autoescape

from .models import Digest
from .publication import validate_public_output


def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("news_assistant"),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml", "j2"),
            default_for_string=True,
        ),
    )


def article_anchor(article_id: str) -> str:
    return f"article-{quote(article_id, safe='')}"


def daily_digest_url(base_url: str, digest_date: date) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in base_url
        or "#" in base_url
    ):
        raise ValueError("digest base URL must be an absolute HTTP(S) URL without query or fragment")
    return f"{base_url.rstrip('/')}/{digest_date.isoformat()}.html"


def render_digest(digest: Digest, output_dir: Path) -> Path:
    validate_public_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    package_dir = Path(__file__).parent
    (assets / "styles.css").write_text((package_dir / "static" / "styles.css").read_text())
    (assets / "feedback.js").write_text((package_dir / "static" / "feedback.js").read_text())
    template = _environment().get_template("digest.html.j2")
    html = template.render(digest=digest, quote=quote, article_anchor=article_anchor)
    dated = output_dir / f"{digest.digest_date.isoformat()}.html"
    dated.write_text(html)
    (output_dir / "index.html").write_text(html)
    render_archive(output_dir, through=digest.digest_date)
    return dated


def render_archive(output_dir: Path, *, through: date | None = None) -> Path:
    cutoff = through or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    dates = []
    for path in output_dir.glob("????-??-??.html"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            value = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if value <= cutoff:
            dates.append(value.isoformat())
    target = output_dir / "archive.html"
    if target.is_symlink():
        raise ValueError("archive target must not be a symlink")
    target.write_text(_environment().get_template("archive.html.j2").render(
        dates=sorted(dates, reverse=True),
    ))
    return target


def render_email_digest(
    digest: Digest, output_path: Path, digest_base_url: str
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = _environment().get_template("email.html.j2")
    html = template.render(
        digest=digest,
        daily_url=daily_digest_url(digest_base_url, digest.digest_date),
        article_anchor=article_anchor,
        quote=quote,
    )
    output_path.write_text(html)
    return output_path
