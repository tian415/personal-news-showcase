"""Bounded public HTML acquisition and attributed, private article-body snapshots.

Extraction is not fact verification. RSS remains unchanged and legacy consumers
continue to use it. Only the explicit editorial path consumes captured bodies.
"""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import httpx
import trafilatura
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 20_000


class ContentFetchError(ValueError):
    def __init__(self, reason: str, *, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class PageCapture:
    final_url: str
    html: bytes


class CapturedContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["extracted", "rss_fallback", "skipped"]
    reason: Literal["http_error", "timeout", "network_error", "blocked_url", "redirect_limit",
                    "non_html", "response_too_large", "empty_extraction", "extractor_error",
                    "budget_limit", "compressed_response"] | None = None
    requested_url: HttpUrl
    final_url: HttpUrl | None = None
    retrieved_at: datetime
    extractor: Literal["trafilatura"] = "trafilatura"
    extractor_version: str = Field(min_length=1)
    text: str = Field(default="", max_length=MAX_TEXT_CHARS)
    text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    html_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    truncated: bool = Field(default=False, strict=True)
    http_status: int | None = Field(default=None, ge=100, le=599, strict=True)

    @model_validator(mode="after")
    def consistent_evidence(self) -> CapturedContent:
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("content capture time must be timezone-aware")
        if self.status == "extracted":
            if (not self.text.strip() or self.reason or not self.html_sha256 or not self.final_url
                    or self.text_sha256 != hashlib.sha256(self.text.encode()).hexdigest()):
                raise ValueError("invalid extracted content evidence")
        elif (self.text or self.text_sha256 or self.truncated or not self.reason):
            raise ValueError("fallback cannot claim extracted body")
        if (self.status == "skipped") != (self.reason == "budget_limit"):
            raise ValueError("invalid skipped content reason")
        return self


class ContentCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["article-content-v1"] = "article-content-v1"
    items: dict[str, CapturedContent]


def captured_contents(audit: dict, snapshot) -> dict[str, CapturedContent]:
    """Validate an optional sidecar against the actual candidate identities."""
    if "article_content" not in audit:
        return {}
    capture = ContentCapture.model_validate(audit["article_content"])
    expected = {f"{s.id}:{a.id}": a for s in snapshot.sources for a in s.candidates}
    if capture.items.keys() != expected.keys():
        raise ValueError("content capture must cover each candidate exactly once")
    for identifier, item in capture.items.items():
        if str(item.requested_url) != str(expected[identifier].url):
            raise ValueError("content capture URL does not match candidate")
        if item.retrieved_at < snapshot.captured_at:
            raise ValueError("body capture predates candidate snapshot")
    return capture.items


def _cloudflare_addresses(host: str, client: httpx.Client) -> list[str]:
    # Fixed HTTPS resolver is the explicit bootstrap trust boundary. Never follow
    # its redirects or disable certificate verification; only domain is sent.
    with client.stream("GET", "https://cloudflare-dns.com/dns-query",
                       params={"name": host, "type": "A"}, follow_redirects=False,
                       headers={"Accept": "application/dns-json",
                                "Accept-Encoding": "identity"}) as response:
        if (response.status_code != 200
                or response.headers.get("content-encoding", "identity").lower() != "identity"):
            raise ContentFetchError("network_error")
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=4096):
            if len(body) + len(chunk) > 65536:
                raise ContentFetchError("network_error")
            body.extend(chunk)
    try:
        result = json.loads(body)
        if not isinstance(result, dict) or result.get("Status") != 0 or result.get("TC", False):
            raise ValueError("invalid DNS answer")
        return [str(ipaddress.IPv4Address(row["data"])) for row in result.get("Answer", [])
                if row["type"] == 1]
    except (ValueError, TypeError, KeyError):
        raise ContentFetchError("network_error") from None


def _public_address(url: httpx.URL, *, client: httpx.Client | None = None,
                    dns_mode: str = "system") -> str:
    """Validate every hop and pin one public DNS answer for the HTTP connection."""
    if (url.scheme not in {"http", "https"} or not url.host or url.username or url.password
            or url.port not in {None, 80, 443} or url.host.endswith((".localhost", ".local"))):
        raise ContentFetchError("blocked_url")
    try:
        literal = ipaddress.ip_address(url.host)
    except ValueError:
        if dns_mode == "cloudflare" and client is not None:
            addresses = _cloudflare_addresses(url.host, client)
        else:
            try:
                addresses = [row[4][0] for row in socket.getaddrinfo(
                    url.host, url.port or (443 if url.scheme == "https" else 80),
                    type=socket.SOCK_STREAM)]
            except OSError:
                raise ContentFetchError("network_error") from None
    else:
        addresses = [str(literal)]
    if not addresses or any(not ipaddress.ip_address(a).is_global
                            or ipaddress.ip_address(a).is_multicast for a in addresses):
        raise ContentFetchError("blocked_url")
    return addresses[0]


def fetch_public_html(
    url: str, *, client: httpx.Client | None = None, max_bytes: int = MAX_HTML_BYTES,
    dns_mode: str = "system",
) -> PageCapture:
    if dns_mode not in {"system", "cloudflare"}:
        raise ValueError("unsupported DNS mode")
    if client is None:
        with httpx.Client(timeout=15, follow_redirects=False, trust_env=False) as owned:
            return fetch_public_html(url, client=owned, max_bytes=max_bytes, dns_mode=dns_mode)
    try:
        current = httpx.URL(url)
        for hop in range(4):
            address = _public_address(current, client=client, dns_mode=dns_mode)
            # Host and TLS SNI retain the publisher identity; connection uses the
            # validated IP, preventing a second DNS lookup/rebinding at connect.
            target = current.copy_with(host=address)
            with client.stream("GET", target, follow_redirects=False,
                               headers={"Host": current.netloc.decode(),
                                        "User-Agent": "PersonalNewsAssistant/0.1",
                                        "Accept": "text/html,application/xhtml+xml",
                                        "Accept-Encoding": "identity"},
                               extensions={"sni_hostname": current.host}) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    if hop == 3 or "location" not in response.headers:
                        raise ContentFetchError("redirect_limit")
                    current = current.join(response.headers["location"])
                    continue
                if response.status_code != 200:
                    raise ContentFetchError("http_error", status_code=response.status_code)
                media_type = response.headers.get("content-type", "").split(";")[0].lower()
                if media_type not in {"text/html", "application/xhtml+xml"}:
                    raise ContentFetchError("non_html")
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ContentFetchError("compressed_response")
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > max_bytes:
                        raise ContentFetchError("response_too_large")
                    body.extend(chunk)
                return PageCapture(str(current), bytes(body))
    except httpx.TimeoutException:
        raise ContentFetchError("timeout") from None
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as error:
        if isinstance(error, ContentFetchError):
            raise
        raise ContentFetchError("network_error") from None
    raise ContentFetchError("redirect_limit")


def extract_snapshot_content(
    audit: dict, *, loader: Callable[[str], PageCapture] | None = None,
    extractor: Callable[[bytes], str | None] | None = None,
    now: datetime | None = None, max_articles: int = 20, dns_mode: str = "system",
) -> dict:
    from .coverage_evaluation import _AuditSnapshot, _aware
    from .reviewed_selection import prepare_review

    if type(max_articles) is not int or not 1 <= max_articles <= 200:
        raise ValueError("max_articles must be an integer from 1 to 200")
    snapshot = _AuditSnapshot.model_validate(audit)
    prepare_review(audit)  # Reject conflicting IDs/metadata before any network.
    if "article_content" in audit:
        raise ValueError("snapshot already has body capture; use the original RSS snapshot")
    reference = _aware(now or datetime.now(UTC))
    if reference < snapshot.captured_at:
        raise ValueError("body capture cannot precede candidate snapshot")
    if dns_mode not in {"system", "cloudflare"}:
        raise ValueError("unsupported DNS mode")
    load = loader or (fetch_public_html if dns_mode == "system" else
                      lambda url: fetch_public_html(url, dns_mode=dns_mode))
    extract = extractor or (lambda data: trafilatura.extract(data, include_comments=False))
    extractor_version = version("trafilatura")
    cached: dict[str, CapturedContent] = {}
    items: dict[str, CapturedContent] = {}
    for source in snapshot.sources:
        for article in source.candidates:
            url = str(article.url)
            if url not in cached:
                common = {"requested_url": url, "retrieved_at": reference,
                          "extractor_version": extractor_version}
                if len(cached) >= max_articles:
                    item = CapturedContent(status="skipped", reason="budget_limit", **common)
                else:
                    try:
                        page = load(url)
                        if len(page.html) > MAX_HTML_BYTES:
                            raise ContentFetchError("response_too_large")
                    except ContentFetchError as error:
                        item = CapturedContent(status="rss_fallback", reason=error.reason,
                                               http_status=error.status_code, **common)
                    else:
                        evidence = {"final_url": page.final_url,
                                    "html_sha256": hashlib.sha256(page.html).hexdigest()}
                        try:
                            text = (extract(page.html) or "").strip()
                        except Exception:  # noqa: BLE001 - isolate third-party parser failures
                            item = CapturedContent(status="rss_fallback", reason="extractor_error",
                                                   **evidence, **common)
                        else:
                            if text:
                                excerpt = text[:MAX_TEXT_CHARS]
                                item = CapturedContent(
                                    status="extracted", text=excerpt,
                                    text_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
                                    truncated=len(text) > MAX_TEXT_CHARS, **evidence, **common)
                            else:
                                item = CapturedContent(status="rss_fallback",
                                                       reason="empty_extraction",
                                                       **evidence, **common)
                cached[url] = item
            items[f"{source.id}:{article.id}"] = cached[url]
    result = copy.deepcopy(audit)
    result["article_content"] = ContentCapture(items=items).model_dump(mode="json")
    captured_contents(result, snapshot)
    return result


def write_content_snapshot(path: Path, text: str) -> None:
    """Publish a complete private file atomically, without replacing any target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.link(temp_path, path)
        finally:
            temp_path.unlink()
