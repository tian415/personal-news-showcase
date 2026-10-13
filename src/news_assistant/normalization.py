from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def normalize_url(value: str) -> str:
    parts = urlsplit(value)
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
            if key.casefold() not in TRACKING_PARAMETERS
        ]
    )
    return urlunsplit((parts.scheme, parts.netloc.casefold(), parts.path.rstrip("/"), query, ""))


def normalize_title(value: str) -> str:
    collapsed = " ".join(value.split()).casefold()
    return re.sub(r"\s+[—–-]\s+[^—–-]+$", "", collapsed).strip()


def article_fingerprint(title: str, summary: str = "") -> str:
    value = f"{normalize_title(title)}\n{' '.join(summary.split()).casefold()}"
    return hashlib.sha256(value.encode()).hexdigest()
