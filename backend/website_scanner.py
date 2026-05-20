"""
Passive HTTP/TLS heuristics for mapping a live URL into the training feature schema.

**Important:** Only request URLs you are authorized to test. This module performs
lightweight GET requests; it does not replace a full security assessment.
"""

from __future__ import annotations

import re
import ssl
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_TIMEOUT = 12


def _xss_header_value(raw: str | None) -> int:
    if not raw:
        return 0
    first = raw.split(";")[0].strip()
    if first == "0":
        return 0
    return 1


def _count_forms(html: str) -> int:
    return len(re.findall(r"<form\b", html, re.IGNORECASE))


def _count_external_scripts(html: str, base_url: str) -> int:
    host = urlparse(base_url).netloc.lower()
    count = 0
    for m in re.finditer(
        r'<script[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE
    ):
        src = m.group(1).strip()
        if src.startswith(("http://", "https://")):
            if urlparse(src).netloc.lower() != host:
                count += 1
        elif src.startswith("//"):
            full = "https:" + src
            if urlparse(full).netloc.lower() != host:
                count += 1
    return count


def _ssl_expiry_days(hostname: str, port: int = 443) -> tuple[int, int]:
    """
    Returns (ssl_valid, ssl_expiry_days). On failure returns (0, 0).
    """
    ctx = ssl.create_default_context()
    try:
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(DEFAULT_TIMEOUT)
            s.connect((hostname, port))
            cert = s.getpeercert()
        if not cert:
            return 0, 0
        not_after = cert.get("notAfter")
        if not not_after:
            return 1, 365
        # cert format e.g. 'Jan  9 12:00:00 2027 GMT'
        exp = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(
            tzinfo=timezone.utc
        )
        days = max(0, int((exp - datetime.now(timezone.utc)).total_seconds() // 86400))
        return 1, days
    except OSError:
        return 0, 0


@dataclass
class ScanResult:
    """Base-schema fields (before derived features)."""

    url_input: str
    final_url: str
    error: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    raw_headers: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def scan_url(url: str) -> ScanResult:
    """
    Fetch URL (GET), collect headers and HTML heuristics.
    Undetectable signals without active scanning use conservative defaults (0).
    """
    notes: list[str] = []
    defaults: dict[str, Any] = {
        "ssl_valid": 0,
        "ssl_expiry_days": 0,
        "has_hsts": 0,
        "has_csp": 0,
        "has_xframe": 0,
        "has_xss_protection": 0,
        "open_ports_count": 1,
        "critical_cves": 0,
        "high_cves": 0,
        "medium_cves": 0,
        "low_cves": 0,
        "sql_injection_detected": 0,
        "xss_detected": 0,
        "csrf_missing": 0,
        "auth_weak": 0,
        "insecure_cookies": 0,
        "mixed_content": 0,
        "server_banner_exposed": 0,
        "directory_listing": 0,
        "default_credentials": 0,
        "response_time_ms": 0,
        "content_length": 0,
        "forms_count": 0,
        "external_scripts_count": 0,
        "redirects_to_http": 0,
    }

    if not url.strip():
        return ScanResult(url_input=url, final_url="", error="Empty URL", features=defaults)

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "CyberRiskScanner/1.0 (authorized assessment only)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        }
    )

    try:
        t0 = datetime.now(timezone.utc)
        resp = session.get(
            url,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=True,
            verify=True,
        )
        t1 = datetime.now(timezone.utc)
        response_time_ms = int((t1 - t0).total_seconds() * 1000)
    except requests.RequestException as e:
        notes.append(f"Request failed: {e}")
        return ScanResult(
            url_input=url,
            final_url=url,
            error=str(e),
            features=defaults,
            notes=notes,
        )

    final_url = str(resp.url)
    parsed = urlparse(final_url)
    headers = {k.lower(): v for k, v in resp.headers.items()}

    h_lower = {k.lower(): v for k, v in resp.headers.items()}
    # normalize for helper funcs expecting Title-Case keys — use lower everywhere
    def gh(key: str) -> str | None:
        return h_lower.get(key.lower())

    hdrs_display = {k: v for k, v in resp.headers.items()}

    insecure_cookie = 0
    set_cookie = gh("set-cookie")
    if set_cookie:
        if "secure" not in set_cookie.lower():
            insecure_cookie = 1
        if "httponly" not in set_cookie.lower():
            insecure_cookie = 1

    mixed = 0
    if final_url.startswith("https://") and "http://" in (resp.text or "")[:50000]:
        mixed = 1

    server_banner = 1 if gh("server") else 0

    html = resp.text or ""
    forms_count = _count_forms(html)
    ext_scripts = _count_external_scripts(html, final_url)

    redirects_to_http = 0
    if final_url.startswith("https://"):
        for h in resp.history:
            if str(h.url).startswith("http://"):
                redirects_to_http = 1
                break

    ssl_valid, ssl_days = 0, 0
    if parsed.scheme == "https" and parsed.hostname:
        ssl_valid, ssl_days = _ssl_expiry_days(parsed.hostname, 443)

    feats = dict(defaults)
    feats.update(
        {
            "ssl_valid": ssl_valid,
            "ssl_expiry_days": ssl_days,
            "has_hsts": 1 if gh("strict-transport-security") else 0,
            "has_csp": 1 if gh("content-security-policy") else 0,
            "has_xframe": 1 if gh("x-frame-options") else 0,
            "has_xss_protection": _xss_header_value(gh("x-xss-protection")),
            "response_time_ms": response_time_ms,
            "content_length": len(resp.content),
            "forms_count": forms_count,
            "external_scripts_count": ext_scripts,
            "redirects_to_http": redirects_to_http,
            "insecure_cookies": insecure_cookie,
            "mixed_content": mixed,
            "server_banner_exposed": server_banner,
        }
    )

    notes.append(
        "CVE counts and injection findings are not inferred from passive fetch; left at 0."
    )
    notes.append(
        "Burp metrics are not produced by this scanner; use Burp integration block."
    )

    return ScanResult(
        url_input=url,
        final_url=final_url,
        features=feats,
        raw_headers={k: v for k, v in hdrs_display.items()},
        notes=notes,
    )
