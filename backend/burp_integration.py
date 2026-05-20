"""
Burp Suite integration: optional REST API, file import stub, or simulated metrics.

Live Burp Enterprise/Scanner REST APIs require licensed deployments and correct
extensions; environment variables:

  BURP_API_URL   — base URL of API (optional)
  BURP_API_KEY   — API token (optional)

When the API is unavailable or returns an error, this module **simulates**
Burp-like scores from passive scan signals plus small random variance.
The field `burp_source` is always one of: ``api`` | ``import`` | ``simulated``.
"""

from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

from backend.website_scanner import ScanResult


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def simulate_from_scan(
    features: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Produce Burp-like numeric fields; clearly synthetic.
    Uses scanner-derived signals only (no real Burp run).
    """
    if seed is not None:
        random.seed(seed)
    # Heuristic score from headers / transport
    h = (
        features.get("has_hsts", 0)
        + features.get("has_csp", 0)
        + features.get("has_xframe", 0)
        + features.get("has_xss_protection", 0)
    )
    ssl_ok = features.get("ssl_valid", 0)
    mixed = features.get("mixed_content", 0)
    insecure_ck = features.get("insecure_cookies", 0)
    base = 55.0 - 8 * h - 10 * ssl_ok + 12 * mixed + 8 * insecure_ck
    base += (features.get("external_scripts_count", 0) or 0) * 0.8
    base += (features.get("forms_count", 0) or 0) * 0.5
    noise = random.uniform(-4.0, 4.0)
    score = int(_clamp(base + noise, 5.0, 99.0))

    # Issue counts loosely correlated (not real Burp)
    crit = 1 if score > 85 and random.random() > 0.7 else 0
    high = max(0, int((score - 40) / 25) + random.randint(0, 2))
    med = max(0, int(score / 20) + random.randint(0, 3))
    low = max(0, int(score / 10) + random.randint(0, 5))

    return {
        "burp_scan_score": score,
        "burp_issues_critical": crit,
        "burp_issues_high": high,
        "burp_issues_medium": med,
        "burp_issues_low": low,
        "burp_source": "simulated",
        "burp_note": "Synthetic Burp-like metrics from scanner heuristics + noise. Not from Burp Suite.",
    }


def try_burp_api() -> dict[str, Any] | None:
    base = os.environ.get("BURP_API_URL", "").rstrip("/")
    key = os.environ.get("BURP_API_KEY", "")
    if not base or not key:
        return None
    try:
        # Generic placeholder path — real Burp REST paths vary by product/version.
        r = requests.get(
            f"{base}/health",
            headers={"Authorization": f"Bearer {key}"},
            timeout=8,
        )
        if r.status_code >= 400:
            return None
        return None  # success ping but no scan payload in this stub
    except requests.RequestException:
        return None


def parse_burp_xml(path: Path) -> dict[str, Any] | None:
    """
    Minimal Burp XML stub: looks for issue counts in Burp-style exports.
    Many Burp XML formats exist; this attempts a forgiving parse.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return None

    # Try common Burp Scanner XML issue tags
    issues = root.findall(".//issue") or root.findall(".//{*}issue")
    if not issues:
        # fallback: count severities by attribute/text
        issues = root.findall(".//*[@severity]")

    crit = high = med = low = 0
    for iss in issues:
        sev = (
            iss.get("severity")
            or iss.findtext("severity", default="")
            or iss.findtext("{*}severity", default="")
        )
        sev_l = (sev or "").lower()
        if "high" in sev_l and "very" not in sev_l:
            high += 1
        elif "critical" in sev_l or "high" in sev_l and "information" not in sev_l:
            if "critical" in sev_l:
                crit += 1
            else:
                high += 1
        elif "medium" in sev_l or "moderate" in sev_l:
            med += 1
        elif "low" in sev_l or "info" in sev_l:
            low += 1
        else:
            low += 1

    total = max(1, crit + high + med + low)
    score = int(_clamp(30 + crit * 12 + high * 6 + med * 2 + low * 0.5, 1, 100))

    return {
        "burp_scan_score": score,
        "burp_issues_critical": crit,
        "burp_issues_high": high,
        "burp_issues_medium": med,
        "burp_issues_low": low,
        "burp_source": "import",
        "burp_note": "Parsed from uploaded Burp XML (best-effort); verify format matches your export.",
    }


def parse_burp_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    # Very loose: look for nested issue lists
    issues = data.get("issues") or data.get("findings") or []
    if isinstance(issues, dict):
        issues = issues.get("items", [])
    if not isinstance(issues, list):
        return None
    crit = high = med = low = 0
    for it in issues:
        if not isinstance(it, dict):
            continue
        s = str(it.get("severity", "")).lower()
        if "critical" in s:
            crit += 1
        elif "high" in s:
            high += 1
        elif "medium" in s:
            med += 1
        else:
            low += 1
    total_i = max(1, crit + high + med + low)
    score = int(_clamp(25 + crit * 15 + high * 7 + med * 3 + low, 1, 100))
    return {
        "burp_scan_score": score,
        "burp_issues_critical": crit,
        "burp_issues_high": high,
        "burp_issues_medium": med,
        "burp_issues_low": low,
        "burp_source": "import",
        "burp_note": "Parsed from uploaded Burp JSON (heuristic keys).",
    }


def merge_burp_into_features(
    scan: ScanResult,
    burp_report_path: str | None = None,
    simulate_seed: int | None = None,
) -> dict[str, Any]:
    """
    Merge scanner base features with Burp block. Priority: import file > API (stub) > simulated.
    """
    feats = dict(scan.features)
    if burp_report_path:
        p = Path(burp_report_path)
        if p.is_file():
            if p.suffix.lower() == ".xml":
                parsed = parse_burp_xml(p)
            elif p.suffix.lower() == ".json":
                parsed = parse_burp_json(p)
            else:
                parsed = None
            if parsed:
                feats.update(
                    {
                        "burp_scan_score": parsed["burp_scan_score"],
                        "burp_issues_critical": parsed["burp_issues_critical"],
                        "burp_issues_high": parsed["burp_issues_high"],
                        "burp_issues_medium": parsed["burp_issues_medium"],
                        "burp_issues_low": parsed["burp_issues_low"],
                    }
                )
                meta = {
                    "burp_source": parsed["burp_source"],
                    "burp_note": parsed["burp_note"],
                }
                return {**meta, "features": feats}

    api = try_burp_api()
    if api and api.get("burp_scan_score") is not None:
        feats.update(api)
        return {
            "burp_source": "api",
            "burp_note": "Burp API response (stub integration — extend for your deployment).",
            "features": feats,
        }

    sim = simulate_from_scan(feats, seed=simulate_seed)
    feats.update(
        {
            "burp_scan_score": sim["burp_scan_score"],
            "burp_issues_critical": sim["burp_issues_critical"],
            "burp_issues_high": sim["burp_issues_high"],
            "burp_issues_medium": sim["burp_issues_medium"],
            "burp_issues_low": sim["burp_issues_low"],
        }
    )
    return {
        "burp_source": sim["burp_source"],
        "burp_note": sim["burp_note"],
        "features": feats,
    }
