"""
Convert Burp XML/JSON exports into a CSV suitable for training.

Usage:
  python tools/import_burp_for_training.py --in-dir path/to/burp_files --out data/burp_import_training.csv

Labeling rule (default): any critical/high issue -> Not Secure, otherwise Secure.
This is a heuristic; provide manual labels if you have human-labeled ground truth.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import csv

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.burp_integration import parse_burp_xml, parse_burp_json
from src.feature_engineering import BASE_FEATURE_NAMES, LABEL_POSITIVE, LABEL_NEGATIVE


def parse_file(path: Path) -> Optional[dict]:
    if path.suffix.lower() == ".xml":
        return parse_burp_xml(path)
    if path.suffix.lower() == ".json":
        return parse_burp_json(path)
    return None


def make_row(parsed: dict, filename: str) -> dict:
    # Build a base-default row; other scanner-derived fields unknown -> defaults
    row = {c: 0 for c in BASE_FEATURE_NAMES}
    row["url"] = f"imported://{filename}"
    # Fill burp fields if present
    if parsed is None:
        row["burp_scan_score"] = 0
        row["burp_issues_critical"] = 0
        row["burp_issues_high"] = 0
        row["burp_issues_medium"] = 0
        row["burp_issues_low"] = 0
    else:
        row["burp_scan_score"] = int(parsed.get("burp_scan_score", 0))
        row["burp_issues_critical"] = int(parsed.get("burp_issues_critical", 0))
        row["burp_issues_high"] = int(parsed.get("burp_issues_high", 0))
        row["burp_issues_medium"] = int(parsed.get("burp_issues_medium", 0))
        row["burp_issues_low"] = int(parsed.get("burp_issues_low", 0))
    # Simple heuristic label
    if row["burp_issues_critical"] > 0 or row["burp_issues_high"] > 0:
        row["label"] = LABEL_POSITIVE
    else:
        row["label"] = LABEL_NEGATIVE
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, help="Directory with Burp XML/JSON files")
    p.add_argument("--out", default=ROOT / "data" / "burp_import_training.csv", help="Output CSV path")
    args = p.parse_args()
    indir = Path(args.in_dir)
    out = Path(args.out)
    if not indir.is_dir():
        print("Input directory not found:", indir)
        raise SystemExit(1)

    files = sorted([x for x in indir.iterdir() if x.is_file()])
    if not files:
        print("No files in:", indir)
        raise SystemExit(1)

    # Write header: BASE_FEATURE_NAMES + ['url','label'] ensuring order
    header = list(BASE_FEATURE_NAMES) + ["url", "label"]
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for fp in files:
            parsed = parse_file(fp)
            row = make_row(parsed, fp.name)
            # Ensure all header keys present
            out_row = {k: row.get(k, 0) for k in header}
            writer.writerow(out_row)
            written += 1
    print(f"Wrote {written} rows to {out}")

if __name__ == "__main__":
    main()
