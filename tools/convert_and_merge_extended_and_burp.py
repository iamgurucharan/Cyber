"""
Convert the extended advisory CSV into legacy-style base feature rows and merge with a Burp-import CSV.

Usage:
  python tools/convert_and_merge_extended_and_burp.py --extended data/extended_realistic_vulnerability_dataset_10000.csv --burp data/burp_import_training.csv --out data/combined_training.csv

The output is a legacy-style CSV with base feature columns + label + url ready for `train_model.py` when you set `CYBER_DATASET_PATH` to it.
"""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import _is_extended_schema, extended_csv_to_training_frame, BASE_FEATURE_NAMES


def convert_extended(path: Path):
    df = pd.read_csv(path, encoding='utf-8')
    if not _is_extended_schema(df.columns):
        raise SystemExit('Provided extended file does not look like the extended advisory schema')
    conv = extended_csv_to_training_frame(df)
    return conv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extended', required=True)
    p.add_argument('--burp', required=True)
    p.add_argument('--out', default=ROOT / 'data' / 'combined_training.csv')
    args = p.parse_args()
    ext_path = Path(args.extended)
    burp_path = Path(args.burp)
    out_path = Path(args.out)

    if not ext_path.is_file():
        print('Extended file missing:', ext_path)
        raise SystemExit(1)
    if not burp_path.is_file():
        print('Burp import CSV missing:', burp_path)
        raise SystemExit(1)

    ext_df = convert_extended(ext_path)
    burp_df = pd.read_csv(burp_path, encoding='utf-8')

    # Ensure burp_df has same columns as ext_df produced rows (it should: base features + label + url)
    # Reorder burp_df columns to match ext_df if possible
    common = [c for c in ext_df.columns if c in burp_df.columns]
    extra = [c for c in burp_df.columns if c not in ext_df.columns]
    if extra:
        # Drop unexpected columns
        burp_df = burp_df[common + [c for c in burp_df.columns if c in common]]
    # Concatenate
    merged = pd.concat([ext_df, burp_df], ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding='utf-8')
    print('Wrote combined CSV to', out_path)

if __name__ == '__main__':
    main()
