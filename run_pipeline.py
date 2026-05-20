#!/usr/bin/env python3
"""
End-to-end ML pipeline: train → interpolation/extrapolation report → CV & edge tests.
Run from project root:

    python run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.dataset_config import resolve_dataset_path
    from src.train_model import train
    from src.interpolation_extrapolation import run_analysis
    from src.model_testing import edge_case_tests, run_cv

    dataset_path = resolve_dataset_path()
    print("== Cybersecurity Risk Prediction pipeline ==")
    print(f"Dataset: {dataset_path}")
    print("[1/3] Training Random Forest + metrics + confusion matrix…")
    train()
    print("[2/3] Interpolation vs extrapolation analysis…")
    run_analysis()
    print("[3/3] Stratified K-fold CV + edge-case checks…")
    run_cv()
    edge_case_tests()
    print("Done. Artifacts in models/ and reports/")
    print("Start API: python backend/app.py")


if __name__ == "__main__":
    main()
