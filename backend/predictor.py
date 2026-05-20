"""
Load persisted scaler + RandomForest and produce risk scores (P(Not Secure)).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    BASE_FEATURE_NAMES,
    add_derived_features,
    get_engineered_feature_names,
)


class RiskPredictor:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else ROOT / "models"
        self._clf = None
        self._scaler = None
        self._artifacts: dict | None = None
        self.feature_names: list[str] = get_engineered_feature_names()

    def load(self) -> None:
        self._clf = joblib.load(self.models_dir / "rf_model.joblib")
        self._scaler = joblib.load(self.models_dir / "scaler.joblib")
        self._artifacts = joblib.load(self.models_dir / "artifacts.joblib")
        saved = list(self._artifacts.get("feature_names", []))
        if saved and saved != self.feature_names:
            self.feature_names = saved

    @property
    def ready(self) -> bool:
        return (
            (self.models_dir / "rf_model.joblib").is_file()
            and (self.models_dir / "scaler.joblib").is_file()
        )

    def ensure_loaded(self) -> None:
        if self._clf is None:
            if not self.ready:
                raise FileNotFoundError(
                    "Model artifacts missing. Run: python run_pipeline.py"
                )
            self.load()

    def vector_from_base_row(self, row: dict) -> np.ndarray:
        """Build one engineered feature row from a dict of base CSV columns."""
        df = pd.DataFrame([row])
        for c in BASE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0
        df = add_derived_features(df)
        return df[self.feature_names].values.astype(np.float64)

    def predict_proba_row(self, row: dict) -> tuple[float, int, np.ndarray]:
        self.ensure_loaded()
        X = self.vector_from_base_row(row)
        Xs = self._scaler.transform(X)
        proba = self._clf.predict_proba(Xs)[0]
        risk = float(proba[1])
        pred_class = int(self._clf.predict(Xs)[0])
        return risk, pred_class, proba

    def explain_confusion_cells(self) -> str:
        return (
            "Confusion matrix (binary: 0=Secure, 1=Not Secure): "
            "TN — true Secure predicted Secure; FP — true Secure predicted Not Secure; "
            "FN — true Not Secure predicted Secure; TP — true Not Secure predicted Not Secure. "
            "High TP and TN with low FP/FN indicate good calibration for this dataset."
        )


def load_top_features_report() -> dict | None:
    p = ROOT / "reports" / "top_features.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
