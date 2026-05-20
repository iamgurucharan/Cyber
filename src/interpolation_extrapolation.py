"""
Interpolation vs extrapolation analysis (distribution shift proxy).

- **Interpolation**: test samples where every feature value falls within the
  training distribution's 10th–90th percentile *per feature* (inclusive).
- **Extrapolation**: remaining test samples (at least one feature outside that band).

Optional synthetic extrapolation rows: random values with at least one dimension
outside train p10–p90.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import load_and_extract  # noqa: E402


def _percentile_bounds(X_train: np.ndarray, low: float = 10.0, high: float = 90.0):
    p_low = np.percentile(X_train, low, axis=0)
    p_high = np.percentile(X_train, high, axis=0)
    return p_low, p_high


def interpolation_mask(X: np.ndarray, p_low: np.ndarray, p_high: np.ndarray) -> np.ndarray:
    """Boolean mask: True if all features within [p_low, p_high]."""
    in_band = (X >= p_low) & (X <= p_high)
    return in_band.all(axis=1)


def generate_synthetic_extrapolation(
    X_train: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Feature vectors with at least one coordinate outside train p10–p90."""
    p_low, p_high = _percentile_bounds(X_train)
    n_features = X_train.shape[1]
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    X_syn = np.empty((n_samples, n_features))
    span = np.maximum(p_high - p_low, 1e-6)
    for i in range(n_samples):
        row = rng.uniform(mins, maxs).astype(np.float64)
        j = int(rng.integers(0, n_features))
        if rng.random() < 0.5:
            row[j] = float(p_low[j] - rng.uniform(0.1, 1.0) * span[j])
        else:
            row[j] = float(p_high[j] + rng.uniform(0.1, 1.0) * span[j])
        X_syn[i] = row
    return X_syn


def run_analysis(
    random_state: int = 42,
    test_size: float = 0.2,
    n_synthetic: int = 200,
) -> dict:
    X, y, _, _ = load_and_extract()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    p_low, p_high = _percentile_bounds(X_train)
    interp = interpolation_mask(X_test, p_low, p_high)
    extrap = ~interp

    models_dir = ROOT / "models"
    model_path = models_dir / "rf_model.joblib"
    scaler_path = models_dir / "scaler.joblib"

    if model_path.is_file() and scaler_path.is_file():
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        clf = RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )
        clf.fit(X_train_s, y_train)

    X_test_s = scaler.transform(X_test)
    pred_test = clf.predict(X_test_s)

    def acc(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return float(accuracy_score(y_test[mask], pred_test[mask]))

    n_interp = int(interp.sum())
    n_extrap = int(extrap.sum())

    rng = np.random.default_rng(random_state)
    X_syn = generate_synthetic_extrapolation(X_train, n_synthetic, rng)
    X_syn_s = scaler.transform(X_syn)
    pred_syn = clf.predict(X_syn_s)
    syn_not_secure_rate = float((pred_syn == 1).mean())

    report = {
        "model_source": "saved_artifacts" if model_path.is_file() else "inline_fit",
        "definition": {
            "interpolation": "Test row where each raw feature is within train p10–p90 for that feature.",
            "extrapolation": "Test row with at least one feature outside train p10–p90.",
        },
        "test_set": {
            "n_total": int(X_test.shape[0]),
            "n_interpolation": n_interp,
            "n_extrapolation": n_extrap,
            "accuracy_interpolation": acc(interp),
            "accuracy_extrapolation": acc(extrap),
            "accuracy_overall": float(accuracy_score(y_test, pred_test)),
        },
        "synthetic_extrapolation": {
            "n_samples": n_synthetic,
            "note": "No labels; reports fraction predicted as Not Secure (class 1).",
            "predicted_not_secure_rate": syn_not_secure_rate,
        },
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "interpolation_extrapolation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    r = run_analysis()
    print(json.dumps(r, indent=2))
