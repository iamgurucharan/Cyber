"""
Train RandomForest on engineered features; persist model + scaler + metrics.

RandomForest regularization (reduces overfit vs fully grown trees):
- max_depth=28 caps tree depth (typical useful range ~20–35 for tabular data).
- min_samples_leaf=3 requires a few samples per leaf (smoother, less memorization than 1).
- max_features='sqrt' decorrelates trees compared to using all 34 features each split.
- class_weight='balanced' reweights classes by inverse frequency (safe when already
  balanced; differs from 'balanced_subsample' which recomputes weights per bootstrap sample).
  See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- ccp_alpha=0.0 disables cost-complexity pruning; a tiny positive value (e.g. 1e-4) can
  further regularize if validation suggests it.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    get_engineered_feature_names,
    load_and_extract,
    save_preprocessor_info,
)

logger = logging.getLogger(__name__)

# Interpretation thresholds (also written to metrics.json for transparency)
OVERFIT_ACCURACY_GAP_THRESHOLD = 0.08
UNDERFIT_ACCURACY_THRESHOLD = 0.65


def rf_classifier_kwargs(
    n_estimators: int = 250,
    random_state: int = 42,
) -> dict:
    """Shared hyperparameters for production training and CV (see module docstring)."""
    return {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "class_weight": "balanced",
        "n_jobs": -1,
        "max_depth": 28,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "ccp_alpha": 0.0,
    }


def _label_distribution(y: np.ndarray) -> dict[str, int]:
    """Counts per human-readable label (Secure=0, Not Secure=1)."""
    y = np.asarray(y).astype(int).ravel()
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    return {LABEL_NEGATIVE: n0, LABEL_POSITIVE: n1}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    return obj


def _write_dataset_balance(
    path: Path,
    *,
    full_counts: dict[str, int],
    train_counts: dict[str, int],
    test_counts: dict[str, int],
    test_size: float,
    random_state: int,
) -> None:
    def _pct(d: dict[str, int]) -> dict[str, float]:
        tot = sum(d.values()) or 1
        return {k: round(100.0 * v / tot, 4) for k, v in d.items()}

    payload = {
        "full_dataset_class_counts": full_counts,
        "full_dataset_class_percent": _pct(full_counts),
        "train_class_counts": train_counts,
        "train_class_percent": _pct(train_counts),
        "test_class_counts": test_counts,
        "test_class_percent": _pct(test_counts),
        "test_size": test_size,
        "random_state": random_state,
        "stratified_split": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Class balance: full=%s train=%s test=%s", full_counts, train_counts, test_counts)
    print(
        "[train] Class balance - full:",
        full_counts,
        "train:",
        train_counts,
        "test:",
        test_counts,
    )


def _write_model_validity_notes(path: Path) -> None:
    """Document dataset limitations (synthetic URLs, extended CSV mapping, etc.)."""
    notes = {
        "limitations": [
            "Default training uses extended_realistic_vulnerability_dataset_10000.csv: advisory "
            "Title/Summary/Severity rows are mapped into the same 30 base scanner/Burp columns used "
            "at inference; many transport/header/Burp numerics are training-time pseudo-values, not "
            "measured from the advisory link.",
            "Legacy security_vulnerabilities.csv uses synthetic URL host patterns; url is excluded "
            "from X in both schemas.",
            "Labels are dataset-defined (extended: Critical/High -> Not Secure, Moderate/Low -> Secure); "
            "performance on live authorized scans may differ.",
            "Burp-related columns may be simulated unless real imports/API are used at inference.",
        ],
        "recommendation": "Treat hold-out and CV metrics as internal validation; validate on "
        "independently labeled real-world samples before operational use.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes, indent=2), encoding="utf-8")


def _learning_curve_diagnostic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    reports_dir: Path,
) -> dict:
    """
    Plot learning curve (accuracy) on a capped subset with a lighter forest for runtime.
    """
    rng = np.random.default_rng(random_state)
    max_n = min(4000, X_train.shape[0])
    if X_train.shape[0] > max_n:
        idx = rng.choice(X_train.shape[0], size=max_n, replace=False)
        X_lc, y_lc = X_train[idx], y_train[idx]
    else:
        X_lc, y_lc = X_train, y_train

    lc_est = RandomForestClassifier(
        **rf_classifier_kwargs(n_estimators=120, random_state=random_state),
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    train_sizes, train_scores, val_scores = learning_curve(
        lc_est,
        X_lc,
        y_lc,
        train_sizes=np.linspace(0.15, 1.0, 5),
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        shuffle=False,
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(train_sizes, train_mean, "o-", color="tab:blue", label="Train (CV mean)")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="tab:blue",
    )
    plt.plot(train_sizes, val_mean, "o-", color="tab:orange", label="Validation (CV mean)")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="tab:orange",
    )
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning curve (diagnostic; lighter RF, subset if large)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = reports_dir / "learning_curve.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    return {
        "learning_curve_plot": out_png.relative_to(ROOT).as_posix(),
        "learning_curve_note": (
            "Computed with n_estimators=120 on up to 4000 train rows for speed; "
            "full model uses n_estimators from train()."
        ),
        "learning_curve_train_sizes": [int(x) for x in train_sizes],
        "learning_curve_train_scores_mean": [float(x) for x in train_mean],
        "learning_curve_val_scores_mean": [float(x) for x in val_mean],
    }


def train(
    random_state: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 250,
) -> dict:
    X, y, feature_names, pre_info = load_and_extract()
    full_counts = _label_distribution(y)
    names = get_engineered_feature_names()
    assert list(names) == feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    train_counts = _label_distribution(y_train)
    test_counts = _label_distribution(y_test)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_balance(
        reports_dir / "dataset_balance.json",
        full_counts=full_counts,
        train_counts=train_counts,
        test_counts=test_counts,
        test_size=test_size,
        random_state=random_state,
    )
    _write_model_validity_notes(reports_dir / "model_validity_notes.json")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train a RandomForest and calibrate probabilities using cross-validated
    # Platt scaling on the training set (avoids deprecated cv='prefit'). Calibrated
    #ClassifierCV will internally perform cross-validation.
    # Train a RandomForest for feature importances (full-fit)
    clf = RandomForestClassifier(**rf_classifier_kwargs(n_estimators, random_state))
    clf.fit(X_train_s, y_train)

    # Separately train a calibrated classifier (can retrain an RF inside) for
    # better-probability outputs. This duplicates effort but preserves a fitted
    # `clf` for extracting importances.
    calibrator = CalibratedClassifierCV(estimator=RandomForestClassifier(**rf_classifier_kwargs(n_estimators, random_state)), method="sigmoid", cv=3)
    calibrator.fit(X_train_s, y_train)
    model_for_pred = calibrator

    y_train_pred = model_for_pred.predict(X_train_s)
    y_test_pred = model_for_pred.predict(X_test_s)
    # predict_proba might be unavailable for some wrappers; handle defensively
    try:
        y_train_proba = model_for_pred.predict_proba(X_train_s)[:, 1]
        y_test_proba = model_for_pred.predict_proba(X_test_s)[:, 1]
    except Exception:
        # fallback to decision_function->sigmoid if needed, but keep simple here
        y_train_proba = np.zeros(X_train_s.shape[0])
        y_test_proba = np.zeros(X_test_s.shape[0])

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    train_f1_macro = float(f1_score(y_train, y_train_pred, average="macro", zero_division=0))
    test_f1_macro = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

    try:
        train_roc = float(roc_auc_score(y_train, y_train_proba))
    except ValueError:
        train_roc = None
    try:
        test_roc = float(roc_auc_score(y_test, y_test_proba))
    except ValueError:
        test_roc = None

    cm = confusion_matrix(y_test, y_test_pred)
    report_test = classification_report(
        y_test,
        y_test_pred,
        target_names=["Secure", "Not Secure"],
        output_dict=True,
        zero_division=0,
    )
    report_train = classification_report(
        y_train,
        y_train_pred,
        target_names=["Secure", "Not Secure"],
        output_dict=True,
        zero_division=0,
    )

    overfit_risk = bool((train_acc - test_acc) > OVERFIT_ACCURACY_GAP_THRESHOLD)
    underfit_risk = bool(
        train_acc < UNDERFIT_ACCURACY_THRESHOLD and test_acc < UNDERFIT_ACCURACY_THRESHOLD
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Persist calibrated model (or raw clf if calibration not applied).
    joblib.dump(model_for_pred, models_dir / "rf_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    artifacts = {
        "feature_names": feature_names,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "test_size": test_size,
        "label_classes": ["Secure", "Not Secure"],
        "positive_class_index": 1,
        "rf_hyperparameters": rf_classifier_kwargs(n_estimators, random_state),
    }
    joblib.dump(artifacts, models_dir / "artifacts.joblib")
    pre_info["feature_names"] = feature_names
    save_preprocessor_info(models_dir / "preprocessor_info.json", pre_info)

    # Confusion matrix plot (hold-out test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Secure", "Pred Not Secure"],
        yticklabels=["True Secure", "True Not Secure"],
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix (hold-out test)")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    lc_info = _learning_curve_diagnostic(X_train_s, y_train, random_state, reports_dir)

    metrics = {
        "accuracy": test_acc,
        "roc_auc": test_roc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_f1_macro": train_f1_macro,
        "test_f1_macro": test_f1_macro,
        "train_roc_auc": train_roc,
        "test_roc_auc": test_roc,
        "f1_score_type": "macro",
        "overfit_risk": overfit_risk,
        "underfit_risk": underfit_risk,
        "interpretation_thresholds": {
            "overfit_accuracy_gap_trigger": OVERFIT_ACCURACY_GAP_THRESHOLD,
            "underfit_accuracy_both_below": UNDERFIT_ACCURACY_THRESHOLD,
            "overfit_rule": "overfit_risk is true when (train_accuracy - test_accuracy) exceeds overfit_accuracy_gap_trigger",
            "underfit_rule": "underfit_risk is true when both train and test accuracy are below underfit_accuracy_both_below",
        },
        "train_class_distribution": train_counts,
        "test_class_distribution": test_counts,
        "full_dataset_class_distribution": full_counts,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["TN", "FP", "FN", "TP"],
        "confusion_matrix_cells": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
        "classification_report": report_test,
        "classification_report_train": report_train,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(feature_names),
        **lc_info,
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    # Top features: RF importances + permutation importance (hold-out subset)
    imp = clf.feature_importances_
    order_rf = np.argsort(imp)[::-1][:20]
    top_rf = [
        {"feature": feature_names[i], "importance": float(imp[i])}
        for i in order_rf
    ]
    n_pi = min(400, X_train_s.shape[0])
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X_train_s.shape[0], size=n_pi, replace=False)
    pi = permutation_importance(
        clf,
        X_train_s[idx],
        y_train[idx],
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    order_pi = np.argsort(pi.importances_mean)[::-1][:15]
    top_pi = [
        {
            "feature": feature_names[i],
            "importance_mean": float(pi.importances_mean[i]),
            "importance_std": float(pi.importances_std[i]),
        }
        for i in order_pi
    ]
    with open(reports_dir / "top_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_forest_importances": top_rf,
                "permutation_importance_train_sample": top_pi,
                "permutation_n_samples": int(n_pi),
            },
            f,
            indent=2,
        )

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = train()
    print(
        "Training complete. test_accuracy:",
        m["test_accuracy"],
        "train_accuracy:",
        m["train_accuracy"],
        "ROC-AUC (test):",
        m.get("roc_auc"),
        "overfit_risk:",
        m.get("overfit_risk"),
    )
