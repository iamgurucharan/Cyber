"""
Stratified K-Fold cross-validation and simple edge-case sanity checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import load_and_extract  # noqa: E402
from src.train_model import rf_classifier_kwargs  # noqa: E402


def _scale_fold(X_train, X_val):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_val)


def run_cv(n_splits: int = 5, random_state: int = 42) -> dict:
    X, y, feature_names, _ = load_and_extract()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs = []
    rocs = []
    f1_macros = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        X_tr_s, X_va_s = _scale_fold(X_tr, X_va)
        clf = RandomForestClassifier(
            **rf_classifier_kwargs(n_estimators=250, random_state=random_state),
        )
        clf.fit(X_tr_s, y_tr)
        pred = clf.predict(X_va_s)
        proba = clf.predict_proba(X_va_s)[:, 1]
        accs.append(accuracy_score(y_va, pred))
        f1_macros.append(f1_score(y_va, pred, average="macro", zero_division=0))
        try:
            rocs.append(roc_auc_score(y_va, proba))
        except ValueError:
            rocs.append(float("nan"))

    acc_mean = float(np.nanmean(accs))
    acc_std = float(np.nanstd(accs))
    f1_mean = float(np.nanmean(f1_macros))
    f1_std = float(np.nanstd(f1_macros))

    summary = {
        "n_splits": n_splits,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "roc_auc_mean": float(np.nanmean(rocs)),
        "roc_auc_std": float(np.nanstd(rocs)),
        "fold_accuracies": [float(a) for a in accs],
        "fold_f1_macros": [float(f) for f in f1_macros],
        "fold_roc_aucs": [float(r) if not np.isnan(r) else None for r in rocs],
        "n_samples": int(X.shape[0]),
        "n_features": len(feature_names),
        "cv_stability_note": (
            "Stratified K-fold reports mean ± std on out-of-fold predictions. "
            "If single train/test gap in metrics.json is large but CV std is low, "
            "the split may be unlucky; if CV std is high, the model or data may be unstable across folds."
        ),
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def edge_case_tests() -> dict:
    """
    Train on full data (small RF) and check monotonic expectations on hand-built rows.
    Uses same feature order as production.
    """
    from src.feature_engineering import load_and_extract

    X, y, names, _ = load_and_extract()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(
        **rf_classifier_kwargs(n_estimators=100, random_state=0),
    )
    clf.fit(Xs, y)

    median_row = np.median(X, axis=0)
    names_idx = {n: i for i, n in enumerate(names)}

    def set_feature(row: np.ndarray, name: str, value: float) -> np.ndarray:
        r = row.copy()
        r[names_idx[name]] = value
        return r

    # Recompute derived for a perturbed base vector (simplified: clone median and tweak key bases)
    def row_from_dict(overrides: dict) -> np.ndarray:
        r = median_row.copy()
        for k, v in overrides.items():
            r[names_idx[k]] = v
        # rebuild derived in-line matching add_derived_features
        crit = r[names_idx["critical_cves"]]
        high = r[names_idx["high_cves"]]
        med = r[names_idx["medium_cves"]]
        low = r[names_idx["low_cves"]]
        r[names_idx["total_cves"]] = crit + high + med + low
        r[names_idx["total_burp_issues"]] = (
            r[names_idx["burp_issues_critical"]]
            + r[names_idx["burp_issues_high"]]
            + r[names_idx["burp_issues_medium"]]
            + r[names_idx["burp_issues_low"]]
        )
        r[names_idx["header_score"]] = (
            r[names_idx["has_hsts"]]
            + r[names_idx["has_csp"]]
            + r[names_idx["has_xframe"]]
            + r[names_idx["has_xss_protection"]]
        )
        vulns = [
            "sql_injection_detected",
            "xss_detected",
            "csrf_missing",
            "auth_weak",
            "insecure_cookies",
            "mixed_content",
            "server_banner_exposed",
            "directory_listing",
            "default_credentials",
        ]
        r[names_idx["vuln_flag_count"]] = sum(r[names_idx[v]] for v in vulns)
        return r

    low_risk = row_from_dict(
        {
            "ssl_valid": 1,
            "has_hsts": 1,
            "has_csp": 1,
            "has_xframe": 1,
            "has_xss_protection": 1,
            "critical_cves": 0,
            "high_cves": 0,
            "open_ports_count": 1,
            "burp_scan_score": 5,
            "burp_issues_critical": 0,
            "burp_issues_high": 0,
            "burp_issues_medium": 0,
            "burp_issues_low": 0,
        }
    )
    high_risk = row_from_dict(
        {
            "ssl_valid": 0,
            "ssl_expiry_days": 0,
            "has_hsts": 0,
            "has_csp": 0,
            "has_xframe": 0,
            "has_xss_protection": 0,
            "critical_cves": 5,
            "high_cves": 10,
            "open_ports_count": 20,
            "burp_scan_score": 95,
            "burp_issues_critical": 5,
            "burp_issues_high": 8,
            "sql_injection_detected": 1,
            "csrf_missing": 1,
        }
    )

    p_low = clf.predict_proba(scaler.transform(low_risk.reshape(1, -1)))[0, 1]
    p_high = clf.predict_proba(scaler.transform(high_risk.reshape(1, -1)))[0, 1]

    results = {
        "low_risk_handcrafted_not_secure_proba": float(p_low),
        "high_risk_handcrafted_not_secure_proba": float(p_high),
        "expect_higher_risk_for_high_vector": bool(p_high >= p_low),
    }

    reports_dir = ROOT / "reports"
    with open(reports_dir / "edge_case_tests.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("CV:", json.dumps(run_cv(), indent=2))
    print("Edge:", json.dumps(edge_case_tests(), indent=2))
