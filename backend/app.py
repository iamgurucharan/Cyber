"""
Flask API + static frontend for Cybersecurity Risk Prediction.

Run from project root:
    python backend/app.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, request, send_file, send_from_directory

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.burp_integration import merge_burp_into_features  # noqa: E402
from backend.predictor import RiskPredictor, load_top_features_report  # noqa: E402
from backend.voice_assistant import build_voice_response  # noqa: E402
from backend.website_scanner import scan_url  # noqa: E402

FRONTEND = ROOT / "frontend"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

app = Flask(
    __name__,
    static_folder=str(FRONTEND),
    static_url_path="",
)


def _confusion_explanation() -> str:
    return (
        "Cells: TN (true Secure, predicted Secure) and TP (true Not Secure, predicted Not Secure) "
        "are correct. FP predicts Not Secure for a Secure site; FN misses a Not Secure site. "
        "The heatmap PNG is generated during training on the stratified hold-out split."
    )


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf so Flask jsonify and browsers never choke on metrics."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _load_metrics_dict() -> dict | None:
    p = REPORTS / "metrics.json"
    if not p.is_file():
        return None
    try:
        return _sanitize_for_json(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return None


def _confusion_cells_from_metrics(m: dict) -> dict[str, int]:
    cells = m.get("confusion_matrix_cells")
    if isinstance(cells, dict) and cells:
        return {
            "TN": int(cells.get("TN", 0)),
            "FP": int(cells.get("FP", 0)),
            "FN": int(cells.get("FN", 0)),
            "TP": int(cells.get("TP", 0)),
        }
    cm = m.get("confusion_matrix")
    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        return {
            "TN": int(cm[0][0]),
            "FP": int(cm[0][1]),
            "FN": int(cm[1][0]),
            "TP": int(cm[1][1]),
        }
    return {"TN": 0, "FP": 0, "FN": 0, "TP": 0}


def _report_class(report: dict, *names: str) -> dict:
    for name in names:
        block = report.get(name)
        if isinstance(block, dict):
            return block
    return {}


def _confusion_matrix_detail_from_metrics(m: dict) -> dict:
    """Build dashboard payload: counts, label order, and per-cell explanations."""
    cells = _confusion_cells_from_metrics(m)
    tn = cells["TN"]
    fp = cells["FP"]
    fn = cells["FN"]
    tp = cells["TP"]

    label_0 = "Secure"
    label_1 = "Not Secure"
    matrix_layout = (
        "sklearn-style 2×2 confusion matrix on the hold-out test split: "
        f"rows are true class ({label_0} row 0, {label_1} row 1); "
        f"columns are predicted class ({label_0} col 0, {label_1} col 1). "
        "So [0,0]=TN, [0,1]=FP, [1,0]=FN, [1,1]=TP."
    )

    cell_blocks = {
        "TN": {
            "short_name": "True Negative",
            "count": tn,
            "explanation": (
                f"The site was truly {label_0} and the model predicted {label_0}. "
                "These are correct rejections of risk for benign-looking telemetry."
            ),
            "security_implication": (
                "Strong TN volume means the model is not over-alarming on clean signals, "
                "which preserves analyst trust and capacity for real issues."
            ),
        },
        "FP": {
            "short_name": "False Positive",
            "count": fp,
            "explanation": (
                f"True class was {label_0} but the model predicted {label_1}—a false alarm."
            ),
            "security_implication": (
                "Too many FPs drive alert fatigue, wasted triage, and tuning pressure; "
                "they are usually less dangerous than missing a real problem but still costly."
            ),
        },
        "FN": {
            "short_name": "False Negative",
            "count": fn,
            "explanation": (
                f"True class was {label_1} but the model predicted {label_0}—a miss."
            ),
            "security_implication": (
                "False negatives are often the highest-severity failure mode in security screening: "
                "a risky posture is treated as safe, which can delay patching, under-prioritize incidents, "
                "or hide abuse until external discovery."
            ),
        },
        "TP": {
            "short_name": "True Positive",
            "count": tp,
            "explanation": (
                f"The site was truly {label_1} and the model predicted {label_1}."
            ),
            "security_implication": (
                "TPs are the detections you want to act on: they justify deeper review, "
                "compensating controls, or remediation while the signal still matches ground truth."
            ),
        },
    }

    report = m.get("classification_report") or {}
    secure = _report_class(report, "Secure", "0")
    not_secure = _report_class(report, "Not Secure", "1")
    macro = _report_class(report, "macro avg", "macro_avg")
    weighted = _report_class(report, "weighted avg", "weighted_avg")

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "label_index_0": label_0,
        "label_index_1": label_1,
        "positive_class_index": 1,
        "positive_class_name": label_1,
        "matrix_layout": matrix_layout,
        "cells": cell_blocks,
        "heatmap_note": _confusion_explanation(),
        "metrics": {
            "accuracy": m.get("accuracy", m.get("test_accuracy")),
            "roc_auc": m.get("roc_auc", m.get("test_roc_auc")),
            "secure": {
                "precision": secure.get("precision"),
                "recall": secure.get("recall"),
                "f1_score": secure.get("f1-score"),
                "support": secure.get("support"),
            },
            "not_secure": {
                "precision": not_secure.get("precision"),
                "recall": not_secure.get("recall"),
                "f1_score": not_secure.get("f1-score"),
                "support": not_secure.get("support"),
            },
            "macro_avg": {
                "precision": macro.get("precision"),
                "recall": macro.get("recall"),
                "f1_score": macro.get("f1-score"),
            },
            "weighted_avg": {
                "precision": weighted.get("precision"),
                "recall": weighted.get("recall"),
                "f1_score": weighted.get("f1-score"),
            },
        },
        "n_test": m.get("n_test"),
        "model_health": {
            "train_accuracy": m.get("train_accuracy"),
            "test_accuracy": m.get("test_accuracy"),
            "train_f1_macro": m.get("train_f1_macro"),
            "test_f1_macro": m.get("test_f1_macro"),
            "train_roc_auc": m.get("train_roc_auc"),
            "test_roc_auc": m.get("test_roc_auc"),
            "overfit_risk": m.get("overfit_risk"),
            "underfit_risk": m.get("underfit_risk"),
            "interpretation_thresholds": m.get("interpretation_thresholds"),
            "learning_curve_plot": m.get("learning_curve_plot"),
            "train_class_distribution": m.get("train_class_distribution"),
            "test_class_distribution": m.get("test_class_distribution"),
        },
    }


predictor = RiskPredictor(MODELS)


@app.get("/api/health")
def health():
    metrics = _load_metrics_dict()
    return jsonify(
        {
            "status": "ok",
            "api_version": 2,
            "model_ready": predictor.ready,
            "metrics_ready": metrics is not None,
            "confusion_matrix_png": (REPORTS / "confusion_matrix.png").is_file(),
            "project_root": str(ROOT),
        }
    )


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api/model/metrics")
def api_metrics():
    m = _load_metrics_dict()
    if m is None:
        return jsonify({"error": "Run python run_pipeline.py first."}), 404
    resp = make_response(jsonify(m))
    resp.headers["Cache-Control"] = "public, max-age=30, must-revalidate"
    return resp


@app.get("/api/model/confusion-matrix")
def api_confusion_matrix():
    p = REPORTS / "confusion_matrix.png"
    if not p.is_file():
        return jsonify({"error": "Confusion matrix not found. Run pipeline."}), 404
    fmt = (request.args.get("format") or "png").lower()
    if fmt == "base64":
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        resp = make_response(jsonify({"format": "png", "base64": b64}))
        resp.headers["Cache-Control"] = "public, max-age=30, must-revalidate"
        return resp
    resp = make_response(send_file(p, mimetype="image/png"))
    resp.headers["Cache-Control"] = "public, max-age=30, must-revalidate"
    return resp


@app.get("/api/model/confusion-matrix-detail")
def api_confusion_matrix_detail():
    m = _load_metrics_dict()
    if m is None:
        return jsonify({"error": "Run python run_pipeline.py first (metrics.json missing)."}), 404
    cells = _confusion_cells_from_metrics(m)
    if sum(cells.values()) <= 0 and not m.get("confusion_matrix"):
        return jsonify({"error": "metrics.json has no confusion matrix counts."}), 404
    payload = _confusion_matrix_detail_from_metrics(m)
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "public, max-age=30, must-revalidate"
    return resp


@app.get("/api/model/interpolation-extrapolation")
def api_interp():
    p = REPORTS / "interpolation_extrapolation.json"
    if not p.is_file():
        return jsonify({"error": "Run pipeline first."}), 404
    return jsonify(json.loads(p.read_text(encoding="utf-8")))


@app.post("/api/voice/answer")
def api_voice():
    data = request.get_json(silent=True) or {}
    q = str(data.get("question", data.get("query", ""))).strip()
    try:
        payload = build_voice_response(q, REPORTS)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Voice assistant error: {exc}"}), 500
    return jsonify(_sanitize_for_json(payload))


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    burp_path = data.get("burp_report_path")
    if burp_path:
        burp_path = str(burp_path).strip() or None

    if not url:
        return jsonify({"ok": False, "error": "Missing url"}), 400

    # Restrict Burp path to files under project root
    safe_burp = None
    if burp_path:
        cand = (ROOT / burp_path).resolve()
        try:
            cand.relative_to(ROOT.resolve())
        except ValueError:
            return jsonify({"ok": False, "error": "burp_report_path must be under project root"}), 400
        if cand.is_file():
            safe_burp = str(cand)

    scan = scan_url(url)
    burp_block = merge_burp_into_features(scan, burp_report_path=safe_burp)
    feats = burp_block["features"]

    report: dict = {
        "ok": True,
        "url": url,
        "final_url": scan.final_url,
        "scan_error": scan.error,
        "scanner_notes": scan.notes,
        "scanner_headers_sample": dict(list(scan.raw_headers.items())[:12]),
        "burp": {
            "burp_source": burp_block["burp_source"],
            "burp_note": burp_block["burp_note"],
            "burp_scan_score": feats.get("burp_scan_score"),
            "burp_issues_critical": feats.get("burp_issues_critical"),
            "burp_issues_high": feats.get("burp_issues_high"),
            "burp_issues_medium": feats.get("burp_issues_medium"),
            "burp_issues_low": feats.get("burp_issues_low"),
        },
        "scan_features": {k: feats.get(k) for k in list(feats.keys()) if not k.startswith("burp_")},
        "confusion_matrix_explanation": _confusion_explanation(),
    }

    if not predictor.ready:
        report["ml"] = None
        report["warning"] = "Model not trained. Run: python run_pipeline.py"
        return jsonify(report), 200

    try:
        risk, pred_class, proba = predictor.predict_proba_row(feats)
    except Exception as e:  # noqa: BLE001
        report["ok"] = False
        report["ml_error"] = str(e)
        return jsonify(report), 500

    label = "Not Secure" if pred_class == 1 else "Secure"
    top = load_top_features_report() or {}
    report["ml"] = {
        "predicted_label": label,
        "risk_score_not_secure_proba": risk,
        "class_probabilities": {"Secure": float(proba[0]), "Not Secure": float(proba[1])},
        "top_features": top.get("permutation_importance_train_sample", top.get("random_forest_importances", []))[:10],
        "feature_importances_rf": (top.get("random_forest_importances") or [])[:10],
    }
    return jsonify(report)


def main():
    import socket

    port = int(os.environ.get("PORT", 5001))
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(
                f"WARNING: port {port} is already in use. "
                "Another process may be serving an older API — stop it or set PORT=5001."
            )
    finally:
        probe.close()
    print(f"Cyber API v2 — dashboard: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
