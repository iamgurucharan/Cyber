"""
Structured FAQ + optional numeric enrichment for POST /api/voice/answer.

Loads JSON from reports/ when the user asks about metrics, health, balance, etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# --- Triggers: load live numbers from disk ---------------------------------
# Short tokens use word boundaries so "contest" does not match "test".
_METRIC_TRIGGERS_WORD = frozenset(
    {
        "train",
        "test",
        "split",
        "tn",
        "tp",
        "fp",
        "fn",
        "f1",
        "roc",
        "auc",
        "cv",
        "gap",
    }
)
_METRIC_TRIGGERS_SUB = frozenset(
    {
        "metric",
        "metrics",
        "accuracy",
        "accurate",
        "overfit",
        "underfit",
        "balance",
        "balanced",
        "confusion",
        "matrix",
        "performance",
        "health",
        "precision",
        "recall",
        "learning curve",
        "cross-validation",
        "cross validation",
        "k-fold",
        "kfold",
        "stratified",
    }
)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and not x.is_integer():
            return f"{x:.{nd}f}"
        return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    return str(x)


def _question_triggers_metrics(q_lower: str) -> bool:
    for w in _METRIC_TRIGGERS_WORD:
        if re.search(rf"\b{re.escape(w)}\b", q_lower):
            return True
    if any(t in q_lower for t in _METRIC_TRIGGERS_SUB):
        return True
    # "how good" style
    if "how good" in q_lower and ("model" in q_lower or "it" in q_lower):
        return True
    return False


def _build_metrics_injection(reports: Path) -> str:
    m = _safe_load_json(reports / "metrics.json")
    if not m:
        return (
            "Run `python run_pipeline.py` to generate reports/metrics.json; "
            "then ask again for live accuracy, ROC-AUC, confusion counts, and health flags."
        )

    cells = m.get("confusion_matrix_cells") or {}
    tn = cells.get("TN")
    fp = cells.get("FP")
    fn = cells.get("FN")
    tp = cells.get("TP")
    thr = m.get("interpretation_thresholds") or {}
    gap_rule = thr.get("overfit_rule", "train minus test accuracy gap")
    under_rule = thr.get("underfit_rule", "both accuracies low")

    parts = [
        "Current run (from reports/metrics.json): "
        f"hold-out test accuracy {_fmt_num(m.get('accuracy'))}, "
        f"ROC-AUC {_fmt_num(m.get('roc_auc'))}, "
        f"n_train={m.get('n_train')}, n_test={m.get('n_test')}, "
        f"n_features={m.get('n_features')}. "
        f"Train accuracy {_fmt_num(m.get('train_accuracy'))} vs test {_fmt_num(m.get('test_accuracy'))}; "
        f"macro F1 train {_fmt_num(m.get('train_f1_macro'))} vs test {_fmt_num(m.get('test_f1_macro'))}. "
        f"Overfit risk flag: {_fmt_num(m.get('overfit_risk'))}; underfit risk: {_fmt_num(m.get('underfit_risk'))}. "
        f"Thresholds: overfit_accuracy_gap_trigger={thr.get('overfit_accuracy_gap_trigger')}, "
        f"underfit_accuracy_both_below={thr.get('underfit_accuracy_both_below')}. "
        f"Rules: {gap_rule} {under_rule}. "
    ]
    if tn is not None:
        parts.append(
            f"Confusion matrix counts on the test split: TN={tn}, FP={fp}, FN={fn}, TP={tp}. "
        )

    bal = _safe_load_json(reports / "dataset_balance.json")
    if bal:
        fc = bal.get("full_dataset_class_counts") or {}
        parts.append(
            "Class balance (dataset_balance.json): full CSV "
            f"Secure={fc.get('Secure')}, Not Secure={fc.get('Not Secure')}; "
            f"stratified_split={bal.get('stratified_split')}, test_size={bal.get('test_size')}. "
        )

    interp = _safe_load_json(reports / "interpolation_extrapolation.json")
    if interp:
        ts = interp.get("test_set") or {}
        parts.append(
            "Interpolation vs extrapolation (interpolation_extrapolation.json): "
            f"n_interpolation={ts.get('n_interpolation')}, n_extrapolation={ts.get('n_extrapolation')}, "
            f"accuracy_interpolation={_fmt_num(ts.get('accuracy_interpolation'))}, "
            f"accuracy_extrapolation={_fmt_num(ts.get('accuracy_extrapolation'))}. "
        )

    cv = _safe_load_json(reports / "cv_summary.json")
    if cv:
        parts.append(
            "Stratified K-fold (cv_summary.json): "
            f"n_splits={cv.get('n_splits')}, accuracy mean±std "
            f"{_fmt_num(cv.get('accuracy_mean'))} ± {_fmt_num(cv.get('accuracy_std'))}, "
            f"macro F1 {_fmt_num(cv.get('f1_macro_mean'))} ± {_fmt_num(cv.get('f1_macro_std'))}. "
        )

    return "".join(parts).strip()


def _shorten_for_tts(text: str, max_len: int = 520) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) <= max_len:
        return t
    cut = t[:max_len]
    last_period = cut.rfind(". ")
    if last_period > 200:
        return cut[: last_period + 1].strip()
    return cut.rsplit(" ", 1)[0] + "…"


# --- FAQ: (topic_id, keywords, answer, optional spoken override) ----------
# Order matters: first matching keyword set wins (put specific before generic).

_VOICE_FAQ: list[tuple[str, tuple[str, ...], str, str | None]] = [
    (
        "ethics_authorization",
        (
            "ethics",
            "ethical",
            "legal",
            "law",
            "permission",
            "authorize",
            "authorized",
            "unauthorized",
            "responsible",
            "disclaimer",
        ),
        (
            "Only scan or analyze systems you own or have explicit written permission to test. "
            "Unauthorized scanning can violate computer misuse laws and terms of service. "
            "This demo is not a full security assessment; do not present simulated Burp metrics as real Burp Suite output."
        ),
        None,
    ),
    (
        "voice_browser",
        (
            "voice",
            "microphone",
            "mic",
            "speech",
            "tts",
            "text to speech",
            "chrome",
            "browser",
            "webkit",
        ),
        (
            "The dashboard voice tab uses the browser Web Speech API: SpeechRecognition for the mic "
            "(Chrome/Chromium recommended; requires HTTPS or localhost and microphone permission) "
            "and speechSynthesis for read-back. Some browsers block audio until you interact with the page "
            "(e.g. click Send or a chip); if nothing speaks, try Chrome or click again after the answer loads."
        ),
        None,
    ),
    (
        "project_purpose",
        (
            "what is this project",
            "what does this project",
            "purpose of",
            "goal of",
            "demo",
        ),
        (
            "This workspace trains a Random Forest on passive security signals from a CSV, "
            "evaluates it with hold-out metrics and plots, and exposes a Flask API plus a small dashboard "
            "that can passively fetch a URL, merge optional Burp-style fields, and return an ML risk score."
        ),
        None,
    ),
    (
        "folder_structure",
        (
            "folder",
            "directory",
            "layout",
            "structure",
            "where is",
            "repo",
        ),
        (
            "Folders: `src/` holds the ML pipeline (features, training, interpolation study, CV); "
            "`backend/` has Flask (`app.py`), passive scanner, Burp helpers, predictor; "
            "`frontend/` is the static dashboard; `data/` holds `security_vulnerabilities.csv`; "
            "`models/` stores joblib artifacts; `reports/` has metrics JSON, confusion matrix PNG, learning curve, etc. "
            "Orchestrate training with `python run_pipeline.py` from the project root."
        ),
        None,
    ),
    (
        "how_to_run",
        (
            "how to run",
            "run the",
            "start server",
            "vscode",
            "vs code",
            "terminal",
            "pipeline",
            "quick start",
        ),
        (
            "From the project root: create a venv, `pip install -r requirements.txt`, run `python run_pipeline.py` "
            "to engineer features, train, and write `reports/metrics.json` and `models/`. "
            "Then `python backend/app.py` (or set PORT) and open http://127.0.0.1:5000/ . "
            "In VS Code: open the folder, select the interpreter from `.venv`, run the same commands in the integrated terminal."
        ),
        None,
    ),
    (
        "tech_stack",
        (
            "tech stack",
            "stack",
            "dependencies",
            "libraries",
            "flask",
            "sklearn",
            "scikit",
        ),
        (
            "Stack: Python 3, Flask for HTTP API and static frontend, scikit-learn (RandomForest, StandardScaler, "
            "metrics, StratifiedKFold, learning_curve), pandas/numpy for data, requests for passive HTTP, "
            "matplotlib/seaborn for plots, joblib for model persistence. The UI is plain HTML/CSS/JavaScript."
        ),
        None,
    ),
    (
        "api_endpoints",
        (
            "endpoint",
            "api",
            "api route",
            "routes",
            "post /",
            "get /",
        ),
        (
            "Main HTTP routes: GET `/` dashboard; GET `/api/health` liveness and model_ready; "
            "GET `/api/model/metrics` raw metrics.json; GET `/api/model/confusion-matrix` PNG or `?format=base64`; "
            "GET `/api/model/confusion-matrix-detail` TN/FP/FN/TP breakdown; "
            "GET `/api/model/interpolation-extrapolation`; POST `/api/analyze` JSON `{url, burp_report_path?}` scan+ML; "
            "POST `/api/voice/answer` JSON `{question}` FAQ with optional live numbers."
        ),
        None,
    ),
    (
        "csv_columns",
        (
            "csv column",
            "columns in",
            "dataset column",
            "what columns",
            "fields in csv",
        ),
        (
            "The training CSV includes `url` and `label` (Secure or Not Secure) plus numeric/boolean signals such as "
            "`ssl_valid`, `ssl_expiry_days`, header flags (HSTS, CSP, X-Frame, XSS protection), CVE counts by severity, "
            "vulnerability booleans (SQLi, XSS, CSRF, weak auth, cookies, mixed content, etc.), timing and content size, "
            "form/script counts, redirects, and Burp-style columns `burp_scan_score` and `burp_issues_*`. "
            "`url` is excluded from training features."
        ),
        None,
    ),
    (
        "feature_engineering",
        (
            "feature",
            "engineering",
            "how many feature",
            "inputs to",
            "derived",
        ),
        (
            "After engineering there are 34 features: 30 base columns from the CSV (excluding url and label) "
            "plus four derived fields: `total_cves`, `total_burp_issues`, `header_score` (sum of security header flags), "
            "and `vuln_flag_count` (count of vulnerability indicator booleans). Values are scaled with StandardScaler "
            "fit only on the training split."
        ),
        None,
    ),
    (
        "random_forest",
        (
            "random forest",
            "hyperparameter",
            "n_estimator",
            "class_weight",
            "forest",
            "rf ",
            " rf",
        ),
        (
            "The classifier is sklearn RandomForestClassifier with n_estimators=250 (full training), random_state=42, "
            "`class_weight='balanced'` so classes are reweighted by inverse frequency, n_jobs=-1, max_depth=28, "
            "min_samples_leaf=3, max_features='sqrt' (decorrelates trees), ccp_alpha=0.0. "
            "A faster forest is used only for the learning-curve diagnostic (see metrics.json learning_curve_note)."
        ),
        None,
    ),
    (
        "train_test_split",
        (
            "train test",
            "hold-out",
            "holdout",
            "test split",
            "stratified split",
        ),
        (
            "Data are split with stratified train_test_split so each set keeps the same class proportions as the CSV. "
            "Default test_size=0.2 with random_state=42. Metrics on the dashboard reflect the held-out test portion; "
            "train metrics are also logged to compare for overfitting."
        ),
        None,
    ),
    (
        "roc_auc",
        (
            "roc",
            "auc",
            "receiver",
        ),
        (
            "ROC-AUC summarizes how well predicted probabilities rank positives vs negatives on the split where both "
            "classes exist. It is written to reports/metrics.json for the test set (and train when computable). "
            "It complements accuracy, which can look high on imbalanced or easy data."
        ),
        None,
    ),
    (
        "confusion_matrix",
        (
            "confusion",
            "tn ",
            " tn",
            "tp ",
            " tp",
            "fp ",
            " fp",
            "fn ",
            " fn",
            "false positive",
            "false negative",
        ),
        (
            "On the sklearn-style 2×2 matrix for classes Secure (0) vs Not Secure (1): TN = true Secure predicted Secure; "
            "FP = false alarm (true Secure but predicted Not Secure); FN = miss (true Not Secure predicted Secure); "
            "TP = correct high-risk flag. Counts and a heatmap PNG are produced during training under reports/."
        ),
        None,
    ),
    (
        "model_health",
        (
            "overfit",
            "underfit",
            "model health",
            "train accuracy",
            "gap",
        ),
        (
            "Model health flags live in metrics.json: overfit_risk is true when train_accuracy minus test_accuracy "
            "exceeds interpretation_thresholds.overfit_accuracy_gap_trigger (default 0.08). "
            "underfit_risk is true when both train and test accuracy are below underfit_accuracy_both_below (default 0.65). "
            "Compare with cv_summary.json: if the single split gap is large but CV std is low, the hold-out may be unlucky; "
            "high CV std suggests instability across folds."
        ),
        None,
    ),
    (
        "learning_curve",
        (
            "learning curve",
            "more data",
            "bias variance",
        ),
        (
            "The learning curve plot (reports/learning_curve.png, paths in metrics.json) shows training and validation "
            "score vs increasing training set sizes using a smaller forest for speed. "
            "If both curves plateau low, the model may be underfitting; if the training score stays much higher than validation, "
            "variance / memorization is a concern."
        ),
        None,
    ),
    (
        "cross_validation",
        (
            "cross validation",
            "cross-validation",
            "k-fold",
            "kfold",
            "stratified k",
            "cv summary",
        ),
        (
            "StratifiedKFold (e.g. 5 splits) is used to score out-of-fold predictions so each fold keeps class balance. "
            "Mean and standard deviation of accuracy, macro F1, and ROC-AUC are saved to reports/cv_summary.json "
            "to sanity-check the single train/test split in metrics.json."
        ),
        None,
    ),
    (
        "interpolation_extrapolation",
        (
            "interpolation",
            "extrapolation",
            "percentile",
            "p10",
            "p90",
            "out of distribution",
            "ood",
        ),
        (
            "Interpolation rows are test samples whose features all fall inside each feature's training 10th–90th percentile band; "
            "extrapolation means at least one feature is outside that envelope (harder generalization). "
            "Aggregates and optional synthetic extrapolation experiment are in reports/interpolation_extrapolation.json."
        ),
        None,
    ),
    (
        "burp_modes",
        (
            "burp",
            "burp_source",
            "simulated",
            "import burp",
            "burp api",
        ),
        (
            "Burp-style numbers are merged in backend/burp_integration.py. Field `burp_source` is always one of: "
            "`api` (REST ping/configured stub), `import` (parsed Burp XML/JSON under the project if you pass burp_report_path), "
            "or `simulated` (heuristic + noise from passive scan features when no real feed is available). "
            "Environment variables BURP_API_URL and BURP_API_KEY enable a minimal health check; real enterprise APIs vary by product."
        ),
        None,
    ),
    (
        "passive_scan_limits",
        (
            "passive scan",
            "scanner limit",
            "what does scanner",
            "scan do",
            "timeout",
        ),
        (
            "The live scanner issues lightweight GET requests (default timeout about 12 seconds), reads response headers and a "
            "snippet of HTML, checks TLS certificate expiry via a socket handshake, and derives counts like forms and external scripts. "
            "It does not crawl the whole site, run active exploit payloads, or replace Burp/DAST; CVE columns in the CSV are not "
            "recomputed from the live page."
        ),
        None,
    ),
    (
        "synthetic_data",
        (
            "synthetic",
            "real world",
            "dataset caveat",
            "validity",
        ),
        (
            "The bundled CSV uses synthetic URL patterns for privacy; labels and scanner fields are demo-oriented. "
            "See reports/model_validity_notes.json after training. Expect domain shift on real production sites; "
            "treat scores as screening aids, not proof of compromise."
        ),
        None,
    ),
    (
        "dataset_overview",
        (
            "dataset",
            "csv",
            "data file",
            "rows",
            "security_vulnerabilities",
        ),
        (
            "Primary table is data/security_vulnerabilities.csv: thousands of rows with TLS/header signals, vulnerability flags, "
            "CVE and Burp-style aggregates, and a Secure vs Not Secure label for supervised learning."
        ),
        None,
    ),
    (
        "metrics_qa",
        (
            "accuracy",
            "how accurate",
            "macro f1",
            "weighted f1",
            "f1-score",
        ),
        (
            "Accuracy is the fraction of examples on a split whose predicted label matches the true label. "
            "This project stores test and train accuracy, macro F1, per-class precision and recall, ROC-AUC when both classes appear, "
            "confusion-matrix TN/FP/FN/TP counts, and overfit/underfit flags in reports/metrics.json."
        ),
        None,
    ),
]


def _match_topic(q_lower: str) -> tuple[str, str, str | None]:
    for topic_id, keywords, answer, spoken_override in _VOICE_FAQ:
        for kw in keywords:
            if kw in q_lower:
                return topic_id, answer, spoken_override
    return (
        "general",
        (
            "You can ask about: project purpose and folders; how to run the pipeline and server; CSV columns and the 34 engineered "
            "features; Random Forest settings and class_weight; train vs test split and ROC-AUC; confusion matrix TN/FP/FN/TP; "
            "model health thresholds and learning curves; stratified K-fold CV; Burp simulated vs API vs import and burp_source; "
            "passive scanner limits; synthetic data caveats; API routes; ethics and authorization; voice/Web Speech requirements."
        ),
        None,
    )


def build_voice_response(question: str, reports_dir: Path) -> dict[str, Any]:
    q_raw = (question or "").strip()
    q_lower = q_raw.lower()

    topic_id, answer, spoken_override = _match_topic(q_lower)

    inject = _question_triggers_metrics(q_lower) or topic_id in {
        "model_health",
        "roc_auc",
        "confusion_matrix",
        "train_test_split",
        "cross_validation",
        "interpolation_extrapolation",
        "metrics_qa",
    }
    if inject:
        extra = _build_metrics_injection(reports_dir)
        if extra:
            answer = f"{answer} {extra}"

    spoken = spoken_override if spoken_override else _shorten_for_tts(answer)

    return {
        "question": q_raw,
        "answer": answer.strip(),
        "spoken_answer": spoken.strip(),
        "matched_topic": topic_id,
    }
