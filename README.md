# Cybersecurity Risk Prediction

End-to-end demo: engineered features from **`data/extended_realistic_vulnerability_dataset_10000.csv`** by default (advisory-style rows mapped into the same 34-feature schema used at inference), **RandomForest** classifier, Flask API with passive URL scanning, optional Burp-style metrics, and a small web dashboard (including a **voice FAQ** tab).

## Quick start

```bash
cd Cyber
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python run_pipeline.py
python backend/app.py
```

Open **http://127.0.0.1:5001/** in your browser.

### Training dataset (default vs override)

| Source | Path |
|--------|------|
| **Default** | `data/extended_realistic_vulnerability_dataset_10000.csv` |
| **Legacy (optional)** | `data/security_vulnerabilities.csv` — tabular scanner simulation with `Secure` / `Not Secure` labels |

Override without editing code:

```bash
# Windows PowerShell
$env:CYBER_DATASET_PATH = "data\security_vulnerabilities.csv"
python run_pipeline.py

# Linux/macOS
export CYBER_DATASET_PATH=data/security_vulnerabilities.csv
python run_pipeline.py
```

The extended file has columns `Title`, `Date`, `Severity`, `Summary`, `Link`. Training maps them into the same base features as `backend/website_scanner.py` + Burp merge (`models/preprocessor_info.json` → `training_dataset` after `run_pipeline.py`). Extended labels: **Critical** or **High** → `Not Secure`; **Moderate** or **Low** → `Secure`.

## Voice assistant

- Open the **Voice Assistant** tab. **Google Chrome** (desktop) is recommended: Web Speech **recognition** and **synthesis** are most reliable there.
- Allow **microphone** permission when the browser prompts. If speech recognition is unavailable, use **Type your question** + **Send** — the same FAQ endpoint handles both.
- **Text-to-speech** may stay silent until you interact with the page (e.g. click **Send** or a suggestion chip); that is normal browser autoplay policy.
- Answers can include **live numbers** from `reports/metrics.json` (and related JSON) when you ask about accuracy, F1, ROC-AUC, overfit, balance, etc.

## Legal / ethics — read this

- **Only scan targets you own or have explicit written authorization to test.** Unauthorized scanning may violate computer misuse laws and site terms of service.
- This project performs **light passive HTTP/TLS checks** and **machine learning on a fixed dataset**. It is **not** a complete security assessment.
- The **extended training CSV** contains realistic-sounding advisory text and vendor names for ML practice; it is **not** a claim that those products were scanned or that listed CVEs were verified against live systems. **Live `/api/analyze` scans** must still be limited to authorized targets; realism of training rows does not grant permission to probe third-party sites.
- **Burp Suite** results are **simulated** unless you supply a supported import file or wire a real API. Do not present simulated Burp metrics as results from PortSwigger products.

## API (selected)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + model files present |
| POST | `/api/analyze` | JSON `{ "url", "burp_report_path?"} ` — scan + ML + Burp block |
| GET | `/api/model/metrics` | `reports/metrics.json` |
| GET | `/api/model/confusion-matrix?format=png\|base64` | PNG or base64 |
| GET | `/api/model/interpolation-extrapolation` | JSON report |
| POST | `/api/voice/answer` | JSON `{ "question" }` — FAQ + optional `spoken_answer`, `matched_topic`; injects live metrics when relevant |

Optional env: `BURP_API_URL`, `BURP_API_KEY` (stub REST check only), `PORT`.

## Model validity and bias

- Class counts for the full CSV and stratified train/test splits are written to `reports/dataset_balance.json` (and summarized in `reports/metrics.json`).
- `reports/model_validity_notes.json` calls out limitations of the extended CSV mapping, legacy synthetic URLs, and simulated scanner fields.
- `metrics.json` includes train vs test accuracy/F1 (macro), optional ROC-AUC, `overfit_risk` / `underfit_risk` flags, and paths to diagnostic plots (e.g. `learning_curve.png`). Compare a large train–test gap with `reports/cv_summary.json` mean ± std to judge stability.

## Layout

- `src/` — feature engineering, training, interpolation/extrapolation, CV tests  
- `backend/` — Flask app, scanner, Burp helpers, predictor  
- `frontend/` — dashboard + Web Speech API  
- `models/` — `joblib` artifacts after training  
- `reports/` — metrics, plots, JSON studies  

## License note

Dataset and code are for education/research. Burp is a trademark of PortSwigger Ltd.; this repo does not ship Burp software.

## Git & Contributing

This repository includes a sensible `.gitignore` and `.gitattributes` to keep large artifacts, local env files, and generated reports out of source control.

Recommended workflow:

```bash
git status
git add .gitignore .gitattributes .env.example
git add src backend frontend tools README.md
git commit -m "chore: add gitignore, gitattributes, env example and tooling"
git push origin main
```

Notes:
- Do not commit your real `.env` or any Burp export files containing sensitive data.
- Large artifacts (models, reports, data) are ignored by default. If you want to track a specific trained model or report, add it intentionally and document it in a release.
- If you'd like, I can create a commit for you locally (and push) — tell me whether to run git commands in this workspace and which remote/branch to use.

