/**
 * Dashboard + Web Speech API voice flow.
 */

const $ = (id) => document.getElementById(id);

/** Same-origin API base (empty) when served by Flask; file:// needs http://127.0.0.1:5000 */
const API_BASE =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:5000"
    : "";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function showApiBanner(message) {
  let el = document.getElementById("api-banner");
  if (!el) {
    el = document.createElement("div");
    el.id = "api-banner";
    el.className = "api-banner";
    el.setAttribute("role", "alert");
    const shell = document.querySelector(".shell");
    if (shell) shell.prepend(el);
  }
  el.textContent = message;
}

/** @type {object | null} Last successful /api/analyze JSON for optional read-aloud. */
let lastAnalyzeReport = null;

const SUGGESTIONS = [
  "What is this project for?",
  "What are the CSV columns and labels?",
  "How many engineered features are there?",
  "What Random Forest hyperparameters and class_weight are used?",
  "Explain train vs test split and stratified sampling.",
  "What is ROC-AUC and where is it stored?",
  "Explain TN FP FN TP on the confusion matrix.",
  "What are overfit_risk and underfit_risk thresholds?",
  "What does the learning curve plot mean?",
  "How does stratified K-fold CV work here?",
  "What is interpolation vs extrapolation?",
  "What are Burp simulated vs API vs import and burp_source?",
  "What are passive scanner limits and timeouts?",
  "What API endpoints exist?",
  "What are ethics and scanning authorization rules?",
];

function renderChips() {
  const wrap = $("voice-chips");
  if (!wrap) return;
  wrap.innerHTML = "";
  SUGGESTIONS.forEach((text) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = text;
    b.addEventListener("click", () => askVoice(text));
    wrap.appendChild(b);
  });
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.id === `panel-${name}`);
  });
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function speak(text) {
  if (!window.speechSynthesis || !text) return;
  try {
    window.speechSynthesis.cancel();
    window.speechSynthesis.resume();
  } catch {
    /* noop */
  }
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1;
  window.speechSynthesis.speak(u);
}

function setVoiceStatus(mode, extra) {
  const el = $("voice-status");
  if (!el) return;
  el.className = `voice-status ${mode}`;
  const labels = {
    idle: "Idle",
    listening: "Listening…",
    processing: "Asking server…",
    error: "Error",
  };
  const base = labels[mode] || mode;
  el.textContent = extra ? `${base} ${extra}` : base;
}

async function askVoice(question, opts = {}) {
  const { skipSpeak = false } = opts;
  const q = String(question || "").trim();
  const transcriptEl = $("voice-transcript");
  const answerEl = $("voice-answer");
  if (transcriptEl) transcriptEl.textContent = q || "—";
  if (answerEl) answerEl.textContent = "…";
  setVoiceStatus("processing", "");
  if (!q) {
    if (answerEl) {
      answerEl.textContent =
        "Type or speak a question, or click a suggestion chip.";
    }
    setVoiceStatus("idle", "");
    return;
  }
  try {
    const res = await fetch(apiUrl("/api/voice/answer"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error("Server returned a non-JSON response. Restart python backend/app.py.");
    }
    if (!res.ok) {
      throw new Error(data.error || res.statusText || "Request failed");
    }
    const body =
      data.answer || data.response || data.message || "";
    if (answerEl) answerEl.textContent = body || "(empty answer)";
    const topic = data.matched_topic ? ` (topic: ${data.matched_topic})` : "";
    setVoiceStatus("idle", topic.trim());
    if (!data.matched_topic && body.length < 200) {
      showApiBanner(
        "Voice assistant is running an older server build. Stop Flask (Ctrl+C) and run: python backend/app.py"
      );
    }
    const toSpeak = data.spoken_answer || data.answer || data.response || "";
    if (!skipSpeak && toSpeak) speak(toSpeak);
  } catch (e) {
    setVoiceStatus("error", "");
    if (answerEl) answerEl.textContent = String(e);
    if (window.location.protocol === "file:") {
      showApiBanner(
        "Open http://127.0.0.1:5000/ after starting python backend/app.py (file:// cannot call the API)."
      );
    }
  }
}

function setupSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  const hint = $("voice-speech-hint");
  const btn = $("btn-voice-listen");

  if (!SpeechRecognition) {
    if (btn) btn.disabled = true;
    if (hint) {
      hint.textContent =
        "Speech recognition is not available in this browser — use typed questions below. Try Google Chrome on desktop for microphone input.";
    }
    return;
  }

  if (hint) {
    hint.textContent =
      "Microphone: click Start listening, speak, then Stop listening (or pause). Your transcript is sent to the same FAQ endpoint as typed questions.";
  }

  const rec = new SpeechRecognition();
  rec.lang = "en-US";
  rec.interimResults = false;
  rec.continuous = false;

  let expectResult = false;

  rec.onresult = (ev) => {
    expectResult = false;
    const text = ev.results[0][0].transcript;
    askVoice(text);
  };

  rec.onerror = (ev) => {
    expectResult = false;
    setVoiceStatus("error", `(${ev.error || "unknown"})`);
  };

  rec.onend = () => {
    if (btn) btn.textContent = "Start listening";
    const noSpeech = expectResult;
    expectResult = false;
    if (noSpeech) {
      setVoiceStatus("idle", "(no speech captured)");
    }
  };

  if (!btn) return;

  btn.addEventListener("click", () => {
    const label = btn.textContent || "";
    if (label.startsWith("Stop")) {
      expectResult = false;
      try {
        rec.stop();
      } catch {
        /* noop */
      }
      btn.textContent = "Start listening";
      setVoiceStatus("idle", "");
      return;
    }
    try {
      expectResult = true;
      btn.textContent = "Stop listening";
      setVoiceStatus("listening", "");
      rec.start();
    } catch {
      expectResult = false;
      btn.textContent = "Start listening";
      setVoiceStatus("error", "(could not start mic)");
    }
  });
}

function fmtNum(n) {
  if (n == null || Number.isNaN(n)) return "—";
  if (typeof n === "number" && !Number.isInteger(n)) return n.toFixed(3);
  return String(n);
}

function riskClass(score) {
  if (score == null) return "risk-low";
  return score >= 0.5 ? "risk-high" : "risk-low";
}

function renderReport(data) {
  const el = $("report");
  el.classList.remove("hidden");

  if (data.ok) {
    lastAnalyzeReport = data;
    const readBtn = $("btn-read-last");
    if (readBtn) readBtn.classList.remove("hidden");
  }

  if (!data.ok) {
    el.innerHTML = `<div class="card"><h2>Error</h2><p>${escapeHtml(
      data.error || data.scan_error || "Unknown"
    )}</p></div>`;
    return;
  }

  const ml = data.ml;
  const risk = ml ? ml.risk_score_not_secure_proba : null;
  const pred = ml ? ml.predicted_label : "—";

  let featuresHtml = "";
  if (ml && ml.top_features && ml.top_features.length) {
    featuresHtml = `<ul class="feature-list">${ml.top_features
      .map(
        (f) =>
          `<li><span>${escapeHtml(f.feature)}</span><span>${fmtNum(
            f.importance_mean != null ? f.importance_mean : f.importance
          )}</span></li>`
      )
      .join("")}</ul>`;
  }

  el.innerHTML = `
    <div class="card">
      <h2>Machine learning</h2>
      <p><span class="risk-pill ${riskClass(risk)}">Predicted: ${escapeHtml(
    pred
  )}</span></p>
      <p class="kv"><span>Risk score (P Not Secure)</span><strong>${fmtNum(
        risk
      )}</strong></p>
      ${
        ml && ml.class_probabilities
          ? `<p class="muted">Secure: ${fmtNum(
              ml.class_probabilities.Secure
            )} · Not Secure: ${fmtNum(ml.class_probabilities["Not Secure"])}</p>`
          : ""
      }
      ${data.warning ? `<p class="muted">${escapeHtml(data.warning)}</p>` : ""}
      <h3>Top model drivers (permutation / importance)</h3>
      ${featuresHtml || "<p class='muted'>Train pipeline to populate.</p>"}
      <h3>Confusion matrix</h3>
      <p class="muted">${escapeHtml(data.confusion_matrix_explanation || "")} See <strong>Model evaluation</strong> at the top of the dashboard for the heatmap and cell-by-cell counts.</p>
    </div>
    <div class="card">
      <h2>Live scan</h2>
      <p class="kv"><span>Final URL</span>${escapeHtml(data.final_url || "")}</p>
      ${
        data.scan_error
          ? `<p class="muted">Scan note: ${escapeHtml(data.scan_error)}</p>`
          : ""
      }
      <div class="grid">
        ${Object.entries(data.scan_features || {})
          .map(
            ([k, v]) =>
              `<div class="kv"><span>${escapeHtml(k)}</span><strong>${fmtNum(
                v
              )}</strong></div>`
          )
          .join("")}
      </div>
    </div>
    <div class="card">
      <h2>Burp block</h2>
      <p class="kv"><span>Source</span><strong>${escapeHtml(
        data.burp.burp_source
      )}</strong></p>
      <p class="muted">${escapeHtml(data.burp.burp_note || "")}</p>
      <div class="grid">
        <div class="kv"><span>burp_scan_score</span><strong>${fmtNum(
          data.burp.burp_scan_score
        )}</strong></div>
        <div class="kv"><span>critical</span><strong>${fmtNum(
          data.burp.burp_issues_critical
        )}</strong></div>
        <div class="kv"><span>high</span><strong>${fmtNum(
          data.burp.burp_issues_high
        )}</strong></div>
        <div class="kv"><span>medium</span><strong>${fmtNum(
          data.burp.burp_issues_medium
        )}</strong></div>
        <div class="kv"><span>low</span><strong>${fmtNum(
          data.burp.burp_issues_low
        )}</strong></div>
      </div>
    </div>
  `;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildLastReportSpeech() {
  if (!lastAnalyzeReport || !lastAnalyzeReport.ok) return "";
  const d = lastAnalyzeReport;
  const ml = d.ml;
  const pred = ml ? ml.predicted_label : "unknown";
  const risk = ml ? fmtNum(ml.risk_score_not_secure_proba) : "n/a";
  const src = d.burp ? d.burp.burp_source : "unknown";
  const url = d.final_url || d.url || "";
  return `Last analyze result for ${url}. Predicted label ${pred}. Risk score for not secure is ${risk}. Burp source ${src}.`;
}

const CM_ORDER = ["TN", "FP", "FN", "TP"];

function renderModelHealth(detail) {
  const strip = $("model-health-strip");
  if (!strip) return;
  const mh = detail.model_health || {};
  const tr =
    mh.train_accuracy != null ? mh.train_accuracy : detail.train_accuracy;
  const te =
    mh.test_accuracy != null ? mh.test_accuracy : detail.test_accuracy;
  const overfit = mh.overfit_risk ?? detail.overfit_risk;
  const underfit = mh.underfit_risk ?? detail.underfit_risk;
  if (tr == null && te == null) {
    strip.classList.add("hidden");
    strip.innerHTML = "";
    return;
  }
  strip.classList.remove("hidden");
  const gap =
    tr != null && te != null && !Number.isNaN(tr - te)
      ? (tr - te).toFixed(3)
      : "—";
  const overClass = overfit ? "bad" : "ok";
  const underClass = underfit ? "bad" : "ok";
  const overLabel = overfit ? "Overfit risk" : "Overfit OK";
  const underLabel = underfit ? "Underfit risk" : "Underfit OK";
  const trainF1 = mh.train_f1_macro ?? detail.train_f1_macro;
  const testF1 = mh.test_f1_macro ?? detail.test_f1_macro;
  strip.innerHTML = `
    <span class="mh-title">Model health</span>
    <span class="mh-muted">Train acc</span><strong>${fmtNum(tr)}</strong>
    <span class="mh-muted">· Test acc</span><strong>${fmtNum(te)}</strong>
    <span class="mh-muted">· Gap (train−test)</span><strong>${gap}</strong>
    <span class="mh-muted">· Macro F1 train/test</span><strong>${fmtNum(
      trainF1
    )}</strong> <span class="mh-muted">/</span> <strong>${fmtNum(
    testF1
  )}</strong>
    <span class="mh-pill ${overClass}">${escapeHtml(overLabel)}</span>
    <span class="mh-pill ${underClass}">${escapeHtml(underLabel)}</span>
  `;
}

function renderModelEvalMetrics(meta) {
  const el = $("model-eval-metrics");
  if (!el || !meta) return;
  const m = meta.metrics || {};
  const acc = m.accuracy;
  const roc = m.roc_auc;
  const ns = m.not_secure || {};
  const ma = m.macro_avg || {};
  el.innerHTML = `
    <div class="kv"><span>Accuracy</span><strong>${fmtNum(acc)}</strong></div>
    <div class="kv"><span>ROC-AUC</span><strong>${fmtNum(roc)}</strong></div>
    <div class="kv"><span>Precision (Not Secure)</span><strong>${fmtNum(ns.precision)}</strong></div>
    <div class="kv"><span>Recall (Not Secure)</span><strong>${fmtNum(ns.recall)}</strong></div>
    <div class="kv"><span>F1 (Not Secure)</span><strong>${fmtNum(ns.f1_score)}</strong></div>
    <div class="kv"><span>Macro F1</span><strong>${fmtNum(ma.f1_score)}</strong></div>
  `;
}

function reportBlock(report, ...names) {
  for (const name of names) {
    if (report && report[name]) return report[name];
  }
  return {};
}

function buildDetailFromMetrics(m) {
  const cellsRaw = m.confusion_matrix_cells || {};
  let tn = cellsRaw.TN;
  let fp = cellsRaw.FP;
  let fn = cellsRaw.FN;
  let tp = cellsRaw.TP;
  const cm = m.confusion_matrix;
  if (
    (tn == null || fp == null || fn == null || tp == null) &&
    Array.isArray(cm) &&
    cm.length === 2
  ) {
    tn = cm[0][0];
    fp = cm[0][1];
    fn = cm[1][0];
    tp = cm[1][1];
  }
  const label0 = "Secure";
  const label1 = "Not Secure";
  const report = m.classification_report || {};
  const secure = reportBlock(report, "Secure", "0");
  const notSecure = reportBlock(report, "Not Secure", "1");
  const macro = reportBlock(report, "macro avg", "macro_avg");
  const cell = (key, shortName, explanation) => ({
    short_name: shortName,
    count: { TN: tn, FP: fp, FN: fn, TP: tp }[key] ?? 0,
    explanation,
    security_implication: "",
  });
  return {
    tn: Number(tn) || 0,
    fp: Number(fp) || 0,
    fn: Number(fn) || 0,
    tp: Number(tp) || 0,
    n_test: m.n_test,
    matrix_layout: `Hold-out test split: rows true class (${label0}, ${label1}); columns predicted.`,
    heatmap_note:
      "Cells: TN/TP correct; FP false alarm; FN missed risk. Heatmap from training pipeline.",
    metrics: {
      accuracy: m.accuracy ?? m.test_accuracy,
      roc_auc: m.roc_auc ?? m.test_roc_auc,
      not_secure: {
        precision: notSecure.precision,
        recall: notSecure.recall,
        f1_score: notSecure["f1-score"] ?? notSecure.f1_score,
      },
      macro_avg: {
        f1_score: macro["f1-score"] ?? macro.f1_score,
      },
    },
    model_health: {
      train_accuracy: m.train_accuracy,
      test_accuracy: m.test_accuracy ?? m.accuracy,
      train_f1_macro: m.train_f1_macro,
      test_f1_macro: m.test_f1_macro,
      overfit_risk: m.overfit_risk,
      underfit_risk: m.underfit_risk,
    },
    train_accuracy: m.train_accuracy,
    test_accuracy: m.test_accuracy ?? m.accuracy,
    train_f1_macro: m.train_f1_macro,
    test_f1_macro: m.test_f1_macro,
    overfit_risk: m.overfit_risk,
    underfit_risk: m.underfit_risk,
    cells: {
      TN: cell(
        "TN",
        "True Negative",
        `Truly ${label0}, predicted ${label0} (correct).`
      ),
      FP: cell(
        "FP",
        "False Positive",
        `Truly ${label0}, predicted ${label1} (false alarm).`
      ),
      FN: cell(
        "FN",
        "False Negative",
        `Truly ${label1}, predicted ${label0} (missed risk).`
      ),
      TP: cell(
        "TP",
        "True Positive",
        `Truly ${label1}, predicted ${label1} (correct detection).`
      ),
    },
  };
}

async function fetchMetricsRaw() {
  const res = await fetch(apiUrl("/api/model/metrics"));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.error || "Model metrics unavailable — run python run_pipeline.py."
    );
  }
  return res.json();
}

function renderConfusionCells(detail) {
  const wrap = $("cm-cells");
  if (!wrap || !detail.cells) return;
  wrap.innerHTML = CM_ORDER.map((key) => {
    const c = detail.cells[key] || {
      short_name: key,
      count: 0,
      explanation: "",
      security_implication: "",
    };
    const extra = key === "FN" ? " fn-cell" : "";
    return `
      <div class="cm-cell${extra}">
        <h4>${escapeHtml(key)} — ${escapeHtml(c.short_name)}</h4>
        <div class="cm-count">${escapeHtml(String(c.count))}</div>
        <p>${escapeHtml(c.explanation)}</p>
        <p>${escapeHtml(c.security_implication)}</p>
      </div>
    `;
  }).join("");
}

async function loadModelEvaluation() {
  const status = $("model-eval-status");
  const body = $("model-eval-body");
  const img = $("cm-image");
  const cap = $("cm-caption");
  const layout = $("cm-layout");
  if (!status || !body) return;

  status.textContent = "Loading evaluation…";
  body.classList.add("hidden");

  try {
    let detail = null;
    let usedFallback = false;
    const detailRes = await fetch(apiUrl("/api/model/confusion-matrix-detail"));
    if (detailRes.ok) {
      detail = await detailRes.json();
    } else if (detailRes.status === 404) {
      const metrics = await fetchMetricsRaw();
      detail = buildDetailFromMetrics(metrics);
      usedFallback = true;
      showApiBanner(
        "Restart Flask (Ctrl+C, then python backend/app.py) for full confusion-matrix detail and voice FAQ."
      );
    } else {
      const err = await detailRes.json().catch(() => ({}));
      throw new Error(
        err.error || "Model metrics unavailable — run python run_pipeline.py."
      );
    }

    status.textContent = usedFallback
      ? "Showing metrics from /api/model/metrics (restart server for full detail API)."
      : "";
    body.classList.remove("hidden");

    renderModelHealth(detail);
    renderModelEvalMetrics(detail);
    if (layout) layout.textContent = detail.matrix_layout || "";
    if (cap) cap.textContent = detail.heatmap_note || "";

    if (img) {
      img.src = `${apiUrl("/api/model/confusion-matrix")}?v=${encodeURIComponent(
        String(detail.n_test ?? Date.now())
      )}`;
      img.onerror = () => {
        status.textContent =
          "Confusion matrix image missing — re-run the pipeline to regenerate reports/confusion_matrix.png.";
      };
      img.onload = () => {
        if (!usedFallback) img.onerror = null;
      };
    }

    renderConfusionCells(detail);
  } catch (e) {
    status.textContent = String(e);
    body.classList.add("hidden");
  }
}

async function runAnalyze() {
  const url = $("url-input").value.trim();
  const burp = $("burp-path").value.trim() || null;
  $("loading").classList.remove("hidden");
  $("report").classList.add("hidden");
  $("btn-analyze").disabled = true;

  try {
    const res = await fetch(apiUrl("/api/analyze"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, burp_report_path: burp }),
    });
    const data = await res.json();
    renderReport(data);
  } catch (e) {
    renderReport({ ok: false, error: String(e) });
  } finally {
    $("loading").classList.add("hidden");
    $("btn-analyze").disabled = false;
  }
}

function wireAnalyzeButton() {
  const btn = $("btn-analyze");
  if (btn) btn.addEventListener("click", runAnalyze);
}

function wireVoicePanel() {
  const send = $("btn-voice-send");
  const input = $("voice-text-input");
  if (send && input) {
    send.addEventListener("click", () => {
      askVoice(input.value);
    });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        askVoice(input.value);
      }
    });
  }
  const readLast = $("btn-read-last");
  if (readLast) {
    readLast.addEventListener("click", () => {
      const s = buildLastReportSpeech();
      if (s) {
        $("voice-transcript").textContent = "(read last report)";
        $("voice-answer").textContent = s;
        setVoiceStatus("idle", "");
        speak(s);
      }
    });
  }
}

async function checkApiCapabilities() {
  if (window.location.protocol === "file:") {
    showApiBanner(
      "You opened the HTML file directly. Start the server (python backend/app.py) and open http://127.0.0.1:5000/"
    );
    return;
  }
  try {
    const res = await fetch(apiUrl("/api/health"));
    if (!res.ok) return;
    const h = await res.json();
    if (h.api_version != null && h.api_version < 2) {
      showApiBanner(
        "Backend is outdated. Stop Flask and run: python backend/app.py"
      );
    }
    if (!h.metrics_ready) {
      const status = $("model-eval-status");
      if (status) {
        status.textContent =
          "No metrics yet — run: python run_pipeline.py";
      }
    }
    if (!h.model_ready) {
      const foot = document.querySelector(".footer");
      if (foot) {
        foot.innerHTML += ` <span class="muted">Model not loaded — run pipeline.</span>`;
      }
    }
  } catch {
    showApiBanner(
      "Cannot reach the API. Start the server: python backend/app.py"
    );
  }
}

function initDashboard() {
  renderChips();
  setupSpeechRecognition();
  wireVoicePanel();
  wireAnalyzeButton();
  loadModelEvaluation();
  checkApiCapabilities();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initDashboard);
} else {
  initDashboard();
}
