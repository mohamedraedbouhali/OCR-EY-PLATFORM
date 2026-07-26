import { useEffect, useMemo, useRef, useState } from "react";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";
import { runOCR, getDocument, meta } from "../api.js";
import useHistoryList from "../hooks/useHistoryList.js";

const MODE_LABELS = {
  genai_vision: "VLM Vision",
  hybrid: "Hybrid (Tesseract + VLM)",
  tesseract_only: "Tesseract only",
};

const ENGINE_LABELS = {
  local: "Local VLM",
  groq: "Cloud (Groq)",
};

const ENGINE_UNAVAILABLE_HINT = {
  local: "Ollama is not running (or the model is not pulled) on the server",
  groq: "No GROQ_API_KEY configured on the server",
};

const PROCESS_STEPS = {
  genai_vision: ["Enhancing image", "VLM vision reading", "Structuring output"],
  hybrid: ["Preprocessing", "Tesseract pass", "VLM correction"],
  tesseract_only: ["Preprocessing", "Tesseract recognition", "Finalizing"],
};

const PREFS_KEY = "ey_ocr_prefs";

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
  } catch {
    return {};
  }
}

function isArabicHeavy(text) {
  if (!text) return false;
  const arabic = (text.match(/[؀-ۿ]/g) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  return arabic > latin;
}

/** Animate a number towards its target with an ease-out curve. */
function useCountUp(target, duration = 800) {
  const [val, setVal] = useState(0);
  const prev = useRef(0);
  useEffect(() => {
    const from = prev.current;
    prev.current = target;
    if (from === target) {
      setVal(target);
      return;
    }
    const t0 = performance.now();
    let raf;
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(Math.round(from + (target - from) * eased));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return val;
}

export default function Dashboard({ user, navigate, notify }) {
  const [info, setInfo] = useState(null);
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState(() => loadPrefs().mode || "genai_vision");
  const [engine, setEngine] = useState(() => loadPrefs().engine || "local");
  const [langs, setLangs] = useState(() => loadPrefs().langs || []);
  const [busy, setBusy] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [pageDrag, setPageDrag] = useState(false);
  const fileInput = useRef(null);
  const { items, refresh } = useHistoryList();

  useEffect(() => {
    meta().then(setInfo).catch(() => {});
  }, []);

  // open the document named in the URL (deep link / refresh / back-forward)
  useEffect(() => {
    const initialId = new URLSearchParams(window.location.search).get("doc");
    if (initialId) openItem(initialId, { updateUrl: false });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    function onPopState() {
      const id = new URLSearchParams(window.location.search).get("doc");
      if (id) openItem(id, { updateUrl: false });
      else setResult(null);
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // settings memory — engine + language choices survive reloads
  useEffect(() => {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ mode, engine, langs }));
  }, [mode, engine, langs]);

  // if the remembered mode/engine is no longer offered by the server, fall back
  useEffect(() => {
    if (!info) return;
    if (!info.modes.includes(mode)) setMode(info.modes[0]);
    if (info.engines && info.engines[engine] === false) {
      const firstUp = Object.keys(info.engines).find((e) => info.engines[e]);
      if (firstUp) setEngine(firstUp);
    }
  }, [info]); // eslint-disable-line react-hooks/exhaustive-deps

  // processing step indicator while the engine works
  useEffect(() => {
    if (!busy) {
      setStepIdx(0);
      return;
    }
    const id = setInterval(
      () => setStepIdx((i) => Math.min(i + 1, PROCESS_STEPS[mode].length - 1)),
      2800
    );
    return () => clearInterval(id);
  }, [busy, mode]);

  // full-page drag & drop overlay
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e) => e.dataTransfer?.types?.includes("Files");
    const enter = (e) => {
      if (!hasFiles(e)) return;
      depth++;
      setPageDrag(true);
    };
    const leave = () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) setPageDrag(false);
    };
    const over = (e) => e.preventDefault();
    const drop = (e) => {
      e.preventDefault();
      depth = 0;
      setPageDrag(false);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);

  /** Reflects the open document in the URL — real link, shareable, back/forward-able. */
  function pushDocUrl(id) {
    const url = id ? `?doc=${id}` : window.location.pathname;
    window.history.pushState({ docId: id || null }, "", url);
  }

  const firstName = (user.name || user.email).split(/[\s@]/)[0];
  const list = items || [];

  const stats = useMemo(() => {
    const docs = list.length;
    const avg =
      docs > 0 ? list.reduce((s, i) => s + (i.duration_s || 0), 0) / docs : 0;
    const langSet = new Set(list.flatMap((i) => i.languages || []));
    return {
      docs,
      avg: avg ? `${avg.toFixed(1)}s` : "—",
      langs: langSet.size
        ? [...langSet].map((l) => l.toUpperCase()).join(" · ")
        : "—",
    };
  }, [list]);

  const docsAnimated = useCountUp(stats.docs);

  function pickFile(f) {
    if (f) {
      setFile(f);
      setResult(null);
      setError("");
      pushDocUrl(null);
    }
  }

  async function handleRun() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const res = await runOCR(file, mode, langs, engine);
      setResult(res);
      refresh();
      pushDocUrl(res.id);
      notify("Extraction complete");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function openItem(id, { updateUrl = true } = {}) {
    setError("");
    try {
      const doc = await getDocument(id);
      setResult(doc);
      if (updateUrl) pushDocUrl(id);
    } catch (err) {
      setError(err.message);
    }
  }

  function copyText() {
    if (!result) return;
    navigator.clipboard
      .writeText(result.text)
      .then(() => notify("Copied to clipboard"));
  }

  async function downloadPDF() {
    if (!result) return;
    const stem = result.filename.replace(/\.[^.]+$/, "");
    const isRtl = isArabicHeavy(result.text);
    const esc = (s) =>
      s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const langs =
      (result.languages || []).map((l) => l.toUpperCase()).join(" · ") || "—";
    const engine = ENGINE_LABELS[result.engine] || result.engine || "";
    const date = new Date().toLocaleDateString();

    // Render an A4-width node the browser paints (Arabic shaped + RTL correct),
    // snapshot it with html2canvas, and emit a real .pdf file via jsPDF — a
    // direct download, no print dialog. Fully offline (both libs are bundled).
    const el = document.createElement("div");
    el.style.cssText =
      "position:fixed;left:-99999px;top:0;width:794px;padding:48px;" +
      "background:#fff;color:#111;font-family:'Segoe UI',Arial,sans-serif;";
    el.innerHTML = `
      <div style="border-bottom:3px solid #FFE600;padding-bottom:10px;margin-bottom:18px;">
        <span style="font-weight:900;background:#2E2E38;color:#FFE600;padding:2px 8px;font-size:15px;">EY</span>
        <span style="font-size:15px;"> Intelligent OCR</span>
        <div style="font-size:18px;font-weight:700;margin:8px 0 2px;">${esc(stem)}</div>
        <div style="font-size:11px;color:#666;">${esc(engine)} · ${esc(langs)} · ${esc(date)}</div>
      </div>
      <pre style="white-space:pre-wrap;word-wrap:break-word;font-family:inherit;font-size:13px;line-height:1.55;margin:0;${
        isRtl ? "direction:rtl;text-align:right;" : ""
      }">${esc(result.text)}</pre>`;
    document.body.appendChild(el);
    try {
      const canvas = await html2canvas(el, { scale: 2, backgroundColor: "#ffffff" });
      const pdf = new jsPDF({ unit: "pt", format: "a4" });
      const pageW = pdf.internal.pageSize.getWidth();
      const pageH = pdf.internal.pageSize.getHeight();
      const imgH = (canvas.height * pageW) / canvas.width;
      const imgData = canvas.toDataURL("image/jpeg", 0.92);
      let pos = 0;
      let left = imgH;
      pdf.addImage(imgData, "JPEG", 0, pos, pageW, imgH);
      left -= pageH;
      while (left > 0) {
        pos -= pageH;
        pdf.addPage();
        pdf.addImage(imgData, "JPEG", 0, pos, pageW, imgH);
        left -= pageH;
      }
      pdf.save(`${stem}_ocr.pdf`);
      notify("PDF downloaded");
    } catch (err) {
      setError("PDF export failed: " + err.message);
    } finally {
      el.remove();
    }
  }

  const rtl = result && isArabicHeavy(result.text);
  const steps = PROCESS_STEPS[mode] || PROCESS_STEPS.genai_vision;

  return (
    <div className="body-row">
      {/* ================= sidebar ================= */}
      <aside className="sidebar">
        <div className="sidebar-scroll">
          <h2 className="panel-title">New extraction</h2>

          <div
            className={`dropzone ${dragOver ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              pickFile(e.dataTransfer.files[0]);
            }}
            onClick={() => fileInput.current?.click()}
          >
            <input
              ref={fileInput}
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.pdf"
              hidden
              onChange={(e) => pickFile(e.target.files[0])}
            />
            {file ? (
              <>
                <div className="drop-filename">{file.name}</div>
                <div className="drop-hint">
                  {(file.size / 1024).toFixed(0)} KB — click to replace
                </div>
              </>
            ) : (
              <>
                <div className="drop-main">Drop a scan here or browse</div>
                <div className="drop-hint">PNG · JPG · WEBP · PDF</div>
              </>
            )}
          </div>

          <h2 className="panel-title engine-title">Engine</h2>
          <div className="engine-toggle" role="radiogroup" aria-label="OCR engine">
            {Object.keys(ENGINE_LABELS).map((e) => {
              const down = info?.engines && info.engines[e] === false;
              return (
                <button
                  key={e}
                  type="button"
                  className={`engine-opt ${engine === e ? "active" : ""}`}
                  disabled={down}
                  title={down ? ENGINE_UNAVAILABLE_HINT[e] : ""}
                  onClick={() => setEngine(e)}
                >
                  {ENGINE_LABELS[e]}
                </button>
              );
            })}
          </div>
          <p className="engine-hint">
            {engine === "local"
              ? `Runs on this machine — ${info?.models?.local || "Ollama"} · offline`
              : `Groq cloud API — ${info?.models?.groq || "Llama 4 vision"}`}
          </p>
        </div>

        {/* run zone — always visible, never clipped */}
        <div className="sidebar-run">
          <button
            className={`btn-primary run-btn ${file && !busy ? "ready" : ""}`}
            disabled={!file || busy}
            onClick={handleRun}
          >
            {busy ? "Processing…" : "Run OCR"}
          </button>
          {!file && !busy && (
            <p className="run-hint">Select a document to enable</p>
          )}
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
        </div>

        {/* historique summary — enhanced empty/loaded state, links to the full page */}
        <div className="sidebar-history">
          <h2 className="panel-title history-title">Historique</h2>
          <div className="history-summary">
            <svg
              className="history-summary-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              aria-hidden="true"
            >
              <circle cx="12" cy="13" r="8" />
              <path d="M12 9v4l2.5 1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M9 2h6M5 5l1.5 1.5M19 5l-1.5 1.5" strokeLinecap="round" />
            </svg>

            {items === null ? (
              <div className="skeleton small" />
            ) : (
              <>
                <div className="history-summary-count">{docsAnimated}</div>
                <p className="history-summary-text">
                  {stats.docs === 0
                    ? "No documents yet — every extraction is saved here permanently."
                    : `document${stats.docs > 1 ? "s" : ""} processed, always searchable.`}
                </p>
              </>
            )}

            <button
              className="btn-primary history-view-all"
              onClick={() => navigate("/history")}
            >
              {stats.docs === 0 ? "View Historique" : "View Historique →"}
            </button>
          </div>
        </div>
      </aside>

      {/* ================= main column ================= */}
      <div className="main-col">
        <section className="welcome">
          <div className="welcome-beam" aria-hidden="true" />
          <div className="welcome-inner">
            <div>
              <h1 className="welcome-title">
                Welcome back, <span className="accent">{firstName}</span>
              </h1>
              <p className="welcome-sub">
                Upload a scanned document and let the GenAI engine read it.
              </p>
            </div>
            <div className="stat-row">
              <div className="stat">
                <span className="stat-value">{docsAnimated}</span>
                <span className="stat-label">Documents</span>
              </div>
              <div className="stat">
                <span className="stat-value">{stats.avg}</span>
                <span className="stat-label">Avg. duration</span>
              </div>
              <div className="stat">
                <span className="stat-value">{stats.langs}</span>
                <span className="stat-label">Languages</span>
              </div>
            </div>
          </div>
        </section>

        <main className="content">
          <section className="panel result">
            {busy && (
              <div className="result-empty">
                <div className="scan-doc" aria-hidden="true">
                  <span className="scan-doc-line" />
                  <span className="scan-doc-line short" />
                  <span className="scan-doc-line" />
                  <span className="scan-doc-line short" />
                  <span className="scan-beam" />
                </div>
                <div className="steps">
                  {steps.map((s, i) => (
                    <div
                      key={s}
                      className={`step ${i === stepIdx ? "active" : ""} ${
                        i < stepIdx ? "done" : ""
                      }`}
                    >
                      <span className="step-dot" />
                      {s}
                    </div>
                  ))}
                </div>
                <p className="muted-note">
                  Complex or multi-page documents can take up to a minute.
                </p>
              </div>
            )}

            {!busy && !result && (
              <div className="result-empty">
                <img
                  className="empty-logo"
                  src="/ey.png"
                  alt=""
                  aria-hidden="true"
                />
                <p className="empty-title">No document selected</p>
                <p className="muted-note">
                  Upload a scan in the sidebar, or open one from your historique.
                </p>
              </div>
            )}

            {!busy && result && (
              <div className="result-body">
                <div className="result-head">
                  <h2 className="result-filename">{result.filename}</h2>
                  <div className="result-actions">
                    <button className="btn-ghost" onClick={copyText}>
                      Copy
                    </button>
                    <button className="btn-primary" onClick={downloadPDF}>
                      Download PDF
                    </button>
                  </div>
                </div>

                {/* metrics row hidden per request */}

                <pre className={`ocr-text reveal ${rtl ? "rtl" : ""}`}>
                  {result.text}
                </pre>
              </div>
            )}
          </section>
        </main>
      </div>

      {/* full-page drop overlay */}
      {pageDrag && (
        <div
          className="drop-overlay"
          onDrop={(e) => {
            e.preventDefault();
            setPageDrag(false);
            pickFile(e.dataTransfer.files[0]);
          }}
          onDragOver={(e) => e.preventDefault()}
        >
          <div className="drop-overlay-box">
            <img src="/ey.png" alt="" aria-hidden="true" />
            <p>Drop your document to upload</p>
          </div>
        </div>
      )}
    </div>
  );
}
