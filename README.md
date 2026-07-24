# EY Intelligent OCR — Multilingual Document Intelligence

> Enhancing OCR performance on scanned documents (Français · English · العربية)
> by leveraging recent **vision-language models (VLMs)**.

Traditional OCR struggles with low-quality, noisy, or structurally complex
documents. This project combines **advanced image preprocessing**, a
**vision-language model**, and **VLM post-correction** to deliver robust,
high-accuracy text extraction across French, English, and Arabic — including
mixed-language and right-to-left content.

Two interchangeable VLM engines back the pipeline:

- **Local VLM** *(default)* — Qwen3-VL served by [Ollama](https://ollama.com)
  on this machine's GPU. 100% offline: no API key, no cost, no document ever
  leaves the computer.
- **Groq cloud** *(optional)* — Llama 4 vision via the Groq API, kept for
  benchmarking local vs cloud quality/latency (requires `GROQ_API_KEY`).

It ships two interfaces over the same OCR core:

- A **full-stack web platform** — React (Vite) SPA + FastAPI + MongoDB, with
  accounts, authentication, and a permanent per-user document history. This
  is the primary deliverable.
- A **Streamlit prototype** (`app.py`) and a **CLI** (`cli.py`), useful for
  quick one-off tests, scripting, and CER/WER benchmarking.

## Quick start

Prerequisites already in place on a configured machine: Python deps installed
(`pip install -r requirements.txt`), `frontend/node_modules` present
(`npm install`), MongoDB running (Windows service), and Ollama installed with
the vision model pulled (`ollama pull qwen3-vl:4b-instruct`). Ollama starts
automatically with Windows — verify with `ollama list`.

Then start the two processes:

```powershell
# terminal 1 — API (from the project root)
python -m uvicorn backend.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** and sign in (default account:
`raedbouhali@gmail.com` / `raed`). Pick **Local VLM** in the sidebar (default),
drop a scan, Run OCR. No API key or internet needed.

One-off alternatives: `streamlit run app.py` (prototype UI) or
`python cli.py scan.png` (terminal).

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Recognition modes](#recognition-modes)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Running the web platform](#running-the-web-platform)
- [Web platform features](#web-platform-features)
- [Backend API reference](#backend-api-reference)
- [Environment variables](#environment-variables)
- [Streamlit prototype](#streamlit-prototype)
- [CLI](#cli)
- [Evaluation (CER / WER)](#evaluation-cer--wer)
- [Notes & known limitations](#notes--known-limitations)

## Architecture

```
                 ┌────────────────────────────────────────────────────────┐
 scanned image / │  1. PREPROCESSING          2. RECOGNITION              │
 scanned PDF ───▶│  resize / grayscale        A) VLM vision               │
                 │  denoise (NL-means)           local (Ollama/Qwen3-VL)  │
                 │  deskew · CLAHE               or cloud (Groq/Llama 4)  │
                 │  adaptive binarization*    B) Hybrid: Tesseract        │
                 │                               + VLM correction         │
                 │  3. POST-PROCESSING        C) Tesseract baseline       │
                 │  language detection                                    │
                 │  RTL handling · Markdown   4. EVALUATION               │
                 │                            CER / WER vs ground truth   │
                 └────────────────────────────────────────────────────────┘
                              * applied for the Tesseract path only
```

The OCR core (`src/`) is interface-agnostic — the web backend, the Streamlit
app, and the CLI all call the same `OCRPipeline`.

```
Browser (React SPA, :5173)
      │  fetch /api/*  (Vite dev proxy → :8000)
      ▼
FastAPI backend (:8000)  ── JWT auth, PBKDF2 password hashing
      │
      ├─▶ MongoDB (ey_ocr.users / ey_ocr.documents)
      │
      └─▶ src.pipeline.OCRPipeline
                │
                ├─▶ src.preprocessing   (OpenCV / Pillow)
                ├─▶ src.ocr_engine      (Tesseract, optional)
                ├─▶ src.local_vlm.OllamaOCR ──▶ Ollama server (Qwen3-VL, local GPU)
                └─▶ src.genai_ocr.GroqOCR   ──▶ Groq API (Llama 4 vision, optional)
```

## Recognition modes

1. **VLM vision** *(recommended, default)* — the preprocessed page is read
   directly by the vision-language model. PDFs are rasterized locally, one
   JPEG image per page (~200 DPI via PyMuPDF) — pages are transcribed
   individually and joined with a `---` separator. Best on degraded scans,
   complex layouts, tables, and Arabic script.
2. **Hybrid** — Tesseract produces a fast raw transcription; the VLM then
   fixes character confusions, broken words, and accent/diacritic errors
   **without paraphrasing**. Requires Tesseract to be installed
   (auto-detected; hidden from both UIs otherwise).
3. **Tesseract only** — classical baseline, no VLM involved. Useful for
   benchmarking against the VLM modes.

Modes 1 and 2 run on either **engine** — `local` (Ollama, default) or `groq`
(cloud) — selectable per request in the web UI, the Streamlit sidebar, and
via `--engine` on the CLI.

## Project layout

```
OCR/
├── src/                    OCR core (used by backend, Streamlit, and CLI)
│   ├── config.py             env vars, engine/model names, supported languages
│   ├── preprocessing.py      OpenCV enhancement pipeline (Pillow fallback)
│   ├── vlm_base.py           shared VLM engine base (prompts, PDF loop)
│   ├── local_vlm.py          OllamaOCR — local VLM engine (default, offline)
│   ├── genai_ocr.py          GroqOCR — optional cloud VLM engine
│   ├── ocr_engine.py         optional Tesseract baseline (fra+eng+ara)
│   ├── language.py           FR/EN/AR detection, RTL helpers
│   ├── evaluation.py         CER / WER metrics (Levenshtein)
│   └── pipeline.py           end-to-end orchestration (the 3 modes above)
│
├── backend/                 FastAPI REST API
│   ├── main.py                app factory, CORS, default-user seeding
│   ├── db.py                  MongoDB connection (PyMongo)
│   ├── security.py            PBKDF2 hashing + JWT issuing/verification
│   ├── schemas.py             Pydantic request/response models
│   ├── routes_auth.py         /api/auth/* (login, me, profile)
│   └── routes_ocr.py          /api/meta, /api/ocr, /api/history*
│
├── frontend/                 React 18 + Vite 6 SPA, EY-branded
│   ├── index.html
│   ├── vite.config.js         dev server + /api proxy to :8000
│   └── src/
│       ├── App.jsx              auth gate + minimal client-side router
│       ├── api.js               fetch wrapper, JWT storage
│       ├── styles.css           EY theme (dark/yellow), all component CSS
│       ├── pages/
│       │   ├── Login.jsx          split-screen sign-in
│       │   ├── Dashboard.jsx      upload, engine picker, result viewer
│       │   └── HistoryPage.jsx    full searchable historique (route: /history)
│       ├── components/
│       │   ├── TopBar.jsx         shared header (logo, account, sign out)
│       │   └── AccountModal.jsx   edit name / photo / password
│       └── hooks/
│           └── useHistoryList.js  shared MongoDB-backed history fetch/delete
│
├── app.py                   Streamlit prototype (single-file UI over src/)
├── cli.py                   command-line interface
└── requirements.txt          Python dependencies (backend + src + Streamlit)
```

## Setup

**Python side** (OCR core, backend, Streamlit, CLI):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Local VLM engine** (default — no API key, fully offline): install
[Ollama](https://ollama.com/download), then pull a vision-capable model:

```powershell
ollama pull qwen3-vl:4b-instruct     # ~3.3 GB, fits a 6 GB laptop GPU
```

Any vision model works — override with `OLLAMA_MODEL` in `.env`
(e.g. `qwen3-vl:8b-instruct` on bigger GPUs, `qwen2.5vl:3b` on smaller
ones). Prefer `*-instruct` tags: the base `qwen3-vl` tags run in "thinking"
mode and burn the whole token budget on reasoning instead of transcription.

**Groq cloud engine** (optional): `copy .env.example .env` and paste your
`GROQ_API_KEY` (free at console.groq.com).

Optional classical baseline: install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
with the `fra`, `eng`, `ara` language packs, then `pip install pytesseract`.
Once installed, the **Hybrid** and **Tesseract only** modes appear
automatically in both UIs — nothing else to configure.

**MongoDB**: install locally (Windows service, Docker, or Atlas) and point
`MONGODB_URI` in `.env` at it — defaults to `mongodb://localhost:27017`.

**Frontend**:

```powershell
cd frontend
npm install
```

## Running the web platform

Before starting, make sure the two background services are up:

- **MongoDB** — installed as a Windows service, starts automatically.
- **Ollama** — starts automatically with Windows (tray icon). If the UI shows
  the Local VLM engine as unavailable, launch Ollama and check the model is
  present with `ollama list`.

Then two processes, both from the project root unless noted:

```powershell
# terminal 1 — API
python -m uvicorn backend.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**. On first backend startup a default account is
seeded automatically:

- email: `raedbouhali@gmail.com`
- password: `raed`
- override with `DEFAULT_USER_EMAIL` / `DEFAULT_USER_PASSWORD` /
  `DEFAULT_USER_NAME` in `.env` before the first run

Every extraction is stored per-user in MongoDB (`ey_ocr.documents`) — nothing
is lost between sessions.

## Web platform features

- **Authentication** — email/password login, JWT (24 h expiry, `HS256`),
  passwords hashed with PBKDF2-SHA256 (200k iterations, random salt per
  user). No signup flow by design — accounts are seeded or created directly
  in MongoDB; this is an internal/prototype tool.
- **Account management** — click the avatar in the top bar to open the
  account modal: change display name, upload/remove a profile photo
  (auto-cropped to a square and downscaled client-side before upload), or
  change password (requires the current password).
- **Upload & run** — drag-and-drop or browse (PNG/JPG/WEBP/PDF), drop
  anywhere on the page (full-screen drop overlay), pick the engine and an
  optional language hint, run. A live step indicator and a document-scan
  animation show progress instead of a generic spinner.
- **Settings memory** — the last-used engine and language hint are
  remembered in `localStorage` and restored on your next visit.
- **Result view** — extracted text (RTL-rendered automatically when
  Arabic-heavy), per-run metrics (engine, languages, duration, token usage,
  Tesseract confidence when applicable), copy-to-clipboard, download as
  `.txt`.
- **Historique** — a compact live summary (document count, quick stats) sits
  in the dashboard sidebar with a button to the dedicated `/history` page: a
  full searchable, deletable grid of every extraction you've run. History
  entries are real links (`?doc=<id>` / navigable URLs) — right-click to open
  in a new tab, browser back/forward works, refreshing keeps the open
  document.
- **No page scroll on desktop** — the app fills the viewport exactly; only
  the historique list and the extracted-text panel scroll internally. Narrow
  screens (≤900px) fall back to normal full-page scrolling.
- **EY branding** — dark (`#2E2E38`) / yellow (`#FFE600`) theme throughout,
  the real EY logo (`frontend/public/ey.png`) as favicon and in the UI, the
  signature diagonal "beam" motif animated on the login and welcome screens.

## Backend API reference

All routes are prefixed with `/api`. Protected routes require
`Authorization: Bearer <token>` (obtained from `/auth/login`).

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /health` | — | liveness + MongoDB connectivity check |
| `POST /auth/login` | — | `{email, password}` → JWT + user profile |
| `GET /auth/me` | ✓ | current user profile |
| `PATCH /auth/profile` | ✓ | update `name` / `avatar` / password |
| `GET /meta` | ✓ | available modes/engines, supported languages, active models |
| `POST /ocr` | ✓ | multipart upload (`file`, `mode`, `engine`, `langs`) → extraction, stored |
| `GET /history` | ✓ | last 50 documents for the current user |
| `GET /history/{id}` | ✓ | full extraction record |
| `DELETE /history/{id}` | ✓ | delete a record (204) |

## Environment variables

Set in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OCR_ENGINE` | `local` | default VLM engine: `local` (Ollama) or `groq` (cloud) |
| `OLLAMA_MODEL` | `qwen3-vl:4b-instruct` | vision model served by Ollama for the local engine |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server address |
| `OLLAMA_NUM_CTX` | `8192` | local model context window per page — Ollama's ~4k default truncates dense pages; raising it above 8192 forces CPU spill on small GPUs (~3x slower) for no gain |
| `GROQ_API_KEY` | *(only for the cloud engine)* | Groq API key — free at [console.groq.com](https://console.groq.com) |
| `OCR_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | cloud vision model (Groq engine) |
| `OCR_MAX_TOKENS` | `8000` | max completion tokens per VLM call |
| `OCR_PDF_DPI` | `300` | PDF rasterization resolution — 200 is faster but drops fine print (form helper text, footnotes) |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | `ey_ocr` | database name |
| `JWT_SECRET` | *(random per install)* | HMAC secret for signing JWTs — keep private |
| `DEFAULT_USER_EMAIL` | `raedbouhali@gmail.com` | seeded account email |
| `DEFAULT_USER_PASSWORD` | `raed` | seeded account password |
| `DEFAULT_USER_NAME` | `Raed Bouhali` | seeded account display name |

## Streamlit prototype

```powershell
streamlit run app.py
```

Upload a scan (PNG/JPG/WEBP or PDF), pick an engine and a mode, run, review
the side-by-side original vs preprocessed images, read the extraction
(RTL-aware for Arabic), download the text, and optionally paste a ground
truth to get CER/WER. The local engine needs only a running Ollama; the cloud
engine reads `GROQ_API_KEY` from `.env` or accepts it in the sidebar.

## CLI

```powershell
python cli.py scan.png                           # local VLM (offline)
python cli.py contrat.pdf -o contrat.txt
python cli.py scan.jpg --engine groq             # cloud VLM instead
python cli.py scan.jpg --mode hybrid --langs fr ar
python cli.py scan.png --truth reference.txt     # prints CER / WER
```

## Evaluation (CER / WER)

`src/evaluation.py` implements Character Error Rate and Word Error Rate with
Unicode NFC normalization (required for consistent Arabic comparison). Use
the Streamlit app's *Evaluate against ground truth* panel or `--truth` on the
CLI to compare modes on your own corpus.

## Notes & known limitations

- The vision model handles skew, noise, and layout natively; preprocessing
  still measurably helps on faint or low-contrast scans.
- Local-engine speed depends on the GPU and, critically, on `OLLAMA_NUM_CTX`:
  the context window's KV cache competes with the model for VRAM, so an
  oversized value spills model layers to CPU and slows every page ~3x. The
  default (8192) is tuned to keep a 4B model fully GPU-resident on a 6 GB card
  (~30 s per dense page). The model loads into VRAM on the first request
  (one-time delay after Ollama starts).
- Scanned PDFs are read one page per model call, sequentially — on a single
  GPU that is the fast path (concurrent calls just contend for the same VRAM).
  Total time ≈ pages × per-page time.
- Multi-page scanned PDFs are rasterized locally with PyMuPDF (no Poppler
  dependency) and read one page at a time, at 300 DPI by default — fine print
  (form placeholders, footnotes) needs that resolution to be read reliably;
  lower `OCR_PDF_DPI` to 200 to trade small-print recall for speed.
- Images are automatically resized to the model's optimal resolution
  (≤ 2576 px long edge) and sent as JPEG to stay under Groq's payload limits;
  tiny scans are upscaled before recognition.
- Groq's model catalog changes over time — check
  [console.groq.com/docs/models](https://console.groq.com/docs/models) for
  current vision-capable model IDs and override via `OCR_MODEL` in `.env` if
  the default ever 404s.
- No user signup flow — accounts are seeded on backend startup or inserted
  directly into the `ey_ocr.users` MongoDB collection. Appropriate for an
  internal prototype, not for public deployment as-is.
- The web platform is Windows/dev-oriented as configured (Vite proxy target
  is `127.0.0.1`, not `localhost`, to sidestep an IPv6/IPv4 resolution
  mismatch observed on Windows). Adjust `frontend/vite.config.js` if
  deploying elsewhere.
