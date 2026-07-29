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
- **Groq cloud** *(optional)* — Llama 4 vision via the Groq API, for
  benchmarking local vs cloud quality/latency (requires `GROQ_API_KEY`). Note:
  Groq removed its Llama-4 vision models from general access in mid-2026, so
  this engine may 404 until a vision-capable model is available again — the
  local engine is the reliable default (see [limitations](#notes--known-limitations)).

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
- [Completeness QA (two-model comparison)](#completeness-qa-two-model-comparison)
- [Recall verification (`--verify`)](#recall-verification---verify)
- [Format validation (`--check`)](#format-validation---check)
- [Applying corrections (`--fix`)](#applying-corrections---fix)
- [Evaluation (CER / WER)](#evaluation-cer--wer)
- [Performance](#performance)
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
                 │  language detection        + optional --verify pass    │
                 │  RTL handling · Markdown                               │
                 │                            4. QUALITY CONTROL          │
                 │                            CER / WER vs ground truth   │
                 │                            format validation (no truth)│
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
   JPEG image per page (300 DPI via PyMuPDF, see `OCR_PDF_DPI`) — pages are
   transcribed individually and joined with a `---` separator. Best on degraded
   scans, complex layouts, tables, and Arabic script. The prompt captures
   document structure as Markdown and reports checkbox/tick-box states (☑/☐).
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
│   ├── compare.py            two-model completeness QA (word-level diff)
│   ├── validators.py         format checks for postcodes/phones/emails
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
python cli.py contrat.pdf --compare --langs fr   # two-model completeness QA
python cli.py form.pdf --verify                  # second recall pass (~2x slower)
python cli.py form.pdf --check                   # postcode/phone/email format check
python cli.py form.pdf --fix                     # apply the unambiguous corrections
python cli.py form.pdf --verify --fix            # both
```

`--verify`, `--check` and `--fix` are currently **CLI-only** — the web platform
and the Streamlit prototype always run the single-pass extraction. The
underlying `verify` flag is already plumbed through `OCRPipeline.run_image` /
`run_pdf`, so exposing it in the other interfaces is a one-line change.

## Completeness QA (two-model comparison)

No OCR engine — local or cloud — can *guarantee* it never drops a word on a
scanned document; accuracy is always a measured rate, not an absolute. The
honest way to push toward completeness is to read each page with **two
independent models** and review wherever they disagree: agreement = high
confidence, disagreement = a spot worth a human glance.

`python cli.py <file> --compare` does exactly this, fully offline, using the
two local models (`qwen3-vl:4b-instruct` vs `qwen3-vl:8b-instruct` — pull the
8b once with `ollama pull qwen3-vl:8b-instruct`). It prints a word-level
agreement percentage and every disagreement span, marking `·(missing)·` when
one model skipped text the other caught. In practice the two models have
*different* blind spots — the 4b is stronger on some checkboxes/fine print,
the 8b catches some headers the 4b drops — so the diff reliably surfaces real
omissions a single pass would hide. `src/compare.py` exposes the same as a
library (`compare_image` / `compare_pdf`).

## Recall verification (`--verify`)

The dominant failure mode of a vision model on a dense form is **omission**,
not hallucination: faint print, marginal notes, table cells and checkboxes get
silently skipped. A second read of the same page rarely helps on its own — the
model tends to skip the same things twice.

`--verify` instead runs a two-pass loop (`src/vlm_base.py`, `extract_text`):

1. **Pass 1** — normal transcription with the extraction prompt.
2. **Pass 2** — the model receives *the same image plus its own draft* wrapped
   in `<draft>` tags, under a prompt whose only job is to find what the draft
   missed and fix what it got wrong, while being forbidden to remove correct
   text, translate, reorder or summarize.

Giving the model its own output as a checklist turns an open-ended
transcription task into a targeted search, which is what recovers the skipped
spans. On PDFs the pass runs per page.

**Cost: exactly 2× the model calls** — so ~2× the time and ~2× the tokens, and
on a PDF that is per page.

**Measured, honestly.** On a 10-page handwritten safeguarding form (7.6 min
single-pass, 7.6 → 12.7 min with `--verify`) the second pass corrected one
occluded phone number that the first pass had guessed wrong, but it did **not**
recover a known dropped clause — a line of handwriting the first pass had
skipped was still missing after verification. The pass is worth running on
documents where completeness matters, but treat it as reducing the omission
rate, not closing it. `--compare` and a human reviewer remain the backstops.

## Format validation (`--check`)

Digits and proper nouns are where a vision model fails first: they carry no
linguistic context to fall back on, so a misread `B` → `8` produces something
that looks perfectly plausible. On a form, those are exactly the fields that
matter — a wrong postcode or phone number is worse than a wrong adjective.

Those fields are *format-constrained*, so a chunk of that error class is
catchable **with no ground truth at all**. `src/validators.py` runs three
checks over the transcription:

| Check | What it catches | Example |
|---|---|---|
| **Structural** | token violates the field's grammar | `TS25 28Q` — a UK postcode cannot hold a digit in its final two positions |
| **Consistency** | the same entity transcribed two ways | `TS25 2SQ` in section 1 vs `TS2S 2SQ` in section 2 — both legal, so one is misread |
| **Plausibility** | wrong length | a 9- or 10-digit UK phone number means a dropped digit |
| **Term drift** | one proper noun spelled two ways | `Harlepool` against `Hartlepool` elsewhere in the same document |

The term check counts occurrences **case-insensitively as substrings**, so
email domains vote. That detail decides real cases: on the test form both
spellings appeared 3× as words, and only `hartlepool.gov.uk` broke the tie 6–3.
A pair where the shorter term is a prefix of the longer is skipped as
inflection — otherwise `Concerns` "corrects" to `Concern` and the check
corrupts text that was right.

When exactly **one** OCR confusion substitution repairs a malformed token
(`8`→`B`, `5`→`S`, `2`→`Z`…), the fix is unambiguous and is printed as a
suggestion. Ambiguous repairs are reported without one.

```
Format validation: 2 error(s), 1 warning(s)
------------------------------------------------------------------------
  [ERROR  ] postcode: TS25 28Q
            not a valid UK postcode — the final two characters must be letters (excluding C, I, K, M, O, V)
            -> did you mean: TS25 2BQ
  [ERROR  ] phone: 01642 4536
            UK numbers have 11 digits; this has 9 — a character was dropped or invented
  [WARNING] postcode: TS252BQ / TS252SQ
            these postcodes differ by only 1-2 characters — likely one address misread two different ways; check against the page
```

Note the third finding: the repaired `TS25 2BQ` and a `TS25 2SQ` read elsewhere
in the document are both structurally valid, but they cannot both be right —
the consistency check surfaces that where the structural check alone would not.

**What this cannot do:** it validates *form, not truth*. `01422 065770` is a
perfectly well-formed UK number that happens to be Halifax rather than the
Teesside `01642` the document actually says — no format rule catches that.
Only a lookup against known-good values would, and that is document-specific.
Treat a clean report as "no detectable format errors", never as "no errors".

The rules are UK-specific (postcode grammar, `0[12378]` phone prefixes) because
that is what the test corpus contains; other locales need their own patterns.

## Applying corrections (`--fix`)

`--check` reports; `--fix` acts. It applies **only** the findings that carry a
suggestion — the ones where exactly one candidate satisfied the field's
grammar, so there is nothing to choose between. Everything else (occlusion,
disagreeing siblings, well-formed but wrong values) is left alone and reported
for a human.

Corrections are applied *before* the text is written, so `-o` output and any
`--truth` CER/WER reflect what you actually receive. `--fix` implies `--check`.

```
[fixed] postcode: TS25 28Q  -> TS25 2BQ
[fixed] postcode: TS19 28Q  -> TS19 2BQ
[fixed] term:     Harlepool -> Hartlepool
```

Measured on the 10-page form: 4 structural errors → **0 errors, 3 warnings**,
with no correct text altered. The three remaining warnings were genuine
human-review items, not false alarms.

`src/validators.py` exposes the same as a library:

```python
from src.validators import validate, apply_fixes, format_report

findings      = validate(text)          # report only
fixed, applied = apply_fixes(text)      # returns corrected text + what changed
```

**Auto-correction is a deliberate risk.** A wrong auto-fix is worse than no
fix, because it is invisible. The suggestion rule is therefore strict — one
surviving candidate, or nothing — and every applied change is printed to
stderr so it can be audited. Both false-positive classes found so far
(`CALL 101` read as a postcode, `Concerns` "corrected" to `Concern`) surfaced
only by running against real 10-page output, not synthetic test strings.

## Evaluation (CER / WER)

`src/evaluation.py` implements Character Error Rate and Word Error Rate with
Unicode NFC normalization (required for consistent Arabic comparison). Use
the Streamlit app's *Evaluate against ground truth* panel or `--truth` on the
CLI to compare modes on your own corpus.

## Performance

Measured on a 6 GB NVIDIA laptop GPU (RTX 4050) with the default local engine
(`qwen3-vl:4b-instruct`, 300 DPI, `OLLAMA_NUM_CTX=8192`):

| Workload | Time |
|---|---|
| Printed page | ~25–29 s |
| Dense handwritten page | ~32–47 s |
| 3-page printed PDF | ~86 s (measured) |
| 10-page form | ~5 min |
| First page after Ollama is idle | **+ ~10 s** one-time (model loads into VRAM) |
| `--verify` recall pass | ~2× (two model calls per page) |
| `--compare` QA mode | ~2.5–3× (runs the 4b **and** 8b models) |
| `--check` / `--fix` | negligible (pure regex, no model call) |

Rule of thumb: **time ≈ pages × ~30 s**, plus a one-time ~10 s on the first
scan after a reboot. Two knobs trade speed for completeness:

- `OLLAMA_NUM_CTX` — the dominant lever. The default 8192 keeps the 4B model
  fully GPU-resident; larger values spill layers to CPU and slow every page
  ~3× for no accuracy gain on single pages.
- `OCR_PDF_DPI` — lowering 300→250 gives ~20 s/page but can drop the finest
  print; 300 is the default to prioritize completeness over speed.

Pages are processed sequentially: on a single GPU that is the fast path
(concurrent calls just contend for the same VRAM).

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
- No OCR engine can guarantee a complete transcription; three independent
  tools address that from different angles — `--verify` (two-pass recall),
  `--compare` (two-model disagreement), and `--check` (format validation).
  None replaces a human review of a critical document.
- `--check` validates **form, not truth**: a well-formed but wrong value (a
  valid postcode for the wrong town) passes clean. Its rules are UK-specific
  and would need extending for other locales.
- **Occluded characters are guessed, not flagged.** Where a watermark or stamp
  is printed across a value, the local 4b model fills the gap with a plausible
  digit rather than marking it unreadable — so a phone number can come back
  confident and wrong. The extraction prompt asks for `[?]` per hidden
  character; the 4b model does not comply (measured: zero markers emitted on a
  form whose watermark crosses six phone numbers). The instruction is kept for
  larger models, but **do not rely on it** — `--check` catches the resulting
  wrong-length numbers, which is currently the real defence.
- Grading OCR output by asking a chatbot "does this match the PDF?" does not
  work: the judge has to re-read the document to answer, and reports its own
  misreads as your errors. Hand-write a `truth.txt` for one page and use
  `--truth` instead.
- The **Hybrid** and **Tesseract only** modes require the Tesseract binary plus
  `pip install pytesseract` (commented out in `requirements.txt` by default).
  Without it both modes are hidden from the web UI and the Streamlit sidebar,
  and only **VLM vision** is offered — this is the normal configuration. They
  are kept for benchmarking the GenAI approach against a classical one.
- Images are automatically resized to the model's optimal resolution
  (≤ 2576 px long edge) and sent as JPEG to stay under Groq's payload limits;
  tiny scans are upscaled before recognition.
- Groq's model catalog changes over time — check
  [console.groq.com/docs/models](https://console.groq.com/docs/models) for
  current vision-capable model IDs and override via `OCR_MODEL` in `.env` if
  the default ever 404s. **As of July 2026 Groq had removed all Llama-4 vision
  models from general access**, so the cloud *vision* engine may be
  unavailable; the local Ollama engine is the reliable default and the
  `--compare` QA uses two local models, needing no cloud at all.
- No user signup flow — accounts are seeded on backend startup or inserted
  directly into the `ey_ocr.users` MongoDB collection. Appropriate for an
  internal prototype, not for public deployment as-is.
- The web platform is Windows/dev-oriented as configured (Vite proxy target
  is `127.0.0.1`, not `localhost`, to sidestep an IPv6/IPv4 resolution
  mismatch observed on Windows). Adjust `frontend/vite.config.js` if
  deploying elsewhere.
