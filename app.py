"""Intelligent OCR — multilingual document intelligence (FR / EN / AR).

EY-styled Streamlit interface over the GenAI OCR pipeline.
Run with:  streamlit run app.py
"""

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src import local_vlm, ocr_engine
from src.config import (OLLAMA_MODEL, SUPPORTED_IMAGE_TYPES, SUPPORTED_LANGUAGES,
                        api_key_present)
from src.evaluation import accuracy_summary
from src.language import is_mostly_rtl
from src.pipeline import Engine, Mode, OCRPipeline
from src.preprocessing import preprocess

# ------------------------------------------------------------------ EY styling

EY_YELLOW = "#FFE600"
EY_DARK = "#2E2E38"
EY_GRAY = "#747480"

st.set_page_config(page_title="Intelligent OCR | Document Intelligence",
                   page_icon="📄", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {EY_DARK}; }}
    h1, h2, h3 {{ color: #FFFFFF !important; font-weight: 700; letter-spacing: -0.5px; }}
    .ey-brandbar {{
        height: 6px; background: {EY_YELLOW}; margin: -1rem -1rem 1.2rem -1rem;
    }}
    .ey-header {{
        display: flex; align-items: baseline; gap: 14px; margin-bottom: 0.2rem;
    }}
    .ey-logo {{
        background: {EY_YELLOW}; color: {EY_DARK}; font-weight: 900;
        font-size: 1.6rem; padding: 2px 12px; line-height: 1.9rem;
    }}
    .ey-title {{ color: #fff; font-size: 1.55rem; font-weight: 700; }}
    .ey-sub {{ color: {EY_GRAY}; font-size: 0.95rem; margin-bottom: 1.4rem; }}
    .stButton>button, .stDownloadButton>button {{
        background: {EY_YELLOW}; color: {EY_DARK}; font-weight: 700;
        border: none; border-radius: 0;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: #e6cf00; color: {EY_DARK};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #1A1A24; border-right: 2px solid {EY_YELLOW};
    }}
    div[data-testid="stMetricValue"] {{ color: {EY_YELLOW}; }}
    .rtl-text {{
        direction: rtl; text-align: right; font-size: 1.05rem; line-height: 1.9;
        background: #23232F; padding: 1rem; border-left: 3px solid {EY_YELLOW};
        white-space: pre-wrap;
    }}
</style>
<div class="ey-brandbar"></div>
<div class="ey-header">
    <span class="ey-logo">EY</span>
    <span class="ey-title">Intelligent OCR &mdash; Document Intelligence</span>
</div>
<div class="ey-sub">Multilingual scanned-document recognition (Français · English · العربية),
enhanced by generative AI &mdash; Building a better working world.</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown(f"<h3 style='color:{EY_YELLOW} !important;'>Configuration</h3>",
                unsafe_allow_html=True)

    engine_labels = {
        f"Local VLM (Ollama · {OLLAMA_MODEL}) — offline": Engine.LOCAL,
        "Groq cloud API (Llama 4 vision)": Engine.GROQ,
    }
    engine_label = st.radio("VLM engine", list(engine_labels))
    engine = engine_labels[engine_label]

    key_ok = api_key_present()
    local_ok = local_vlm.is_available()
    if engine is Engine.LOCAL:
        if local_ok:
            st.success("Ollama detected — 100% local, no data leaves this machine",
                       icon="💻")
        else:
            st.error(f"Ollama not reachable or model missing. Start Ollama and run "
                     f"`ollama pull {OLLAMA_MODEL}`.", icon="⚠️")
    elif not key_ok:
        api_key_input = st.text_input("Groq API key", type="password",
                                      help="Or set GROQ_API_KEY in a .env file")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            key_ok = True
    else:
        st.success("API key detected", icon="🔑")

    tess_ok = ocr_engine.is_available()
    mode_labels = {
        "VLM vision (recommended)": Mode.GENAI_VISION,
        "Hybrid: Tesseract + VLM correction": Mode.HYBRID,
        "Tesseract only (baseline)": Mode.TESSERACT_ONLY,
    }
    available = list(mode_labels) if tess_ok else list(mode_labels)[:1]
    if not tess_ok:
        st.caption("Tesseract not detected — classical modes disabled. "
                   "The VLM engine covers the full workflow.")
    mode_label = st.radio("Recognition mode", available)
    mode = mode_labels[mode_label]

    lang_names = st.multiselect(
        "Expected languages (optional hint)",
        list(SUPPORTED_LANGUAGES.values()),
        help="Leave empty for automatic detection.",
    )
    lang_hint = [code for code, name in SUPPORTED_LANGUAGES.items()
                 if name in lang_names] or None

    st.divider()
    st.caption("Pipeline: preprocessing (deskew · denoise · CLAHE) → recognition "
               "→ GenAI post-processing → evaluation (CER/WER)")

# ------------------------------------------------------------------ main

uploaded = st.file_uploader(
    "Upload a scanned document",
    type=[ext.lstrip(".") for ext in SUPPORTED_IMAGE_TYPES] + ["pdf"],
    help="Images (PNG, JPG, WEBP, GIF) or scanned PDFs",
)

if uploaded is None:
    st.info("Upload a scanned document to start. Low-quality, noisy, skewed and "
            "mixed-language (FR/EN/AR) documents are supported.")
    st.stop()

file_bytes = uploaded.getvalue()
is_pdf = Path(uploaded.name).suffix.lower() == ".pdf"

col_doc, col_out = st.columns([5, 7], gap="large")

with col_doc:
    st.subheader("Document")
    if is_pdf:
        st.markdown(f"📕 **{uploaded.name}** — {len(file_bytes)/1024:.0f} KB "
                    "(processed natively, multi-page)")
    else:
        tab_orig, tab_pre = st.tabs(["Original", "Preprocessed"])
        with tab_orig:
            st.image(file_bytes, use_container_width=True)
        with tab_pre:
            with st.spinner("Enhancing image..."):
                pre_preview = preprocess(file_bytes,
                                         binarize=mode != Mode.GENAI_VISION)
            st.image(pre_preview.image, use_container_width=True)
            st.caption(" → ".join(pre_preview.steps))

with col_out:
    st.subheader("Extraction")

    vlm_needed = mode != Mode.TESSERACT_ONLY
    if vlm_needed and engine is Engine.GROQ and not key_ok:
        st.warning("Enter your Groq API key in the sidebar to run the cloud engine.")
        st.stop()
    if vlm_needed and engine is Engine.LOCAL and not local_ok:
        st.warning("Start Ollama (and pull the model) to run the local VLM engine.")
        st.stop()
    if is_pdf and mode != Mode.GENAI_VISION:
        st.warning("PDF input is handled by the VLM vision engine — switch mode.")
        st.stop()

    if st.button("▶  Run OCR", type="primary", use_container_width=True):
        pipeline = OCRPipeline()
        with st.spinner("Reading the document..."):
            try:
                if is_pdf:
                    result = pipeline.run_pdf(file_bytes, language_hint=lang_hint,
                                              engine=engine)
                else:
                    result = pipeline.run_image(file_bytes, mode=mode,
                                                language_hint=lang_hint,
                                                engine=engine)
            except Exception as exc:  # surface API/engine errors cleanly
                st.error(f"OCR failed: {exc}")
                st.stop()
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result:
        langs = ", ".join(SUPPORTED_LANGUAGES.get(c, c) for c in result.languages) or "—"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Languages", langs)
        m2.metric("Duration", f"{result.duration_s}s")
        m3.metric("Tokens (in/out)", f"{result.input_tokens}/{result.output_tokens}"
                  if result.model else "—")
        m4.metric("Tesseract conf.", f"{result.tesseract_confidence}%"
                  if result.tesseract_confidence is not None else "—")

        if result.raw_text is not None:
            t_final, t_raw = st.tabs(["✨ Corrected (VLM)", "Raw OCR"])
            with t_raw:
                st.code(result.raw_text or "(empty)", language=None)
        else:
            (t_final,) = st.tabs(["✨ Extracted text"])

        with t_final:
            if is_mostly_rtl(result.final_text):
                st.markdown(f"<div class='rtl-text'>{result.final_text}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(result.final_text or "*(empty)*")

        st.download_button("⬇  Download text",
                           result.final_text.encode("utf-8"),
                           file_name=f"{Path(uploaded.name).stem}_ocr.txt",
                           mime="text/plain", use_container_width=True)

        with st.expander("📏 Evaluate against ground truth (CER / WER)"):
            truth = st.text_area("Paste the reference (ground-truth) text:",
                                 height=160)
            if truth.strip():
                s = accuracy_summary(truth, result.final_text)
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("CER", f"{s['cer']:.2%}")
                e2.metric("WER", f"{s['wer']:.2%}")
                e3.metric("Char accuracy", f"{s['char_accuracy']}%")
                e4.metric("Word accuracy", f"{s['word_accuracy']}%")
