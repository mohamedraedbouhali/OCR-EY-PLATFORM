"""Central configuration for the OCR pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

# Which VLM engine backs the vision/hybrid modes: "local" (Ollama) or "groq" (cloud)
DEFAULT_ENGINE = os.getenv("OCR_ENGINE", "local")

# Local VLM served by Ollama — the default, fully offline engine.
# Use an *-instruct tag: the base qwen3-vl tags default to "thinking" mode,
# which spends the whole token budget reasoning and returns empty content.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Context window for the local model. Each page is a separate call, so this
# only needs to fit ONE page: ~4300 image tokens + the transcription. Ollama's
# ~4k default truncates dense pages; 8192 leaves ample headroom. Going higher
# (e.g. 16384) wastes VRAM on KV cache and forces model layers onto the CPU on
# a 6 GB GPU — measured ~3x slower per page for zero accuracy gain.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# Cloud model used when the Groq engine is selected (Groq — Llama 4 vision)
MODEL = os.getenv("OCR_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Max completion tokens per VLM call (both engines)
MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "8000"))

# PDF rasterization resolution. Fine print (form helper text, footnotes) sits
# right at the model's legibility threshold: measured, 300 DPI reliably reads
# tiny placeholder boxes that 200-250 DPI drop. Kept at 300 to prioritize
# completeness over speed; lower it only if you don't need the finest print.
PDF_RASTER_DPI = int(os.getenv("OCR_PDF_DPI", "300"))

# Languages the system is tuned for
SUPPORTED_LANGUAGES = {
    "fr": "French",
    "en": "English",
    "ar": "Arabic",
}

# Tesseract language pack string (used only when Tesseract is installed)
TESSERACT_LANGS = "fra+eng+ara"

# Image handling
MAX_IMAGE_LONG_EDGE = 2576  # model's max useful resolution — larger images are downscaled
SUPPORTED_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg",
                         ".jpeg": "image/jpeg", ".webp": "image/webp",
                         ".gif": "image/gif"}


def api_key_present() -> bool:
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and key != "gsk_..."  # ignore the .env.example placeholder
