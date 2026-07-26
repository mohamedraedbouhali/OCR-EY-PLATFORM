"""Cross-model OCR comparison for completeness QA.

Runs the same page through two vision models and diffs the results word by
word. Where the models AGREE, confidence is high; where they DISAGREE, a human
should check the scan. This is the honest answer to "don't miss a word": no
single model guarantees completeness, but two independent readers flag exactly
the spots worth reviewing.

Both models are local by default (qwen3-vl 4b vs 8b), so the check is fully
offline. Any Ollama vision tag works.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from .config import OLLAMA_MODEL
from .local_vlm import OllamaOCR

# Second reader: the larger local model if present, else a sensible default.
DEFAULT_MODEL_A = OLLAMA_MODEL
DEFAULT_MODEL_B = "qwen3-vl:8b-instruct"


@dataclass
class Disagreement:
    a: str          # words from model A (may be empty = A missed it)
    b: str          # words from model B (may be empty = B missed it)


@dataclass
class CompareResult:
    model_a: str
    model_b: str
    text_a: str
    text_b: str
    agreement: float                       # 0..1, word-level
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def agreement_pct(self) -> float:
        return round(self.agreement * 100, 1)


def _words(text: str) -> list[str]:
    return text.split()


def compare_image(image_bytes: bytes, media_type: str = "image/jpeg",
                  languages: list[str] | None = None,
                  model_a: str = DEFAULT_MODEL_A,
                  model_b: str = DEFAULT_MODEL_B) -> CompareResult:
    """Transcribe one page with two models and diff them word by word."""
    text_a = OllamaOCR(model=model_a).extract_text(image_bytes, media_type, languages).text
    text_b = OllamaOCR(model=model_b).extract_text(image_bytes, media_type, languages).text

    wa, wb = _words(text_a), _words(text_b)
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    agreement = sm.ratio()  # 2*matches / (len_a + len_b)

    disagreements: list[Disagreement] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        disagreements.append(Disagreement(a=" ".join(wa[i1:i2]),
                                           b=" ".join(wb[j1:j2])))
    return CompareResult(model_a=model_a, model_b=model_b, text_a=text_a,
                         text_b=text_b, agreement=agreement,
                         disagreements=disagreements)


def compare_pdf(pdf_bytes: bytes, languages: list[str] | None = None,
                model_a: str = DEFAULT_MODEL_A,
                model_b: str = DEFAULT_MODEL_B) -> list[CompareResult]:
    """Two-model comparison of a scanned PDF, page by page."""
    import io

    import fitz
    from PIL import Image

    from .config import PDF_RASTER_DPI

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    matrix = fitz.Matrix(PDF_RASTER_DPI / 72, PDF_RASTER_DPI / 72)
    results: list[CompareResult] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            buf = io.BytesIO()
            Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").save(
                buf, format="JPEG", quality=88)
            results.append(compare_image(buf.getvalue(), "image/jpeg", languages,
                                         model_a, model_b))
    finally:
        doc.close()
    return results


def format_report(r: CompareResult) -> str:
    """Human-readable comparison report for the terminal."""
    lines = [
        f"Model A: {r.model_a}",
        f"Model B: {r.model_b}",
        f"Word-level agreement: {r.agreement_pct}%  "
        f"({len(r.disagreements)} disagreement span(s) to review)",
        "",
        "Disagreements (A | B) — empty side means that model missed the text:",
        "-" * 72,
    ]
    if not r.disagreements:
        lines.append("  (none — both models produced identical text)")
    for d in r.disagreements:
        lines.append(f"  A: {d.a or '·(missing)·'}")
        lines.append(f"  B: {d.b or '·(missing)·'}")
        lines.append("")
    return "\n".join(lines)
