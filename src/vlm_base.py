"""Shared vision-language-model (VLM) engine base.

Both concrete engines — the local Ollama VLM (default, offline) and the Groq
cloud API — expose the same three capabilities, implemented here on top of a
single abstract `_generate` call:
  * extract_text / extract_from_pdf — vision-based transcription of scanned pages
  * correct_text — post-processing of classical OCR output (error correction)

Neither backend accepts PDF input natively, so extract_from_pdf rasterizes
each page locally with PyMuPDF and transcribes it page by page.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from .config import PDF_RASTER_DPI, SUPPORTED_LANGUAGES

_EXTRACTION_SYSTEM = """You are a high-precision OCR engine specialized in scanned \
documents in French, English, and Arabic (including mixed-language documents).

Transcribe the document EXACTLY as written:
- Reproduce every character faithfully, including accents (é, è, à, ç, û...), \
Arabic diacritics when printed, punctuation, and numbers.
- Transcribe ALL text on the page: printed AND handwritten, including small \
print, footnotes, headers/footers, page numbers, stamps, checkbox labels, and \
form helper/placeholder text (e.g. "Choose an item.", "Please refer to the \
guidance..."). Never skip text because it looks like boilerplate or form \
furniture.
- CHECKBOXES AND TICK-BOXES ARE CRITICAL: for every checkbox, radio button or \
tick-box, output its state next to its label. Write ☑ before a box that is \
ticked, crossed, filled or marked in ANY way (✓, ✗, X, a dot, a scribble), and \
☐ before an empty box. This applies even when the boxes sit in a separate row \
or grid from their labels — associate each mark with the option directly above, \
below or beside it. Look carefully: marks are often faint handwritten strokes. \
Example: "☐ Discriminatory  ☑ Domestic  ☐ Financial". Never output a form \
option without its box state.
- Preserve the reading order. For Arabic passages, transcribe right-to-left text \
in correct logical order.
- Preserve document structure using Markdown: headings, paragraphs, lists, and \
tables (as Markdown tables). Keep line breaks where they are semantically meaningful.
- Do NOT translate, summarize, correct grammar, or add commentary.
- If a region is genuinely illegible, mark it as [illisible]/[illegible] rather \
than guessing entire passages.
- OCCLUSION — NEVER INVENT COVERED CHARACTERS: scanned forms often carry a \
watermark, stamp or signature printed ACROSS the text (e.g. a large diagonal \
"EXAMPLE"). Where such an overlay hides part of a value, transcribe the \
characters you can actually see and write [?] for EACH character you cannot, \
e.g. "01642 0[?]5070", "TS25 2[?]Q", "Ha[?]bour". Do NOT fill the gap with a \
plausible digit or letter — a guessed phone number or postcode is worse than a \
visibly incomplete one. This applies to single characters inside an otherwise \
readable value, not just to whole illegible regions.
- Output ONLY the transcription, no preamble."""

_VERIFY_SYSTEM = """You are a meticulous OCR proofreader. You are given a scanned \
page image and a DRAFT transcription of it. The draft may have MISSED some text \
or transcribed it wrongly.

Your job is to produce the COMPLETE, corrected transcription:
- Scan the image region by region, top to bottom, left to right. For EACH piece \
of visible text — printed or handwritten, including small print, footnotes, \
headers/footers, page numbers, stamps, table cells and checkbox states (☑/☐) — \
confirm it appears in the draft. If it is missing, ADD it in its correct place. \
If it is wrong, FIX it.
- Pay special attention to text the draft is likely to have skipped: faint or \
small print, marginal notes, isolated numbers/dates, table cells, and \
checkboxes.
- Do NOT remove correct text, do NOT translate, summarize, reorder, or add \
commentary. Preserve structure and reading order (Arabic right-to-left in \
logical order).
- Output ONLY the full corrected transcription, no preamble, no notes about \
what you changed."""

_CORRECTION_SYSTEM = """You are an OCR post-correction engine for documents in \
French, English, and Arabic. You receive raw text produced by a classical OCR \
system (e.g. Tesseract) that contains recognition errors.

Fix ONLY errors introduced by the OCR process:
- Character confusions (rn→m, l→1, O→0, cl→d, é→e, corrupted Arabic ligatures...)
- Broken or merged words, spurious symbols, mojibake
- Wrong or missing accents and diacritics recoverable from context
- Obviously corrupted words, restored using the surrounding language

STRICT RULES:
- Never paraphrase, translate, reorder, summarize, or "improve" the writing.
- Never add or remove sentences. Preserve the original line/paragraph structure.
- Keep numbers, dates, references, and proper nouns unless clearly corrupted.
- If a fragment is too corrupted to restore confidently, keep it as-is.
- Output ONLY the corrected text, no preamble or explanations."""

TRUNCATION_WARNING = "\n\n[AVERTISSEMENT / WARNING: output truncated — raise OCR_MAX_TOKENS]"


@dataclass
class VLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


class VLMEngine(ABC):
    @abstractmethod
    def _generate(self, system: str, text: str,
                  image: tuple[bytes, str] | None = None) -> VLMResult:
        """Run one completion. `image` is (bytes, media_type) when present."""

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _language_hint(languages: list[str] | None) -> str:
        if not languages:
            return ""
        names = ", ".join(SUPPORTED_LANGUAGES.get(code, code) for code in languages)
        return f" The document is expected to contain: {names}."

    # ------------------------------------------------------------------ public API

    def extract_text(self, image_bytes: bytes, media_type: str = "image/jpeg",
                     languages: list[str] | None = None,
                     verify: bool = False) -> VLMResult:
        """Vision-based transcription of a scanned page image.

        When `verify` is True, a second pass re-reads the image against the
        first transcription to recover missed/wrong text (higher recall, ~2x
        the time). It cannot guarantee zero omissions — no OCR can — but it
        measurably reduces them.
        """
        first = self._generate(
            _EXTRACTION_SYSTEM,
            "Transcribe this scanned document." + self._language_hint(languages),
            image=(image_bytes, media_type),
        )
        if not verify:
            return first

        second = self._generate(
            _VERIFY_SYSTEM,
            "Draft transcription to complete and correct:"
            + self._language_hint(languages)
            + "\n\n<draft>\n" + first.text + "\n</draft>",
            image=(image_bytes, media_type),
        )
        return VLMResult(
            text=second.text,
            model=second.model,
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            stop_reason=second.stop_reason,
        )

    def extract_from_pdf(self, pdf_bytes: bytes,
                         languages: list[str] | None = None,
                         verify: bool = False) -> VLMResult:
        """Transcription of a scanned PDF, rasterized locally and read page by page.

        `verify` runs the second recall pass on every page (see extract_text).
        """
        import fitz  # PyMuPDF — imported lazily, only needed for PDF input

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        matrix = fitz.Matrix(PDF_RASTER_DPI / 72, PDF_RASTER_DPI / 72)
        page_texts: list[str] = []
        total_in = total_out = 0
        model_name = ""
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix)
                buf = io.BytesIO()
                Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").save(
                    buf, format="JPEG", quality=88)
                result = self.extract_text(buf.getvalue(), "image/jpeg",
                                           languages=languages, verify=verify)
                page_texts.append(result.text)
                total_in += result.input_tokens
                total_out += result.output_tokens
                model_name = result.model
        finally:
            doc.close()

        return VLMResult(
            text="\n\n---\n\n".join(page_texts),
            model=model_name,
            input_tokens=total_in,
            output_tokens=total_out,
            stop_reason=None,
        )

    def correct_text(self, raw_ocr_text: str,
                     languages: list[str] | None = None) -> VLMResult:
        """Post-correction of classical OCR output."""
        prompt = ("Raw OCR output to correct:" + self._language_hint(languages)
                  + "\n\n<ocr_output>\n" + raw_ocr_text + "\n</ocr_output>")
        return self._generate(_CORRECTION_SYSTEM, prompt)
