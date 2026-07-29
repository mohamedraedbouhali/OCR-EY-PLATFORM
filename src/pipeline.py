"""End-to-end OCR pipeline: preprocessing → recognition → VLM post-processing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from . import ocr_engine
from .config import DEFAULT_ENGINE
from .language import detect_languages
from .preprocessing import preprocess
from .vlm_base import VLMEngine


class Mode(str, Enum):
    GENAI_VISION = "genai_vision"          # the VLM reads the image directly (recommended)
    HYBRID = "hybrid"                      # Tesseract extraction + VLM correction
    TESSERACT_ONLY = "tesseract_only"      # classical baseline, no VLM


class Engine(str, Enum):
    LOCAL = "local"                        # Ollama VLM on this machine (default, offline)
    GROQ = "groq"                          # Groq cloud API (needs GROQ_API_KEY)


@dataclass
class PipelineResult:
    mode: Mode
    final_text: str
    raw_text: str | None = None            # pre-correction text (hybrid mode)
    languages: list[str] = field(default_factory=list)
    preprocessing_steps: list[str] = field(default_factory=list)
    tesseract_confidence: float | None = None
    engine: str | None = None              # which VLM ran (None for tesseract_only)
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0


class OCRPipeline:
    def __init__(self, api_key: str | None = None):
        self._engines: dict[Engine, VLMEngine] = {}
        self._api_key = api_key  # Groq only; the local engine needs no key

    def engine(self, engine: Engine | str | None = None) -> VLMEngine:
        """Lazily construct and cache the requested VLM engine."""
        eng = Engine(engine) if engine else Engine(DEFAULT_ENGINE)
        if eng not in self._engines:
            if eng is Engine.GROQ:
                from .genai_ocr import GroqOCR
                self._engines[eng] = GroqOCR(api_key=self._api_key)
            else:
                from .local_vlm import OllamaOCR
                self._engines[eng] = OllamaOCR()
        return self._engines[eng]

    def run_image(self, image_bytes: bytes, mode: Mode = Mode.GENAI_VISION,
                  language_hint: list[str] | None = None,
                  engine: Engine | str | None = None,
                  verify: bool = False) -> PipelineResult:
        start = time.perf_counter()

        needs_binarization = mode in (Mode.HYBRID, Mode.TESSERACT_ONLY)
        pre = preprocess(image_bytes, binarize=needs_binarization)
        result = PipelineResult(mode=mode, final_text="",
                                preprocessing_steps=pre.steps)

        if mode == Mode.GENAI_VISION:
            gen = self.engine(engine).extract_text(pre.to_jpeg_bytes(), "image/jpeg",
                                                   languages=language_hint,
                                                   verify=verify)
            result.final_text = gen.text
            result.engine = (Engine(engine) if engine else Engine(DEFAULT_ENGINE)).value
            result.model = gen.model
            result.input_tokens, result.output_tokens = gen.input_tokens, gen.output_tokens
        else:
            tess = ocr_engine.extract(pre.image)
            result.raw_text = tess.text
            result.tesseract_confidence = tess.mean_confidence
            if mode == Mode.HYBRID:
                langs = language_hint or detect_languages(tess.text)
                gen = self.engine(engine).correct_text(tess.text, languages=langs)
                result.final_text = gen.text
                result.engine = (Engine(engine) if engine else Engine(DEFAULT_ENGINE)).value
                result.model = gen.model
                result.input_tokens, result.output_tokens = gen.input_tokens, gen.output_tokens
            else:
                result.final_text = tess.text

        result.languages = language_hint or detect_languages(result.final_text)
        result.duration_s = round(time.perf_counter() - start, 2)
        return result

    def run_pdf(self, pdf_bytes: bytes,
                language_hint: list[str] | None = None,
                engine: Engine | str | None = None,
                verify: bool = False) -> PipelineResult:
        """Each page is rasterized locally and read by the vision model."""
        start = time.perf_counter()
        gen = self.engine(engine).extract_from_pdf(pdf_bytes, languages=language_hint,
                                                   verify=verify)
        steps = ["PDF rasterized locally, page by page"]
        if verify:
            steps.append("second-pass recall verification per page")
        return PipelineResult(
            mode=Mode.GENAI_VISION,
            final_text=gen.text,
            languages=language_hint or detect_languages(gen.text),
            preprocessing_steps=steps,
            engine=(Engine(engine) if engine else Engine(DEFAULT_ENGINE)).value,
            model=gen.model,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            duration_s=round(time.perf_counter() - start, 2),
        )
