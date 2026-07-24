"""Optional classical OCR baseline (Tesseract).

The pipeline works without it — this engine exists to (a) benchmark the GenAI
approach against a traditional one and (b) feed the GenAI post-correction mode.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from PIL import Image

from .config import TESSERACT_LANGS

try:
    import pytesseract

    _PYTESSERACT = True
except ImportError:  # pragma: no cover
    pytesseract = None
    _PYTESSERACT = False


@dataclass
class TesseractResult:
    text: str
    mean_confidence: float  # 0-100, from Tesseract word confidences


def is_available() -> bool:
    return _PYTESSERACT and shutil.which("tesseract") is not None


def extract(image: Image.Image, langs: str = TESSERACT_LANGS) -> TesseractResult:
    if not is_available():
        raise RuntimeError(
            "Tesseract is not installed. Install the binary (with fra/eng/ara "
            "traineddata) and `pip install pytesseract`, or use the GenAI engine."
        )
    data = pytesseract.image_to_data(image, lang=langs,
                                     output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
    text = pytesseract.image_to_string(image, lang=langs)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return TesseractResult(text=text.strip(), mean_confidence=round(mean_conf, 1))
