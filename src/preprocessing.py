"""Image preprocessing for scanned documents.

Uses OpenCV when available (deskew, denoise, CLAHE, adaptive threshold);
degrades gracefully to a Pillow-only pipeline otherwise.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False

from .config import MAX_IMAGE_LONG_EDGE


@dataclass
class PreprocessResult:
    image: Image.Image
    steps: list[str] = field(default_factory=list)

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return buf.getvalue()

    def to_jpeg_bytes(self, quality: int = 88) -> bytes:
        """Smaller than PNG — keeps the base64 payload safely under vision-API size caps."""
        buf = io.BytesIO()
        self.image.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _resize_long_edge(img: Image.Image, steps: list[str]) -> Image.Image:
    long_edge = max(img.size)
    if long_edge > MAX_IMAGE_LONG_EDGE:
        scale = MAX_IMAGE_LONG_EDGE / long_edge
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
        steps.append(f"resized to {img.width}x{img.height}")
    elif long_edge < 1000:
        # Tiny scans benefit from upscaling before recognition
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        steps.append(f"upscaled 2x to {img.width}x{img.height}")
    return img


def _deskew_cv2(gray: "np.ndarray", steps: list[str]) -> "np.ndarray":
    """Estimate the dominant text angle and rotate the page upright."""
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray  # negligible skew, or likely a misdetection
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    steps.append(f"deskewed {angle:+.2f} deg")
    return rotated


def preprocess(image_bytes: bytes, *, binarize: bool = False) -> PreprocessResult:
    """Run the enhancement pipeline on raw image bytes.

    `binarize` is off by default: the vision model performs better on
    contrast-enhanced grayscale than on hard-thresholded images, while
    Tesseract prefers binarized input.
    """
    steps: list[str] = []
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img = _resize_long_edge(img, steps)

    if CV2_AVAILABLE:
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        steps.append("grayscale")
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7,
                                        searchWindowSize=21)
        steps.append("denoised (NL-means)")
        gray = _deskew_cv2(gray, steps)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        steps.append("contrast enhanced (CLAHE)")
        if binarize:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, blockSize=31, C=15)
            steps.append("adaptive binarization")
        out = Image.fromarray(gray)
    else:
        out = ImageOps.grayscale(img)
        steps.append("grayscale")
        out = out.filter(ImageFilter.MedianFilter(size=3))
        steps.append("denoised (median)")
        out = ImageOps.autocontrast(out, cutoff=1)
        out = ImageEnhance.Contrast(out).enhance(1.4)
        steps.append("contrast enhanced")
        if binarize:
            out = out.point(lambda p: 255 if p > 150 else 0)
            steps.append("global binarization")

    return PreprocessResult(image=out, steps=steps)
