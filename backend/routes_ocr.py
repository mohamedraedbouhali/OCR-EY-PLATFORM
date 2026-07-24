"""OCR processing and document-history endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from src import local_vlm, ocr_engine
from src.config import (DEFAULT_ENGINE, MODEL, OLLAMA_MODEL, SUPPORTED_IMAGE_TYPES,
                        SUPPORTED_LANGUAGES, api_key_present)
from src.pipeline import Engine, Mode, OCRPipeline

from . import db
from .schemas import HistoryItem, MetaResponse, OCRResponse
from .security import get_current_user

router = APIRouter(tags=["ocr"])

_pipeline = OCRPipeline()
_MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # aligned with the model's request limit


@router.get("/meta", response_model=MetaResponse)
def meta(user: dict = Depends(get_current_user)):
    tess = ocr_engine.is_available()
    modes = [Mode.GENAI_VISION.value]
    if tess:
        modes += [Mode.HYBRID.value, Mode.TESSERACT_ONLY.value]
    return MetaResponse(
        modes=modes,
        engines={Engine.LOCAL.value: local_vlm.is_available(),
                 Engine.GROQ.value: api_key_present()},
        default_engine=DEFAULT_ENGINE,
        models={Engine.LOCAL.value: OLLAMA_MODEL, Engine.GROQ.value: MODEL},
        tesseract_available=tess,
        languages=SUPPORTED_LANGUAGES,
        model=MODEL,
    )


def _doc_to_response(doc: dict) -> OCRResponse:
    return OCRResponse(
        id=str(doc["_id"]),
        filename=doc["filename"],
        mode=doc["mode"],
        engine=doc.get("engine"),
        text=doc["text"],
        raw_text=doc.get("raw_text"),
        languages=doc.get("languages", []),
        preprocessing_steps=doc.get("preprocessing_steps", []),
        tesseract_confidence=doc.get("tesseract_confidence"),
        model=doc.get("model"),
        input_tokens=doc.get("input_tokens", 0),
        output_tokens=doc.get("output_tokens", 0),
        duration_s=doc.get("duration_s", 0.0),
        created_at=doc["created_at"],
    )


@router.post("/ocr", response_model=OCRResponse)
def run_ocr(
    file: UploadFile,
    mode: str = Form(Mode.GENAI_VISION.value),
    engine: str = Form(DEFAULT_ENGINE),  # "local" (Ollama) or "groq" (cloud)
    langs: str = Form(""),  # comma-separated ISO codes, e.g. "fr,ar"
    user: dict = Depends(get_current_user),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    is_pdf = suffix == ".pdf"
    if not is_pdf and suffix not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Unsupported file type: {suffix or '(none)'}")
    try:
        mode_enum = Mode(mode)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown mode: {mode}")
    try:
        engine_enum = Engine(engine)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown engine: {engine}")
    if mode_enum != Mode.TESSERACT_ONLY:
        if engine_enum == Engine.GROQ and not api_key_present():
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "GROQ_API_KEY is not configured on the server")
        if engine_enum == Engine.LOCAL and not local_vlm.is_available():
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Local VLM unavailable — is Ollama running with '{OLLAMA_MODEL}' "
                f"pulled? (ollama pull {OLLAMA_MODEL})")
    if is_pdf and mode_enum != Mode.GENAI_VISION:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "PDF input requires the GenAI vision mode")

    data = file.file.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            "File exceeds the 30 MB limit")

    hint = [code.strip() for code in langs.split(",")
            if code.strip() in SUPPORTED_LANGUAGES] or None
    try:
        if is_pdf:
            result = _pipeline.run_pdf(data, language_hint=hint, engine=engine_enum)
        else:
            result = _pipeline.run_image(data, mode=mode_enum, language_hint=hint,
                                         engine=engine_enum)
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"OCR failed: {exc}")

    doc = {
        "user_email": user["email"],
        "filename": file.filename or "upload",
        "mode": result.mode.value,
        "engine": result.engine,
        "text": result.final_text,
        "raw_text": result.raw_text,
        "languages": result.languages,
        "preprocessing_steps": result.preprocessing_steps,
        "tesseract_confidence": result.tesseract_confidence,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "duration_s": result.duration_s,
        "created_at": datetime.now(timezone.utc),
    }
    doc["_id"] = db.documents.insert_one(doc).inserted_id
    return _doc_to_response(doc)


@router.get("/history", response_model=list[HistoryItem])
def history(user: dict = Depends(get_current_user)):
    cursor = (db.documents.find({"user_email": user["email"]})
              .sort("created_at", -1).limit(50))
    return [
        HistoryItem(
            id=str(d["_id"]),
            filename=d["filename"],
            mode=d["mode"],
            languages=d.get("languages", []),
            preview=(d.get("text") or "")[:160],
            duration_s=d.get("duration_s", 0.0),
            created_at=d["created_at"],
        )
        for d in cursor
    ]


def _owned_doc_or_404(doc_id: str, user: dict) -> dict:
    try:
        oid = ObjectId(doc_id)
    except InvalidId:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    doc = db.documents.find_one({"_id": oid, "user_email": user["email"]})
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.get("/history/{doc_id}", response_model=OCRResponse)
def get_document(doc_id: str, user: dict = Depends(get_current_user)):
    return _doc_to_response(_owned_doc_or_404(doc_id, user))


@router.delete("/history/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = _owned_doc_or_404(doc_id, user)
    db.documents.delete_one({"_id": doc["_id"]})
