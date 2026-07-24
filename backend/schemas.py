"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    email: str
    name: str
    avatar: str | None = None  # data-URI image, set from the account modal


class ProfileUpdate(BaseModel):
    name: str | None = None
    avatar: str | None = None          # data URI; empty string removes the photo
    current_password: str | None = None
    new_password: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class OCRResponse(BaseModel):
    id: str
    filename: str
    mode: str
    engine: str | None = None
    text: str
    raw_text: str | None = None
    languages: list[str]
    preprocessing_steps: list[str]
    tesseract_confidence: float | None = None
    model: str | None = None
    input_tokens: int
    output_tokens: int
    duration_s: float
    created_at: datetime


class HistoryItem(BaseModel):
    id: str
    filename: str
    mode: str
    languages: list[str]
    preview: str
    duration_s: float
    created_at: datetime


class MetaResponse(BaseModel):
    modes: list[str]
    engines: dict[str, bool]     # engine value -> currently available
    default_engine: str
    models: dict[str, str]       # engine value -> model name
    tesseract_available: bool
    languages: dict[str, str]
    model: str                   # legacy: cloud model name
