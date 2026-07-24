"""EY Intelligent OCR — FastAPI application.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from . import db  # noqa: E402
from .routes_auth import router as auth_router  # noqa: E402
from .routes_ocr import router as ocr_router  # noqa: E402
from .security import hash_password  # noqa: E402

logger = logging.getLogger("uvicorn.error")

DEFAULT_USER_EMAIL = os.getenv("DEFAULT_USER_EMAIL", "raedbouhali@gmail.com").lower()
DEFAULT_USER_PASSWORD = os.getenv("DEFAULT_USER_PASSWORD", "raed")
DEFAULT_USER_NAME = os.getenv("DEFAULT_USER_NAME", "Raed Bouhali")


def seed_default_user() -> None:
    """Create the default account if it doesn't exist yet (idempotent)."""
    if db.users.find_one({"email": DEFAULT_USER_EMAIL}) is None:
        db.users.insert_one({
            "email": DEFAULT_USER_EMAIL,
            "name": DEFAULT_USER_NAME,
            "password_hash": hash_password(DEFAULT_USER_PASSWORD),
        })
        logger.info("Seeded default user %s", DEFAULT_USER_EMAIL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if db.ping():
        db.ensure_indexes()
        seed_default_user()
    else:
        logger.warning("MongoDB unreachable at startup — auth will fail until it is up")
    yield


app = FastAPI(title="EY Intelligent OCR API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(ocr_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "database": "up" if db.ping() else "down"}
