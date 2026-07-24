"""MongoDB connection (sync PyMongo — routes run in FastAPI's threadpool)."""

import os

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB", "ey_ocr")

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
db = client[DB_NAME]

users = db.users
documents = db.documents


def ping() -> bool:
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False


def ensure_indexes() -> None:
    users.create_index([("email", ASCENDING)], unique=True)
    documents.create_index([("user_email", ASCENDING), ("created_at", DESCENDING)])
