"""Authentication and account endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from . import db
from .schemas import LoginRequest, ProfileUpdate, TokenResponse, UserOut
from .security import create_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

_MAX_AVATAR_CHARS = 400_000  # ~300 KB image as base64 — plenty for a 192px avatar


def _user_out(user: dict) -> UserOut:
    return UserOut(email=user["email"], name=user.get("name", ""),
                   avatar=user.get("avatar"))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    if not db.ping():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Database unavailable — is MongoDB running?")
    user = db.users.find_one({"email": body.email.lower()})
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Incorrect email or password")
    return TokenResponse(access_token=create_token(user["email"]),
                         user=_user_out(user))


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    return _user_out(user)


@router.patch("/profile", response_model=UserOut)
def update_profile(body: ProfileUpdate, user: dict = Depends(get_current_user)):
    updates: dict = {}

    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name cannot be empty")
        updates["name"] = name[:80]

    if body.avatar is not None:
        if body.avatar == "":
            updates["avatar"] = None
        else:
            if not body.avatar.startswith("data:image/") or ";base64," not in body.avatar:
                raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                    "Avatar must be a base64 image data URI")
            if len(body.avatar) > _MAX_AVATAR_CHARS:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    "Avatar image too large")
            updates["avatar"] = body.avatar

    if body.new_password:
        if not body.current_password or not verify_password(
                body.current_password, user["password_hash"]):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                "Current password is incorrect")
        if len(body.new_password) < 4:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "New password must be at least 4 characters")
        updates["password_hash"] = hash_password(body.new_password)

    if not updates:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to update")

    db.users.update_one({"_id": user["_id"]}, {"$set": updates})
    return _user_out(db.users.find_one({"_id": user["_id"]}))
