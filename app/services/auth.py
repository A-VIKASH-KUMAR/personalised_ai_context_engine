"""Authentication service: password hashing, JWT creation/verification, MongoDB user store."""

from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING

from app.config import settings
from app.models.auth_schemas import UserOut, UserRegister

logger = logging.getLogger(__name__)

_BCRYPT_MAX_BYTES = 72

_mongo_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def _normalize_password(plain: str) -> bytes:
    digest = hashlib.sha256(plain.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(plain: str) -> str:
    import bcrypt

    normalized = _normalize_password(plain)
    assert len(normalized) <= _BCRYPT_MAX_BYTES
    return bcrypt.hashpw(normalized, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    import bcrypt

    normalized = _normalize_password(plain)
    try:
        return bcrypt.checkpw(normalized, hashed.encode("utf-8"))
    except ValueError:
        return False


async def init_db() -> None:
    """Open the MongoDB connection, ensure indexes, and seed mock users once."""
    global _mongo_client, _db

    _mongo_client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
        uuidRepresentation="standard",
    )
    _db = _mongo_client[settings.mongodb_db_name]

    try:
        await _mongo_client.admin.command("ping")
        logger.info(
            "Connected to MongoDB db=%s collection=%s",
            settings.mongodb_db_name,
            settings.mongodb_users_collection,
        )
    except Exception as exc:
        logger.error("Could not reach MongoDB at %s: %s", settings.mongodb_uri, exc)
        raise

    users = _db[settings.mongodb_users_collection]
    await users.create_index([("email", ASCENDING)], unique=True)
    await users.create_index([("id", ASCENDING)], unique=True)

    _MOCK_SEED = [
        {"id": "user_101", "email": "aarav@example.com", "name": "Aarav Sharma", "password": "password123"},
        {"id": "user_102", "email": "priya@example.com", "name": "Priya Patel", "password": "password123"},
        {"id": "user_103", "email": "rohan@example.com", "name": "Rohan Mehta", "password": "password123"},
        {"id": "user_104", "email": "vikash@example.com", "name": "vikash kumar", "password": "password123"},
    ]

    inserted = 0
    for seed in _MOCK_SEED:
        existing = await users.find_one({"email": seed["email"]})
        if existing:
            continue
        doc = {
            "id": seed["id"],
            "email": seed["email"],
            "name": seed["name"],
            "hashed_password": hash_password(seed["password"]),
            "created_at": datetime.now(timezone.utc),
        }
        await users.insert_one(doc)
        inserted += 1

    logger.info("MongoDB users seed complete: inserted=%s", inserted)


async def close_db() -> None:
    """Close the MongoDB connection (call on shutdown)."""
    global _mongo_client, _db
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        _db = None
        logger.info("MongoDB connection closed")


def _get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB is not initialised. Did lifespan startup run?")
    return _db


def _user_doc_to_out(doc: dict[str, Any]) -> UserOut:
    return UserOut(id=doc["id"], email=doc["email"], name=doc["name"])


def _create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


async def create_user(register: UserRegister) -> UserOut:
    users = _get_db()[settings.mongodb_users_collection]
    existing = await users.find_one({"email": register.email})
    if existing:
        raise ValueError("Email already registered")

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    doc = {
        "id": user_id,
        "email": register.email,
        "name": register.name,
        "hashed_password": hash_password(register.password),
        "date_of_birth": register.date_of_birth,
        "time_of_birth": register.time_of_birth,
        "place_of_birth": register.place_of_birth,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        await users.insert_one(doc)
    except Exception as exc:
        logger.exception("Failed to insert user %s: %s", register.email, exc)
        raise ValueError("Could not register user") from exc

    logger.info("Registered new user id=%s email=%s", user_id, register.email)
    return _user_doc_to_out(doc)


async def authenticate_user(email: str, password: str) -> UserOut | None:
    users = _get_db()[settings.mongodb_users_collection]
    doc = await users.find_one({"email": email})
    if not doc:
        return None
    if not verify_password(password, doc["hashed_password"]):
        return None
    return _user_doc_to_out(doc)


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    users = _get_db()[settings.mongodb_users_collection]
    return await users.find_one({"email": email})


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    users = _get_db()[settings.mongodb_users_collection]
    return await users.find_one({"id": user_id})


async def _decode_token(token: str) -> UserOut:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception from None
    user = await get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return _user_doc_to_out(user)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    return await _decode_token(token)


async def get_current_user_ws(token: str) -> UserOut:
    return await _decode_token(token)