"""Authentication routes: register and login."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models.auth_schemas import Token, UserLogin, UserOut, UserRegister
from app.services.auth import _create_access_token, authenticate_user, create_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister) -> UserOut:
    try:
        user = await create_user(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return user


@router.post("/login", response_model=Token)
async def login(body: UserLogin) -> Token:
    user = await authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = _create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )
    return Token(access_token=access_token, token_type="bearer")