"""Pydantic request / response schemas for authentication."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100, description="User password")
    name: str = Field(..., min_length=1, max_length=100, description="Display name")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    time_of_birth: str = Field(..., description="Time of birth in HH:MM format")
    place_of_birth: str | None = Field(None, description="Place of birth (city)")


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=100, description="User password")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: EmailStr
    name: str
