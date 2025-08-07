"""
Pydantic models used for request/response bodies.

These schemas define the shapes of incoming and outgoing data for
FastAPI endpoints. They ensure that data is validated and provide a
clear contract between the client and server.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for creating a new user."""

    email: EmailStr = Field(..., description="User email used for login.")
    password: str = Field(..., min_length=6, description="Password for the account.")


class UserResponse(BaseModel):
    """Schema returned after creating or retrieving a user."""

    id: int
    email: EmailStr

    class Config:
        orm_mode = True


class Token(BaseModel):
    """Schema representing an authentication token."""

    access_token: str
    token_type: str


class Query(BaseModel):
    """Schema for a question asked by the user."""

    question: str = Field(..., description="The question to ask the system.")


class Answer(BaseModel):
    """Schema returned for an answer to a question."""

    answer: str
