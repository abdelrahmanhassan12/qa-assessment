"""
Authentication utilities for the QA application.

This module centralises functions related to password hashing,
authentication token generation and retrieval of the current user from a
JWT. The implementation uses `passlib` for secure password hashing and
`PyJWT` for creating and verifying JSON Web Tokens.
"""

from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from . import models
from .database import SessionLocal


# Secret key used to sign JWTs. In a real application this should
# absolutely not be hard-coded – instead load from environment
# variables or a secrets manager. The key must be long and random.
SECRET_KEY = "a0d3be69f7be426596a5bd6b3fd3c9b9d6f7e280e9a5dbba3f0f5f97477a3c06"
ALGORITHM = "HS256"
# Token lifetime in minutes
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours by default


# Configure passlib to use bcrypt for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def get_password_hash(password: str) -> str:
    """Hash a plaintext password for storage."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token containing the given data.

    The JWT specification recommends that the `sub` claim is a
    string. To avoid decoding errors the function coerces `sub` to a
    string if present. An `exp` claim is always added to enforce
    expiry.

    Args:
        data: A dictionary of claims to encode into the JWT. Should
            include the subject (user ID) and any additional claims.
        expires_delta: Optional timedelta specifying token lifetime. If
            omitted, a default expiry defined by
            ``ACCESS_TOKEN_EXPIRE_MINUTES`` is used.

    Returns:
        A signed JWT string.
    """
    to_encode = data.copy()
    # Ensure subject is a string
    sub = to_encode.get("sub")
    if sub is not None:
        to_encode["sub"] = str(sub)
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    """Return a user object by email, or None if not found."""
    return db.query(models.User).filter(models.User.email == email).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
    """Validate a user's email and password.

    Returns the user if authentication is successful, otherwise
    ``None``.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> models.User:
    """Retrieve the currently authenticated user from the JWT.

    A new database session is created internally to avoid FastAPI
    attempting to treat the session factory as a dependency. This
    prevents the inadvertent injection of `local_kw` query parameters
    seen when `Depends(SessionLocal)` is used directly. The session
    is closed automatically before returning.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    # Create a new session to load the user
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user is None:
            raise credentials_exception
        return user
    finally:
        db.close()
