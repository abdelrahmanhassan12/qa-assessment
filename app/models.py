"""
SQLAlchemy models for the QA application.

These classes define the tables used by the application:

* **User** – registered users with an email and hashed password. A user
  can own many documents and logs.
* **Document** – uploaded documents belonging to a user. Each document
  can have many embedding chunks representing portions of the text.
* **EmbeddingChunk** – stores individual chunks of text along with
  their pickled vector representation for similarity search.
* **Log** – records each question asked by a user along with the
  response and performance metrics.

Relationships are defined to simplify navigation between models. If
you add new models or fields, remember to update the corresponding
Pydantic schemas in `schemas.py`.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Text, LargeBinary
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    """Represents a user of the application."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    logs = relationship("Log", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    """Represents an uploaded document containing textual content."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    owner = relationship("User", back_populates="documents")
    embeddings = relationship("EmbeddingChunk", back_populates="document", cascade="all, delete-orphan")


class EmbeddingChunk(Base):
    """Represents a chunk of text and its vector embedding."""

    __tablename__ = "embedding_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    vector = Column(LargeBinary, nullable=False)  # pickled numpy array

    # Relationships
    document = relationship("Document", back_populates="embeddings")


class Log(Base):
    """Represents a single query log entry."""

    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    response_time = Column(Float, nullable=False)

    # Relationships
    user = relationship("User", back_populates="logs")
