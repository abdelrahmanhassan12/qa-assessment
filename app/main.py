"""
Entry point for the QA FastAPI application.

This module wires together the database models, authentication
utilities and embedding manager to expose a simple REST API. Users can
register and login, upload documents to build a knowledge base, and
pose questions against their own uploaded data.

The implementation uses a TF‑IDF + FAISS based vector search as a
lightweight alternative to state‑of‑the‑art embeddings. If you wish to
integrate a transformer model or external API you can extend
`generate_answer` accordingly. See the README for instructions on
providing an OpenAI API key.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timedelta
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from . import auth, embeddings, models, schemas
from .database import Base, SessionLocal, engine


# Create all database tables on startup. In a production system you
# would manage migrations separately, but for a small demo this is
# convenient.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI QA Service", version="0.1.0")


def get_db():
    """Provide a SQLAlchemy session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for embedding.

    The function iterates through the text in steps of ``chunk_size - overlap``
    and yields substrings of length ``chunk_size``. Overlap between
    chunks helps preserve context across boundaries. Whitespace is
    preserved as is. If the text is shorter than ``chunk_size`` a
    single chunk is returned.
    """
    chunks: List[str] = []
    if not text:
        return chunks
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        # Move start forward by chunk_size - overlap
        if end == length:
            break
        start += max(chunk_size - overlap, 1)
    return chunks


@app.post("/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user.

    Checks whether the email already exists before creating a new user.
    Passwords are hashed using bcrypt via passlib. Returns the newly
    created user without exposing the hashed password.
    """
    existing = auth.get_user_by_email(db, user.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    hashed = auth.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate a user and return an access token.

    Uses OAuth2 password flow. The client must send ``username`` and
    ``password`` fields. If authentication succeeds a JWT is returned
    which must be included as a Bearer token in subsequent requests.
    """
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    # Use user ID as the JWT subject
    access_token = auth.create_access_token({"sub": user.id})
    return schemas.Token(access_token=access_token, token_type="bearer")


@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a text or PDF document and add its contents to the vector store.

    The endpoint accepts files with `.txt` or `.pdf` extensions. The
    text is extracted and split into overlapping chunks. Each chunk is
    embedded using a TF‑IDF vectoriser and stored both in the database
    and in-memory FAISS index. Returns a message indicating how many
    chunks were processed.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".txt", ".pdf"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

    try:
        content_bytes = await file.read()
        content = ""
        if ext == ".txt":
            content = content_bytes.decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            # Use PyPDF2 to extract text from each page
            from PyPDF2 import PdfReader

            with open("/tmp/_upload.pdf", "wb") as tmpf:
                tmpf.write(content_bytes)
            reader = PdfReader("/tmp/_upload.pdf")
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            content = "\n".join(pages)
        else:
            content = ""
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read uploaded file")

    # Split the extracted content into chunks
    chunks = split_text(content)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document contains no text")

    # Create a new document record
    document = models.Document(user_id=current_user.id, filename=filename)
    db.add(document)
    db.commit()
    db.refresh(document)

    # Prepare to store embedding chunks
    chunk_ids = []
    chunk_vectors = []
    for idx, chunk in enumerate(chunks):
        # We'll embed later via embeddings.add_text_chunks; first store raw text
        # Use a dummy empty vector placeholder for now (updated below)
        placeholder_vector = pickle.dumps(b"")
        emb = models.EmbeddingChunk(document_id=document.id, chunk_index=idx, text=chunk, vector=placeholder_vector)
        db.add(emb)
        db.flush()  # assign an ID without committing all yet
        chunk_ids.append(emb.id)
    db.commit()

    # Compute embeddings for the new chunks using TF‑IDF + FAISS
    # Use the text chunks themselves as input; vectoriser will be fit later on all texts
    embeddings.add_text_chunks(current_user.id, chunks, chunk_ids)

    # After computing embeddings we need to update the stored vectors in the database
    # Retrieve the entry for this user to access the vectoriser and matrix
    entry = embeddings._user_indexes[current_user.id]
    vectorizer = entry["vectorizer"]
    dense = vectorizer.transform(entry["texts"]).toarray().astype("float32")
    dense = embeddings._normalise_matrix(dense)
    # dense rows correspond to entry["chunk_ids"] in order
    # Update each EmbeddingChunk record with the pickled vector
    id_to_row = {cid: row for row, cid in enumerate(entry["chunk_ids"])}
    for cid in chunk_ids:
        row = id_to_row.get(cid)
        if row is None:
            continue
        vec = dense[row]
        vec_bytes = pickle.dumps(vec)
        db.query(models.EmbeddingChunk).filter(models.EmbeddingChunk.id == cid).update({"vector": vec_bytes})
    db.commit()

    return {"message": f"Processed {len(chunk_ids)} chunks from document."}


def generate_answer(context_chunks: List[str], question: str) -> str:
    """Generate an answer from the context and question using an LLM.

    If the environment variable ``OPENAI_API_KEY`` is defined, the
    function will attempt to call the OpenAI ChatCompletion API. To
    avoid exposing secrets in code, users must set this variable
    themselves before running the app. When no API key is provided the
    function simply returns the concatenation of the top context
    chunks. This fallback allows the system to run end-to-end without
    external dependencies.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    # Join context into a single string
    context_text = "\n\n".join(context_chunks)
    if api_key:
        try:
            import openai

            openai.api_key = api_key
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception:
            # If the OpenAI call fails fall back to returning context
            pass
    # Fallback: return the most relevant context as the "answer"
    return context_text


@app.post("/ask", response_model=schemas.Answer)
def ask_question(
    query: schemas.Query,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Answer a question using the user's uploaded documents.

    Performs a similarity search over the user's vector store to
    identify relevant text chunks. The top results are passed to a
    language model to generate an answer. The request and response are
    logged along with the time taken to respond.
    """
    start_time = time.perf_counter()
    # Search for similar chunks
    chunk_ids = embeddings.search(current_user.id, query.question, top_k=3)
    if not chunk_ids:
        answer_text = "No relevant documents found. Please upload some documents first."
        response_time = time.perf_counter() - start_time
        # Log the query with empty context
        log = models.Log(
            user_id=current_user.id,
            question=query.question,
            response=answer_text,
            response_time=response_time,
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
        return {"answer": answer_text}

    # Fetch the corresponding chunk texts from the database
    chunks = (
        db.query(models.EmbeddingChunk)
        .filter(models.EmbeddingChunk.id.in_(chunk_ids))
        .order_by(models.EmbeddingChunk.id)
        .all()
    )
    context_texts = [c.text for c in chunks]
    # Generate answer using LLM or fallback
    answer_text = generate_answer(context_texts, query.question)
    response_time = time.perf_counter() - start_time
    # Save log
    log = models.Log(
        user_id=current_user.id,
        question=query.question,
        response=answer_text,
        response_time=response_time,
        timestamp=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    return {"answer": answer_text}
