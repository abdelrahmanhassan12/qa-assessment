# AI Question Answering Service

This project is a minimal question–answering API built as part of an AI engineering assignment.  Users can upload small text or PDF documents, ask questions about their own documents and receive responses generated from the most relevant text.  The service implements basic authentication, stores per‐query logs in a local database and uses a vector search backend to find related chunks of text.

## Architecture Overview

```
┌─────────────┐    upload    ┌─────────────┐
│  Client/UI  │ ───────────▶ │ FastAPI app │
└─────────────┘              │             │
      ▲  ▲  ask question    │  • `/register` – create new user accounts
      │  │  ┌────────────┐  │  • `/login` – obtain JWT access token
      │  └──┤ Authentication │
      │     └──────────────┘  │  • `/upload` – upload text/PDF documents
      │                        │  • `/ask` – ask questions against your data
      │                        └─────────────┘
      │                             │
      │                       ┌──────────┐  holds user/session state
      │                       │ Database │◀──────────────┐
      │                       └──────────┘               │
      │                                                 │
      │    FAISS vector search (per user)              │
      └──────────────────────────────────────────────────┘
```

When a document is uploaded, its text is extracted and split into overlapping chunks.  Each chunk is converted into a TF‑IDF vector and stored in an in‑memory FAISS index dedicated to the uploading user.  When a question is asked the system embeds the query with the same vectoriser, retrieves the most similar chunks and either forwards them to an external LLM (if an `OPENAI_API_KEY` is provided) or returns the relevant context directly.  Every query is logged with a timestamp, question, response and response time.

## Tech Stack

- **FastAPI** – web framework used to expose REST endpoints.
- **SQLite** via **SQLAlchemy** – simple relational database for storing users, documents and logs.
- **Passlib (bcrypt)** and **PyJWT** – secure password hashing and JWT authentication.
- **FAISS** and **scikit‑learn** – vector search and TF‑IDF embeddings for similarity search.
- **PyPDF2** – PDF text extraction.

No external AI models are bundled with the project.  If you wish to generate more natural answers you can provide an OpenAI API key via the `OPENAI_API_KEY` environment variable; otherwise the system returns the most relevant document text as the answer.

## Setup Instructions

1. **Install dependencies**

   From the project root run:

   ```bash
   pip install fastapi uvicorn sqlalchemy passlib[bcrypt] PyJWT PyPDF2 faiss-cpu python-multipart
   ```

2. **(Optional) Set an OpenAI API key**

   To enable language model answers export your key before starting the server:

   ```bash
   export OPENAI_API_KEY=sk-...your key...
   ```

3. **Start the server**

   ```bash
   python -m uvicorn qa_app.app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`.  FastAPI automatically exposes an interactive API explorer at `http://localhost:8000/docs`.

4. **Create a user and authenticate**

   Use the `/register` endpoint to create a new user, then obtain a token via `/login`.  The returned JWT must be supplied as a Bearer token in the `Authorization` header for the protected endpoints.

5. **Upload documents and ask questions**

   Send a `multipart/form-data` POST request to `/upload` with a `file` field containing a `.txt` or `.pdf`.  After uploading documents you can POST to `/ask` with JSON `{"question": "..."}` to retrieve answers.

## API Usage

Below is a brief overview of the available endpoints.  See the auto‑generated documentation at `/docs` for full request/response schemas.

| Method | Endpoint      | Description                           | Authentication |
|------:|---------------|---------------------------------------|---------------|
| POST  | `/register`   | Create a new user account             | No            |
| POST  | `/login`      | Obtain a JWT token                    | No            |
| POST  | `/upload`     | Upload a `.txt` or `.pdf` document    | Yes           |
| POST  | `/ask`        | Ask a question against your documents | Yes           |

## Known Limitations

- **Simple embeddings** – TF‑IDF vectors can capture keyword overlap but may perform poorly on nuanced queries compared to transformer‑based embeddings.
- **In‑memory vector index** – all document vectors live in memory and are rebuilt on each upload.  This is acceptable for small datasets but will not scale to large corpora.
- **No concurrency guarantees** – the vectoriser and FAISS index are global per user and not protected by locks.  Concurrent uploads or queries by the same user may cause race conditions in a production environment.
- **Authentication** – tokens never expire while the server runs; token revocation and refresh flows are not implemented.
- **UI/Deployment** – no front‑end or containerisation is provided out of the box.  These are left as extensions for further work.

