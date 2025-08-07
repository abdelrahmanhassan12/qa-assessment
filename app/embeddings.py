"""
Utility functions to create and search vector embeddings for text.

The current implementation uses a TF‑IDF bag-of-words model to
represent text documents. Although this is a simple technique compared
to modern transformer-based embeddings, it has no external
dependencies and performs adequately on small corpora. Vectors are
stored in a FAISS index to allow fast similarity search.

Embeddings and indexes are maintained per user to ensure that
documents uploaded by one user do not influence the results returned
for another. All state is held in memory; if persistence across
process restarts is desired, the structures could be serialised to
disk.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss


# In-memory structure keyed by user ID. Each entry contains:
# - vectorizer: fitted TfidfVectorizer
# - texts: list of chunk strings
# - chunk_ids: list of identifiers corresponding to embedding_chunks rows
# - index: FAISS IndexFlatIP for similarity search
_user_indexes: Dict[int, dict] = {}


def _normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    """L2 normalise each row of the given matrix.

    This ensures that cosine similarity can be computed using inner
    product in FAISS. If the matrix is empty (zero rows or zero
    columns) the same object is returned.
    """
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return matrix / norms


def _ensure_user_index(user_id: int) -> None:
    """Create an empty entry for a user if one does not exist."""
    if user_id not in _user_indexes:
        _user_indexes[user_id] = {
            "vectorizer": None,
            "texts": [],
            "chunk_ids": [],
            "index": None,
        }


def add_text_chunks(user_id: int, chunks: List[str], chunk_ids: List[int]) -> None:
    """Add new text chunks for a given user and rebuild the index.

    The TF‑IDF vectoriser is refit on all existing texts plus new
    chunks. A new FAISS index is built from scratch. For small
    datasets the cost of rebuilding is negligible. On larger corpora
    consider incremental indexing or a more advanced vector store.

    Args:
        user_id: ID of the user these chunks belong to.
        chunks: List of chunk strings to add.
        chunk_ids: List of database IDs for each chunk. Must match
            length of ``chunks``.
    """
    _ensure_user_index(user_id)
    entry = _user_indexes[user_id]

    # Append the new texts and ids
    entry["texts"].extend(chunks)
    entry["chunk_ids"].extend(chunk_ids)

    # Fit or refit the vectoriser
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(entry["texts"])
    entry["vectorizer"] = vectorizer
    # Convert to float32 numpy matrix and normalise
    dense = matrix.toarray().astype("float32")
    dense = _normalise_matrix(dense)

    # Build a new FAISS index
    if dense.size == 0:
        # No data to index yet
        entry["index"] = None
    else:
        dim = dense.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(dense)
        entry["index"] = index


def search(user_id: int, query: str, top_k: int = 3) -> List[int]:
    """Search for the most relevant text chunks for a user's query.

    Args:
        user_id: ID of the user to search within.
        query: Query string to embed and search.
        top_k: Number of results to return.

    Returns:
        A list of chunk IDs sorted by relevance (most relevant first).
        If no chunks exist for the user, an empty list is returned.
    """
    if user_id not in _user_indexes:
        return []
    entry = _user_indexes[user_id]
    vectorizer = entry.get("vectorizer")
    index = entry.get("index")
    if vectorizer is None or index is None or index.ntotal == 0:
        return []
    vec = vectorizer.transform([query]).toarray().astype("float32")
    vec = _normalise_matrix(vec)
    # Search for top_k results
    scores, idxs = index.search(vec, top_k)
    idx_list = idxs[0]
    # Filter out invalid indices (-1 if fewer than k results)
    result_ids = []
    for i in idx_list:
        if i < 0 or i >= len(entry["chunk_ids"]):
            continue
        result_ids.append(entry["chunk_ids"][i])
    return result_ids
