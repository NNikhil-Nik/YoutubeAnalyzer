"""
Persistent Vector Database Module
===================================
Manages a local ChromaDB store at ./chroma_db_storage.

Responsibilities:
  - Check whether a video's chunks are already cached (by video_id metadata).
  - Add new chunks when a video is seen for the first time.
  - Retrieve the top-3 most relevant chunks for a query, filtered to the active video.
  - Handle corrupted / locked DB gracefully.
"""

import os
import shutil
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_DIR = "./chroma_db_storage"
COLLECTION_NAME = "youtube_transcripts"


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )


def _load_db() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=DB_DIR,
    )


def is_video_cached(video_id: str) -> bool:
    """Return True if chunks for this video_id already exist in the DB."""
    try:
        db = _load_db()
        results = db.get(where={"video_id": video_id}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def add_chunks(chunks: list[Document], video_id: str, batch_size: int = 50) -> None:
    """
    Embed and persist chunks in batches to stay within the free-tier limit of
    100 embedding requests/minute. Retries automatically on 429 rate-limit errors.
    """
    for chunk in chunks:
        chunk.metadata["video_id"] = video_id

    db = _load_db()
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        print(f"  Embedding chunks {i + 1}–{min(i + batch_size, total)} of {total}…")

        retries = 0
        while True:
            try:
                db.add_documents(batch)
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 60 * (retries + 1)
                    print(f"  Rate limit hit — waiting {wait}s before retrying…")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise RuntimeError(f"Failed to write to vector database: {exc}") from exc


def retrieve_chunks(question: str, video_id: str, k: int = 3) -> list[Document]:
    """
    Similarity-search the DB, restricted to chunks from the active video.
    Returns exactly k documents (or fewer if the DB has less than k).
    """
    try:
        db = _load_db()
        return db.similarity_search(
            question,
            k=k,
            filter={"video_id": video_id},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to query vector database: {exc}") from exc


def reset_db() -> None:
    """Delete and recreate the local DB directory — used when corruption is detected."""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    print("Vector database has been reset.")
