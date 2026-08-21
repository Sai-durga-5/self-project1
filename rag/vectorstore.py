import os
from typing import Optional
import chromadb
from dotenv import load_dotenv
from rag.embeddings import get_embedding_function

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

COLLECTIONS = ["resumes", "job_descriptions", "interview_questions"]

_client: Optional[chromadb.PersistentClient] = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_collection(collection_name: str):
    if collection_name not in COLLECTIONS:
        raise ValueError(f"Unknown collection '{collection_name}'. Valid: {COLLECTIONS}")
    client = get_client()
    embedding_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    return collection


def add_documents(
    collection_name: str,
    texts: list[str],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
) -> None:
    collection = get_collection(collection_name)
    if not texts:
        return

    if ids is None:
        existing_count = collection.count()
        ids = [f"{collection_name}_{existing_count + i}" for i in range(len(texts))]

    if metadatas is None:
        metadatas = [{} for _ in texts]

    # ChromaDB has a batch size limit; chunk large inserts
    batch_size = 500
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.add(
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )


def query(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
) -> list[str]:
    collection = get_collection(collection_name)
    count = collection.count()
    if count == 0:
        return []
    actual_n = min(n_results, count)
    results = collection.query(
        query_texts=[query_text],
        n_results=actual_n,
    )
    documents = results.get("documents", [[]])[0]
    return documents


def collection_count(collection_name: str) -> int:
    collection = get_collection(collection_name)
    return collection.count()
