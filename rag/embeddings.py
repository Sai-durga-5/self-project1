from typing import Optional
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings
import numpy as np


class SentenceTransformerEmbeddings(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using sentence-transformers.
    Uses 'all-MiniLM-L6-v2' model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(list(input), convert_to_numpy=True)
        return embeddings.tolist()


# Singleton instance for reuse across the application
_embedding_function: Optional[SentenceTransformerEmbeddings] = None


def get_embedding_function() -> SentenceTransformerEmbeddings:
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = SentenceTransformerEmbeddings()
    return _embedding_function
