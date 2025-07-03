# rag_arxiv/vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int = None):
        self.index = None
        self.metadata = []
        self.dim = dim

    def add(self, embeddings: list[list[float]], metadatas: list[dict]):
        vectors = np.array(embeddings).astype("float32")
        if self.index is None:
            # Infer dimension from first batch
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            # Ensure new vectors match existing dimension
            if vectors.shape[1] != self.dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        query_vec = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        return [self.metadata[i] for i in indices[0]]
