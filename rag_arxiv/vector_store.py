# rag_arxiv/vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings: list[list[float]], metadatas: list[dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        query_vec = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        return [self.metadata[i] for i in indices[0]]