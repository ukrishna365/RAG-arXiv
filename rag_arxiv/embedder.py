# rag_arxiv/embedder.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Embedder:
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", chunk_size=250, chunk_overlap=50):
        self.model = SentenceTransformer(model_name)
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        return self.model.encode(chunks, convert_to_numpy=True).tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_chunks(texts)

    def process(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> tuple[list[str], list[list[float]]]:
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        embeddings = self.embed_chunks(chunks)
        return chunks, embeddings
