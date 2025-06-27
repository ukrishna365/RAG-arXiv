# rag_arxiv/rag_pipeline.py

import os
import openai
from dotenv import load_dotenv

from rag_arxiv.embedder import Embedder
from rag_arxiv.vector_store import VectorStore

class RAGPipeline:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, model="gpt-3.5-turbo"):
        self.embedder = embedder
        self.vector_store = vector_store
        self.model = model

        # Load API key from .env
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def query(self, question: str, top_k: int = 3) -> str:
        # Embed the question
        question_embedding = self.embedder.embed([question])[0]

        # Retrieve top-k similar chunks
        top_chunks = self.vector_store.search(question_embedding, top_k=top_k)

        # Format context
        context = "\n---\n".join(chunk["chunk"] for chunk in top_chunks)

        # Construct prompt
        prompt = (
            "You are a helpful academic assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        # Query OpenAI
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        return response.choices[0].message["content"].strip()
