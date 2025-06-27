from rag_arxiv.embedder import Embedder
from rag_arxiv.vector_store import VectorStore
from rag_arxiv.rag_pipeline import RAGPipeline


def test_rag_pipeline():
    # Sample docs
    docs = [
        {"chunk": "Transformers are great for machine translation.", "source": "doc0"},
        {"chunk": "Graph neural networks work well on structured data.", "source": "doc1"},
        {"chunk": "Convolutional networks perform well on image data.", "source": "doc2"},
        {"chunk": "Large language models excel in few-shot learning.", "source": "doc3"},
        {"chunk": "Attention mechanisms are key in sequence modeling.", "source": "doc4"},
    ]

    # Initialize components
    embedder = Embedder()
    store = VectorStore()
    chunks = [doc["chunk"] for doc in docs]
    embeddings = embedder.embed(chunks)
    store.add(embeddings, docs)

    # Run pipeline
    pipeline = RAGPipeline(embedder=embedder, vector_store=store)
    question = "What helps sequence models handle dependencies?"
    answer = pipeline.query(question)

    print("== Test RAGPipeline ==")
    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    test_rag_pipeline()
