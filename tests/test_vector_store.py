from rag_arxiv.embedder import Embedder
from rag_arxiv.vector_store import VectorStore
import pprint

def test_vector_search():
    texts = [
        "Transformers are great for machine translation.",
        "Graph neural networks work well on structured data.",
        "Convolutional networks dominate in image tasks.",
        "Self-supervised learning improves data efficiency.",
        "Attention mechanisms are key in sequence modeling."
    ]

    metadata = [{"chunk": t, "source": f"doc{i}"} for i, t in enumerate(texts)]

    embedder = Embedder()
    chunks, embeddings = [], []
    for text in texts:
        c, e = embedder.process(text)
        chunks.extend(c)
        embeddings.extend(e)

    store = VectorStore(dim=384)
    store.add(embeddings, metadata[:len(embeddings)])

    # Search
    query = "What models are good for text generation?"
    query_embedding = embedder.embed_chunks([query])[0]

    top_matches = store.search(query_embedding, top_k=3)
    print("\n== Top 3 Matches ==")
    for match in top_matches:
        pprint.pprint(match)

if __name__ == "__main__":
    test_vector_search()
