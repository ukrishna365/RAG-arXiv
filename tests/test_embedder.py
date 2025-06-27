from rag_arxiv.embedder import Embedder
import pprint

def test_chunking():
    print("\n== Chunking Test ==")
    text = (
        "Transformer models have revolutionized natural language processing. "
        "Their ability to capture long-range dependencies and scale to massive datasets "
        "has made them the architecture of choice for many tasks including translation, summarization, and question answering."
        "This test sentence is just to ensure that we have enough text to split across multiple chunks."
    )

    embedder = Embedder()
    chunks = embedder.chunk_text(text, chunk_size = 50)
    chunks = embedder.chunk_text(text)

    print(f"Chunk count: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk #{i+1} ---\n{chunk}")

def test_embedding():
    print("\n== Embedding Test ==")
    text = "Deep learning models are changing how we interpret and generate language."
    embedder = Embedder()
    chunks, embeddings = embedder.process(text)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")
    print("\nFirst embedding preview:\n")
    pprint.pprint(embeddings[0][:10])  # show first 10 dims of first embedding

if __name__ == "__main__":
    test_chunking()
    test_embedding()
