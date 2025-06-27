from rag_arxiv.embedder import Embedder

def run_test(text, chunk_size, label):
    print(f"\n== {label} (chunk_size={chunk_size}) ==")
    embedder = Embedder()
    chunks, embeddings = embedder.process(text, chunk_size=chunk_size)

    print(f"Chunk count: {len(chunks)}")
    print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk #{i+1} ---\n{chunk[:200]}{'...' if len(chunk) > 200 else ''}")

if __name__ == "__main__":
    text = (
        "Transformer models have revolutionized NLP. "
        "They can capture long-range dependencies and scale to large datasets. "
        "This makes them the go-to architecture for translation, summarization, question answering, and more. "
        "In this test case, we’re adding more sentences to make sure the text is long enough to require chunking. "
        "Chunking is critical for RAG pipelines, especially when documents exceed token limits. "
        "Let’s continue adding text just to simulate a more realistic academic abstract or paper section. "
        "Eventually, this should be enough to trigger multiple chunks under a small chunk size."
    )

    run_test(text, chunk_size=150, label="Small Chunk Test")
    run_test(text, chunk_size=600, label="Large Chunk Test")
