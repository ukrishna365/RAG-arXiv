import time
from rag_arxiv.query_arXiv import query_arxiv_fulltext
from rag_arxiv.embedder import Embedder

def test_pipeline_with_chunk_size(question, chunk_size, num_papers=10, max_pages=10):
    # Adjust Embedder globally
    embedder = Embedder(chunk_size=chunk_size)
    start_time = time.time()
    answer = query_arxiv_fulltext(
        question,
        num_papers=num_papers,
        max_pages=max_pages
    )
    elapsed = time.time() - start_time
    print(f"\n== Chunk Size: {chunk_size} ==")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Answer:\n{answer}\n")
    return answer

if __name__ == "__main__":
    question = "Are there models for detecting misinformation on social media?"
    for chunk_size in [200, 225, 250, 275, 300]:
        test_pipeline_with_chunk_size(question, chunk_size)
