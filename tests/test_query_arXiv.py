# tests/test_query_arxiv_fulltext.py

from rag_arxiv.query_arXiv import query_arxiv_fulltext  # fixed casing on 'arxiv'

def test_query():
    question = "Are there models for detecting misinformation on social media"
    print("== Fulltext RAG Test ==")
    print(f"Question: {question}\n")

    # Updated to match new signature
    answer = query_arxiv_fulltext(question, num_papers=10, max_pages=10)

    print("== Answer ==")
    print(answer)

if __name__ == "__main__":
    test_query()
