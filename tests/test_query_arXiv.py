# tests/test_query_arxiv_fulltext.py

from rag_arxiv.query_arXiv import query_arxiv_fulltext

def test_query():
    question = "What are common recession indicators?"
    print("== Fulltext RAG Test ==")
    print(f"Question: {question}\n")

    answer = query_arxiv_fulltext(question, top_n=3, max_pages=3)

    print("== Answer ==")
    print(answer)

if __name__ == "__main__":
    test_query()
