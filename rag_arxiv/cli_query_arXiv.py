# cli_query_arxiv.py

import time
from rag_arxiv.query_arXiv import query_arxiv_fulltext

def main():
    print("Welcome to the arXiv RAG-arXiv QA system!")
    question = input("Enter your question: ").strip()

    num_papers = input("Number of PDFs to retrieve [default 5]: ").strip()
    num_papers = int(num_papers) if num_papers else 5

    max_pages = input("Max pages per PDF [default 5]: ").strip()
    max_pages = int(max_pages) if max_pages else 5

    print("\nProcessing your query... This may take a moment.\n")

    start_time = time.time()
    answer = query_arxiv_fulltext(
        question,
        num_papers=num_papers,
        max_pages=max_pages
    )
    elapsed_time = time.time() - start_time

    print("\n== Answer ==")
    print(answer)
    print(f"\nTime taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
