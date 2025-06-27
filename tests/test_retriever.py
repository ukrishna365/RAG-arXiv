from rag_arxiv.retriever import Retriever
import pprint

def test_retriever_basic():
    print("\n== Abstract-only Test ==")
    retriever = Retriever(max_results=1)
    results = retriever.fetch("neural networks", extract_text=False)

    for i, paper in enumerate(results, 1):
        print(f"\n--- Result #{i} ---")
        pprint.pprint(paper)

def test_retriever_with_fulltext():
    print("\n== Abstract + Fulltext Test ==")
    retriever = Retriever(max_results=1)
    results = retriever.fetch("transformer models", extract_text=True, max_pdf_pages=3)

    for i, paper in enumerate(results, 1):
        print(f"\n--- Result #{i} ---")
        print("Title:", paper["title"])
        print("Summary (Abstract):", paper["summary"][:500], "...")
        print("Fulltext Preview (first 1000 chars):")
        print(paper.get("fulltext", "[no fulltext]")[:2000])
        print("\n--- End of result ---")

if __name__ == "__main__":
    test_retriever_basic()
    test_retriever_with_fulltext()
