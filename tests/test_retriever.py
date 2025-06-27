from rag_arxiv.retriever import Retriever
import pprint

def test_retriever():
    retriever = Retriever(max_results=2)
    results = retriever.fetch("machine learning")

    for i, paper in enumerate(results, 1):
        print(f"\n--- Result #{i} ---")
        pprint.pprint(paper)

if __name__ == "__main__":
    test_retriever()
