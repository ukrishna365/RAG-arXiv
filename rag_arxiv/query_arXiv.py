# rag_arxiv/query_arxiv.py
from rag_arxiv.retriever import Retriever
from rag_arxiv.embedder import Embedder
from rag_arxiv.vector_store import VectorStore
from rag_arxiv.rag_pipeline import RAGPipeline
from rag_arxiv.utils import extract_text_from_pdf_url, filter_by_similarity


def query_arxiv_fulltext(question: str, num_papers=10, max_pages=10, max_tokens_per_doc=3000):
    retriever = Retriever(max_results=num_papers)
    papers = retriever.fetch(question)

    embedder = Embedder()

    filtered_papers = filter_by_similarity(question, papers, embedder, top_k=num_papers, threshold=0.4)
    if not filtered_papers:
        print("No papers passed initial semantic filtering, trying with lower threshold.")
        filtered_papers = filter_by_similarity(question, papers, embedder, top_k=num_papers, threshold=0.2)
    if not filtered_papers:
        print("Still no semantically relevant papers found, using top abstracts without filter.")
        filtered_papers = papers[:num_papers]

    docs, metadatas, used_papers = [], [], []
    for paper in filtered_papers:
        pdf_text = extract_text_from_pdf_url(paper["pdf_url"], max_pages=max_pages)
        if "PDF extraction failed" not in pdf_text:
            if len(pdf_text.split()) > max_tokens_per_doc:
                print(f"Trimming long document from {len(pdf_text.split())} tokens to {max_tokens_per_doc} tokens.")
                pdf_text = " ".join(pdf_text.split()[:max_tokens_per_doc])
            docs.append(pdf_text)
            metadatas.append({"chunk": pdf_text, "source": paper["title"]})
            used_papers.append((paper["title"], paper["pdf_url"]))
        else:
            print(f"Failed to extract PDF for: {paper['title']}")

    if not docs:
        print("No valid PDFs found, falling back to abstracts only.")
        texts = [p["summary"] for p in filtered_papers]
        metadatas = [{"chunk": t, "source": p["title"]} for t, p in zip(texts, filtered_papers)]
        docs = texts

    if not docs:
        return "No text could be extracted from the retrieved papers. Try a different query."

    # Estimate total token count
    total_words = sum(len(doc.split()) for doc in docs)
    estimated_tokens = int(total_words / 0.75)
    print(f"Estimated total prompt size: ~{estimated_tokens} tokens ({total_words} words across all chunks).")

    embeddings = embedder.embed(docs)
    store = VectorStore()
    store.add(embeddings, metadatas)

    pipeline = RAGPipeline(embedder, store, model="gpt-4o")
    answer = pipeline.query(question, top_k=min(num_papers, 5))

    formatted_sources = "\n\nArticles/papers used:\n"
    for i, (title, url) in enumerate(used_papers if used_papers else [(p["title"], p["pdf_url"]) for p in filtered_papers], 1):
        formatted_sources += f"{i}. {title.strip()} — {url}\n"

    return f"{answer.strip()}\n{formatted_sources}"


if __name__ == "__main__":
    q = "How are transformer models applied in biology?"
    print("== Question ==")
    print(q)
    print("\n== Answer ==")
    print(query_arxiv_fulltext(q, num_papers=10, max_pages=10))
