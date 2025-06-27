# query_arxiv.py

from rag_arxiv.retriever import Retriever
from rag_arxiv.embedder import Embedder
from rag_arxiv.vector_store import VectorStore
from rag_arxiv.rag_pipeline import RAGPipeline
from rag_arxiv.utils import extract_text_from_pdf_url

def query_arxiv_fulltext(question: str, top_n: int = 10, max_pages: int = 10):
    # Step 1: Retrieve top-N papers
    retriever = Retriever(max_results=top_n)
    papers = retriever.fetch(question)

    # Step 2: Extract full-text and metadata
    docs = []
    metadatas = []
    used_papers = []

    for paper in papers:
        pdf_text = extract_text_from_pdf_url(paper["pdf_url"], max_pages=max_pages)
        if "PDF extraction failed" not in pdf_text:
            docs.append(pdf_text)
            metadatas.append({"chunk": pdf_text, "source": paper["title"]})
            used_papers.append((paper["title"], paper["pdf_url"]))
        else:
            print(f"⚠️ Failed to extract PDF for: {paper['title']}")

    if not docs:
        return "No valid PDFs found for embedding."

    # Step 3: Embed and store
    embedder = Embedder()
    embeddings = embedder.embed(docs)
    store = VectorStore()
    store.add(embeddings, metadatas)

    # Step 4: Query via RAG
    pipeline = RAGPipeline(embedder, store, model="gpt-4o")
    answer = pipeline.query(question)

    # Step 5: Format sources used
    formatted_sources = "\n\nArticles/papers used:\n"
    for i, (title, url) in enumerate(used_papers, 1):
        formatted_sources += f"{i}. {title.strip()} — {url}\n"

    return f"{answer.strip()}\n{formatted_sources}"
