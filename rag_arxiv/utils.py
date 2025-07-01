import requests
import fitz  # PyMuPDF
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf_url(pdf_url: str, max_pages: int = 10):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        pages = [doc[i].get_text() for i in range(min(max_pages, len(doc)))]
        return "\n".join(pages)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def filter_by_similarity(question: str, docs: list[dict], embedder, top_k=10, threshold=0.2):
    """
    Filter documents based on semantic similarity to the question.
    Each doc should have a 'summary' key (or 'fulltext' if you build on that later).

    Returns the top_k docs above the similarity threshold.
    """
    question_vec = embedder.embed([question])[0]
    scored = []

    for doc in docs:
        text = doc.get("summary") or doc.get("fulltext") or ""
        doc_vec = embedder.embed([text])[0]
        sim = cosine_similarity([question_vec], [doc_vec])[0][0]
        scored.append((sim, doc))

    # Sort by similarity descending
    scored.sort(reverse=True, key=lambda x: x[0])

    # Keep only above threshold, return up to top_k
    filtered_docs = [doc for sim, doc in scored if sim >= threshold][:top_k]
    return filtered_docs
