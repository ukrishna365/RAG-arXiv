import requests
from io import BytesIO
import pdfplumber
from pdf2image import convert_from_bytes
import easyocr
import fitz  # PyMuPDF
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf_url(pdf_url: str, max_pages: int = 10):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        pdf_bytes = BytesIO(response.content)

        # Try pdfplumber first for structured text + tables
        text_parts = []
        with pdfplumber.open(pdf_bytes) as pdf:
            num_pages = min(max_pages, len(pdf.pages))
            for i in range(num_pages):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                text_parts.append(text.strip())

                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join([" | ".join(row) for row in table])
                    text_parts.append("[Extracted Table]\n" + table_text)

        combined_text = "\n\n".join(filter(None, text_parts))

        # If barely any text, fallback to OCR using EasyOCR
        if len(combined_text.strip()) < 100:
            print("Minimal text detected, attempting OCR fallback with EasyOCR...")
            reader = easyocr.Reader(['en'], gpu=False)
            images = convert_from_bytes(response.content, first_page=1, last_page=max_pages)
            ocr_text = []
            for img in images:
                results = reader.readtext(np.array(img), detail=0, paragraph=True)
                ocr_text.append("\n".join(results))
            combined_text = "\n".join(ocr_text)

        return combined_text if combined_text.strip() else "[No extractable content found.]"

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
