import requests
import fitz  # PyMuPDF
from io import BytesIO

def extract_text_from_pdf_url(pdf_url: str, max_pages: int = 5):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        pages = [doc[i].get_text() for i in range(min(max_pages, len(doc)))]
        return "\n".join(pages)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"
