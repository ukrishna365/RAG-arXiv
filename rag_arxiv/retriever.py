# rag_arxiv/retriever.py
import arxiv
from .utils import extract_text_from_pdf_url

class Retriever:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def fetch(self, query: str, extract_text: bool = False, max_pdf_pages: int = 5):
        search = arxiv.Search(query=query, max_results=self.max_results)
        results = []

        for result in search.results():
            entry = {
                "title": result.title,
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "authors": [a.name for a in result.authors],
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id
            }

            if extract_text:
                entry["fulltext"] = extract_text_from_pdf_url(result.pdf_url, max_pages=max_pdf_pages)

            results.append(entry)

        return results
