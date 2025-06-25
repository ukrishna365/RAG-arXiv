import arxiv

import arxiv
from ragflow.preprocessing.parsers import Pdf

class Retriever:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def fetch_with_fulltext(self, query: str, extract_text: bool = True):
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

            if extract_text and result.pdf_url:
                try:
                    pdf = Pdf.from_url(result.pdf_url)
                    entry["fulltext_chunks"] = pdf.chunk()
                except Exception as e:
                    entry["fulltext_chunks"] = [f"[PDF extraction failed: {e}]"]

            results.append(entry)

        return results