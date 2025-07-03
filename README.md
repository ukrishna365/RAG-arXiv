# RAG-arXiv

**A Retrieval-Augmented Generation (RAG) pipeline that answers questions using full-text scholarly papers from [arXiv.org](https://arxiv.org).**
This project combines vector-based semantic search with large language models (LLMs) to generate grounded, research-backed answers.

---

## Features

- **arXiv Retriever:** Fetches top-N papers by keyword query (title, abstract, and PDF link).
- **Full-text PDF Extraction:** Downloads and extracts text from arXiv PDFs (up to configurable max pages).
- **Chunking & Embedding:** Splits text into overlapping chunks and generates embeddings using `paraphrase-MiniLM-L3-v2` for efficiency.
- **FAISS Vector Store:** Enables fast similarity search to find the most relevant document chunks.
- **OpenAI RAG Generation:** Prompts GPT-4o using retrieved chunks as context to ensure answers are grounded in actual arXiv papers.

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```


---

## Setup your OpenAI API Key

Create a `.env` file in the project root directory:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

Run the interactive CLI:

```bash
python cli_query_arxiv.py
```

- Enter your question when prompted.
- Configure how many papers to retrieve and how many pages to extract from each PDF.
- The program will generate an answer along with references to the exact arXiv papers used.

---

## Example

```
Welcome to the arXiv RAG-arXiv QA system!
Enter your question: How are transformer models applied in biology?
Number of PDFs to retrieve [default 5]: 8
Max pages per PDF [default 5]: 10

== Answer ==
[Grounded explanation based on retrieved papers]

Articles/papers used:
1. ...
2. ...
Time taken: 42.78 seconds
```

---

## How it works

1. Retrieves top matching arXiv papers.
2. Filters them by semantic similarity to your question.
3. Extracts full PDF text (truncated if too large for safe context size).
4. Embeds and indexes chunks in a FAISS vector store.
5. Generates an answer using GPT-4o, citing the retrieved papers.

---

## Project Structure

```
rag_arxiv/
├── retriever.py        # arXiv search
├── utils.py            # PDF extraction & filtering helpers
├── embedder.py         # Chunking & embeddings
├── vector_store.py     # FAISS similarity search
├── rag_pipeline.py     # Integrates everything with OpenAI
```

Includes test scripts in `/tests` and an interactive CLI in `cli_query_arxiv.py`.

---