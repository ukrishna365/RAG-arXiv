# tests/test_compare_rag_vs_gpt.py

import os
from openai import OpenAI
from rag_arxiv.query_arXiv import query_arxiv_fulltext

def test_comparison():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    question = "How are transformer models applied in biology?"

    print("== QUESTION ==")
    print(question)

    # RAG Answer
    print("\n== RAG-Augmented Answer (arXiv PDFs) ==")
    rag_answer = query_arxiv_fulltext(question, top_n=8, max_pages=10)
    print(rag_answer)

    # GPT-4o Answer
    print("\n== GPT-4o Answer (no context) ==")
    gpt_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    print(gpt_answer.choices[0].message.content)

if __name__ == "__main__":
    test_comparison()