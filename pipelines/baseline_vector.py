"""
Frozen vector-only RAG pipeline (Chroma PDF + LLM).
Uses the same Ollama embeddings as ingest (nomic-embed-text, 768-dim).
"""

from langchain_core.prompts import ChatPromptTemplate

from retrieval.clients import model
from retrieval.vector_retriever import retrieve_from_vector

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Jesteś inteligentnym asystentem uczelnianym. Twoim celem jest przygotowanie odpowiedzi, \
bez zbędnego tekstu. Odpowiadaj tylko na bazie kontekstu, jeśli w kontekście z bazy wiedzy \
nie znajdziesz wystarczających informacji aby odpowiedzieć na pytanie, poinformuj o tym użytkownika.

Treść zapytania: {question}

Poniżej masz kontekst z bazy wiedzy:
{context}

ODPOWIEDŹ:
""")


def get_context_and_preview(query_text: str, n_results: int = 3) -> tuple[str, list[dict]]:
    return retrieve_from_vector(query_text, n_results=n_results)


def generate_answer_detailed(question: str, verbose: bool = False) -> dict:
    context, chunks = get_context_and_preview(question)

    if verbose:
        print("\n[Vector — sources]")
        for i, ch in enumerate(chunks, 1):
            print(f"  {i}. {ch['tag']} (score: {ch['score']})")

    chain = ANSWER_PROMPT | model
    response = chain.invoke({"question": question, "context": context or ""})

    return {
        "final_answer": response.content,
        "context": context,
        "chunks": chunks,
    }


def generate_answer(question: str, verbose: bool = False) -> str:
    return generate_answer_detailed(question, verbose=verbose)["final_answer"]
