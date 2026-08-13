"""
Frozen baseline hybrid pipeline (AS-IS orchestration).
Used for benchmarks — do not add guard/filter logic here.
"""

from langchain_core.prompts import ChatPromptTemplate

from retrieval.clients import model
from retrieval.graph_retriever import retrieve_from_graph
from retrieval.normalize import analyze_query
from retrieval.vector_retriever import retrieve_from_vector

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Jesteś inteligentnym asystentem uczelnianym. Odpowiadaj rzeczowo i zwięźle, \
wyłącznie na podstawie dostarczonego kontekstu.
Jeśli kontekst nie zawiera wystarczających informacji, poinformuj o tym użytkownika.

Pytanie: {question}

=== KONTEKST Z GRAFU WIEDZY (dane strukturalne) ===
{graph_context}

=== KONTEKST Z DOKUMENTÓW PDF (dane tekstowe) ===
{vector_context}

ODPOWIEDŹ:
""")


def generate_answer_detailed(question: str, verbose: bool = False) -> dict:
    from retrieval.normalize import normalize_analysis

    intent_raw = analyze_query(question)
    intent = normalize_analysis(intent_raw)
    if verbose:
        print(f"\n[Baseline — analiza] raw={intent_raw} normalized={intent}")

    graph_context = retrieve_from_graph(intent_raw)
    if verbose:
        preview = graph_context[:600] if graph_context else "Brak wyników"
        print(f"\n[Baseline — graph]\n{preview}")

    vector_context, chunks = retrieve_from_vector(question)
    if verbose:
        print("\n[Baseline — vector]")
        for i, ch in enumerate(chunks, 1):
            print(f"  {i}. {ch['tag']} (score: {ch['score']})")

    chain = ANSWER_PROMPT | model
    response = chain.invoke({
        "question": question,
        "graph_context": graph_context or "Brak danych strukturalnych.",
        "vector_context": vector_context or "Brak pasujących dokumentów.",
    })

    return {
        "final_answer": response.content,
        "intent_raw": intent_raw,
        "intent": intent,
        "graph_context": graph_context,
        "vector_context": vector_context,
        "chunks": chunks,
    }


def generate_answer(question: str, verbose: bool = False) -> str:
    return generate_answer_detailed(question, verbose=verbose)["final_answer"]
