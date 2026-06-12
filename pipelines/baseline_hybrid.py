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


def generate_answer(question: str, verbose: bool = False) -> str:
    analysis = analyze_query(question)
    if verbose:
        print(f"\n[Baseline — analiza] {analysis}")

    graph_context = retrieve_from_graph(analysis)
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
    return response.content
