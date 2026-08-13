from retrieval.clients import graph
from retrieval.cypher import CYPHER_QUERIES, WYMAGA_KIERUNKU
from retrieval.models import ContextItem
from retrieval.normalize import normalize_analysis


def retrieve_from_graph(analysis: dict) -> str:
    """Baseline: formatted string context from Neo4j."""
    analysis = normalize_analysis(analysis)
    kategorie = analysis.get("kategorie", [])
    kierunek = analysis.get("kierunek")
    kontekst = analysis.get("kontekst")
    graph_parts = []

    for kat in kategorie:
        query = CYPHER_QUERIES.get(kat)
        if not query:
            continue
        if kat in WYMAGA_KIERUNKU and not kierunek:
            graph_parts.append(
                f"[GRAPH — {kat.upper()}] Podaj nazwę kierunku, aby uzyskać szczegóły."
            )
            continue
        params = {"kierunek": kierunek, "kontekst": kontekst}
        try:
            results = graph.query(query, params)
            if results:
                formatted = f"[GRAPH — {kat.upper()}]\n"
                for row in results:
                    formatted += "  " + " | ".join(
                        f"{k}: {v}" for k, v in row.items() if v is not None
                    ) + "\n"
                graph_parts.append(formatted.strip())
            else:
                graph_parts.append(f"[GRAPH — {kat.upper()}] Brak wyników w bazie.")
        except Exception as e:
            graph_parts.append(f"[GRAPH — {kat.upper()}] Błąd zapytania: {e}")

    return "\n\n".join(graph_parts)


def retrieve_graph_items(analysis: dict) -> list[ContextItem]:
    """Agentic: atomic graph rows as ContextItem list."""
    analysis = normalize_analysis(analysis)
    kategorie = analysis.get("kategorie", [])
    kierunek = analysis.get("kierunek")
    kontekst = analysis.get("kontekst")
    items: list[ContextItem] = []

    for kat in kategorie:
        query = CYPHER_QUERIES.get(kat)
        if not query:
            continue
        if kat in WYMAGA_KIERUNKU and not kierunek:
            items.append(ContextItem(
                id=f"graph:{kat}:missing_kierunek",
                source="graph",
                content=f"Podaj nazwę kierunku, aby uzyskać szczegóły dla kategorii {kat}.",
                metadata={"kategoria": kat, "status": "needs_kierunek"},
            ))
            continue
        params = {"kierunek": kierunek, "kontekst": kontekst}
        try:
            results = graph.query(query, params)
            for i, row in enumerate(results):
                content = " | ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
                items.append(ContextItem(
                    id=f"graph:{kat}:{i}",
                    source="graph",
                    content=content,
                    metadata={"kategoria": kat, **{k: v for k, v in row.items() if v is not None}},
                ))
        except Exception as e:
            items.append(ContextItem(
                id=f"graph:{kat}:error",
                source="graph",
                content=f"Błąd zapytania: {e}",
                metadata={"kategoria": kat, "status": "error"},
            ))

    return items
