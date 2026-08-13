from retrieval.clients import vector_store
from retrieval.models import ContextItem


def retrieve_from_vector(question: str, n_results: int = 3) -> tuple[str, list[dict]]:
    """Baseline: formatted string + preview metadata."""
    docs = vector_store.similarity_search_with_score(question, k=n_results)
    context_parts = []
    preview_data = []

    for doc, score in docs:
        source = doc.metadata.get("source", "N/A")
        tag = f"[PDF] {source}"
        context_parts.append(f"{tag}:\n{doc.page_content}")
        preview_data.append({
            "tag": tag,
            "content": doc.page_content[:500].replace("\n", " ") + "...",
            "score": round(float(score), 4),
        })

    return "\n\n".join(context_parts), preview_data


def retrieve_vector_items(question: str, n_results: int = 5) -> list[ContextItem]:
    """Agentic: PDF chunks as ContextItem list."""
    docs = vector_store.similarity_search_with_score(question, k=n_results)
    items: list[ContextItem] = []

    for i, (doc, score) in enumerate(docs):
        source = doc.metadata.get("source", "N/A")
        items.append(ContextItem(
            id=f"vector:pdf:{i}",
            source="vector",
            content=doc.page_content,
            metadata={"source": source, "score": round(float(score), 4)},
        ))

    return items
