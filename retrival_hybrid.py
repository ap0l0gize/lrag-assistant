"""
Baseline hybrid RAG entrypoint.
Delegates to frozen pipelines.baseline_hybrid (for benchmark comparison).
"""

from pipelines.baseline_hybrid import generate_answer

__all__ = ["generate_answer"]

if __name__ == "__main__":
    questions = [
        "Ile wynosi próg punktowy na Informatykę Stosowaną stacjonarną?",
        # "Jakie przedmioty maturalne są wymagane na Informatykę i Systemy Inteligentne?",
    ]
    q = questions[0]
    print(f"Pytanie: {q}\n{'=' * 60}")
    answer = generate_answer(q, verbose=True)
    print("\n--- ODPOWIEDŹ (baseline) ---")
    print(answer)
