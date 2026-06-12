"""
Demo: porównanie 3 podejść RAG na liście przykładowych pytań.

1. Vector-only  — retrieval PDF (Chroma) + LLM
2. Baseline hybrid — graf Neo4j + wektor, bez agentów
3. Agentic — LangGraph z guard / filter / fact-check (+ trace)
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.prompts import ChatPromptTemplate

from agents.graph import run as agentic_run
from pipelines.baseline_hybrid import generate_answer as baseline_answer
from retrieval.clients import model
from retrieval.vector_retriever import retrieve_from_vector

SAMPLE_QUESTIONS_PATH = ROOT / "examples" / "sample_questions.json"

VECTOR_PROMPT = ChatPromptTemplate.from_template("""
Jesteś inteligentnym asystentem uczelnianym. Odpowiadaj rzeczowo i zwięźle, \
wyłącznie na podstawie dostarczonego kontekstu z dokumentów PDF.
Jeśli kontekst nie zawiera wystarczających informacji, poinformuj o tym użytkownika.

Pytanie: {question}

=== KONTEKST Z DOKUMENTÓW PDF ===
{vector_context}

ODPOWIEDŹ:
""")


def load_sample_questions() -> list[dict]:
    return json.loads(SAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))


def run_vector_only(question: str) -> tuple[str, int]:
    t0 = time.perf_counter()
    vector_context, _ = retrieve_from_vector(question)
    chain = VECTOR_PROMPT | model
    response = chain.invoke({
        "question": question,
        "vector_context": vector_context or "Brak pasujących dokumentów.",
    })
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return response.content, latency_ms


def run_baseline(question: str) -> tuple[str, int]:
    t0 = time.perf_counter()
    answer = baseline_answer(question, verbose=False)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return answer, latency_ms


def run_agentic(question: str) -> tuple[dict, int]:
    t0 = time.perf_counter()
    result = agentic_run(question, verbose=True)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    result["latency_ms"] = latency_ms
    return result, latency_ms


def print_agentic_details(result: dict) -> None:
    if result.get("blocked"):
        guard = result.get("guard_decision", {})
        print(f"[Blocked] category={guard.get('category')} reason={guard.get('reason')}")

    for step in result.get("agent_trace", []):
        print(f"  -> {step}")

    if result.get("filter_decisions"):
        keep = sum(1 for d in result["filter_decisions"] if d.get("verdict") == "keep")
        reject = len(result["filter_decisions"]) - keep
        print(f"\n[Filter] keep={keep} reject={reject}")

    verification = result.get("answer_verification", {})
    if verification:
        print(f"[Fact check] grounded={verification.get('grounded')}")
        if verification.get("issues"):
            print(f"  issues: {verification['issues']}")

    print(f"\nODPOWIEDŹ: {result.get('final_answer', '')}")


def run_question(question: str, index: int | None = None, total: int | None = None) -> None:
    header = f"[{index}/{total}] {question}" if index is not None else question
    print(f"\n{'=' * 70}")
    print(header)
    print("=" * 70)

    print("\n--- Vector-only ---")
    try:
        answer, ms = run_vector_only(question)
        print(f"({ms} ms)\n{answer}")
    except Exception as e:
        print(f"BŁĄD: {e}")

    print("\n--- Baseline hybrid ---")
    try:
        answer, ms = run_baseline(question)
        print(f"({ms} ms)\n{answer}")
    except Exception as e:
        print(f"BŁĄD: {e}")

    print("\n--- Agentic ---")
    try:
        result, ms = run_agentic(question)
        print(f"({ms} ms)")
        print_agentic_details(result)
    except Exception as e:
        print(f"BŁĄD: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: vector-only vs baseline hybrid vs agentic"
    )
    parser.add_argument("--index", type=int, help="Indeks pytania z examples/sample_questions.json (0-based)")
    parser.add_argument("--question", help="Własne pytanie (zamiast listy)")
    parser.add_argument("--skip-preflight", action="store_true", help="Pomiń diagnostykę środowiska")
    args = parser.parse_args()

    if not args.skip_preflight:
        eval_dir = ROOT / "evaluation"
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        from benchmark_runner import preflight_checks

        if not preflight_checks():
            print("\nPreflight FAILED — uzupełnij .env, uruchom Neo4j/Ollama i ingest_hybrid.py")
            sys.exit(1)
        print("Preflight OK\n")

    if args.question:
        run_question(args.question)
        return

    samples = load_sample_questions()
    if args.index is not None:
        if args.index < 0 or args.index >= len(samples):
            parser.error(f"--index musi być 0..{len(samples) - 1}")
        run_question(samples[args.index]["question"], args.index + 1, len(samples))
        return

    for i, case in enumerate(samples, 1):
        run_question(case["question"], i, len(samples))


if __name__ == "__main__":
    main()
