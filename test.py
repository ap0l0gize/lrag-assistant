"""
Demo: porównanie 3 podejść RAG na liście przykładowych pytań.

1. Vector-only  — Chroma PDF + LLM (baseline_vector)
2. Hybrid GraphRAG — Neo4j + wektor (baseline_hybrid)
3. Agentic — LangGraph z guard / filter / fact-check
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agents.graph import run as agentic_run
from agents.stages import build_agentic_stages
from pipelines.baseline_hybrid import generate_answer as hybrid_answer
from pipelines.baseline_vector import generate_answer as vector_answer

QUESTIONS_PATH = ROOT / "evaluation" / "questions.json"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def print_agentic_stages(result: dict) -> None:
    stages = build_agentic_stages(result)
    for name, data in stages.items():
        if name == "answer" and isinstance(data, dict):
            print(f"\n[{name}] draft: {data.get('draft_answer', '')[:300]}...")
        elif name == "final" and isinstance(data, dict):
            print(f"\n[{name}] via={data.get('via')}: {data.get('text', '')}")
        elif name == "filter" and isinstance(data, dict):
            s = data.get("summary", {})
            print(f"\n[{name}] keep={s.get('keep')} reject={s.get('reject')}")
        else:
            print(f"\n[{name}] {data}")

    for step in result.get("agent_trace", []):
        print(f"  -> {step}")


def run_question(question: str, index: int | None = None, total: int | None = None) -> None:
    header = f"[{index}/{total}] {question}" if index is not None else question
    print(f"\n{'=' * 70}")
    print(header)
    print("=" * 70)

    print("\n--- Vector ---")
    try:
        print(vector_answer(question, verbose=False))
    except Exception as e:
        print(f"BŁĄD: {e}")

    print("\n--- Hybrid ---")
    try:
        print(hybrid_answer(question, verbose=False))
    except Exception as e:
        print(f"BŁĄD: {e}")

    print("\n--- Agentic ---")
    try:
        result = agentic_run(question, verbose=False)
        print_agentic_stages(result)
    except Exception as e:
        print(f"BŁĄD: {e}")


def main():
    parser = argparse.ArgumentParser(description="Demo: vector vs hybrid vs agentic")
    parser.add_argument("--index", type=int, help="Indeks pytania z evaluation/questions.json (0-based)")
    parser.add_argument("--question", help="Własne pytanie")
    parser.add_argument("--skip-preflight", action="store_true")
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

    samples = load_questions()
    if args.index is not None:
        if args.index < 0 or args.index >= len(samples):
            parser.error(f"--index musi być 0..{len(samples) - 1}")
        run_question(samples[args.index]["question"], args.index + 1, len(samples))
        return

    for i, case in enumerate(samples, 1):
        run_question(case["question"], i, len(samples))


if __name__ == "__main__":
    main()
