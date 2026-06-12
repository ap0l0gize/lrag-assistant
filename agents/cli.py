import argparse
import json
import sys

from agents.graph import run

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="AGH-AURA agentic hybrid RAG")
    parser.add_argument("question", nargs="?", help="Pytanie użytkownika")
    parser.add_argument("-v", "--verbose", action="store_true", help="Pokaż trace agentów")
    parser.add_argument("--json", action="store_true", help="Wynik jako JSON")
    args = parser.parse_args()

    question = args.question
    if not question:
        question = sys.stdin.read().strip()
    if not question:
        parser.error("Podaj pytanie jako argument lub na stdin")

    result = run(question, verbose=args.verbose)

    if args.json:
        out = {
            "question": question,
            "blocked": result.get("blocked", False),
            "guard_decision": result.get("guard_decision"),
            "intent": result.get("intent"),
            "filter_decisions": result.get("filter_decisions"),
            "draft_answer": result.get("draft_answer"),
            "answer_verification": result.get("answer_verification"),
            "final_answer": result.get("final_answer"),
            "agent_trace": result.get("agent_trace"),
            "latency_ms": result.get("latency_ms"),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("\n--- ODPOWIEDŹ (agentic) ---")
        print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
