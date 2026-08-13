"""
Run vector / hybrid / agentic on a simple questions JSON and save side-by-side results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVAL_DIR / "questions.json"
COMPARE_DIR = EVAL_DIR / "results" / "compare"
TRACES_DIR = COMPARE_DIR / "traces"


def load_questions(path: Path | None = None) -> list[dict]:
    p = path or DEFAULT_QUESTIONS
    cases = json.loads(p.read_text(encoding="utf-8"))
    for case in cases:
        if "id" not in case or "question" not in case:
            raise ValueError(f"Każdy case wymaga id i question: {case}")
    return cases


def _run_with_retry(fn, label: str, max_retries: int = 6):
    from benchmark_runner import _run_with_llm_retry

    return _run_with_llm_retry(fn, label, max_retries=max_retries)


def run_vector(question: str) -> tuple[dict, int]:
    from pipelines.baseline_vector import generate_answer_detailed as vector_detailed

    t0 = time.perf_counter()
    detail = _run_with_retry(lambda: vector_detailed(question), "vector")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {**detail, "latency_ms": latency_ms}, latency_ms


def run_hybrid(question: str) -> tuple[dict, int]:
    from pipelines.baseline_hybrid import generate_answer_detailed as hybrid_detailed

    t0 = time.perf_counter()
    detail = _run_with_retry(lambda: hybrid_detailed(question), "hybrid")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {**detail, "latency_ms": latency_ms}, latency_ms


def run_agentic(question: str) -> tuple[dict, int]:
    from agents.graph import run as agentic_run
    from agents.stages import build_agentic_stages

    t0 = time.perf_counter()
    state = _run_with_retry(lambda: agentic_run(question, verbose=False), "agentic")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    stages = build_agentic_stages(state)
    detail = {
        "final_answer": state.get("final_answer", ""),
        "blocked": bool(state.get("blocked", False)),
        "agent_trace": state.get("agent_trace", []),
        "stages": stages,
        "latency_ms": latency_ms,
    }
    return detail, latency_ms


def build_trace(case: dict, vector: dict, hybrid: dict, agentic: dict) -> dict:
    return {
        "id": case["id"],
        "question": case["question"],
        "side_by_side": {
            "vector": {
                "final_answer": vector.get("final_answer", ""),
                "latency_ms": vector.get("latency_ms"),
            },
            "hybrid": {
                "final_answer": hybrid.get("final_answer", ""),
                "latency_ms": hybrid.get("latency_ms"),
            },
            "agentic": {
                "final_answer": agentic.get("final_answer", ""),
                "latency_ms": agentic.get("latency_ms"),
            },
        },
        "vector": {
            "chunks": vector.get("chunks", []),
            "context": vector.get("context", ""),
            "final_answer": vector.get("final_answer", ""),
            "latency_ms": vector.get("latency_ms"),
        },
        "hybrid": {
            "intent_raw": hybrid.get("intent_raw"),
            "intent": hybrid.get("intent"),
            "graph_context": hybrid.get("graph_context", ""),
            "vector_context": hybrid.get("vector_context", ""),
            "chunks": hybrid.get("chunks", []),
            "final_answer": hybrid.get("final_answer", ""),
            "latency_ms": hybrid.get("latency_ms"),
        },
        "agentic": {
            "stages": agentic.get("stages", {}),
            "agent_trace": agentic.get("agent_trace", []),
            "final_answer": agentic.get("final_answer", ""),
            "blocked": agentic.get("blocked", False),
            "latency_ms": agentic.get("latency_ms"),
        },
    }


def run_compare(
    questions_path: Path | None = None,
    case_id: str | None = None,
    dry_run: bool = False,
    skip_preflight: bool = False,
) -> tuple[list[dict], Path]:
    from compare_reporting import generate_compare_report

    cases = load_questions(questions_path)
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise ValueError(f"Nie znaleziono case id={case_id}")

    if dry_run:
        print(f"Dry-run: {len(cases)} pytań")
        for c in cases:
            print(f"  {c['id']}: {c['question'][:80]}")
        return [], COMPARE_DIR

    if not skip_preflight:
        from benchmark_runner import preflight_checks

        if not preflight_checks():
            raise SystemExit("Preflight failed — napraw środowisko przed compare.")

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    n = len(cases)

    for i, case in enumerate(cases, 1):
        qid = case["id"]
        print(f"\n[{i}/{n}] {qid}")
        print(f"  Q: {case['question'][:100]}{'...' if len(case['question']) > 100 else ''}")

        print("  → vector...")
        vector, v_ms = run_vector(case["question"])
        print(f"     {v_ms} ms")

        print("  → hybrid...")
        hybrid, h_ms = run_hybrid(case["question"])
        print(f"     {h_ms} ms")

        print("  → agentic...")
        agentic, a_ms = run_agentic(case["question"])
        print(f"     {a_ms} ms")

        trace = build_trace(case, vector, hybrid, agentic)
        trace_path = TRACES_DIR / f"{qid}.json"
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

        summaries.append(trace)
        print(f"  trace: {trace_path}")

    results_path = COMPARE_DIR / "results.json"
    results_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = generate_compare_report(summaries)
    print(f"\nWyniki: {results_path}")
    print(f"Raport: {report_path}")
    return summaries, report_path
