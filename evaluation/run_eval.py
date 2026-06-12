"""
Compare baseline hybrid vs agentic LangGraph pipeline.

Legacy (no flags): evaluation/benchmark.json
v2 (--suite): evaluation/benchmark_suite/*.json with explainability + reporting
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agents.graph import run as agentic_run
from pipelines.baseline_hybrid import generate_answer as baseline_answer

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"


def accuracy_contains(answer: str, expected: list[str]) -> bool | None:
    if not expected:
        return None
    answer_lower = answer.lower()
    return all(exp.lower() in answer_lower for exp in expected)


def run_benchmark(pipeline: str, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        if pipeline == "baseline":
            answer = baseline_answer(case["question"], verbose=False)
            blocked = False
        else:
            result = agentic_run(case["question"], verbose=False)
            answer = result.get("final_answer", "")
            blocked = result.get("blocked", False)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        acc = accuracy_contains(answer, case.get("expected_answer_contains", []))
        expect_blocked = case.get("expect_blocked", False)
        blocked_ok = (blocked == expect_blocked) if expect_blocked else None

        rows.append({
            "id": case["id"],
            "pipeline": pipeline,
            "category": case.get("category"),
            "latency_ms": latency_ms,
            "accuracy_contains": acc,
            "blocked": blocked,
            "blocked_ok": blocked_ok,
            "answer_preview": answer[:200],
        })
        print(f"  [{pipeline}] {case['id']} — {latency_ms}ms")

    return rows


def summarize(rows: list[dict], pipeline: str) -> dict:
    subset = [r for r in rows if r["pipeline"] == pipeline]
    acc_rows = [r for r in subset if r["accuracy_contains"] is not None]
    block_rows = [r for r in subset if r["blocked_ok"] is not None]
    return {
        "pipeline": pipeline,
        "count": len(subset),
        "accuracy_rate": (
            sum(1 for r in acc_rows if r["accuracy_contains"]) / len(acc_rows)
            if acc_rows else None
        ),
        "blocked_rate_ok": (
            sum(1 for r in block_rows if r["blocked_ok"]) / len(block_rows)
            if block_rows else None
        ),
        "avg_latency_ms": (
            sum(r["latency_ms"] for r in subset) / len(subset) if subset else 0
        ),
    }


def run_legacy():
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Benchmark (legacy): {len(cases)} pytań\n")
    print("=== Baseline hybrid ===")
    baseline_rows = run_benchmark("baseline", cases)
    print("\n=== Agentic LangGraph ===")
    agentic_rows = run_benchmark("agentic", cases)

    all_rows = baseline_rows + agentic_rows
    out_path = RESULTS_DIR / "eval_results.json"
    out_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Podsumowanie ===")
    for pipeline in ("baseline", "agentic"):
        s = summarize(all_rows, pipeline)
        print(f"\n{pipeline}:")
        print(f"  accuracy@contains: {s['accuracy_rate']}")
        print(f"  blocked_ok:        {s['blocked_rate_ok']}")
        print(f"  avg latency:       {s['avg_latency_ms']:.0f} ms")

    print(f"\nWyniki zapisane: {out_path}")


def run_v2(
    suite_arg: str,
    case_id: str | None,
    verbose: bool,
    dry_run: bool,
    with_report: bool,
    resume: bool,
):
    eval_dir = Path(__file__).parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from benchmark_runner import run_suites, save_results
    from reporting import generate_report

    suite_ids = [s.strip() for s in suite_arg.split(",")]
    rows, summary = run_suites(
        suite_ids=suite_ids,
        case_id=case_id,
        verbose=verbose,
        dry_run=dry_run,
        resume=resume,
    )

    if dry_run:
        return

    results_path = save_results(rows, summary)

    print("\n=== Podsumowanie v2 ===")
    for pipeline in ("baseline", "agentic"):
        s = summary.get("pipelines", {}).get(pipeline, {})
        print(f"\n{pipeline}:")
        acc = s.get("accuracy_rate")
        print(f"  accuracy@assertions: {acc:.1%}" if acc is not None else "  accuracy@assertions: n/a")
        br = s.get("blocked_rate_ok")
        print(f"  blocked_ok:          {br:.1%}" if br is not None else "  blocked_ok:          n/a")
        print(f"  avg latency:         {s.get('avg_latency_ms', 0):.0f} ms")
        if s.get("failure_stages"):
            print(f"  failure_stages:      {s['failure_stages']}")

    print(f"\nWyniki zapisane: {results_path}")

    if with_report:
        report_path = generate_report(rows, summary)
        print(f"Raport: {report_path}")
        print(f"Wykresy: {Path(__file__).parent / 'results' / 'v2' / 'charts'}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline hybrid vs agentic")
    parser.add_argument(
        "--suite",
        help="Suite v2: all | rag_benchmark,security,... (omit for legacy benchmark.json)",
    )
    parser.add_argument("--case", help="Run single case id (v2 only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose agent trace")
    parser.add_argument("--dry-run", action="store_true", help="List cases without API calls")
    parser.add_argument("--report", action="store_true", help="Generate report.md + charts (v2)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Wznów z evaluation/results/v2/eval_results.json (pomiń ukończone id+pipeline)",
    )
    args = parser.parse_args()

    if args.suite:
        run_v2(args.suite, args.case, args.verbose, args.dry_run, args.report, args.resume)
    else:
        if args.case or args.dry_run or args.report:
            parser.error("--case, --dry-run, --report require --suite")
        run_legacy()


if __name__ == "__main__":
    main()
