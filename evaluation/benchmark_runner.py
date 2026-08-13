"""
Benchmark runner v2 — baseline hybrid vs agentic with rich assertions and explainability.
"""

from __future__ import annotations

import json
import os
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

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from agents.graph import run as agentic_run
from pipelines.baseline_hybrid import generate_answer as baseline_answer

SUITE_DIR = Path(__file__).parent / "benchmark_suite"
MANIFEST_PATH = SUITE_DIR / "manifest.json"
RESULTS_V2_DIR = Path(__file__).parent / "results" / "v2"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_with_llm_retry(fn, label: str, max_retries: int = 6):
    """Retry przy 429 / rate limit od providera LLM."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_name = type(e).__name__
            is_rate_limit = "RateLimit" in err_name or "429" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait = min(30 * (attempt + 1), 120)
                _log(f"  [RETRY] {label} — rate limit LLM, czekam {wait}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def preflight_checks() -> bool:
    """Szybka diagnostyka przed długim runem. Zwraca False jeśli coś krytycznego nie działa."""
    from dotenv import load_dotenv

    load_dotenv()
    _log("--- Preflight ---")
    ok = True

    or_key = os.getenv("OPENROUTER_API_KEY", "")
    if not or_key or not or_key.startswith("sk-or-"):
        _log("  [FAIL] OPENROUTER_API_KEY: brak lub nieprawidłowy format (oczekiwane sk-or-...)")
        ok = False
    else:
        model_name = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
        _log(f"  [OK]   OPENROUTER_API_KEY: ustawiony, model={model_name}")

    try:
        from config import Config
        from langchain_neo4j import Neo4jGraph

        c = Config.from_env()
        g = Neo4jGraph(
            url=c.neo4j_url,
            username=c.neo4j_username,
            password=c.neo4j_password,
            refresh_schema=False,
        )
        n = g.query("MATCH (n:Kierunek) RETURN count(n) AS c")[0]["c"]
        _log(f"  [OK]   Neo4j: połączono, Kierunek={n} węzłów")
        if n == 0:
            _log("  [WARN] Neo4j pusty — uruchom: python ingest_hybrid.py")
    except Exception as e:
        _log(f"  [FAIL] Neo4j: {e}")
        ok = False

    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        from config import Config

        c = Config.from_env()
        vs = Chroma(
            collection_name="pdfs",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=c.db_path,
        )
        docs = vs._collection.count()
        _log(f"  [OK]   Chroma: {docs} dokumentów")
        if docs == 0:
            _log("  [WARN] Chroma pusta — uruchom: python ingest_hybrid.py")
    except Exception as e:
        _log(f"  [FAIL] Chroma/Ollama: {e}")
        ok = False

    _log("--- Preflight koniec ---\n")
    return ok


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_suite(suite_id: str) -> list[dict]:
    manifest = load_manifest()
    for suite in manifest["suites"]:
        if suite["id"] == suite_id:
            path = SUITE_DIR / suite["file"]
            return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown suite: {suite_id}")


def load_cases(
    suite_ids: list[str] | None = None,
    case_id: str | None = None,
) -> list[dict]:
    manifest = load_manifest()
    all_ids = [s["id"] for s in manifest["suites"]]
    ids = suite_ids if suite_ids and suite_ids != ["all"] else all_ids

    cases: list[dict] = []
    for sid in ids:
        cases.extend(load_suite(sid))

    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise ValueError(f"Case not found: {case_id}")

    return cases


def _check_must_contain(answer: str, terms: list[str]) -> bool:
    lower = answer.lower()
    return all(t.lower() in lower for t in terms)


def _check_must_contain_any_flat(answer: str, groups: list[list[str]]) -> bool:
    lower = answer.lower()
    for group in groups:
        if not any(term.lower() in lower for term in group):
            return False
    return True


def evaluate_assertions(answer: str, assertions: dict) -> dict:
    """Return accuracy details; accuracy is None if no content assertions."""
    if not assertions:
        return {"accuracy": None, "details": {}}

    details: dict = {}
    has_content_checks = False

    if "must_contain" in assertions:
        has_content_checks = True
        details["must_contain"] = _check_must_contain(answer, assertions["must_contain"])

    if "must_contain_any" in assertions:
        has_content_checks = True
        details["must_contain_any"] = _check_must_contain_any_flat(
            answer, assertions["must_contain_any"]
        )

    if "must_not_contain" in assertions:
        has_content_checks = True
        lower = answer.lower()
        violations = [t for t in assertions["must_not_contain"] if t.lower() in lower]
        details["must_not_contain"] = len(violations) == 0
        if violations:
            details["must_not_contain_violations"] = violations

    if not has_content_checks:
        return {"accuracy": None, "details": details}

    checks = [v for k, v in details.items() if not k.endswith("_violations") and isinstance(v, bool)]
    accuracy = all(checks) if checks else None
    return {"accuracy": accuracy, "details": details}


def _count_filter_keep(state: dict) -> tuple[int, int, int]:
    decisions = state.get("filter_decisions") or []
    kept = sum(1 for d in decisions if d.get("verdict") == "keep")
    rejected = len(decisions) - kept
    graph_kept = sum(
        1 for d in decisions
        if d.get("verdict") == "keep" and str(d.get("id", "")).startswith("graph:")
    )
    return kept, rejected, graph_kept


def _is_clarify_answer(answer: str) -> bool:
    markers = [
        "nie znalazłem wystarczających danych",
        "nie mogę potwierdzić tej odpowiedzi",
        "spróbuj doprecyzować",
    ]
    lower = answer.lower()
    return any(m in lower for m in markers)


def determine_failure_stage(state: dict, expectations: dict) -> str:
    if state.get("blocked"):
        return "guard"

    trace = state.get("agent_trace") or []
    trace_str = " ".join(trace)

    if not state.get("filtered_items") and "filter" in trace_str:
        return "filter_empty"

    verification = state.get("answer_verification") or {}
    if verification and not verification.get("grounded", True):
        return "fact_check_reject"

    answer = state.get("final_answer") or ""
    if _is_clarify_answer(answer) or "clarify" in trace_str:
        return "clarify"

    if expectations.get("expect_grounded") is True:
        if not verification.get("grounded", True):
            return "fact_check_reject"

    if state.get("draft_answer") and not state.get("final_answer"):
        return "answer"

    return "pass"


def check_agentic_expectations(state: dict, expectations: dict) -> dict:
    if not expectations:
        return {}

    kept, rejected, graph_kept = _count_filter_keep(state)
    verification = state.get("answer_verification") or {}
    trace = state.get("agent_trace") or []
    trace_str = " ".join(trace)
    answer = state.get("final_answer") or ""

    checks: dict = {}

    if "min_filter_keep" in expectations:
        checks["filter_keep_ok"] = kept >= expectations["min_filter_keep"]

    if "min_filter_keep_graph" in expectations:
        checks["filter_keep_graph_ok"] = graph_kept >= expectations["min_filter_keep_graph"]

    if expectations.get("expect_grounded") is True:
        checks["grounded_ok"] = bool(verification.get("grounded", False))

    if expectations.get("expect_grounded") is False:
        checks["grounded_ok"] = not verification.get("grounded", True)

    if "expect_clarify" in expectations:
        checks["clarify_ok"] = _is_clarify_answer(answer) == expectations["expect_clarify"]

    if "expect_trace_has" in expectations:
        for needle in expectations["expect_trace_has"]:
            key = f"trace_has_{needle.replace(' ', '_').replace('->', '_')[:40]}"
            checks[key] = needle in trace_str

    checks["failure_stage"] = determine_failure_stage(state, expectations)
    return checks


def run_case_baseline(case: dict, verbose: bool = False) -> dict:
    t0 = time.perf_counter()
    answer = baseline_answer(case["question"], verbose=verbose)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    assertions = case.get("assertions", {})
    acc_result = evaluate_assertions(answer, assertions)
    expect_blocked = assertions.get("expect_blocked", False)
    blocked = False
    blocked_ok = (blocked == expect_blocked) if expect_blocked else None

    return {
        "id": case["id"],
        "question": case["question"],
        "suite": case.get("suite"),
        "pipeline": "baseline",
        "category": case.get("category"),
        "tags": case.get("tags", []),
        "latency_ms": latency_ms,
        "accuracy": acc_result["accuracy"],
        "assertion_details": acc_result["details"],
        "blocked": blocked,
        "blocked_ok": blocked_ok,
        "answer_preview": answer[:300],
        "reasoning": case.get("reasoning"),
        "explainability": {},
    }


def run_case_agentic(case: dict, verbose: bool = False) -> dict:
    t0 = time.perf_counter()
    state = agentic_run(case["question"], verbose=verbose)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer = state.get("final_answer") or ""
    assertions = case.get("assertions", {})
    acc_result = evaluate_assertions(answer, assertions)
    expect_blocked = assertions.get("expect_blocked", False)
    blocked = bool(state.get("blocked", False))
    blocked_ok = (blocked == expect_blocked) if expect_blocked else None

    kept, rejected, graph_kept = _count_filter_keep(state)
    expectations = case.get("agentic_expectations", {})
    agentic_checks = check_agentic_expectations(state, expectations)

    explainability = {
        "guard_decision": state.get("guard_decision"),
        "intent_categories": (state.get("intent") or {}).get("kategorie"),
        "filter_summary": {"keep": kept, "reject": rejected, "graph_keep": graph_kept},
        "filter_decisions": (state.get("filter_decisions") or [])[:6],
        "answer_verification": state.get("answer_verification"),
        "agent_trace": state.get("agent_trace"),
        "failure_stage": agentic_checks.get("failure_stage", "pass"),
    }

    return {
        "id": case["id"],
        "question": case["question"],
        "suite": case.get("suite"),
        "pipeline": "agentic",
        "category": case.get("category"),
        "tags": case.get("tags", []),
        "latency_ms": latency_ms,
        "accuracy": acc_result["accuracy"],
        "assertion_details": acc_result["details"],
        "blocked": blocked,
        "blocked_ok": blocked_ok,
        "answer_preview": answer[:300],
        "reasoning": case.get("reasoning"),
        "explainability": explainability,
        "agentic_checks": agentic_checks,
    }


def load_existing_results() -> list[dict]:
    path = RESULTS_V2_DIR / "eval_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _done_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["id"], r["pipeline"]) for r in rows}


def run_suites(
    suite_ids: list[str] | None = None,
    case_id: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> tuple[list[dict], dict]:
    cases = load_cases(suite_ids, case_id)

    if dry_run:
        print(f"Dry-run: {len(cases)} cases")
        for c in cases:
            print(f"  [{c.get('suite')}] {c['id']} — {c.get('category')}")
        return [], {"dry_run": True, "count": len(cases)}

    if not preflight_checks():
        raise SystemExit("Preflight failed — napraw środowisko przed benchmarkiem.")

    n = len(cases)
    rows: list[dict] = load_existing_results() if resume else []
    done = _done_keys(rows)
    t_start = time.perf_counter()
    remaining = n * 2 - len(done)
    _log(f"Benchmark v2: {n} pytań × 2 pipeline'y = {n * 2} runów")
    if resume:
        _log(f"Resume: {len(done)} gotowych, do zrobienia: {remaining}")
    _log(f"Szacunek: ~{remaining * 8}s–{remaining * 25}s (zależnie od LLM API)\n")

    _log("=== Baseline hybrid ===")
    for i, case in enumerate(cases, 1):
        if (case["id"], "baseline") in done:
            _log(f"  [{i}/{n}] baseline SKIP   {case['id']} (resume)")
            continue
        _log(f"  [{i}/{n}] baseline START {case['id']} ...")
        try:
            row = _run_with_llm_retry(
                lambda c=case: run_case_baseline(c, verbose=verbose),
                f"baseline {case['id']}",
            )
            rows.append(row)
            acc = row["accuracy"]
            acc_str = "n/a" if acc is None else ("OK" if acc else "FAIL")
            _log(
                f"  [{i}/{n}] baseline DONE  {case['id']} — "
                f"{row['latency_ms']}ms — acc={acc_str}"
            )
        except Exception as e:
            _log(f"  [{i}/{n}] baseline ERROR {case['id']} — {type(e).__name__}: {e}")
            raise
        save_results(rows, build_summary(rows))
        time.sleep(2)

    _log("\n=== Agentic LangGraph ===")
    for i, case in enumerate(cases, 1):
        if (case["id"], "agentic") in done:
            _log(f"  [{i}/{n}] agentic SKIP    {case['id']} (resume)")
            continue
        _log(f"  [{i}/{n}] agentic START  {case['id']} ...")
        try:
            row = _run_with_llm_retry(
                lambda c=case: run_case_agentic(c, verbose=verbose),
                f"agentic {case['id']}",
            )
            rows.append(row)
            acc = row["accuracy"]
            acc_str = "n/a" if acc is None else ("OK" if acc else "FAIL")
            stage = row.get("explainability", {}).get("failure_stage", "?")
            _log(
                f"  [{i}/{n}] agentic DONE   {case['id']} — "
                f"{row['latency_ms']}ms — acc={acc_str} — stage={stage}"
            )
        except Exception as e:
            _log(f"  [{i}/{n}] agentic ERROR  {case['id']} — {type(e).__name__}: {e}")
            raise
        save_results(rows, build_summary(rows))
        time.sleep(2)

    elapsed = int(time.perf_counter() - t_start)
    _log(f"\nBenchmark zakończony w {elapsed}s ({elapsed // 60}m {elapsed % 60}s)")

    summary = build_summary(rows)
    return rows, summary


def build_summary(rows: list[dict]) -> dict:
    summary: dict = {"pipelines": {}}
    for pipeline in ("baseline", "agentic"):
        subset = [r for r in rows if r["pipeline"] == pipeline]
        acc_rows = [r for r in subset if r["accuracy"] is not None]
        block_rows = [r for r in subset if r["blocked_ok"] is not None]

        failure_stages: dict[str, int] = {}
        if pipeline == "agentic":
            for r in subset:
                stage = r.get("explainability", {}).get("failure_stage", "unknown")
                failure_stages[stage] = failure_stages.get(stage, 0) + 1

        by_category: dict[str, dict] = {}
        for r in subset:
            cat = r.get("category") or "unknown"
            if cat not in by_category:
                by_category[cat] = {"count": 0, "accuracy_hits": 0, "accuracy_total": 0}
            by_category[cat]["count"] += 1
            if r["accuracy"] is not None:
                by_category[cat]["accuracy_total"] += 1
                if r["accuracy"]:
                    by_category[cat]["accuracy_hits"] += 1

        summary["pipelines"][pipeline] = {
            "count": len(subset),
            "accuracy_rate": (
                sum(1 for r in acc_rows if r["accuracy"]) / len(acc_rows) if acc_rows else None
            ),
            "blocked_rate_ok": (
                sum(1 for r in block_rows if r["blocked_ok"]) / len(block_rows)
                if block_rows else None
            ),
            "avg_latency_ms": (
                sum(r["latency_ms"] for r in subset) / len(subset) if subset else 0
            ),
            "by_category": by_category,
            "failure_stages": failure_stages if pipeline == "agentic" else None,
        }

    return summary


def save_results(rows: list[dict], summary: dict) -> Path:
    RESULTS_V2_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_V2_DIR / "eval_results.json"
    summary_path = RESULTS_V2_DIR / "summary.json"

    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results_path
