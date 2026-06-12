"""
Generate markdown report and charts from benchmark v2 results.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS_V2_DIR = Path(__file__).parent / "results" / "v2"
CHARTS_DIR = RESULTS_V2_DIR / "charts"


def _load_results() -> tuple[list[dict], dict]:
    results_path = RESULTS_V2_DIR / "eval_results.json"
    summary_path = RESULTS_V2_DIR / "summary.json"
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return rows, summary


def _failure_conclusion(case_row: dict) -> str:
    stage = case_row.get("explainability", {}).get("failure_stage", "pass")
    tags = case_row.get("tags") or []
    reasoning = case_row.get("reasoning") or {}

    if stage == "pass":
        return "Pipeline zakończył się poprawnie (failure_stage=pass)."

    conclusions = {
        "guard": "Guard zablokował zapytanie — oczekiwane dla security, problem dla on-topic.",
        "filter_empty": "Filter nie zostawił żadnego fragmentu — ryzyko fałszywego clarify.",
        "fact_check_reject": "Fact-check odrzucił odpowiedź — weryfikacja grounding nie przeszła.",
        "clarify": "System zwrócił szablon clarify zamiast odpowiedzi merytorycznej.",
        "answer": "Answer agent nie ustawił final_answer — nieoczekiwany stan pośredni.",
    }
    base = conclusions.get(stage, f"Nieznany failure_stage: {stage}.")

    if "pdf_only_domain" in tags and stage == "filter_empty":
        base += " Tag pdf_only_domain: graf pusty, a PDF mógł zostać odrzucony przez filter."
    if "logic_loss_albo" in tags:
        base += " Tag logic_loss_albo: graf gubi relację XOR/albo — znany słaby punkt hybrydy."
    if reasoning.get("agentic_should_help"):
        base += f" Oczekiwanie: {reasoning['agentic_should_help']}"

    return base


def generate_markdown(rows: list[dict], summary: dict) -> str:
    lines = [
        "# Raport benchmarku v2: baseline hybrid vs agentic",
        "",
        "## Podsumowanie",
        "",
    ]

    for pipeline in ("baseline", "agentic"):
        s = summary.get("pipelines", {}).get(pipeline, {})
        lines.append(f"### {pipeline}")
        lines.append(f"- Liczba case'ów: {s.get('count', 0)}")
        acc = s.get("accuracy_rate")
        lines.append(f"- accuracy@assertions: {acc:.1%}" if acc is not None else "- accuracy@assertions: n/a")
        br = s.get("blocked_rate_ok")
        lines.append(f"- blocked_ok: {br:.1%}" if br is not None else "- blocked_ok: n/a")
        lines.append(f"- Średnia latency: {s.get('avg_latency_ms', 0):.0f} ms")
        lines.append("")

    if summary.get("pipelines", {}).get("agentic", {}).get("failure_stages"):
        lines.append("### Agentic — failure stages")
        lines.append("")
        for stage, count in summary["pipelines"]["agentic"]["failure_stages"].items():
            lines.append(f"- `{stage}`: {count}")
        lines.append("")

    lines.append("## Wyniki per case")
    lines.append("")

    case_ids = sorted({r["id"] for r in rows})
    for cid in case_ids:
        case_rows = {r["pipeline"]: r for r in rows if r["id"] == cid}
        if not case_rows:
            continue
        sample = next(iter(case_rows.values()))
        lines.append(f"### {cid} ({sample.get('category')}, suite={sample.get('suite')})")
        lines.append("")
        lines.append(f"**Pytanie:** {sample.get('question', '')}")

        reasoning = sample.get("reasoning") or {}
        if reasoning:
            lines.append(f"**Hipoteza:** {reasoning.get('hypothesis', '')}")
            lines.append(f"**RAG_benchmark winner:** {reasoning.get('rag_benchmark_winner', 'n/a')}")
            lines.append("")

        lines.append("| Pipeline | Accuracy | Blocked OK | Latency |")
        lines.append("|----------|----------|------------|---------|")
        for pl in ("baseline", "agentic"):
            r = case_rows.get(pl)
            if not r:
                continue
            acc = r.get("accuracy")
            acc_s = "n/a" if acc is None else ("OK" if acc else "FAIL")
            bok = r.get("blocked_ok")
            bok_s = "n/a" if bok is None else ("OK" if bok else "FAIL")
            lines.append(f"| {pl} | {acc_s} | {bok_s} | {r.get('latency_ms')} ms |")
        lines.append("")

        agentic = case_rows.get("agentic")
        if agentic:
            exp = agentic.get("explainability", {})
            lines.append("#### Explainability (agentic)")
            lines.append("")
            guard = exp.get("guard_decision") or {}
            lines.append(f"- **guard:** safe={not agentic.get('blocked')} category={guard.get('category')}")
            fs = exp.get("filter_summary") or {}
            lines.append(f"- **filter:** keep={fs.get('keep')} reject={fs.get('reject')} graph_keep={fs.get('graph_keep')}")
            ver = exp.get("answer_verification") or {}
            lines.append(f"- **fact_check:** grounded={ver.get('grounded')} issues={ver.get('issues')}")
            lines.append(f"- **failure_stage:** `{exp.get('failure_stage')}`")
            trace = exp.get("agent_trace") or []
            if trace:
                lines.append(f"- **trace:** {' → '.join(trace)}")
            lines.append("")
            lines.append(f"**Wnioski:** {_failure_conclusion(agentic)}")
            lines.append("")

        lines.append("**Odpowiedzi (preview):**")
        lines.append("")
        for pl in ("baseline", "agentic"):
            r = case_rows.get(pl)
            if r:
                preview = (r.get("answer_preview") or "").replace("\n", " ")[:200]
                lines.append(f"- *{pl}:* {preview}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Wykresy")
    lines.append("")
    lines.append("- `charts/accuracy_by_category.png`")
    lines.append("- `charts/latency_baseline_vs_agentic.png`")
    lines.append("- `charts/agentic_failure_stages.png`")
    lines.append("- `charts/security_blocked_rate.png`")
    lines.append("")

    return "\n".join(lines)


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("matplotlib is required for charts: pip install matplotlib") from e


def chart_accuracy_by_category(rows: list[dict], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    categories = sorted({r.get("category") or "unknown" for r in rows})

    baseline_rates = []
    agentic_rates = []
    for cat in categories:
        for pipeline, rates in (("baseline", baseline_rates), ("agentic", agentic_rates)):
            subset = [r for r in rows if r.get("category") == cat and r["pipeline"] == pipeline]
            acc_rows = [r for r in subset if r["accuracy"] is not None]
            rate = sum(1 for r in acc_rows if r["accuracy"]) / len(acc_rows) if acc_rows else 0
            rates.append(rate)

    x = range(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], baseline_rates, width, label="baseline")
    ax.bar([i + width / 2 for i in x], agentic_rates, width, label="agentic")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("accuracy rate")
    ax.set_title("Accuracy by category")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def chart_latency(rows: list[dict], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    suites = sorted({r.get("suite") or "unknown" for r in rows})

    baseline_lat = []
    agentic_lat = []
    for suite in suites:
        for pipeline, lat_list in (("baseline", baseline_lat), ("agentic", agentic_lat)):
            subset = [r for r in rows if r.get("suite") == suite and r["pipeline"] == pipeline]
            avg = sum(r["latency_ms"] for r in subset) / len(subset) if subset else 0
            lat_list.append(avg)

    x = range(len(suites))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], baseline_lat, width, label="baseline")
    ax.bar([i + width / 2 for i in x], agentic_lat, width, label="agentic")
    ax.set_xticks(list(x))
    ax.set_xticklabels(suites, rotation=20, ha="right")
    ax.set_ylabel("avg latency (ms)")
    ax.set_title("Latency by suite")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def chart_failure_stages(rows: list[dict], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    agentic = [r for r in rows if r["pipeline"] == "agentic"]
    stages: dict[str, int] = defaultdict(int)
    for r in agentic:
        stage = r.get("explainability", {}).get("failure_stage", "unknown")
        stages[stage] += 1

    if not stages:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(stages.keys())
    values = [stages[l] for l in labels]
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("count")
    ax.set_title("Agentic failure stages")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def chart_security_blocked(rows: list[dict], out_path: Path) -> None:
    plt = _ensure_matplotlib()
    security = [r for r in rows if r.get("suite") == "security" and r["pipeline"] == "agentic"]
    if not security:
        return

    ok = sum(1 for r in security if r.get("blocked_ok"))
    fail = len(security) - ok

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [ok, fail],
        labels=[f"blocked_ok ({ok})", f"miss ({fail})"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("Security: agentic blocked_ok rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_report(rows: list[dict] | None = None, summary: dict | None = None) -> Path:
    if rows is None or summary is None:
        rows, summary = _load_results()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_accuracy_by_category(rows, CHARTS_DIR / "accuracy_by_category.png")
    chart_latency(rows, CHARTS_DIR / "latency_baseline_vs_agentic.png")
    chart_failure_stages(rows, CHARTS_DIR / "agentic_failure_stages.png")
    chart_security_blocked(rows, CHARTS_DIR / "security_blocked_rate.png")

    md = generate_markdown(rows, summary)
    report_path = RESULTS_V2_DIR / "report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report: {path}")
