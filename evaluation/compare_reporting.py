"""Generate side-by-side markdown report from compare results."""

from __future__ import annotations

import json
from pathlib import Path

COMPARE_DIR = Path(__file__).parent / "results" / "compare"


def _cell(text: str, max_len: int = 500) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) > max_len:
        return t[:max_len] + "..."
    return t


def _format_guard(stage: dict | None) -> str:
    if not stage:
        return "n/a"
    safe = stage.get("safe", True)
    cat = stage.get("category", "?")
    reason = stage.get("reason", "")
    return f"safe={safe}, category={cat}, reason={reason}"


def _format_intent(stage: dict | None) -> str:
    if not stage:
        return "n/a"
    norm = stage.get("normalized") or {}
    cats = norm.get("kategorie", [])
    parts = [f"kategorie={cats}"]
    for key in ("kierunek", "tryb", "przedmiot", "egzamin"):
        if norm.get(key):
            parts.append(f"{key}={norm[key]}")
    return ", ".join(parts)


def _format_filter(stage: dict | None) -> str:
    if not stage:
        return "n/a"
    summary = stage.get("summary") or {}
    line = f"keep={summary.get('keep', 0)} reject={summary.get('reject', 0)}"
    rejects = [d for d in (stage.get("decisions") or []) if d.get("verdict") == "reject"][:3]
    if rejects:
        examples = "; ".join(f"{d.get('id')}: {d.get('reason', '')[:60]}" for d in rejects)
        line += f" | przykłady reject: {examples}"
    return line


def generate_compare_report(summaries: list[dict]) -> Path:
    lines = [
        "# Porównanie: vector RAG vs hybrid GraphRAG vs agentic",
        "",
        f"Liczba pytań: {len(summaries)}",
        "",
    ]

    for item in summaries:
        qid = item["id"]
        question = item.get("question", "")
        sb = item.get("side_by_side", {})
        v = sb.get("vector", {})
        h = sb.get("hybrid", {})
        a = sb.get("agentic", {})

        lines.append(f"## {qid}")
        lines.append("")
        lines.append(f"**Pytanie:** {question}")
        lines.append("")
        lines.append(
            f"Latency: vector {v.get('latency_ms', '?')} ms | "
            f"hybrid {h.get('latency_ms', '?')} ms | "
            f"agentic {a.get('latency_ms', '?')} ms"
        )
        lines.append("")
        lines.append("### Odpowiedzi (side-by-side)")
        lines.append("")
        lines.append("| vector | hybrid | agentic |")
        lines.append("|--------|--------|---------|")
        lines.append(
            f"| {_cell(v.get('final_answer', ''))} "
            f"| {_cell(h.get('final_answer', ''))} "
            f"| {_cell(a.get('final_answer', ''))} |"
        )
        lines.append("")

        stages = (item.get("agentic") or {}).get("stages") or {}
        if stages:
            lines.append("### Agentic — etapy")
            lines.append("")
            lines.append("| Etap | Wynik |")
            lines.append("|------|-------|")
            if stages.get("guard") is not None:
                lines.append(f"| guard | {_cell(_format_guard(stages.get('guard')), 200)} |")
            if stages.get("intent"):
                lines.append(f"| intent | {_cell(_format_intent(stages.get('intent')), 200)} |")
            retrieve = stages.get("retrieve")
            if retrieve:
                lines.append(
                    f"| retrieve | graph={retrieve.get('graph_count', 0)} "
                    f"+ vector={retrieve.get('vector_count', 0)} "
                    f"(łącznie {retrieve.get('items_total', 0)}) |"
                )
            if stages.get("filter"):
                lines.append(f"| filter | {_cell(_format_filter(stages.get('filter')), 300)} |")
            answer = stages.get("answer")
            if answer:
                lines.append(f"| answer (draft) | {_cell(answer.get('draft_answer', ''), 400)} |")
            fc = stages.get("fact_check")
            if fc:
                issues = fc.get("issues") or []
                lines.append(
                    f"| fact_check | grounded={fc.get('grounded')} "
                    f"issues={issues if issues else '[]'} |"
                )
            final = stages.get("final")
            if final:
                lines.append(
                    f"| final ({final.get('via', '?')}) | "
                    f"{_cell(final.get('text', ''), 400)} |"
                )
            lines.append("")
            trace = (item.get("agentic") or {}).get("agent_trace") or []
            if trace:
                lines.append(f"**Trace:** {' → '.join(trace)}")
                lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("Pełne trace JSON: `traces/{id}.json`")
    lines.append("")

    report_path = COMPARE_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def load_and_regenerate_report() -> Path:
    results_path = COMPARE_DIR / "results.json"
    summaries = json.loads(results_path.read_text(encoding="utf-8"))
    return generate_compare_report(summaries)
