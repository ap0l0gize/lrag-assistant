import json
import re

from retrieval.clients import model
from retrieval.graph_retriever import retrieve_graph_items
from retrieval.normalize import analyze_query, normalize_analysis
from retrieval.vector_retriever import retrieve_vector_items

from agents.prompts import (
    ANSWER_PROMPT,
    CLARIFY_TEMPLATE,
    FACT_CHECK_PROMPT,
    FACT_CHECK_REJECT_TEMPLATE,
    FILTER_PROMPT,
    GUARD_PROMPT,
    REFUSE_TEMPLATE,
)
from agents.state import AgentState


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _append_trace(state: AgentState, step: str) -> list[str]:
    trace = list(state.get("agent_trace", []))
    trace.append(step)
    return trace


def guard_agent(state: AgentState) -> dict:
    question = state["question"]
    chain = GUARD_PROMPT | model
    response = chain.invoke({"question": question})
    decision = _parse_json(response.content.strip())

    if decision is None:
        decision = {"safe": True, "category": "safe", "reason": "parse fallback (fail-open)"}

    safe = bool(decision.get("safe", True))
    return {
        "guard_decision": decision,
        "blocked": not safe,
        "agent_trace": _append_trace(state, f"guard -> {'safe' if safe else 'blocked'}"),
    }


def refuse_agent(state: AgentState) -> dict:
    reason = state.get("guard_decision", {}).get("reason", "Niedozwolone zapytanie.")
    return {
        "final_answer": REFUSE_TEMPLATE.format(reason=reason),
        "agent_trace": _append_trace(state, "refuse"),
    }


def intent_agent(state: AgentState) -> dict:
    raw = analyze_query(state["question"])
    normalized = normalize_analysis(raw)
    return {
        "intent_raw": raw,
        "intent": normalized,
        "agent_trace": _append_trace(state, f"intent -> {normalized.get('kategorie')}"),
    }


def hybrid_retrieve_agent(state: AgentState) -> dict:
    intent = state.get("intent", {})
    graph_items = retrieve_graph_items(intent)
    vector_items = retrieve_vector_items(state["question"], n_results=5)
    raw_items = [item.to_dict() for item in graph_items + vector_items]
    return {
        "raw_items": raw_items,
        "agent_trace": _append_trace(
            state,
            f"retrieve -> {len(graph_items)} graph + {len(vector_items)} vector",
        ),
    }


def _format_items_for_filter(items: list[dict]) -> str:
    lines = []
    for item in items:
        preview = item["content"][:400].replace("\n", " ")
        lines.append(f"- id={item['id']} source={item['source']}: {preview}")
    return "\n".join(lines) if lines else "(brak fragmentów)"


def context_filter_agent(state: AgentState) -> dict:
    raw_items = state.get("raw_items", [])
    if not raw_items:
        return {
            "filtered_items": [],
            "filter_decisions": [],
            "filtered_context": "",
            "agent_trace": _append_trace(state, "filter -> empty"),
        }

    chain = FILTER_PROMPT | model
    response = chain.invoke({
        "question": state["question"],
        "intent": json.dumps(state.get("intent", {}), ensure_ascii=False),
        "items": _format_items_for_filter(raw_items),
    })
    parsed = _parse_json(response.content.strip())
    decisions = parsed.get("decisions", []) if parsed else []

    decision_map = {d["id"]: d for d in decisions if "id" in d}
    filtered_items = []
    filter_decisions = []

    for item in raw_items:
        dec = decision_map.get(item["id"])
        if dec:
            verdict = dec.get("verdict", "keep")
            reason = dec.get("reason", "")
        else:
            verdict = "keep"
            reason = "brak decyzji LLM — domyślnie keep"

        filter_decisions.append({
            "id": item["id"],
            "verdict": verdict,
            "reason": reason,
        })
        if verdict == "keep":
            filtered_items.append(item)

    if not filter_decisions and raw_items:
        filtered_items = raw_items
        filter_decisions = [
            {"id": i["id"], "verdict": "keep", "reason": "filter parse fallback"}
            for i in raw_items
        ]

    context_parts = []
    for item in filtered_items:
        tag = f"[{item['source'].upper()}] {item['id']}"
        context_parts.append(f"{tag}:\n{item['content']}")

    kept = sum(1 for d in filter_decisions if d["verdict"] == "keep")
    rejected = len(filter_decisions) - kept

    return {
        "filtered_items": filtered_items,
        "filter_decisions": filter_decisions,
        "filtered_context": "\n\n".join(context_parts),
        "agent_trace": _append_trace(state, f"filter -> keep={kept} reject={rejected}"),
    }


def answer_agent(state: AgentState) -> dict:
    chain = ANSWER_PROMPT | model
    response = chain.invoke({
        "question": state["question"],
        "filtered_context": state.get("filtered_context", ""),
    })
    return {
        "draft_answer": response.content,
        "agent_trace": _append_trace(state, "answer"),
    }


def fact_check_agent(state: AgentState) -> dict:
    draft = state.get("draft_answer", "")
    filtered_context = state.get("filtered_context", "")

    chain = FACT_CHECK_PROMPT | model
    response = chain.invoke({
        "question": state["question"],
        "filtered_context": filtered_context[:4000],
        "draft_answer": draft,
    })
    verification = _parse_json(response.content.strip())
    if verification is None:
        verification = {
            "grounded": True,
            "issues": [],
            "notes": "parse fallback (fail-open)",
        }

    grounded = bool(verification.get("grounded", True))
    result: dict = {
        "answer_verification": verification,
        "agent_trace": _append_trace(
            state,
            f"fact_check -> {'grounded' if grounded else 'rejected'}",
        ),
    }
    if grounded:
        result["final_answer"] = draft
    return result


def clarify_agent(state: AgentState) -> dict:
    verification = state.get("answer_verification")
    if verification and not verification.get("grounded", True):
        issues = verification.get("issues") or []
        detail = "; ".join(issues) if issues else verification.get("notes", "")
        final_answer = FACT_CHECK_REJECT_TEMPLATE.format(issues=detail)
    elif not state.get("filtered_items"):
        final_answer = CLARIFY_TEMPLATE.format(
            notes="Brak pasujących fragmentów w bazie wiedzy."
        )
    else:
        final_answer = CLARIFY_TEMPLATE.format(notes="Spróbuj doprecyzować pytanie.")

    return {
        "final_answer": final_answer,
        "agent_trace": _append_trace(state, "clarify"),
    }


def route_after_guard(state: AgentState) -> str:
    return "unsafe" if state.get("blocked") else "safe"


def route_after_filter(state: AgentState) -> str:
    if not state.get("filtered_items"):
        return "clarify"
    return "answer"


def route_after_fact_check(state: AgentState) -> str:
    verification = state.get("answer_verification", {})
    if verification.get("grounded", True):
        return "accept"
    return "clarify"
