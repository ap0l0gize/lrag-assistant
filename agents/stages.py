"""Build per-stage agentic output for compare traces and reports."""


def build_agentic_stages(state: dict) -> dict:
    trace = state.get("agent_trace") or []
    trace_str = " ".join(trace)
    stages: dict = {}

    stages["guard"] = state.get("guard_decision")

    if state.get("blocked"):
        stages["final"] = {
            "text": state.get("final_answer", ""),
            "via": "refuse",
        }
        return stages

    stages["intent"] = {
        "raw": state.get("intent_raw"),
        "normalized": state.get("intent"),
    }

    raw_items = state.get("raw_items") or []
    graph_count = sum(1 for i in raw_items if i.get("source") == "graph")
    vector_count = sum(1 for i in raw_items if i.get("source") == "vector")
    stages["retrieve"] = {
        "graph_count": graph_count,
        "vector_count": vector_count,
        "items_total": len(raw_items),
        "items_preview": [
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "content_preview": (item.get("content") or "")[:200],
            }
            for item in raw_items[:10]
        ],
    }

    decisions = state.get("filter_decisions") or []
    kept = sum(1 for d in decisions if d.get("verdict") == "keep")
    stages["filter"] = {
        "summary": {"keep": kept, "reject": len(decisions) - kept},
        "decisions": decisions,
    }

    if "answer" not in trace_str and "clarify" in trace_str:
        stages["final"] = {
            "text": state.get("final_answer", ""),
            "via": "filter_empty",
        }
        return stages

    draft = state.get("draft_answer")
    if draft:
        stages["answer"] = {"draft_answer": draft}

    verification = state.get("answer_verification")
    if verification:
        stages["fact_check"] = verification

    final_text = state.get("final_answer", "")
    if final_text:
        if "refuse" in trace_str:
            via = "refuse"
        elif "fact_check -> rejected" in trace_str or (
            verification and not verification.get("grounded", True)
        ):
            via = "fact_check_reject"
        elif "clarify" in trace_str:
            via = "clarify"
        else:
            via = "fact_check_accept"
        stages["final"] = {"text": final_text, "via": via}

    return stages
