import time

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    answer_agent,
    clarify_agent,
    context_filter_agent,
    fact_check_agent,
    guard_agent,
    hybrid_retrieve_agent,
    intent_agent,
    refuse_agent,
    route_after_fact_check,
    route_after_filter,
    route_after_guard,
)
from agents.state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guard", guard_agent)
    g.add_node("refuse", refuse_agent)
    g.add_node("intent", intent_agent)
    g.add_node("hybrid_retrieve", hybrid_retrieve_agent)
    g.add_node("context_filter", context_filter_agent)
    g.add_node("answer", answer_agent)
    g.add_node("fact_check", fact_check_agent)
    g.add_node("clarify", clarify_agent)

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard, {
        "safe": "intent",
        "unsafe": "refuse",
    })
    g.add_edge("intent", "hybrid_retrieve")
    g.add_edge("hybrid_retrieve", "context_filter")
    g.add_conditional_edges("context_filter", route_after_filter, {
        "answer": "answer",
        "clarify": "clarify",
    })
    g.add_edge("answer", "fact_check")
    g.add_conditional_edges("fact_check", route_after_fact_check, {
        "accept": END,
        "clarify": "clarify",
    })
    g.add_edge("clarify", END)
    g.add_edge("refuse", END)

    return g.compile()


def run(question: str, verbose: bool = False) -> AgentState:
    app = build_graph()
    t0 = time.perf_counter()
    result: AgentState = app.invoke({
        "question": question,
        "agent_trace": [],
    })
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)

    if verbose:
        _print_trace(result)

    return result


def _print_trace(state: AgentState) -> None:
    print("\n[Agent trace]")
    for step in state.get("agent_trace", []):
        print(f"  -> {step}")

    guard = state.get("guard_decision", {})
    if guard:
        print(f"\n[Guard] safe={guard.get('safe')} category={guard.get('category')}")
        print(f"  reason: {guard.get('reason')}")

    if state.get("filter_decisions"):
        print("\n[Context filter]")
        for d in state["filter_decisions"]:
            print(f"  {d['verdict'].upper():6} {d['id']} — {d['reason']}")

    verification = state.get("answer_verification", {})
    if verification:
        print(f"\n[Fact check] grounded={verification.get('grounded')}")
        if verification.get("issues"):
            print(f"  issues: {verification.get('issues')}")
        if verification.get("notes"):
            print(f"  notes: {verification.get('notes')}")

    if state.get("draft_answer") and not state.get("blocked"):
        preview = state["draft_answer"][:200].replace("\n", " ")
        print(f"\n[Draft answer] {preview}...")

    if "latency_ms" in state:
        print(f"\n[Latency] {state['latency_ms']} ms")
