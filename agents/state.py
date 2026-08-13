from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str

    guard_decision: dict
    blocked: bool

    intent: dict
    intent_raw: dict

    raw_items: list[dict]
    filtered_items: list[dict]
    filter_decisions: list[dict]
    filtered_context: str

    draft_answer: str
    answer_verification: dict

    final_answer: str
    agent_trace: list[str]
    latency_ms: int
