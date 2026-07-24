"""
app/graphs/demo_graph.py
-------------------------
Minimal LangGraph StateGraph demonstrating the two-node workflow pattern.

Node layout:
    START ──► start_node ──► respond_node ──► END

State:
    A plain TypedDict is used for the LangGraph state because LangGraph's
    reducer / checkpointer system works most reliably with plain dicts.
    Pydantic models (AgentRequest / AgentState) live at the API boundary
    in main.py — they are serialised to this dict before entering the graph
    and deserialised after.

Extending this graph:
    - Add new nodes (functions) and wire them with graph.add_edge().
    - Add conditional edges with graph.add_conditional_edges().
    - Swap generate_mock_response() for a real model call when ready.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from services.mock_response_service import generate_mock_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state — plain TypedDict so LangGraph can freely serialise it
# ---------------------------------------------------------------------------


class DemoGraphState(TypedDict):
    """Data carried between nodes inside the LangGraph workflow."""

    request_id: str
    message: str
    user_id: str
    mode: str
    answer: str | None


# ---------------------------------------------------------------------------
# Node functions — each receives the full state dict and returns a partial
# update dict.  LangGraph merges the returned dict into the current state.
# ---------------------------------------------------------------------------


def start_node(state: DemoGraphState) -> dict:
    """Log request metadata and pass state through unchanged."""
    logger.info(
        "request_id=%s actor_id_present=%s mode=%s — graph started",
        state["request_id"],
        bool(state["user_id"]),
        state["mode"],
    )
    # Return empty dict — nothing to change, just logging.
    return {}


def respond_node(state: DemoGraphState) -> dict:
    """Generate the answer and store it in the state."""
    answer = generate_mock_response(state["message"])
    logger.debug("request_id=%s  answer produced", state["request_id"])
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_demo_graph() -> StateGraph:
    """Construct and compile the demo StateGraph.

    Returns a *compiled* graph ready to call with .invoke(state_dict).

    Example::

        graph = build_demo_graph()
        result = graph.invoke({
            "request_id": "abc",
            "message": "hello",
            "user_id": "local-user",
            "mode": "demo",
            "answer": None,
        })
        print(result["answer"])
    """
    builder = StateGraph(DemoGraphState)

    builder.add_node("start_node", start_node)
    builder.add_node("respond_node", respond_node)

    builder.add_edge(START, "start_node")
    builder.add_edge("start_node", "respond_node")
    builder.add_edge("respond_node", END)

    return builder.compile()
