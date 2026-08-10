"""LangGraph routing for durable practice-generation operations."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.schemas import PracticeGraphCommand


class PracticeGenerationGraphState(TypedDict):
    message: dict[str, object]
    processed: bool


def build_practice_generation_graph(orchestrator: PracticeGenerationOrchestrator):
    def route(state: PracticeGenerationGraphState) -> str:
        command = PracticeGraphCommand.model_validate(state["message"])
        if command.operation == "plan_and_fill":
            return "plan_and_fill"
        if command.operation == "finalize":
            return "finalize"
        if command.operation == "generate_wave":
            return "generate_wave"
        return "generate_group"

    def plan_and_fill(state: PracticeGenerationGraphState) -> dict[str, bool]:
        command = PracticeGraphCommand.model_validate(state["message"])
        orchestrator.plan_and_fill(command.test_id)
        return {"processed": True}

    def generate_group(state: PracticeGenerationGraphState) -> dict[str, bool]:
        command = PracticeGraphCommand.model_validate(state["message"])
        if command.group_id is None:
            raise ValueError("generate_group requires group_id")
        orchestrator.generate_group(command.test_id, command.group_id)
        return {"processed": True}

    def generate_wave(state: PracticeGenerationGraphState) -> dict[str, bool]:
        command = PracticeGraphCommand.model_validate(state["message"])
        if not command.group_ids:
            raise ValueError("generate_wave requires group_ids")
        orchestrator.generate_wave(command.test_id, command.group_ids)
        return {"processed": True}

    def finalize(state: PracticeGenerationGraphState) -> dict[str, bool]:
        command = PracticeGraphCommand.model_validate(state["message"])
        orchestrator.finalize(command.test_id, attempt=command.attempt)
        return {"processed": True}

    builder = StateGraph(PracticeGenerationGraphState)
    builder.add_node("plan_and_fill", plan_and_fill)
    builder.add_node("generate_group", generate_group)
    builder.add_node("generate_wave", generate_wave)
    builder.add_node("finalize", finalize)
    builder.add_conditional_edges(
        START,
        route,
        {
            "plan_and_fill": "plan_and_fill",
            "generate_group": "generate_group",
            "generate_wave": "generate_wave",
            "finalize": "finalize",
        },
    )
    builder.add_edge("plan_and_fill", END)
    builder.add_edge("generate_group", END)
    builder.add_edge("generate_wave", END)
    builder.add_edge("finalize", END)
    return builder.compile()


class PracticeGraphRunner:
    def __init__(self, orchestrator: PracticeGenerationOrchestrator) -> None:
        self._graph = build_practice_generation_graph(orchestrator)

    def process(self, command: PracticeGraphCommand) -> None:
        self._graph.invoke({"message": command.model_dump(), "processed": False})
