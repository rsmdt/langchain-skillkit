"""Agent state definition for LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


def merge_available_tools(
    current: list[str] | None,
    update: list[str],
) -> list[str]:
    """Reducer that unions and sorts tool name lists."""
    return sorted(set(current or []) | set(update))


class AgentState(TypedDict, total=False):
    """Shared state for multi-agent graphs with progressive skill loading.

    Fields:
        messages: Conversation history with LangGraph message aggregation.
        sender: Name of the last node that called tools (for routing back).
        available_tools: Dynamically expanding set of tool names.
        loaded_skills: Accumulated list of loaded skill names.
    """

    messages: Annotated[list, add_messages]
    sender: str
    available_tools: Annotated[list[str], merge_available_tools]
    loaded_skills: Annotated[list[str], operator.add]
