"""Factory for creating LangGraph-compatible agent node functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from langchain_agent_skills.registry import ToolRegistry
from langchain_agent_skills.skill_catalog import build_skill_catalog
from langchain_agent_skills.skill_toolkit import SkillToolkit
from langchain_agent_skills.state import AgentState
from langchain_agent_skills.types import NodeConfig
from langchain_agent_skills.validate import validate_node_config


def create_agent(
    config_path: str | Path,
    *,
    toolkit: SkillToolkit,
    llm: BaseChatModel,
    extra_tools: list[BaseTool] | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """Create a LangGraph node function from a node config markdown file.

    The returned function has the same shape as a hand-written LangGraph
    node: it accepts ``AgentState`` and returns a state update dict with
    ``messages`` and ``sender``.

    Routing strategy is determined by the node's frontmatter:
    - **No ``allowed-tools``** (Approach 1): all registry tools are bound.
      The user routes a shared ``ToolNode`` back via ``state["sender"]``.
    - **Has ``allowed-tools``** (Approach 2): only those tools are bound.
      The user wires a dedicated ``ToolNode`` with hardcoded edge back.

    In both cases, the toolkit's ``load_skill`` and ``read_reference``
    tools are always included, enabling progressive skill loading.

    Args:
        config_path: Path to a node ``.md`` file with YAML frontmatter.
        toolkit: SkillToolkit providing load_skill and read_reference.
        llm: The language model to use for inference.
        extra_tools: Additional tools to include beyond the registry and toolkit.

    Returns:
        A callable ``(AgentState) -> dict`` suitable for
        ``workflow.add_node(name, fn)``.

    Raises:
        ValueError: If the node config is invalid.
    """
    config = NodeConfig.from_file(Path(config_path))
    skills_dir = Path(toolkit.skills_dir)

    errors = validate_node_config(config, toolkit.registry, skills_dir)
    if errors:
        raise ValueError(
            f"Invalid node config '{config.name}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    system_prompt = config.system_prompt
    if config.skills:
        catalog = build_skill_catalog(config.skills, skills_dir)
        if catalog:
            system_prompt = f"{system_prompt}\n\n{catalog}"

    toolkit_tools = toolkit.get_tools()
    extras = list(extra_tools or [])

    if config.has_allowed_tools:
        node_tools = toolkit.registry.get_tools(config.allowed_tools)
    else:
        node_tools = toolkit.registry.all_tools()

    all_tools = node_tools + toolkit_tools + extras
    all_tool_names = {t.name for t in all_tools}

    node_name = config.name

    def agent_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])

        available = set(state.get("available_tools", []))
        if available:
            active_tools = [t for t in all_tools if t.name in available]
        else:
            active_tools = list(all_tools)

        if active_tools:
            bound_llm = llm.bind_tools(active_tools)
        else:
            bound_llm = llm

        messages_for_llm = [SystemMessage(content=system_prompt)] + list(messages)

        response = bound_llm.invoke(messages_for_llm)

        return {
            "messages": [response],
            "sender": node_name,
        }

    agent_node.__name__ = node_name
    agent_node.__doc__ = config.description
    agent_node.__qualname__ = f"create_agent.<locals>.{node_name}"

    agent_node.config = config
    agent_node.tool_names = all_tool_names

    return agent_node
