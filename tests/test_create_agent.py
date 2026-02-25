"""Tests for the create_agent factory."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from langchain_agent_skills.create_agent import create_agent
from langchain_agent_skills.registry import ToolRegistry
from langchain_agent_skills.skill_toolkit import SkillToolkit

FIXTURES = Path(__file__).parent / "fixtures"


def _make_tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: "ok",
        name=name,
        description=f"A mock {name} tool",
    )


def _make_toolkit() -> SkillToolkit:
    registry = ToolRegistry()
    registry.register(
        _make_tool("web_search"),
        _make_tool("calculate"),
        _make_tool("sql_query"),
    )
    return SkillToolkit(skills_dir=str(FIXTURES / "skills"), registry=registry)


def _make_llm(response_text: str = "I am a research assistant.") -> MagicMock:
    """Create a mock LLM that returns a predictable response."""
    mock_response = AIMessage(content=response_text)

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.invoke.return_value = mock_response
    mock_llm.bind_tools.return_value = mock_bound
    mock_llm.invoke.return_value = mock_response

    return mock_llm


class TestCreateAgentFactory:
    def test_returns_callable(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        assert callable(agent)

    def test_function_name_matches_config(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        assert agent.__name__ == "researcher"

    def test_raises_on_invalid_config(self):
        registry = ToolRegistry()
        toolkit = SkillToolkit(
            skills_dir=str(FIXTURES / "skills"), registry=registry
        )
        llm = _make_llm()

        with pytest.raises(ValueError, match="Invalid node config"):
            create_agent(
                FIXTURES / "prompts/nodes/researcher.md",
                toolkit=toolkit,
                llm=llm,
            )

    def test_attaches_tool_names(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        assert "load_skill" in agent.tool_names
        assert "read_reference" in agent.tool_names
        assert "web_search" in agent.tool_names

    def test_analyst_only_gets_allowed_tools(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        # Need stakeholder_mapping skill for analyst
        (FIXTURES / "skills/stakeholder_mapping").mkdir(exist_ok=True)
        (FIXTURES / "skills/stakeholder_mapping/SKILL.md").write_text(
            "---\nname: stakeholder-mapping\ndescription: Map stakeholders\n---\nInstructions"
        )

        try:
            agent = create_agent(
                FIXTURES / "prompts/nodes/analyst.md",
                toolkit=toolkit,
                llm=llm,
            )

            # Should have allowed-tools + toolkit tools, NOT all registry tools
            assert "sql_query" in agent.tool_names
            assert "calculate" in agent.tool_names
            assert "load_skill" in agent.tool_names
            # web_search is NOT in analyst's allowed-tools
            assert "web_search" not in agent.tool_names
        finally:
            import shutil

            shutil.rmtree(FIXTURES / "skills/stakeholder_mapping", ignore_errors=True)


class TestAgentNodeExecution:
    def test_stamps_sender(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        result = agent({"messages": [HumanMessage(content="Hello")]})

        assert result["sender"] == "researcher"

    def test_returns_llm_response_in_messages(self):
        toolkit = _make_toolkit()
        llm = _make_llm("Market analysis complete.")

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        result = agent({"messages": [HumanMessage(content="Analyze the market")]})

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Market analysis complete."

    def test_binds_tools_to_llm(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        agent({"messages": [HumanMessage(content="Hello")]})

        llm.bind_tools.assert_called_once()
        bound_tools = llm.bind_tools.call_args[0][0]
        tool_names = {t.name for t in bound_tools}
        assert "load_skill" in tool_names
        assert "web_search" in tool_names

    def test_filters_tools_by_available_tools_state(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        # Only load_skill and web_search available in state
        agent({
            "messages": [HumanMessage(content="Hello")],
            "available_tools": ["load_skill", "web_search"],
        })

        bound_tools = llm.bind_tools.call_args[0][0]
        tool_names = {t.name for t in bound_tools}
        assert "load_skill" in tool_names
        assert "web_search" in tool_names
        assert "calculate" not in tool_names

    def test_injects_system_prompt(self):
        toolkit = _make_toolkit()
        llm = _make_llm()

        agent = create_agent(
            FIXTURES / "prompts/nodes/researcher.md",
            toolkit=toolkit,
            llm=llm,
        )

        agent({"messages": [HumanMessage(content="Hello")]})

        bound_llm = llm.bind_tools.return_value
        call_args = bound_llm.invoke.call_args[0][0]
        # First message should be SystemMessage
        assert call_args[0].content.startswith("You are a highly capable Research Assistant")
