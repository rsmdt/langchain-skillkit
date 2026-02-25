"""Tests for the SkillToolkit."""

from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool
from langgraph.types import Command

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
    registry.register(_make_tool("web_search"), _make_tool("calculate"))
    return SkillToolkit(skills_dir=str(FIXTURES / "skills"), registry=registry)


class TestSkillToolkitGetTools:
    def test_returns_two_tools(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools()

        assert len(tools) == 2

    def test_tool_names(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools()
        names = {t.name for t in tools}

        assert names == {"load_skill", "read_reference"}

    def test_tools_are_structured_tools(self):
        toolkit = _make_toolkit()
        tools = toolkit.get_tools()

        for tool in tools:
            assert isinstance(tool, StructuredTool)


class TestLoadSkill:
    def test_returns_command_with_updated_tools(self):
        toolkit = _make_toolkit()
        load_skill = next(t for t in toolkit.get_tools() if t.name == "load_skill")

        result = load_skill.invoke({"skill_name": "market_sizing"})

        assert isinstance(result, Command)
        update = result.update
        assert "available_tools" in update
        assert "web_search" in update["available_tools"]
        assert "calculate" in update["available_tools"]
        assert "load_skill" in update["available_tools"]
        assert "read_reference" in update["available_tools"]

    def test_records_loaded_skill(self):
        toolkit = _make_toolkit()
        load_skill = next(t for t in toolkit.get_tools() if t.name == "load_skill")

        result = load_skill.invoke({"skill_name": "market_sizing"})

        assert result.update["loaded_skills"] == ["market_sizing"]

    def test_rejects_invalid_skill_name(self):
        toolkit = _make_toolkit()
        load_skill = next(t for t in toolkit.get_tools() if t.name == "load_skill")

        result = load_skill.invoke({"skill_name": "../etc/passwd"})

        assert isinstance(result, str)
        assert "Invalid skill name" in result

    def test_rejects_nonexistent_skill(self):
        toolkit = _make_toolkit()
        load_skill = next(t for t in toolkit.get_tools() if t.name == "load_skill")

        result = load_skill.invoke({"skill_name": "nonexistent"})

        assert isinstance(result, str)
        assert "not found" in result


class TestReadReference:
    def test_reads_reference_file(self):
        toolkit = _make_toolkit()
        read_ref = next(t for t in toolkit.get_tools() if t.name == "read_reference")

        result = read_ref.invoke(
            {"skill_name": "market_sizing", "file_name": "calculator.py"}
        )

        assert "calculate_tam" in result
        assert "calculate_sam" in result

    def test_rejects_path_traversal_in_filename(self):
        toolkit = _make_toolkit()
        read_ref = next(t for t in toolkit.get_tools() if t.name == "read_reference")

        result = read_ref.invoke(
            {"skill_name": "market_sizing", "file_name": "../../etc/passwd"}
        )

        assert isinstance(result, str)
        assert "Invalid" in result

    def test_rejects_missing_file(self):
        toolkit = _make_toolkit()
        read_ref = next(t for t in toolkit.get_tools() if t.name == "read_reference")

        result = read_ref.invoke(
            {"skill_name": "market_sizing", "file_name": "nonexistent.py"}
        )

        assert isinstance(result, str)
        assert "not found" in result
