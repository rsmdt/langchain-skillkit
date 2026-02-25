"""Tests for node configuration validation."""

from pathlib import Path

from langchain_core.tools import StructuredTool

from langchain_agent_skills.registry import ToolRegistry
from langchain_agent_skills.types import NodeConfig
from langchain_agent_skills.validate import validate_node_config

FIXTURES = Path(__file__).parent / "fixtures"


def _make_tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: "ok",
        name=name,
        description=f"A mock {name} tool",
    )


class TestValidateNodeConfig:
    def test_valid_researcher_config(self):
        registry = ToolRegistry()
        registry.register(_make_tool("web_search"), _make_tool("calculate"))
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/researcher.md")

        errors = validate_node_config(config, registry, FIXTURES / "skills")

        assert errors == []

    def test_valid_analyst_config(self):
        registry = ToolRegistry()
        registry.register(
            _make_tool("sql_query"),
            _make_tool("calculate"),
            _make_tool("web_search"),
        )
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/analyst.md")

        errors = validate_node_config(
            config, registry, FIXTURES / "skills"
        )

        # stakeholder_mapping skill doesn't exist in fixtures
        assert any("missing skill" in e for e in errors)

    def test_reports_unknown_allowed_tool(self):
        registry = ToolRegistry()
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/analyst.md")

        errors = validate_node_config(config, registry, FIXTURES / "skills")

        assert any("sql_query" in e for e in errors)

    def test_reports_unknown_skill_tool(self):
        registry = ToolRegistry()
        # Register no tools — skill's allowed-tools will fail
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/researcher.md")

        errors = validate_node_config(config, registry, FIXTURES / "skills")

        assert any("web_search" in e for e in errors)
