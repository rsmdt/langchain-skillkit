"""Tests for the tool registry."""

from langchain_core.tools import StructuredTool

from langchain_agent_skills.registry import ToolRegistry


def _make_tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: "ok",
        name=name,
        description=f"A mock {name} tool",
    )


class TestToolRegistry:
    def test_register_and_retrieve(self):
        registry = ToolRegistry()
        tool = _make_tool("web_search")
        registry.register(tool)

        assert registry.get("web_search") is tool

    def test_register_multiple(self):
        registry = ToolRegistry()
        t1 = _make_tool("web_search")
        t2 = _make_tool("calculate")
        registry.register(t1, t2)

        assert len(registry) == 2

    def test_get_returns_none_for_missing(self):
        registry = ToolRegistry()

        assert registry.get("missing") is None

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(_make_tool("sql_query"))

        assert "sql_query" in registry
        assert "missing" not in registry

    def test_all_tools(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"), _make_tool("b"))

        assert len(registry.all_tools()) == 2

    def test_all_names(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"), _make_tool("b"))

        assert registry.all_names() == {"a", "b"}

    def test_get_tools_filters_by_names(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"), _make_tool("b"), _make_tool("c"))

        result = registry.get_tools(["a", "c"])

        assert [t.name for t in result] == ["a", "c"]

    def test_get_tools_skips_missing(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"))

        result = registry.get_tools(["a", "missing"])

        assert [t.name for t in result] == ["a"]
