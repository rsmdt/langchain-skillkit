"""Tool registry for managing available LangChain tools."""

from __future__ import annotations

from langchain_core.tools import BaseTool


class ToolRegistry:
    """Registry mapping tool names to LangChain BaseTool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, *tools: BaseTool) -> None:
        """Register one or more tools by their ``name`` attribute."""
        for tool in tools:
            self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return a tool by name, or ``None`` if not registered."""
        return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[BaseTool]:
        """Return tools matching *names*, skipping unregistered ones."""
        return [self._tools[n] for n in names if n in self._tools]

    def all_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def all_names(self) -> set[str]:
        """Return the set of all registered tool names."""
        return set(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
