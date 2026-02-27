# langchain-skillkit

Skill-driven agent toolkit for LangGraph with semantic skill discovery.

[![PyPI version](https://img.shields.io/pypi/v/langchain-skillkit.svg)](https://pypi.org/project/langchain-skillkit/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-skillkit.svg)](https://pypi.org/project/langchain-skillkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Give your LangGraph agents reusable, discoverable skills defined as markdown files. Two paths to use: `SkillKit` as a standalone toolkit you wire yourself, or the `node` metaclass that gives you a complete ReAct subgraph with dependency injection.

## Table of Contents

- [Installation & Quick Start](#installation--quick-start)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Security](#security)
- [Why This Toolkit?](#why-this-toolkit)
- [Contributing](#contributing)

## Installation & Quick Start

Requires **Python 3.11+**, `langchain-core>=0.3`, `langgraph>=0.4`.

```bash
pip install langchain-skillkit
```

Skills follow the [AgentSkills.io specification](https://agentskills.io/specification) — each skill is a directory with a `SKILL.md` and optional reference files:

```
skills/
  market-sizing/
    SKILL.md                # Instructions + frontmatter (name, description)
    calculator.py           # Template — loaded on demand via SkillRead
  competitive-analysis/
    SKILL.md
    swot-template.md        # Reference doc — loaded on demand via SkillRead
    examples/
      output.json           # Example output
```

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_skillkit import node, AgentState

# --- Define tools ---

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# --- Declare an agent ---
# Subclassing `node` produces a CompiledStateGraph, not a class.
# The agent gets Skill and SkillRead tools automatically from the skills directory.

class researcher(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "researcher"}

# --- Use standalone ---

result = researcher.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})

# --- Or compose into a parent graph ---

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)
graph = workflow.compile()
```

## Examples

See [`examples/`](examples/) for complete working code:

- **[`standalone_node.py`](examples/standalone_node.py)** — Simplest usage: declare a node class, invoke it
- **[`manual_wiring.py`](examples/manual_wiring.py)** — Use `SkillKit` as a standalone toolkit with full graph control
- **[`multi_agent.py`](examples/multi_agent.py)** — Compose multiple agents in a parent graph

## API Reference

### `SkillKit(skills_dirs)`

Toolkit that provides `Skill` and `SkillRead` tools.

```python
from langchain_skillkit import SkillKit

kit = SkillKit("skills/")
all_tools = [web_search] + kit.tools  # [web_search, Skill, SkillRead]
```

**Parameters:**
- `skills_dirs` (str | list[str]): Directory or list of directories containing skill subdirectories

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `tools` | `list[BaseTool]` | `[Skill, SkillRead]` — built once, cached |

### `node`

Declarative agent builder. Subclassing produces a `CompiledStateGraph`.

```python
from langchain_skillkit import node

class my_agent(node):
    llm = ChatOpenAI(model="gpt-4o")    # Required
    tools = [web_search]                  # Optional
    skills = "skills/"                    # Optional

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "my_agent"}

my_agent.invoke({"messages": [HumanMessage("...")]})
```

**Class attributes:**

| Attribute | Required | Description |
|-----------|----------|-------------|
| `llm` | Yes | Language model instance |
| `tools` | No | List of LangChain tools |
| `skills` | No | Path(s) to skill directories, or a `SkillKit` instance |

**Handler signature:**

```python
async def handler(state, *, llm, tools, runtime): ...
```

`state` is positional. Everything after `*` is keyword-only and injected by name — declare only what you need:

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict` | LangGraph state (positional, required) |
| `llm` | `BaseChatModel` | LLM pre-bound with all tools via `bind_tools()` |
| `tools` | `list[BaseTool]` | All tools available to the agent |
| `runtime` | `Any` | LangGraph runtime context (passed through from config) |

### `AgentState`

Minimal LangGraph state type for composing nodes in a parent graph:

```python
from langchain_skillkit import AgentState
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
```

Extend it with your own fields:

```python
class MyState(AgentState):
    current_project: str
    iteration_count: int
```

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `Annotated[list, add_messages]` | Conversation history with LangGraph message reducer |
| `sender` | `str` | Name of the last node that produced output |

## Security

- **Path traversal prevention**: File paths resolved to absolute and checked against skill directories.
- **Name validation**: Skill names validated per [AgentSkills.io spec](https://agentskills.io/specification) — lowercase alphanumeric + hyphens, 1-64 chars, must match directory name.
- **Tool scoping**: Each `node` subclass only has access to the tools declared in its `tools` attribute.

## Why This Toolkit?

Developers building multi-agent LangGraph systems face these problems:

1. **Prompt reuse is manual.** The same domain instructions get copy-pasted across agents with no versioning or structure.
2. **Agents lack discoverability.** There's no standard way for an LLM to find and select relevant instructions at runtime.
3. **Agent wiring is repetitive.** Every ReAct agent needs the same graph boilerplate: handler node, tool node, conditional edges.
4. **Reference files are inaccessible.** Templates, scripts, and examples referenced in prompts can't be loaded on demand.

This toolkit solves all four with:

- Skill-as-markdown: reusable instructions with structured frontmatter
- Semantic discovery: the LLM matches user intent to skill descriptions at runtime
- Declarative agents: `class my_agent(node)` gives you a complete ReAct subgraph
- On-demand file loading: `SkillRead` lets the LLM pull reference files when needed
- AgentSkills.io spec compliance: portable skills that work across toolkits
- Full type safety: mypy strict mode support

## Contributing

This toolkit is extracted from a production codebase and is actively maintained. Issues, feature requests, and pull requests are welcome.

```bash
git clone https://github.com/rsmdt/langchain-skillkit.git
cd langchain-skillkit
uv sync --extra dev
uv run pytest --tb=short -q
uv run ruff check src/ tests/
uv run mypy src/
```

GitHub: https://github.com/rsmdt/langchain-skillkit
