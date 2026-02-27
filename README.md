# langchain-skillkit

Skill-driven agent toolkit for LangGraph with semantic skill discovery.

Two paths to use: `SkillKit` as a standalone toolkit you wire yourself, or the `node` metaclass that gives you a complete ReAct subgraph with dependency injection.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Skills](#skills)
- [Examples](#examples)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Security](#security)
- [Development](#development)

## Installation

```bash
pip install langchain-skillkit
```

**Requires:** Python 3.11+, `langchain-core>=0.3`, `langgraph>=0.4`

## Quick Start

### 1. Create a skill

Skills follow the [AgentSkills.io specification](https://agentskills.io/specification). Each lives in its own directory with a `SKILL.md`.

`skills/market-sizing/SKILL.md`

```markdown
---
name: market-sizing
description: Calculate TAM, SAM, and SOM for market analysis
---
# Market Sizing Methodology

## Step 1: Define Market Boundaries
Identify the total addressable market by defining geographic and demographic scope.

## Step 2: Top-Down Analysis
Use industry reports and macro data to estimate TAM.

## Step 3: Bottom-Up Validation
Cross-reference with unit economics and customer segments.

**Reference Documents:**
- `calculator.py`: Python template for market calculations. Use `SkillRead` to view it.
```

### 2. Build an agent

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_skillkit import node

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

class researcher(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "researcher"}
```

`researcher` is a `CompiledStateGraph` — use it standalone or as a node in a larger graph:

```python
from langchain_core.messages import HumanMessage

# Standalone
result = researcher.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})

# As a node in a parent graph
from langgraph.graph import StateGraph, START, END
from langchain_skillkit import AgentState

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)
graph = workflow.compile()
```

## Skills

### Directory structure

```
skills/
  market-sizing/
    SKILL.md              # Required — instructions + frontmatter
    calculator.py         # Optional — loaded via SkillRead
  competitive-analysis/
    SKILL.md
```

### SKILL.md format

```yaml
---
name: market-sizing                 # Must match directory name
description: Calculate TAM/SAM/SOM  # Shown in Skill tool description
---
Instructions returned when the agent calls Skill("market-sizing").
```

| Field | Required | Constraints |
|-------|----------|-------------|
| `name` | Yes | 1-64 chars, lowercase alphanumeric + hyphens, must match directory name |
| `description` | Yes | Shown in the `Skill` tool description for semantic discovery |

### Reference files

Skills can include reference files (templates, scripts, examples) in their directory. The `SkillRead` tool lets the LLM read them on demand:

```markdown
**Reference Documents:**
- `calculator.py`: Python template. Use `SkillRead` to view it.
```

## Examples

See [`examples/`](examples/) for complete working code:

- **[`standalone_node.py`](examples/standalone_node.py)** — Simplest usage: declare a node class, invoke it
- **[`manual_wiring.py`](examples/manual_wiring.py)** — Use `SkillKit` as a standalone toolkit with full graph control
- **[`multi_agent.py`](examples/multi_agent.py)** — Compose multiple agents in a parent graph

## How It Works

### Semantic skill discovery

The `Skill` tool's description dynamically lists all available skills:

```xml
<available_skills>
<skill>
  <name>market-sizing</name>
  <description>Calculate TAM, SAM, and SOM for market analysis</description>
</skill>
</available_skills>
```

The LLM discovers skills at runtime. If your prompt says "size the market" and a `market-sizing` skill exists, the LLM connects the dots and calls `Skill("market-sizing")` — returning the skill's instructions as a tool message.

### The `node` metaclass

When you declare `class researcher(node):`, the metaclass:

1. Extracts `llm`, `tools`, `skills`, and `handler` from the class body
2. Validates the handler signature
3. Builds a `StateGraph` with the handler node + `ToolNode` + conditional edges
4. Returns a `CompiledStateGraph` (not a class)

Each node is a self-contained ReAct subgraph with its own tools.

## API Reference

### `SkillKit(skills_dirs)`

Toolkit that provides `Skill` and `SkillRead` tools.

```python
from langchain_skillkit import SkillKit

kit = SkillKit("skills/")
all_tools = [web_search] + kit.tools  # [web_search, Skill, SkillRead]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `skills_dirs` | `str \| list[str]` | Directory or list of directories containing skill subdirectories |

| Property | Type | Description |
|----------|------|-------------|
| `tools` | `list[BaseTool]` | `[Skill, SkillRead]` — built once, cached |

### `node`

Metaclass base class. Subclassing produces a `CompiledStateGraph`.

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

#### Class attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `llm` | Yes | Language model instance |
| `tools` | No | List of LangChain tools |
| `skills` | No | Path(s) to skill directories, or a `SkillKit` instance |

#### Handler signature

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

Minimal LangGraph state type used internally by the `node` metaclass. Import it when composing nodes in a parent graph so your state is compatible:

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

## Development

```bash
git clone https://github.com/rsmdt/langchain-skillkit.git
cd langchain-skillkit
uv sync --extra dev
uv run pytest --tb=short -q
uv run ruff check src/ tests/
uv run mypy src/
```

## License

MIT
