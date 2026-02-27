# langchain-skillkit

Skill-driven agent toolkit for LangGraph with semantic skill discovery.

Two paths to use: `SkillKit` as a standalone LangChain toolkit you wire yourself, or the `node` metaclass that gives you a complete ReAct subgraph with dependency injection.

## Install

```bash
pip install langchain-skillkit
```

**Requires:** Python 3.11+, `langchain-core>=0.3`, `langgraph>=0.4`

## Quick Start

### 1. Define a skill

Each skill lives in its own directory with a `SKILL.md`.

`skills/market_sizing/SKILL.md`

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

### 2. Use with the `node` metaclass (convenience)

Declare a class — get a `CompiledStateGraph` with an automatic ReAct loop (handler <-> ToolNode):

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

`researcher` is a compiled graph — use it standalone or as a node in a larger graph:

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

#### Handler signature

```python
async def handler(state, *, llm, tools, runtime): ...
```

`state` is positional. Everything after `*` is keyword-only and injected by name — declare only what you need, in any order:

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict` | LangGraph state (positional, required) |
| `llm` | `BaseChatModel` | LLM pre-bound with all tools via `bind_tools()` |
| `tools` | `list[BaseTool]` | All tools available to the agent |
| `runtime` | `Any` | LangGraph runtime context (passed through from config) |

#### Class attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `llm` | Yes | Language model instance |
| `tools` | No | List of LangChain tools |
| `skills` | No | Path(s) to skill directories, or a `SkillKit` instance |

### 3. Use with `SkillKit` (manual wiring)

Use `SkillKit` as a standard LangChain toolkit when you want full control over your graph:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_skillkit import SkillKit

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

llm = ChatOpenAI(model="gpt-4o")
kit = SkillKit("skills/")

# Combine your tools with skill tools
all_tools = [web_search] + kit.get_tools()  # [web_search, Skill, SkillRead]
bound_llm = llm.bind_tools(all_tools)
```

`SkillKit` accepts a single path or a list of paths:

```python
# Multiple skill directories
kit = SkillKit(["skills/", "shared_skills/"])
```

## How It Works

### Semantic skill discovery

The `Skill` tool's description dynamically lists all available skills from the skills directories:

```xml
<available_skills>
<skill>
  <name>market-sizing</name>
  <description>Calculate TAM, SAM, and SOM for market analysis</description>
</skill>
</available_skills>
```

The LLM discovers skills at runtime. If your prompt says "size the market" and a `market-sizing` skill exists, the LLM connects the dots and calls `Skill("market-sizing")` — returning the skill's instructions as a tool message.

### Reference files

Skills can include reference files (templates, scripts, examples) in their directory. The `SkillRead` tool lets the LLM read them:

```
skills/
  market_sizing/
    SKILL.md
    calculator.py       # Loaded via SkillRead
  competitive_analysis/
    SKILL.md
```

### Multi-agent routing

Use the `node` metaclass to create multiple agents and compose them:

```python
from langchain_skillkit import node, AgentState
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class researcher(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "researcher"}

class analyst(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [sql_query, calculate]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "analyst"}

# Compose in a parent graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)
graph = workflow.compile()
```

Each `node` subclass is a self-contained subgraph with its own tools and ReAct loop.

## File Formats

### Skill (`SKILL.md`)

```yaml
---
name: market-sizing                 # Defaults to directory name if omitted.
description: Calculate TAM/SAM/SOM  # Shown in Skill tool's available_skills list.
---
Instructions returned to the agent when Skill("market-sizing") is called.
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Skill identifier. Defaults to directory name. |
| `description` | Yes | One-line summary shown in the `Skill` tool description. |

## API

### `SkillKit(skills_dirs)`

LangChain toolkit that provides `Skill` and `SkillRead` tools.

```python
from langchain_skillkit import SkillKit

kit = SkillKit("skills/")
tools = kit.get_tools()  # [Skill, SkillRead]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `skills_dirs` | `str \| list[str]` | Directory or list of directories containing skill subdirectories |

### `node`

Metaclass base. Subclassing produces a `CompiledStateGraph` with a ReAct loop.

```python
from langchain_skillkit import node

class my_agent(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "my_agent"}

# my_agent is a CompiledStateGraph — not a class
my_agent.invoke({"messages": [HumanMessage("...")]})
```

### `AgentState`

Minimal LangGraph state. Extend with your own fields:

```python
from langchain_skillkit import AgentState

class MyState(AgentState):
    current_project: str
    iteration_count: int
```

| Field | Description |
|-------|-------------|
| `messages` | Conversation history with `add_messages` reducer |
| `sender` | Name of the last node that produced output |

## Security

- **Path traversal prevention**: File paths resolved to absolute and checked against skill directories.
- **Name validation**: Skill names match `^[a-z][a-z0-9_-]{0,63}$`. File names match `^[a-zA-Z0-9_.-]{1,255}$`.
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
