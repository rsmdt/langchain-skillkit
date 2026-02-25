# langchain-agent-skills

Frontmatter-driven agent nodes for LangGraph with progressive skill loading.

Write markdown files with YAML frontmatter. Get back LangGraph node functions with tool routing, skill loading, and the sender pattern handled for you. No manual plumbing.

## Install

```bash
pip install langchain-agent-skills
```

**Requires:** Python 3.11+, `langchain-core>=0.3`, `langgraph>=0.4`

## Quick Start

### 1. Define a node

Create a markdown file with YAML frontmatter for your agent's configuration, and the system prompt as the body.

`prompts/nodes/researcher.md`

```markdown
---
name: researcher
description: Research specialist that gathers factual information
skills:
  - market_sizing
---
You are a highly capable Research Assistant.
Your primary goal is to gather factual information.
Use load_skill to activate capabilities before using skill-specific tools.
```

### 2. Define a skill

Skills follow the [AgentSkills.io specification](https://agentskills.io/specification). Each skill lives in its own directory with a `SKILL.md` file.

`skills/market_sizing/SKILL.md`

```markdown
---
name: market-sizing
description: Calculate TAM, SAM, and SOM for market analysis
allowed-tools:
  - web_search
  - calculate
---
# Market Sizing Methodology

Step 1: Define market boundaries...
Step 2: Top-down analysis using industry reports...
Step 3: Bottom-up validation with unit economics...

**Reference Documents:**
- `calculator.py`: Python template. Use `read_reference` to view it.
```

Reference files like `calculator.py` go alongside `SKILL.md` in the skill directory. The agent loads them on demand via the `read_reference` tool.

### 3. Wire it up

```python
from langchain_agent_skills import create_agent, SkillToolkit, ToolRegistry, AgentState
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Define your tools
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Register tools and create the toolkit
registry = ToolRegistry()
registry.register(web_search, calculate)

toolkit = SkillToolkit(skills_dir="skills/", registry=registry)

# Create the agent node
llm = ChatOpenAI(model="gpt-4o", temperature=0)
researcher = create_agent("prompts/nodes/researcher.md", toolkit=toolkit, llm=llm)

# Build a LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("tools", ToolNode(registry.all_tools() + toolkit.get_tools()))

workflow.add_edge(START, "researcher")
workflow.add_conditional_edges(
    "researcher",
    lambda s: "tools" if s["messages"][-1].tool_calls else END,
    ["tools", END],
)
workflow.add_edge("tools", "researcher")

graph = workflow.compile()
```

## How It Works

### Progressive skill loading

Agents start with two tools always available: `load_skill` and `read_reference`. When the LLM calls `load_skill("market_sizing")`, the skill's `allowed-tools` are merged into the agent's `available_tools` state. On the next turn, the agent can use those tools.

```
Turn 1: Agent has [load_skill, read_reference]
        LLM calls load_skill("market_sizing")

Turn 2: Agent has [load_skill, read_reference, web_search, calculate]
        LLM calls web_search("TAM for SaaS market")
```

Skills that declare no `allowed-tools` unlock all registered tools when loaded.

### Skill catalog injection

When a node declares `skills` in its frontmatter, a markdown catalog is appended to the system prompt so the LLM knows what skills are available:

```markdown
## Available Skills

Use `load_skill` to activate a skill before using its capabilities.

| Skill | Description |
|-------|-------------|
| market-sizing | Calculate TAM, SAM, and SOM for market analysis |
```

## Multi-Agent Routing

When multiple agents share tools, you need to solve the **return address problem**: after the `ToolNode` executes, which agent gets the result back?

The `AgentState` includes a `sender` field. Each agent stamps its name into `sender` when it returns. You route based on that.

### Approach 1: Shared ToolNode (default)

When a node has **no `allowed-tools`** in its frontmatter, it gets access to all registered tools. Multiple agents can share a single `ToolNode`, using `sender` for routing.

```python
researcher = create_agent("prompts/nodes/researcher.md", toolkit=toolkit, llm=llm)
analyst = create_agent("prompts/nodes/analyst.md", toolkit=toolkit, llm=llm)

shared_tools = ToolNode(registry.all_tools() + toolkit.get_tools())

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)
workflow.add_node("tools", shared_tools)

# Both agents route to the shared ToolNode
def route_after_llm(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("researcher", route_after_llm, ["tools", END])
workflow.add_conditional_edges("analyst", route_after_llm, ["tools", END])

# ToolNode routes back to whoever called it
workflow.add_conditional_edges(
    "tools",
    lambda s: s["sender"],
    {"researcher": "researcher", "analyst": "analyst"},
)
```

### Approach 2: Dedicated ToolNode

When a node **has `allowed-tools`**, it only gets those specific tools. Give it a dedicated `ToolNode` with a hardcoded edge back. No `sender` routing needed.

`prompts/nodes/analyst.md`

```markdown
---
name: analyst
description: Data analyst with strict tool boundaries
allowed-tools:
  - sql_query
  - calculate
skills:
  - stakeholder_mapping
---
You are a data analyst. You have direct access to sql_query and calculate.
```

```python
analyst = create_agent("prompts/nodes/analyst.md", toolkit=toolkit, llm=llm)

# Dedicated ToolNode with only the analyst's tools
analyst_tools = ToolNode(
    registry.get_tools(["sql_query", "calculate"]) + toolkit.get_tools()
)

workflow.add_node("analyst", analyst)
workflow.add_node("analyst_tools", analyst_tools)

workflow.add_conditional_edges("analyst", route_after_llm, ["analyst_tools", END])
workflow.add_edge("analyst_tools", "analyst")  # Always routes back
```

### Mixing both approaches

The routing strategy is per-node, driven by frontmatter. In the same graph, some agents can share a `ToolNode` (Approach 1) while others have dedicated ones (Approach 2).

## File Formats

### Node config (`.md`)

```yaml
---
name: researcher                    # Required. Used as function name and sender stamp.
description: Research specialist    # Required. Attached as function docstring.
skills:                             # Optional. Skill catalog injected into system prompt.
  - market_sizing
  - stakeholder_mapping
allowed-tools:                      # Optional. If present, restricts to these tools only.
  - web_search
  - calculate
---
System prompt goes here. This is sent as a SystemMessage before the
conversation history on every LLM invocation.
```

| Field | Required | Effect |
|-------|----------|--------|
| `name` | Yes | Node identifier, function name, `sender` value |
| `description` | Yes | Function docstring |
| `skills` | No | Skill names to include in system prompt catalog |
| `allowed-tools` | No | If present: Approach 2 (dedicated tools). If absent: Approach 1 (all tools). |

### Skill (`SKILL.md`)

```yaml
---
name: market-sizing                 # Optional. Defaults to directory name.
description: Calculate TAM/SAM/SOM  # Optional. Shown in skill catalog.
allowed-tools:                      # Optional. Tools unlocked when skill loaded.
  - web_search
  - calculate
---
Instructions shown to the agent when load_skill is called.

**Reference Documents:**
- `calculator.py`: Use `read_reference` to view it.
```

Skills live in directories under your `skills_dir`:

```
skills/
  market_sizing/
    SKILL.md
    calculator.py       # Loaded via read_reference
    template.md         # Any reference files
  stakeholder_mapping/
    SKILL.md
```

## API Reference

### `create_agent(config_path, *, toolkit, llm, extra_tools=None)`

Creates a LangGraph node function from a node config markdown file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str \| Path` | Path to node `.md` file |
| `toolkit` | `SkillToolkit` | Provides `load_skill` and `read_reference` tools |
| `llm` | `BaseChatModel` | Language model for inference |
| `extra_tools` | `list[BaseTool] \| None` | Additional tools beyond registry and toolkit |

**Returns:** A callable `(AgentState) -> dict` suitable for `workflow.add_node()`.

The returned function has these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `__name__` | `str` | Node name from frontmatter |
| `config` | `NodeConfig` | Parsed node configuration |
| `tool_names` | `set[str]` | All tool names available to this agent |

**Raises:** `ValueError` if the config references unknown tools or missing skills.

### `SkillToolkit(skills_dir, registry)`

LangChain `BaseToolkit` subclass that provides two `StructuredTool` instances.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `skills_dir` | `str` | Path to directory containing skill subdirectories |
| `registry` | `ToolRegistry` | Registry of available tools |

**`toolkit.get_tools()`** returns:

| Tool | Input | Returns |
|------|-------|---------|
| `load_skill` | `skill_name: str` | `Command` updating `available_tools` and `loaded_skills` in state |
| `read_reference` | `skill_name: str, file_name: str` | Raw text content of the reference file |

Both tools validate inputs (name patterns, path traversal) and return error strings on failure via `handle_tool_error`.

### `ToolRegistry()`

Registry mapping tool names to LangChain `BaseTool` instances.

```python
registry = ToolRegistry()
registry.register(tool_a, tool_b)       # Register by .name attribute
registry.get("tool_a")                  # BaseTool | None
registry.get_tools(["tool_a", "tool_b"])  # list[BaseTool], skips missing
registry.all_tools()                    # list[BaseTool]
registry.all_names()                    # set[str]
"tool_a" in registry                    # True
```

### `AgentState`

LangGraph `TypedDict` with reducer annotations for multi-agent state management.

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]              # Conversation history
    sender: str                                          # Last node that called tools
    available_tools: Annotated[list[str], merge_available_tools]  # Progressive tool set
    loaded_skills: Annotated[list[str], operator.add]    # Skill audit trail
```

| Field | Reducer | Behavior |
|-------|---------|----------|
| `messages` | `add_messages` | LangGraph message aggregation (merges by ID) |
| `sender` | None | Overwritten each turn by the calling node |
| `available_tools` | `merge_available_tools` | Union + sort (tools only grow, never shrink) |
| `loaded_skills` | `operator.add` | Appends (accumulates loaded skill names) |

## Security

- **Path traversal prevention**: All file paths are resolved to absolute and checked against the `skills_dir` boundary.
- **Name validation**: Skill names must match `^[a-z][a-z0-9_-]{0,63}$`. Reference file names must match `^[a-zA-Z0-9_.-]{1,255}$`.
- **Tool guarding**: Approach 2 nodes physically cannot access tools outside their `allowed-tools` list. Approach 1 nodes can be further restricted via `available_tools` state.

## Development

```bash
git clone https://github.com/your-org/langchain-agent-skills.git
cd langchain-agent-skills
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -v
```

## License

MIT
