"""Skill-driven agent toolkit for LangGraph with semantic skill discovery.

Two paths to use:

**Manual** — use ``SkillKit`` as a standard LangChain toolkit::

    from langchain_skillkit import SkillKit

    kit = SkillKit("skills/")
    tools = [web_search] + kit.tools

**Convenience** — use ``node`` metaclass to get a ReAct subgraph::

    from langchain_skillkit import node

    class researcher(node):
        llm = ChatOpenAI(model="gpt-4o")
        tools = [web_search]
        skills = "skills/"

        async def handler(state, *, llm):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response], "sender": "researcher"}

    graph = researcher.compile()
    graph.invoke({"messages": [HumanMessage("...")]})
"""

from langchain_skillkit.node import node
from langchain_skillkit.skill_kit import SkillKit
from langchain_skillkit.state import AgentState

__all__ = [
    "SkillKit",
    "node",
    "AgentState",
]
