"""Standalone node — the simplest way to use langchain-skillkit.

Declare a class with the node metaclass and get a complete ReAct agent
with skill support. The result is a CompiledStateGraph you can invoke directly.
"""

from langchain_core.messages import HumanMessage
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


# researcher is a CompiledStateGraph — invoke it directly
if __name__ == "__main__":
    result = researcher.invoke(
        {"messages": [HumanMessage("Size the B2B SaaS market in Europe")]}
    )
    print(result["messages"][-1].content)
