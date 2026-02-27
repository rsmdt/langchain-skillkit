"""Tests for the node metaclass."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_skillkit.node import (
    _normalize_skills,
    _validate_handler_signature,
)
from langchain_skillkit.skill_kit import SkillKit

FIXTURES = Path(__file__).parent / "fixtures"


class TestValidateHandlerSignature:
    def test_accepts_state_only(self):
        def handler(state):
            pass

        injectable = _validate_handler_signature(handler, "test")

        assert injectable == set()

    def test_accepts_state_with_llm(self):
        def handler(state, *, llm):
            pass

        injectable = _validate_handler_signature(handler, "test")

        assert injectable == {"llm"}

    def test_accepts_all_injectables(self):
        def handler(state, *, llm, tools, runtime):
            pass

        injectable = _validate_handler_signature(handler, "test")

        assert injectable == {"llm", "tools", "runtime"}

    def test_rejects_empty_signature(self):
        def handler():
            pass

        with pytest.raises(ValueError, match="at least"):
            _validate_handler_signature(handler, "test")

    def test_rejects_unknown_keyword_param(self):
        def handler(state, *, unknown):
            pass

        with pytest.raises(ValueError, match="unknown handler parameter"):
            _validate_handler_signature(handler, "test")

    def test_rejects_positional_after_state(self):
        def handler(state, extra):
            pass

        with pytest.raises(ValueError, match="keyword-only"):
            _validate_handler_signature(handler, "test")

    def test_accepts_kwargs(self):
        def handler(state, *, llm, **kwargs):
            pass

        injectable = _validate_handler_signature(handler, "test")

        assert injectable == {"llm"}


class TestNormalizeSkills:
    def test_none_returns_none(self):
        assert _normalize_skills(None) is None

    def test_string_returns_skill_kit(self):
        result = _normalize_skills(str(FIXTURES / "skills"))

        assert isinstance(result, SkillKit)

    def test_list_returns_skill_kit(self):
        result = _normalize_skills([str(FIXTURES / "skills")])

        assert isinstance(result, SkillKit)

    def test_skill_kit_passes_through(self):
        kit = SkillKit(str(FIXTURES / "skills"))

        result = _normalize_skills(kit)

        assert result is kit

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_skills(123)


class TestNodeMetaclass:
    def test_creates_compiled_graph(self):
        from langchain_skillkit import node

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class test_agent(node):
            llm = mock_llm
            tools = []

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="hello")],
                    "sender": "test_agent",
                }

        # The metaclass returns a CompiledStateGraph, not a class
        assert hasattr(test_agent, "invoke")
        assert hasattr(test_agent, "ainvoke")

    def test_requires_handler(self):
        from langchain_skillkit import node

        with pytest.raises(ValueError, match="must define.*handler"):

            class bad_agent(node):
                llm = MagicMock()

    def test_requires_llm(self):
        from langchain_skillkit import node

        with pytest.raises(ValueError, match="must define.*llm"):

            class bad_agent(node):
                async def handler(state):
                    return {"messages": [], "sender": "bad"}

    def test_handler_must_be_callable(self):
        from langchain_skillkit import node

        with pytest.raises(ValueError, match="callable"):

            class bad_agent(node):
                llm = MagicMock()
                handler = "not a function"

    def test_tools_must_be_list(self):
        from langchain_skillkit import node

        with pytest.raises(ValueError, match="tools must be a list"):

            class bad_agent(node):
                llm = MagicMock()
                tools = "not a list"

                async def handler(state, *, llm):
                    return {"messages": [], "sender": "bad"}

    def test_node_with_skills(self):
        from langchain_skillkit import node

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class skilled_agent(node):
            llm = mock_llm
            skills = str(FIXTURES / "skills")

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="skilled")],
                    "sender": "skilled_agent",
                }

        assert hasattr(skilled_agent, "invoke")


class TestNodeInvocation:
    @pytest.mark.asyncio
    async def test_no_tools_node_invokes_handler(self):
        from langchain_skillkit import node

        mock_llm = MagicMock()

        class simple(node):
            llm = mock_llm

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="done")],
                    "sender": "simple",
                }

        result = await simple.ainvoke({
            "messages": [HumanMessage(content="hi")],
        })

        assert result["messages"][-1].content == "done"
