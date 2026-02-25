"""SkillToolkit — LangChain Toolkit for progressive skill loading."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, ToolException
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_agent_skills.registry import ToolRegistry
from langchain_agent_skills.types import SkillConfig

SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
REFERENCE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,255}$")


class LoadSkillInput(BaseModel):
    """Input schema for the load_skill tool."""

    skill_name: str = Field(
        description="Name of the skill to load (e.g. 'market_sizing')"
    )


class ReadReferenceInput(BaseModel):
    """Input schema for the read_reference tool."""

    skill_name: str = Field(
        description="Name of the skill containing the reference file"
    )
    file_name: str = Field(
        description="Name of the reference file to read (e.g. 'calculator.py')"
    )


class SkillToolkit(BaseToolkit):
    """Toolkit providing load_skill and read_reference as StructuredTools.

    These tools are always available to every agent node, enabling
    progressive skill loading and reference file access.
    """

    skills_dir: str
    registry: ToolRegistry

    model_config = {"arbitrary_types_allowed": True}

    def get_tools(self) -> list[BaseTool]:
        """Return the toolkit's tools: load_skill and read_reference."""
        return [
            self._build_load_skill(),
            self._build_read_reference(),
        ]

    def _resolve_skills_dir(self) -> Path:
        return Path(self.skills_dir).resolve()

    def _validate_skill_name(self, skill_name: str) -> None:
        if not SKILL_NAME_PATTERN.match(skill_name):
            available = self._list_skills()
            raise ToolException(
                f"Invalid skill name '{skill_name}'. "
                f"Available skills: {', '.join(available) or 'none'}"
            )

    def _validate_path_traversal(self, resolved: Path, base: Path) -> None:
        if not str(resolved).startswith(str(base) + os.sep):
            raise ToolException("Path traversal detected")

    def _list_skills(self) -> list[str]:
        skills_dir = self._resolve_skills_dir()
        if not skills_dir.exists():
            return []
        return sorted(
            d.name
            for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )

    def _build_load_skill(self) -> StructuredTool:
        skills_dir = self._resolve_skills_dir()
        registry = self.registry

        def load_skill(skill_name: str) -> Any:
            """Load a skill's instructions and progressively unlock its tools.

            Reads the SKILL.md file, injects its content into messages,
            and merges the skill's allowed-tools into the agent's
            available_tools state.
            """
            self._validate_skill_name(skill_name)

            skill_path = (skills_dir / skill_name / "SKILL.md").resolve()
            self._validate_path_traversal(skill_path, skills_dir)

            if not skill_path.exists():
                available = self._list_skills()
                raise ToolException(
                    f"Skill '{skill_name}' not found. "
                    f"Available skills: {', '.join(available) or 'none'}"
                )

            config = SkillConfig.from_directory(skills_dir / skill_name)

            skill_tools = config.allowed_tools
            always_available = {"load_skill", "read_reference"}
            updated_tools = sorted(
                set(skill_tools) | always_available | registry.all_names()
                if not skill_tools
                else set(skill_tools) | always_available
            )

            return Command(
                update={
                    "available_tools": updated_tools,
                    "loaded_skills": [skill_name],
                },
            )

        return StructuredTool.from_function(
            func=load_skill,
            name="load_skill",
            description=(
                "Load a skill's instructions and unlock its tools. "
                "Call this before using skill-specific capabilities."
            ),
            args_schema=LoadSkillInput,
            handle_tool_error=True,
        )

    def _build_read_reference(self) -> StructuredTool:
        skills_dir = self._resolve_skills_dir()

        def read_reference(skill_name: str, file_name: str) -> str:
            """Read a reference file from within a skill directory.

            Returns the raw content of templates, examples, or scripts
            bundled with a skill.
            """
            self._validate_skill_name(skill_name)
            if not REFERENCE_NAME_PATTERN.match(file_name):
                raise ToolException(f"Invalid file name '{file_name}'")

            file_path = (skills_dir / skill_name / file_name).resolve()
            self._validate_path_traversal(file_path, skills_dir)

            if not file_path.exists():
                raise ToolException(
                    f"Reference file '{file_name}' not found in skill '{skill_name}'"
                )

            return file_path.read_text()

        return StructuredTool.from_function(
            func=read_reference,
            name="read_reference",
            description=(
                "Read a reference file (template, example, script) "
                "from within a skill directory."
            ),
            args_schema=ReadReferenceInput,
            handle_tool_error=True,
        )
