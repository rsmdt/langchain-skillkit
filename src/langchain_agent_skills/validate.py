"""Compile-time validation for node and skill configurations."""

from __future__ import annotations

from pathlib import Path

from langchain_agent_skills.frontmatter import parse_frontmatter
from langchain_agent_skills.registry import ToolRegistry
from langchain_agent_skills.types import NodeConfig


def validate_node_config(
    config: NodeConfig,
    registry: ToolRegistry,
    skills_dir: Path,
) -> list[str]:
    """Validate a node configuration against the registry and skills directory.

    Returns a list of error messages. An empty list means the config is valid.
    """
    errors: list[str] = []

    for tool_name in config.allowed_tools:
        if tool_name not in registry:
            errors.append(
                f"Node '{config.name}' references unknown tool '{tool_name}'"
            )

    for skill_name in config.skills:
        skill_md = Path(skills_dir) / skill_name / "SKILL.md"
        if not skill_md.exists():
            errors.append(
                f"Node '{config.name}' references missing skill '{skill_name}': "
                f"{skill_md} does not exist"
            )
            continue

        result = parse_frontmatter(skill_md)
        for tool_name in result.metadata.get("allowed-tools", []):
            if tool_name not in registry:
                errors.append(
                    f"Skill '{skill_name}' references unknown tool '{tool_name}'"
                )

    return errors
