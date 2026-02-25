"""Builds a markdown skill catalog for injection into system prompts."""

from __future__ import annotations

from pathlib import Path

from langchain_agent_skills.types import SkillConfig


def build_skill_catalog(skill_names: list[str], skills_dir: Path) -> str:
    """Build a markdown table listing available skills.

    Reads each skill's ``SKILL.md`` frontmatter to extract name and
    description. Returns empty string if *skill_names* is empty.
    """
    if not skill_names:
        return ""

    rows: list[str] = []
    for name in skill_names:
        skill_dir = Path(skills_dir) / name
        if not (skill_dir / "SKILL.md").exists():
            continue
        config = SkillConfig.from_directory(skill_dir)
        rows.append(f"| {config.name} | {config.description} |")

    if not rows:
        return ""

    lines = [
        "## Available Skills",
        "",
        "Use `load_skill` to activate a skill before using its capabilities.",
        "",
        "| Skill | Description |",
        "|-------|-------------|",
        *rows,
    ]
    return "\n".join(lines)
