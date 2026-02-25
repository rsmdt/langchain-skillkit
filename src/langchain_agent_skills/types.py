"""Configuration types for nodes and skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_agent_skills.frontmatter import parse_frontmatter


@dataclass(frozen=True)
class NodeConfig:
    """Parsed configuration for a graph node agent."""

    name: str
    description: str
    system_prompt: str = ""
    skills: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)

    @property
    def has_allowed_tools(self) -> bool:
        return len(self.allowed_tools) > 0

    @classmethod
    def from_file(cls, path: Path) -> NodeConfig:
        """Parse a node configuration from a markdown file with frontmatter.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If required fields ``name`` or ``description`` are missing.
        """
        result = parse_frontmatter(path)
        metadata = result.metadata

        if "name" not in metadata:
            raise ValueError(f"Node config {path} is missing required field 'name'")
        if "description" not in metadata:
            raise ValueError(f"Node config {path} is missing required field 'description'")

        return cls(
            name=metadata["name"],
            description=metadata["description"],
            system_prompt=result.content,
            skills=metadata.get("skills", []),
            allowed_tools=metadata.get("allowed-tools", []),
        )


@dataclass(frozen=True)
class SkillConfig:
    """Parsed configuration for an agent skill."""

    name: str
    description: str
    instructions: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    directory: Path = field(default=Path("."))
    reference_files: list[str] = field(default_factory=list)

    @classmethod
    def from_directory(cls, skill_dir: Path) -> SkillConfig:
        """Parse a skill from its directory containing SKILL.md.

        Raises:
            FileNotFoundError: If ``SKILL.md`` does not exist in *skill_dir*.
        """
        skill_md = Path(skill_dir) / "SKILL.md"
        result = parse_frontmatter(skill_md)
        metadata = result.metadata

        ref_files = [
            f.name
            for f in Path(skill_dir).iterdir()
            if f.is_file() and f.name != "SKILL.md"
        ]

        return cls(
            name=metadata.get("name", skill_dir.name),
            description=metadata.get("description", ""),
            instructions=result.content,
            allowed_tools=metadata.get("allowed-tools", []),
            directory=Path(skill_dir),
            reference_files=sorted(ref_files),
        )
