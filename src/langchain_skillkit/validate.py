"""Validation utilities for langchain-skillkit."""

from __future__ import annotations

from langchain_skillkit.types import SkillConfig


def validate_skill_config(config: SkillConfig) -> list[str]:
    """Validate a skill configuration.

    Returns a list of error messages. An empty list means valid.
    """
    errors: list[str] = []

    if not config.name:
        errors.append("Skill is missing required field 'name'")

    if not config.description:
        errors.append(
            f"Skill '{config.name}' is missing required field 'description'"
        )

    return errors
