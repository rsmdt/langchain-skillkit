"""Tests for skill config validation."""

from langchain_skillkit.types import SkillConfig
from langchain_skillkit.validate import validate_skill_config


class TestValidateSkillConfig:
    def test_valid_config_returns_no_errors(self):
        config = SkillConfig(name="market-sizing", description="Size markets")

        errors = validate_skill_config(config)

        assert errors == []

    def test_missing_name_returns_error(self):
        config = SkillConfig(name="", description="Some description")

        errors = validate_skill_config(config)

        assert any("name" in e for e in errors)

    def test_missing_description_returns_error(self):
        config = SkillConfig(name="test-skill", description="")

        errors = validate_skill_config(config)

        assert any("description" in e for e in errors)

    def test_missing_both_returns_two_errors(self):
        config = SkillConfig(name="", description="")

        errors = validate_skill_config(config)

        assert len(errors) == 2
