"""Tests for configuration types."""

from pathlib import Path

import pytest

from langchain_agent_skills.types import NodeConfig, SkillConfig

FIXTURES = Path(__file__).parent / "fixtures"


class TestNodeConfig:
    def test_from_file_parses_researcher(self):
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/researcher.md")

        assert config.name == "researcher"
        assert config.description == "Research specialist that gathers factual information"
        assert config.skills == ["market_sizing"]
        assert config.allowed_tools == []
        assert "You are a highly capable Research Assistant" in config.system_prompt

    def test_from_file_parses_analyst_with_allowed_tools(self):
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/analyst.md")

        assert config.name == "analyst"
        assert config.allowed_tools == ["sql_query", "calculate"]
        assert config.skills == ["stakeholder_mapping"]

    def test_has_allowed_tools_returns_false_for_researcher(self):
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/researcher.md")

        assert config.has_allowed_tools is False

    def test_has_allowed_tools_returns_true_for_analyst(self):
        config = NodeConfig.from_file(FIXTURES / "prompts/nodes/analyst.md")

        assert config.has_allowed_tools is True

    def test_raises_on_missing_name(self, tmp_path):
        md_file = tmp_path / "bad.md"
        md_file.write_text("---\ndescription: no name\n---\nBody")

        with pytest.raises(ValueError, match="name"):
            NodeConfig.from_file(md_file)

    def test_raises_on_missing_description(self, tmp_path):
        md_file = tmp_path / "bad.md"
        md_file.write_text("---\nname: test\n---\nBody")

        with pytest.raises(ValueError, match="description"):
            NodeConfig.from_file(md_file)


class TestSkillConfig:
    def test_from_directory_parses_skill(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market_sizing")

        assert config.name == "market-sizing"
        assert config.description == "Calculate TAM, SAM, and SOM for market analysis"
        assert config.allowed_tools == ["web_search", "calculate"]
        assert "# Market Sizing Methodology" in config.instructions

    def test_raises_on_missing_skill_md(self, tmp_path):
        skill_dir = tmp_path / "empty_skill"
        skill_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            SkillConfig.from_directory(skill_dir)

    def test_lists_reference_files(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market_sizing")

        assert "calculator.py" in config.reference_files
        assert "SKILL.md" not in config.reference_files
