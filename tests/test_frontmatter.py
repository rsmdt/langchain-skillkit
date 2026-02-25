"""Tests for YAML frontmatter parsing."""

from pathlib import Path

import pytest

from langchain_agent_skills.frontmatter import parse_frontmatter

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseFrontmatter:
    def test_parses_name_and_description(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert result.metadata["name"] == "researcher"
        assert result.metadata["description"] == "Research specialist that gathers factual information"

    def test_parses_skills_list(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert result.metadata["skills"] == ["market_sizing"]

    def test_parses_allowed_tools(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/analyst.md")

        assert result.metadata["allowed-tools"] == ["sql_query", "calculate"]

    def test_extracts_body_as_content(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert "You are a highly capable Research Assistant" in result.content

    def test_body_excludes_frontmatter(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert "---" not in result.content
        assert "name:" not in result.content

    def test_returns_empty_metadata_when_no_frontmatter(self, tmp_path):
        md_file = tmp_path / "plain.md"
        md_file.write_text("Just plain markdown content.")

        result = parse_frontmatter(md_file)

        assert result.metadata == {}
        assert result.content == "Just plain markdown content."

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_frontmatter(tmp_path / "nonexistent.md")

    def test_parses_skill_frontmatter(self):
        result = parse_frontmatter(FIXTURES / "skills/market_sizing/SKILL.md")

        assert result.metadata["name"] == "market-sizing"
        assert result.metadata["description"] == "Calculate TAM, SAM, and SOM for market analysis"
        assert result.metadata["allowed-tools"] == ["web_search", "calculate"]
        assert "# Market Sizing Methodology" in result.content
