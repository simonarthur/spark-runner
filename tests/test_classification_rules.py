"""Tests for classification rules loading and prompt building.

Pure function tests — no LLM mocking needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sparky_runner
from sparky_runner import ClassificationRules, load_classification_rules, _build_rules_prompt_section


# ── load_classification_rules ────────────────────────────────────────────


class TestLoadClassificationRules:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_classification_rules(tmp_path / "nonexistent.txt")
        assert result.error_rules == []
        assert result.warning_rules == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text("")
        result = load_classification_rules(path)
        assert result.error_rules == []
        assert result.warning_rules == []

    def test_both_sections_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text(
            "[ERRORS]\n"
            "Search is broken\n"
            "Form fails\n"
            "\n"
            "[WARNINGS]\n"
            "Index shifted\n"
        )
        result = load_classification_rules(path)
        assert result.error_rules == ["Search is broken", "Form fails"]
        assert result.warning_rules == ["Index shifted"]

    def test_comments_and_blanks_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text(
            "# Top comment\n"
            "\n"
            "[ERRORS]\n"
            "# This is an error rule comment\n"
            "Real error rule\n"
            "\n"
            "[WARNINGS]\n"
            "Real warning rule\n"
        )
        result = load_classification_rules(path)
        assert result.error_rules == ["Real error rule"]
        assert result.warning_rules == ["Real warning rule"]

    def test_case_insensitive_headers(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text(
            "[errors]\n"
            "lower error\n"
            "[Warnings]\n"
            "mixed warning\n"
        )
        result = load_classification_rules(path)
        assert result.error_rules == ["lower error"]
        assert result.warning_rules == ["mixed warning"]

    def test_lines_before_sections_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text(
            "This line has no section\n"
            "Neither does this\n"
            "[ERRORS]\n"
            "Real error\n"
        )
        result = load_classification_rules(path)
        assert result.error_rules == ["Real error"]
        assert result.warning_rules == []

    def test_errors_only(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text("[ERRORS]\nOnly error\n")
        result = load_classification_rules(path)
        assert result.error_rules == ["Only error"]
        assert result.warning_rules == []

    def test_warnings_only(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.txt"
        path.write_text("[WARNINGS]\nOnly warning\n")
        result = load_classification_rules(path)
        assert result.error_rules == []
        assert result.warning_rules == ["Only warning"]


# ── _build_rules_prompt_section ──────────────────────────────────────────


class TestBuildRulesPromptSection:
    def test_empty_rules_returns_empty_string(self) -> None:
        assert _build_rules_prompt_section(ClassificationRules()) == ""

    def test_error_only_rules(self) -> None:
        rules = ClassificationRules(error_rules=["Search broken"])
        result = _build_rules_prompt_section(rules)
        assert "PRIORITY CLASSIFICATION RULES" in result
        assert 'classified as "error"' in result
        assert "1. Search broken" in result
        assert 'classified as "warning"' not in result

    def test_warning_only_rules(self) -> None:
        rules = ClassificationRules(warning_rules=["Index shifted"])
        result = _build_rules_prompt_section(rules)
        assert "PRIORITY CLASSIFICATION RULES" in result
        assert 'classified as "warning"' in result
        assert "1. Index shifted" in result
        assert 'classified as "error"' not in result

    def test_both_sections(self) -> None:
        rules = ClassificationRules(
            error_rules=["Error A", "Error B"],
            warning_rules=["Warning X"],
        )
        result = _build_rules_prompt_section(rules)
        assert 'classified as "error"' in result
        assert "1. Error A" in result
        assert "2. Error B" in result
        assert 'classified as "warning"' in result
        assert "1. Warning X" in result
