"""Tests for LLM-calling functions in sparky_runner.

Every test uses ``mock_summary_client`` (from conftest) so no real API calls
are made.  The ``make_llm_response`` helper builds the mock response object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import sparky_runner
from sparky_runner import ClassificationRules
from tests.conftest import make_llm_response


# ── generate_task_name ───────────────────────────────────────────────────

class TestGenerateTaskName:
    def test_sanitized_output(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response(
            "  Write Tea Blog  "
        )
        assert sparky_runner.generate_task_name("Write a blog about tea") == "write-tea-blog"

    def test_strips_quotes(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response(
            '"my-task"'
        )
        assert sparky_runner.generate_task_name("anything") == "my-task"

    def test_fallback_on_empty(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response("   ")
        assert sparky_runner.generate_task_name("x") == "unnamed-task"

    def test_special_chars_replaced(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response(
            "Hello World! @#$"
        )
        result = sparky_runner.generate_task_name("x")
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in result)

    def test_long_name_truncated(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response(
            "a-" * 100  # 200 chars
        )
        result = sparky_runner.generate_task_name("x")
        assert len(result) <= 80
        assert not result.endswith("-")


# ── find_relevant_knowledge ──────────────────────────────────────────────

class TestFindRelevantKnowledge:
    def test_empty_index_returns_empty(self) -> None:
        result = sparky_runner.find_relevant_knowledge("do stuff", [])
        assert result == {"reusable_subtasks": [], "relevant_observations": [], "coverage_notes": ""}

    def test_parses_json_response(self, mock_summary_client: MagicMock) -> None:
        response_json = json.dumps({
            "reusable_subtasks": [{"filename": "login.txt", "phase_name": "Login", "reason": "same"}],
            "relevant_observations": ["UI is slow"],
            "coverage_notes": "needs search",
        })
        mock_summary_client.messages.create.return_value = make_llm_response(response_json)
        index = [{
            "goal_file": "g.json",
            "main_task": "test",
            "key_observations": [],
            "subtasks": [{"filename": "login.txt", "name": "Login", "content": "did login"}],
        }]
        result = sparky_runner.find_relevant_knowledge("new task", index)
        assert len(result["reusable_subtasks"]) == 1
        assert result["relevant_observations"] == ["UI is slow"]

    def test_markdown_wrapped_json(self, mock_summary_client: MagicMock) -> None:
        inner = json.dumps({"reusable_subtasks": [], "relevant_observations": ["x"], "coverage_notes": ""})
        mock_summary_client.messages.create.return_value = make_llm_response(
            f"```json\n{inner}\n```"
        )
        index = [{"goal_file": "g.json", "main_task": "t", "key_observations": [], "subtasks": []}]
        result = sparky_runner.find_relevant_knowledge("q", index)
        assert result["relevant_observations"] == ["x"]

    def test_parse_error_fallback(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response("not json at all")
        index = [{"goal_file": "g.json", "main_task": "t", "key_observations": [], "subtasks": []}]
        result = sparky_runner.find_relevant_knowledge("q", index)
        assert result == {"reusable_subtasks": [], "relevant_observations": [], "coverage_notes": ""}


# ── decompose_task ───────────────────────────────────────────────────────

class TestDecomposeTask:
    def test_returns_phases(self, mock_summary_client: MagicMock) -> None:
        phases = [
            {"name": "Login", "task": "Log in to the app"},
            {"name": "Navigate", "task": "Go to dashboard"},
        ]
        mock_summary_client.messages.create.return_value = make_llm_response(json.dumps(phases))
        result = sparky_runner.decompose_task("Do a thing")
        assert len(result) == 2
        assert result[0]["name"] == "Login"

    def test_reuse_loads_file(
        self, mock_summary_client: MagicMock, fake_tasks_dir: Path
    ) -> None:
        (fake_tasks_dir / "login.txt").write_text("Prior login steps")
        phases = [{"name": "Login", "reuse": "login.txt"}]
        mock_summary_client.messages.create.return_value = make_llm_response(json.dumps(phases))
        result = sparky_runner.decompose_task("test", knowledge_match={"reusable_subtasks": [{"filename": "login.txt", "phase_name": "Login", "reason": "r"}]})
        assert "Prior login steps" in result[0]["task"]
        assert "reuse" not in result[0]

    def test_reuse_missing_file_fallback(
        self, mock_summary_client: MagicMock, fake_tasks_dir: Path
    ) -> None:
        phases = [{"name": "Login", "reuse": "gone.txt"}]
        mock_summary_client.messages.create.return_value = make_llm_response(json.dumps(phases))
        result = sparky_runner.decompose_task("test")
        assert "Login" in result[0]["task"]
        assert "reuse" not in result[0]

    def test_observations_included_in_prompt(self, mock_summary_client: MagicMock) -> None:
        """Relevant observations from prior runs are surfaced in the decomposition prompt."""
        phases = [{"name": "Login", "task": "Log in"}]
        mock_summary_client.messages.create.return_value = make_llm_response(json.dumps(phases))

        sparky_runner.decompose_task(
            "Create a campaign",
            knowledge_match={
                "reusable_subtasks": [],
                "relevant_observations": ["Express tools auto-select platforms"],
                "coverage_notes": "",
            },
        )

        call_args = mock_summary_client.messages.create.call_args
        prompt_text: str = call_args.kwargs["messages"][0]["content"]
        assert "Express tools auto-select platforms" in prompt_text
        assert "OBSERVATIONS FROM PRIOR RUNS" in prompt_text


# ── generate_task_report ─────────────────────────────────────────────────

class TestGenerateTaskReport:
    def test_appends_subtasks(self, mock_summary_client: MagicMock) -> None:
        response_json = json.dumps({
            "main_task": "Did the thing",
            "key_observations": ["obs1"],
        })
        mock_summary_client.messages.create.return_value = make_llm_response(response_json)
        summaries: list[dict[str, str]] = [
            {"name": "Login", "outcome": "SUCCESS", "summary": "ok", "filename": "login.txt"},
            {"name": "Nav", "outcome": "SUCCESS", "summary": "ok", "filename": "nav.txt"},
        ]
        report = sparky_runner.generate_task_report("test", "prompt", summaries)
        assert report["main_task"] == "Did the thing"
        assert len(report["subtasks"]) == 2
        assert report["subtasks"][0] == {"subtask": 1, "filename": "login.txt"}
        assert report["subtasks"][1] == {"subtask": 2, "filename": "nav.txt"}

    def test_markdown_wrapped(self, mock_summary_client: MagicMock) -> None:
        inner = json.dumps({"main_task": "t", "key_observations": []})
        mock_summary_client.messages.create.return_value = make_llm_response(
            f"```json\n{inner}\n```"
        )
        report = sparky_runner.generate_task_report("t", "p", [])
        assert report["main_task"] == "t"


# ── classify_observations ────────────────────────────────────────────────

class TestClassifyObservations:
    def test_empty_list(self) -> None:
        assert sparky_runner.classify_observations("prompt", []) == []

    def test_classifies_correctly(self, mock_summary_client: MagicMock) -> None:
        classified = json.dumps([
            {"text": "search broken", "severity": "error"},
            {"text": "layout shifted", "severity": "warning"},
        ])
        mock_summary_client.messages.create.return_value = make_llm_response(classified)
        result = sparky_runner.classify_observations("task", ["search broken", "layout shifted"])
        assert result[0]["severity"] == "error"
        assert result[1]["severity"] == "warning"

    def test_invalid_severity_normalized(self, mock_summary_client: MagicMock) -> None:
        classified = json.dumps([{"text": "x", "severity": "critical"}])
        mock_summary_client.messages.create.return_value = make_llm_response(classified)
        result = sparky_runner.classify_observations("task", ["x"])
        assert result[0]["severity"] == "warning"

    def test_json_parse_error_fallback(self, mock_summary_client: MagicMock) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response("garbage")
        result = sparky_runner.classify_observations("task", ["a", "b"])
        assert len(result) == 2
        assert all(r["severity"] == "warning" for r in result)

    def test_handles_dict_input(self, mock_summary_client: MagicMock) -> None:
        """Already-classified dicts should be re-normalised to plain text for the LLM."""
        classified = json.dumps([{"text": "obs", "severity": "error"}])
        mock_summary_client.messages.create.return_value = make_llm_response(classified)
        result = sparky_runner.classify_observations(
            "task", [{"text": "obs", "severity": "warning"}]
        )
        assert result[0]["text"] == "obs"


# ── classify_observations with rules ────────────────────────────────────


class TestClassifyObservationsWithRules:
    def test_rules_appear_in_prompt(self, mock_summary_client: MagicMock) -> None:
        """When rules are provided, their text should appear in the LLM prompt."""
        rules = ClassificationRules(
            error_rules=["Search dropdown broken"],
            warning_rules=["Index shifted"],
        )
        classified = json.dumps([{"text": "obs", "severity": "error"}])
        mock_summary_client.messages.create.return_value = make_llm_response(classified)

        sparky_runner.classify_observations("task", ["obs"], rules=rules)

        call_args = mock_summary_client.messages.create.call_args
        prompt_text: str = call_args.kwargs["messages"][0]["content"]
        assert "PRIORITY CLASSIFICATION RULES" in prompt_text
        assert "Search dropdown broken" in prompt_text
        assert "Index shifted" in prompt_text

    def test_empty_rules_no_section_in_prompt(self, mock_summary_client: MagicMock) -> None:
        """Empty rules should not add a rules section to the prompt."""
        rules = ClassificationRules()
        classified = json.dumps([{"text": "obs", "severity": "warning"}])
        mock_summary_client.messages.create.return_value = make_llm_response(classified)

        sparky_runner.classify_observations("task", ["obs"], rules=rules)

        call_args = mock_summary_client.messages.create.call_args
        prompt_text: str = call_args.kwargs["messages"][0]["content"]
        assert "PRIORITY CLASSIFICATION RULES" not in prompt_text

    def test_defaults_to_module_rules(
        self, mock_summary_client: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no rules kwarg is passed, the module-level rules are used."""
        module_rules = ClassificationRules(error_rules=["Module-level error rule"])
        monkeypatch.setattr(sparky_runner, "_classification_rules", module_rules)

        classified = json.dumps([{"text": "obs", "severity": "error"}])
        mock_summary_client.messages.create.return_value = make_llm_response(classified)

        sparky_runner.classify_observations("task", ["obs"])

        call_args = mock_summary_client.messages.create.call_args
        prompt_text: str = call_args.kwargs["messages"][0]["content"]
        assert "Module-level error rule" in prompt_text


# ── merge_observations ───────────────────────────────────────────────────

class TestMergeObservations:
    def test_merges_and_preserves_severity(self, mock_summary_client: MagicMock) -> None:
        merged_texts = json.dumps(["kept obs"])
        mock_summary_client.messages.create.return_value = make_llm_response(merged_texts)
        existing: list[dict[str, str]] = [{"text": "kept obs", "severity": "error"}]
        new: list[str] = ["new obs"]
        result = sparky_runner.merge_observations(existing, new)
        assert len(result) == 1
        assert result[0]["text"] == "kept obs"
        assert result[0]["severity"] == "error"

    def test_default_severity_for_new(self, mock_summary_client: MagicMock) -> None:
        merged_texts = json.dumps(["brand new"])
        mock_summary_client.messages.create.return_value = make_llm_response(merged_texts)
        result = sparky_runner.merge_observations([], ["brand new"])
        assert result[0]["severity"] == "warning"

    def test_markdown_wrapped(self, mock_summary_client: MagicMock) -> None:
        inner = json.dumps(["a", "b"])
        mock_summary_client.messages.create.return_value = make_llm_response(
            f"```json\n{inner}\n```"
        )
        result = sparky_runner.merge_observations(["a"], ["b"])
        assert len(result) == 2
