"""Tests for LLM conversation saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from spark_runner.llm_trace import save_llm_conversation


def _make_mock_response(
    text: str = "response text",
    model: str = "claude-sonnet-4-5-20250929",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """Create a mock Anthropic response object."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.model = model
    response.stop_reason = stop_reason
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


class TestSaveLlmConversation:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
        response = _make_mock_response()

        filename = save_llm_conversation(tmp_path, "test_step", messages, response)

        assert filename == "llm_test_step.json"
        saved = json.loads((tmp_path / filename).read_text())
        assert saved["step"] == "test_step"
        assert saved["model"] == "claude-sonnet-4-5-20250929"
        assert saved["response_text"] == "response text"
        assert saved["stop_reason"] == "end_turn"
        assert saved["input_tokens"] == 100
        assert saved["output_tokens"] == 50
        assert saved["timestamp"]  # non-empty
        assert len(saved["messages"]) == 1
        assert saved["messages"][0]["role"] == "user"

    def test_correct_structure(self, tmp_path: Path) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Prompt text"},
        ]
        response = _make_mock_response(text="LLM response")

        save_llm_conversation(tmp_path, "knowledge_matching", messages, response)
        saved = json.loads((tmp_path / "llm_knowledge_matching.json").read_text())

        expected_keys = {
            "step", "model", "timestamp", "messages",
            "response_text", "stop_reason", "input_tokens", "output_tokens",
        }
        assert set(saved.keys()) == expected_keys

    def test_truncates_large_prompt(self, tmp_path: Path) -> None:
        large_content = "x" * 200_000
        messages: list[dict[str, Any]] = [{"role": "user", "content": large_content}]
        response = _make_mock_response()

        save_llm_conversation(tmp_path, "big_prompt", messages, response)
        saved = json.loads((tmp_path / "llm_big_prompt.json").read_text())

        saved_content = saved["messages"][0]["content"]
        assert len(saved_content) < 110_000
        assert "truncated" in saved_content
