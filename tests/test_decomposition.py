"""Tests for task decomposition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spark_runner.decomposition import decompose_task


def _identity(text: str) -> str:
    return text


class TestDecomposeTaskHints:
    """Tests for hint injection into the decompose_task prompt."""

    def test_decompose_task_includes_hints_in_prompt(self, tmp_path: Path) -> None:
        """When hints are provided, the prompt should contain the OPERATOR HINTS section."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"name": "Login", "task": "Log in"}]')]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = decompose_task(
            prompt="Test the sign-up flow",
            host="https://test.example.com",
            tasks_dir=tasks_dir,
            client=mock_client,
            restore_fn=_identity,
            hints=["Split the form into two phases", "Skip the export step"],
        )

        # Check the prompt sent to the LLM
        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        assert "OPERATOR HINTS" in prompt_content
        assert "Split the form into two phases" in prompt_content
        assert "Skip the export step" in prompt_content
        assert "human reviewer" in prompt_content

    def test_decompose_task_no_hints_omits_section(self, tmp_path: Path) -> None:
        """When hints is None, the prompt should not contain the OPERATOR HINTS section."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"name": "Login", "task": "Log in"}]')]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = decompose_task(
            prompt="Test the sign-up flow",
            host="https://test.example.com",
            tasks_dir=tasks_dir,
            client=mock_client,
            restore_fn=_identity,
            hints=None,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        assert "OPERATOR HINTS" not in prompt_content

    def test_decompose_task_empty_hints_omits_section(self, tmp_path: Path) -> None:
        """When hints is an empty list, the prompt should not contain the OPERATOR HINTS section."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"name": "Login", "task": "Log in"}]')]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = decompose_task(
            prompt="Test the sign-up flow",
            host="https://test.example.com",
            tasks_dir=tasks_dir,
            client=mock_client,
            restore_fn=_identity,
            hints=[],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        assert "OPERATOR HINTS" not in prompt_content
