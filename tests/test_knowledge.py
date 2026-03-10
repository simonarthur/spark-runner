"""Tests for spark_runner.knowledge module."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from spark_runner.knowledge import (
    _MAX_KNOWLEDGE_CHARS,
    find_relevant_knowledge,
)
from tests.conftest import make_llm_response


def _make_task(index: int, content_size: int = 100) -> dict[str, Any]:
    """Build a synthetic knowledge index entry for a task file."""
    return {
        "filename": f"subtask-{index}.txt",
        "name": f"Subtask {index}",
        "content": "x" * content_size,
    }


class TestFindRelevantKnowledgeTruncation:
    """Verify that an oversized knowledge index is truncated before the API call."""

    def test_truncates_large_index(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Each task file produces roughly content_size + overhead chars.
        # Create enough task files to exceed the budget.
        content_per_task = 100_000
        num_tasks = (_MAX_KNOWLEDGE_CHARS // content_per_task) + 5
        knowledge_index = [_make_task(i, content_per_task) for i in range(num_tasks)]

        client = MagicMock()
        client.messages.create.return_value = make_llm_response(
            json.dumps({
                "reusable_subtasks": [],
                "relevant_observations": [],
                "coverage_notes": "",
            })
        )

        result = find_relevant_knowledge("new task", knowledge_index, client)

        # The call should succeed.
        assert result["reusable_subtasks"] == []

        # The index_text passed to the API must be within budget.
        call_args = client.messages.create.call_args
        user_message: str = call_args.kwargs["messages"][0]["content"]
        # The index_text is embedded in the prompt; just check the whole
        # prompt doesn't contain all the goals' content.
        assert len(user_message) < _MAX_KNOWLEDGE_CHARS + 10_000  # some prompt overhead

        # A truncation warning should be printed.
        captured = capsys.readouterr()
        assert "knowledge index truncated" in captured.out
        assert "task file(s) to fit token limit" in captured.out

    def test_no_truncation_when_within_budget(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Small indexes should pass through without truncation."""
        knowledge_index = [_make_task(i, 100) for i in range(3)]

        client = MagicMock()
        client.messages.create.return_value = make_llm_response(
            json.dumps({
                "reusable_subtasks": [],
                "relevant_observations": [],
                "coverage_notes": "",
            })
        )

        find_relevant_knowledge("small task", knowledge_index, client)

        captured = capsys.readouterr()
        assert "truncated" not in captured.out
