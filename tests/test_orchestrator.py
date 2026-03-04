"""Tests for orchestrator behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spark_runner.models import CredentialProfile, SparkConfig, TaskSpec


@pytest.fixture()
def _minimal_config(tmp_path: Path) -> SparkConfig:
    """Build a minimal SparkConfig backed by temp directories."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    gs_dir = tmp_path / "goal_summaries"
    gs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    return SparkConfig(
        base_url="https://test.example.com",
        data_dir=tmp_path,
        tasks_dir=tasks_dir,
        goal_summaries_dir=gs_dir,
        runs_dir=runs_dir,
        credentials={"default": CredentialProfile(email="a@b.com", password="pw")},
        active_credential_profile="default",
        headless=True,
        auto_close=True,
        update_summary=False,
        update_tasks=False,
        knowledge_reuse=False,
    )


class TestRunSingleEmptySubtasks:
    """When a goal file has no subtasks, run_single should decompose the prompt."""

    @pytest.mark.asyncio
    async def test_goal_with_empty_subtasks_calls_decompose(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        config = _minimal_config

        # Create a goal file with empty subtasks (as generate-goals produces)
        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data: dict[str, Any] = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        with (
            patch("spark_runner.orchestrator.decompose_task") as mock_decompose,
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_decompose.return_value = [
                {"name": "Fill form", "task": "Fill out the sign-up form"},
            ]
            mock_run_phase.return_value = (True, MagicMock(action_results=MagicMock(return_value=[])), [])
            mock_summarize.return_value = "Filled the form successfully"

            from spark_runner.orchestrator import run_single

            mock_client = MagicMock()
            await run_single(task, config, client=mock_client)

            mock_decompose.assert_called_once()
            # The prompt from the goal file should be passed to decompose_task
            assert mock_decompose.call_args.args[0] == "Test user sign-up"
