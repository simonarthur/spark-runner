"""Tests for spark_runner.pytest_plugin: SparkTestRunner and spark_config fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spark_runner.models import CredentialProfile, SparkConfig
from spark_runner.pytest_plugin import SparkTestRunner


# ── Local fixture re-export ───────────────────────────────────────────────
# The spark_config fixture is registered via the entry-point when the package
# is installed, but within this project's own test run it is not automatically
# discovered.  We re-expose it here as a local fixture so the
# TestSparkConfigFixture class can exercise it directly.

@pytest.fixture()
def spark_config(request: pytest.FixtureRequest, tmp_path: Path) -> SparkConfig:
    """Local proxy for the plugin's spark_config fixture."""
    from spark_runner import pytest_plugin as _plugin
    return _plugin.spark_config.__wrapped__(request, tmp_path)  # type: ignore[attr-defined]


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path) -> SparkConfig:
    """Build a minimal SparkConfig backed by a temporary directory."""
    return SparkConfig(
        data_dir=tmp_path / "spark_data",
        base_url="https://test.example.com",
        credentials={"default": CredentialProfile(email="t@example.com", password="pw")},
        active_credential_profile="default",
        update_summary=False,
        update_tasks=False,
        auto_close=True,
        headless=True,
    )


# ── SparkTestRunner ─────────────────────────────────────────────────────


class TestSparkTestRunner:
    def test_instantiation_with_config(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)
        assert runner.config is config

    def test_stores_config_as_attribute(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)
        assert isinstance(runner.config, SparkConfig)

    def test_config_base_url_accessible(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)
        assert runner.config.base_url == "https://test.example.com"

    def test_config_credential_profile_accessible(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)
        assert runner.config.active_credential_profile == "default"

    @pytest.mark.asyncio
    async def test_execute_calls_run_single(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)

        mock_result = MagicMock()
        with patch(
            "spark_runner.orchestrator.run_single", new=AsyncMock(return_value=mock_result)
        ) as mock_run:
            result = await runner.execute(prompt="Do something")

        mock_run.assert_called_once()
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_execute_passes_prompt_in_task_spec(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)

        mock_result = MagicMock()
        with patch(
            "spark_runner.orchestrator.run_single", new=AsyncMock(return_value=mock_result)
        ) as mock_run:
            await runner.execute(prompt="Test prompt text")

        call_args = mock_run.call_args
        task_spec = call_args.args[0]
        assert task_spec.prompt == "Test prompt text"

    @pytest.mark.asyncio
    async def test_execute_passes_goal_path_in_task_spec(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)
        goal_path = tmp_path / "my-goal-task.json"

        mock_result = MagicMock()
        with patch(
            "spark_runner.orchestrator.run_single", new=AsyncMock(return_value=mock_result)
        ) as mock_run:
            await runner.execute(goal_path=goal_path)

        call_args = mock_run.call_args
        task_spec = call_args.args[0]
        assert task_spec.goal_path == goal_path

    @pytest.mark.asyncio
    async def test_execute_uses_default_credential_profile(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)

        mock_result = MagicMock()
        with patch(
            "spark_runner.orchestrator.run_single", new=AsyncMock(return_value=mock_result)
        ) as mock_run:
            await runner.execute(prompt="p")

        call_args = mock_run.call_args
        task_spec = call_args.args[0]
        assert task_spec.credential_profile == "default"

    @pytest.mark.asyncio
    async def test_execute_accepts_credential_profile_override(self, tmp_path: Path) -> None:
        profiles = {
            "default": CredentialProfile(),
            "admin": CredentialProfile(email="a@example.com", password="apw"),
        }
        config = SparkConfig(
            data_dir=tmp_path,
            credentials=profiles,
            active_credential_profile="default",
        )
        runner = SparkTestRunner(config)

        mock_result = MagicMock()
        with patch(
            "spark_runner.orchestrator.run_single", new=AsyncMock(return_value=mock_result)
        ) as mock_run:
            await runner.execute(prompt="p", credential_profile="admin")

        call_args = mock_run.call_args
        task_spec = call_args.args[0]
        assert task_spec.credential_profile == "admin"

    @pytest.mark.asyncio
    async def test_execute_passes_config_to_run_single(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        runner = SparkTestRunner(config)

        mock_result = MagicMock()
        with patch(
            "spark_runner.orchestrator.run_single", new=AsyncMock(return_value=mock_result)
        ) as mock_run:
            await runner.execute(prompt="p")

        call_args = mock_run.call_args
        passed_config = call_args.args[1]
        assert passed_config is config


# ── spark_config fixture ────────────────────────────────────────────────


class TestSparkConfigFixture:
    def test_fixture_produces_spark_config(self, spark_config: SparkConfig) -> None:
        assert isinstance(spark_config, SparkConfig)

    def test_fixture_uses_tmp_path_as_data_dir(
        self, spark_config: SparkConfig, tmp_path: Path
    ) -> None:
        # The fixture uses tmp_path internally; the data_dir must be a valid path
        assert spark_config.data_dir.is_absolute()

    def test_fixture_has_sensible_defaults(self, spark_config: SparkConfig) -> None:
        assert spark_config.base_url != ""
        assert spark_config.active_credential_profile == "default"

    def test_fixture_disables_summary_updates(self, spark_config: SparkConfig) -> None:
        # During tests we must not write back to stored summaries
        assert spark_config.update_summary is False

    def test_fixture_disables_task_updates(self, spark_config: SparkConfig) -> None:
        assert spark_config.update_tasks is False

    def test_fixture_enables_auto_close(self, spark_config: SparkConfig) -> None:
        assert spark_config.auto_close is True

    def test_fixture_creates_data_directories(self, spark_config: SparkConfig) -> None:
        assert spark_config.tasks_dir is not None
        assert spark_config.goal_summaries_dir is not None
        assert spark_config.runs_dir is not None
        assert spark_config.tasks_dir.exists()
        assert spark_config.goal_summaries_dir.exists()
        assert spark_config.runs_dir.exists()

    def test_fixture_has_default_models(self, spark_config: SparkConfig) -> None:
        assert "browser_control" in spark_config.models
        assert "summarization" in spark_config.models
