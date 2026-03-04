"""Pytest plugin for spark_runner: fixtures, markers, and configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Generator

import pytest

from spark_runner.models import RunResult, SparkConfig, TaskSpec


def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "spark_goal(path): run a saved goal file")
    config.addinivalue_line("markers", "spark_prompt(text): run a prompt-based task")


@pytest.fixture
def spark_config(request: pytest.FixtureRequest, tmp_path: Path) -> SparkConfig:
    """Configuration for spark_runner tests.

    Defaults to a temporary data directory. Can be overridden via
    ``pyproject.toml`` under ``[tool.spark_runner]``.
    """
    # Check for pyproject.toml config
    ini_config = request.config.inicfg or {}
    tool_config: dict[str, Any] = {}

    # Try to read from pyproject.toml [tool.spark_runner] section
    rootdir = request.config.rootpath
    pyproject = rootdir / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                tool_config = data.get("tool", {}).get("spark_runner", {})
            except Exception:
                pass

    data_dir = Path(tool_config.get("data_dir", str(tmp_path / "spark_runner")))
    if not data_dir.is_absolute():
        data_dir = rootdir / data_dir

    config = SparkConfig(
        data_dir=data_dir,
        base_url=tool_config.get("base_url", "https://sparky-web-dev.vercel.app"),
        active_credential_profile=tool_config.get("credential_profile", "default"),
        headless=tool_config.get("headless", True),
        update_summary=False,  # Don't modify stored data during tests
        update_tasks=False,
        auto_close=True,
    )
    config.ensure_dirs()
    return config


@pytest.fixture
async def spark_runner(spark_config: SparkConfig) -> Any:
    """Configured runner instance for tests.

    Returns an object with an ``execute`` method for running tasks.
    """
    return SparkTestRunner(spark_config)


@pytest.fixture
async def spark_result(
    request: pytest.FixtureRequest, spark_runner: Any
) -> RunResult:
    """Auto-runs the goal/prompt from marker and returns the result.

    Use with ``@pytest.mark.spark_goal(...)`` or ``@pytest.mark.spark_prompt(...)``.
    """
    # Check for spark_goal marker
    goal_marker = request.node.get_closest_marker("spark_goal")
    if goal_marker:
        goal_path = Path(goal_marker.args[0])
        if not goal_path.is_absolute():
            goal_path = request.config.rootpath / goal_path
        return await spark_runner.execute(goal_path=goal_path)

    # Check for spark_prompt marker
    prompt_marker = request.node.get_closest_marker("spark_prompt")
    if prompt_marker:
        return await spark_runner.execute(prompt=prompt_marker.args[0])

    pytest.fail(
        "spark_result fixture requires either "
        "@pytest.mark.spark_goal or @pytest.mark.spark_prompt marker"
    )


class SparkTestRunner:
    """Test-friendly wrapper around the spark_runner orchestrator."""

    def __init__(self, config: SparkConfig) -> None:
        self.config = config

    async def execute(
        self,
        prompt: str | None = None,
        goal_path: Path | None = None,
        credential_profile: str | None = None,
    ) -> RunResult:
        """Execute a task and return the result.

        Args:
            prompt: A task prompt to execute.
            goal_path: A goal file to replay.
            credential_profile: Optional credential profile override.

        Returns:
            A ``RunResult`` with phase outcomes and screenshots.
        """
        from spark_runner.orchestrator import run_single

        task = TaskSpec(
            prompt=prompt,
            goal_path=goal_path,
            credential_profile=credential_profile or self.config.active_credential_profile,
        )
        return await run_single(task, self.config)
