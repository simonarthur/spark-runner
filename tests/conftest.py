"""Shared fixtures for sparky_runner tests.

Sets dummy env vars BEFORE importing sparky_runner so its module-level
``load_dotenv()`` / ``anthropic.Anthropic()`` calls don't fail or leak real
credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Set dummy env vars before the first import of sparky_runner ──────────
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-fake-key")
os.environ.setdefault("USER_EMAIL", "test@example.com")
os.environ.setdefault("USER_PASSWORD", "test-password-123")

import sparky_runner  # noqa: E402  (must come after env-var setup)


# ── Helper function (not a fixture) ─────────────────────────────────────

def make_llm_response(text: str) -> MagicMock:
    """Build a mock Anthropic ``Message`` whose ``.content[0].text`` returns *text*."""
    content_block = MagicMock()
    content_block.text = text
    message = MagicMock()
    message.content = [content_block]
    return message


# ── Autouse fixture: isolate module globals every test ───────────────────

@pytest.fixture(autouse=True)
def _patch_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure HOST / USER_EMAIL / USER_PASSWORD are deterministic for every test."""
    monkeypatch.setattr(sparky_runner, "HOST", "https://test.example.com")
    monkeypatch.setattr(sparky_runner, "USER_EMAIL", "test@example.com")
    monkeypatch.setattr(sparky_runner, "USER_PASSWORD", "test-password-123")


# ── Filesystem isolation fixtures ────────────────────────────────────────

@pytest.fixture()
def fake_tasks_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``sparky_runner.TASKS_DIR`` to a temp directory."""
    d = tmp_path / "tasks"
    d.mkdir()
    monkeypatch.setattr(sparky_runner, "TASKS_DIR", d)
    return d


@pytest.fixture()
def fake_goal_summaries_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``sparky_runner.GOAL_SUMMARIES_DIR`` to a temp directory."""
    d = tmp_path / "goal_summaries"
    d.mkdir()
    monkeypatch.setattr(sparky_runner, "GOAL_SUMMARIES_DIR", d)
    return d


@pytest.fixture()
def fake_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``sparky_runner.RUNS_DIR`` to a temp directory."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(sparky_runner, "RUNS_DIR", d)
    return d


# ── Mock LLM client fixture ─────────────────────────────────────────────

@pytest.fixture()
def mock_summary_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``sparky_runner.summary_client`` with a ``MagicMock``."""
    client = MagicMock()
    monkeypatch.setattr(sparky_runner, "summary_client", client)
    return client
