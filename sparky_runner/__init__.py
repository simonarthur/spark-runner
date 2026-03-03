"""SparkyAI Browser Automation Runner.

This package provides browser automation testing via the ``browser_use`` library
and Anthropic Claude for LLM planning/summarization.

Backward compatibility: All public symbols from the original monolith are
re-exported here so that ``import sparky_runner`` continues to work.
Module-level globals (HOST, USER_EMAIL, etc.) are preserved for tests that
monkeypatch them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# ── Re-export submodule symbols ──────────────────────────────────────
# These imports make ``from sparky_runner import X`` work for all public names.

from sparky_runner.models import (
    ClassificationRules,
    CredentialProfile,
    ModelConfig,
    PhaseResult,
    RunResult,
    ScreenshotRecord,
    SparkyConfig,
    TaskSpec,
)
from sparky_runner.storage import (
    safe_write_path,
    phase_name_to_slug,
    make_run_dir,
)
from sparky_runner.log import (
    log_event,
    log_problem,
)
from sparky_runner.classification import (
    load_classification_rules,
    _build_rules_prompt_section,
)
from sparky_runner.orchestrator import (
    format_phase_plan,
    format_knowledge_match,
)

# ── Module-level side effects (deferred until first import) ──────────

load_dotenv()

# ── Backward-compatible module globals ───────────────────────────────
# Tests monkeypatch these directly (e.g. ``monkeypatch.setattr(sparky_runner, "HOST", ...)``).
# The wrapper functions below read them at call time.

import anthropic

USER_EMAIL: str = os.environ.get("USER_EMAIL", "")
USER_PASSWORD: str = os.environ.get("USER_PASSWORD", "")
summary_client: anthropic.Anthropic = anthropic.Anthropic()
DEFAULT_HOST: str = "https://sparky-web-dev.vercel.app"
HOST: str = ""

# Directories — created lazily; tests redirect these via monkeypatch
TASKS_DIR: Path = Path("tasks")
TASKS_DIR.mkdir(exist_ok=True)
GOAL_SUMMARIES_DIR: Path = Path("goal_summaries")
GOAL_SUMMARIES_DIR.mkdir(exist_ok=True)
RUNS_DIR: Path = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

CLASSIFICATION_RULES_PATH: Path = Path("classification_rules.txt")
_classification_rules: ClassificationRules = load_classification_rules(CLASSIFICATION_RULES_PATH)


# ── Constants re-exported from submodules ─────────────────────────────

from sparky_runner.execution import _PHASE_RULES, _REPLAY_PREFIX  # noqa: E402


# ── Backward-compatible wrapper functions ─────────────────────────────
# Each wrapper reads the current module globals at call time, then delegates
# to the real implementation in a submodule.  This ensures that when tests
# monkeypatch globals (e.g. ``sparky_runner.HOST``), the patched value flows
# through to the implementation.

from sparky_runner import placeholders as _ph  # noqa: E402
from sparky_runner import knowledge as _kn  # noqa: E402
from sparky_runner import decomposition as _dc  # noqa: E402
from sparky_runner import summarization as _sm  # noqa: E402
from sparky_runner import classification as _cl  # noqa: E402
from sparky_runner import observations as _ob  # noqa: E402
from sparky_runner import goals as _gl  # noqa: E402
from sparky_runner import execution as _ex  # noqa: E402
from sparky_runner import storage as _st  # noqa: E402


def _host_to_placeholder(text: str) -> str:
    """Replace the active HOST URL with {BASE_URL}, normalizing trailing slashes."""
    return _ph.host_to_placeholder(text, HOST)


def _placeholder_to_host(text: str) -> str:
    """Replace {BASE_URL} with the active HOST URL, normalizing trailing slashes."""
    return _ph.placeholder_to_host(text, HOST)


def _credentials_to_placeholders(text: str) -> str:
    """Replace literal credentials with {USER_EMAIL}/{USER_PASSWORD} placeholders."""
    return _ph.credentials_to_placeholders(text, USER_EMAIL, USER_PASSWORD)


def _placeholders_to_credentials(text: str) -> str:
    """Replace {USER_EMAIL}/{USER_PASSWORD} placeholders with literal credentials."""
    return _ph.placeholders_to_credentials(text, USER_EMAIL, USER_PASSWORD)


def _sanitize_for_storage(text: str) -> str:
    """Replace host URL and credentials with placeholders for safe storage."""
    return _ph.sanitize_for_storage(text, HOST, USER_EMAIL, USER_PASSWORD)


def _restore_from_storage(text: str) -> str:
    """Restore host URL and credentials from placeholders."""
    return _ph.restore_from_storage(text, HOST, USER_EMAIL, USER_PASSWORD)


def _observation_text(obs: str | dict[str, str]) -> str:
    """Extract text from an observation."""
    return _cl._observation_text(obs)


def load_knowledge_index() -> list[dict[str, Any]]:
    """Read all goal summaries and their referenced task files into a knowledge index."""
    return _kn.load_knowledge_index(GOAL_SUMMARIES_DIR, TASKS_DIR, _restore_from_storage)


def find_relevant_knowledge(
    prompt: str,
    knowledge_index: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use an LLM to find reusable subtasks and observations from prior goals."""
    return _kn.find_relevant_knowledge(prompt, knowledge_index, summary_client)


def generate_task_name(prompt: str) -> str:
    """Use an LLM to generate a short, filename-safe task name."""
    return _dc.generate_task_name(prompt, summary_client)


def decompose_task(
    prompt: str,
    knowledge_match: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to decompose a free-form task into ordered execution phases."""
    return _dc.decompose_task(
        prompt, HOST, USER_EMAIL, USER_PASSWORD,
        TASKS_DIR, summary_client, _restore_from_storage,
        knowledge_match=knowledge_match,
    )


def extract_phase_history(result: Any) -> str:
    """Extract a structured text log from the agent's step-by-step history."""
    return _sm.extract_phase_history(result)


def summarize_phase(
    phase_name: str,
    phase_task: str,
    result: Any,
    success: bool,
) -> str:
    """Use an LLM to generate a structured summary of a completed phase."""
    return _sm.summarize_phase(phase_name, phase_task, result, success, summary_client)


def generate_task_report(
    task_name: str,
    prompt: str,
    summaries: list[dict[str, str]],
) -> dict[str, Any]:
    """Use an LLM to generate the final task report."""
    return _sm.generate_task_report(task_name, prompt, summaries, summary_client)


def classify_observations(
    prompt: str,
    observations: list[str | dict[str, str]],
    rules: ClassificationRules | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to classify each observation as an error or a warning."""
    effective_rules = rules if rules is not None else _classification_rules
    return _cl.classify_observations(prompt, observations, summary_client, rules=effective_rules)


def merge_observations(
    existing: list[str | dict[str, str]],
    new: list[str | dict[str, str]],
) -> list[dict[str, str]]:
    """Use an LLM to merge and de-duplicate two lists of observations."""
    return _ob.merge_observations(existing, new, summary_client)


def _extract_and_log_observations(
    summary: str, phase_name: str, problem_log: Path
) -> None:
    """Extract observations and sub-phase failures from a phase summary and log them."""
    return _ob._extract_and_log_observations(summary, phase_name, problem_log)


def build_augmented_task(
    original_task: str,
    prior_summaries: list[dict[str, str]],
    cross_goal_observations: list[str | dict[str, str]] | None = None,
) -> str:
    """Prepend accumulated context to a phase's task instructions."""
    return _ex.build_augmented_task(
        original_task, prior_summaries, _restore_from_storage,
        cross_goal_observations=cross_goal_observations,
    )


async def run_phase(
    name: str,
    task: str,
    llm: Any,
    browser: Any,
    conversation_log: Path,
    event_log: Path,
    problem_log: Path,
    run_dir: Path,
) -> tuple[bool, Any]:
    """Run a single phase of the workflow using a browser automation agent."""
    return await _ex.run_phase(
        name, task, llm, browser,
        conversation_log, event_log, problem_log, run_dir,
    )


def load_goal_summary(goal_path: Path) -> tuple[str, str, list[dict[str, str]]]:
    """Load a goal summary JSON and reconstruct the prompt, task name, and phases."""
    return _gl.load_goal_summary(goal_path, TASKS_DIR, _restore_from_storage)


def list_goals() -> None:
    """Print a summary of all existing goals."""
    return _gl.list_goals(GOAL_SUMMARIES_DIR, _restore_from_storage)


def classify_existing_goals() -> None:
    """Retroactively classify observations in all existing goal summaries."""
    return _gl.classify_existing_goals(
        GOAL_SUMMARIES_DIR,
        lambda prompt, obs: classify_observations(prompt, obs),
    )


def _get_orphan_tasks() -> list[str]:
    """Return sorted list of task filenames not referenced by any goal summary."""
    return _st.get_orphan_tasks(TASKS_DIR, GOAL_SUMMARIES_DIR)


def find_orphan_tasks() -> None:
    """Print task files not referenced by any goal summary."""
    return _st.find_orphan_tasks(TASKS_DIR, GOAL_SUMMARIES_DIR)


def clean_orphan_tasks() -> None:
    """Delete task files not referenced by any goal summary."""
    return _st.clean_orphan_tasks(TASKS_DIR, GOAL_SUMMARIES_DIR)


# ── USAGE constant (backward compat) ──────────────────────────────────

USAGE: str = """\
usage: sparky_runner.py [options] [goal_summary.json]

SparkyAI Browser Automation Runner

positional arguments:
  goal_summary.json       Replay a previously saved goal summary file

options:
  -h, --help              Show this help message and exit
  -p, --prompt PROMPT     Task prompt (skips interactive input)
  -u, --url URL           Base URL
  --no-update-summary     Don't update the goal summary file on replay
  --no-update-tasks       Don't overwrite existing task files on replay
  --no-knowledge-reuse    Don't reuse knowledge from prior goals
  --auto-close            Close the browser automatically on finish
  --list-goals            List all existing goals and exit
  --classify-existing     Classify observations in all existing goal summaries
  --find-orphans          List task files not referenced by any goal summary
  --clean-orphans         Delete task files not referenced by any goal summary
"""
