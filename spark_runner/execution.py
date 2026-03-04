"""Phase execution: build_augmented_task and run_phase."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from browser_use import Agent, Browser, ChatBrowserUse
from browser_use.agent.views import AgentHistoryList

from spark_runner.classification import _observation_text
from spark_runner.log import log_event, log_problem
from spark_runner.models import ScreenshotRecord
from spark_runner.storage import phase_name_to_slug
from spark_runner.summarization import extract_phase_history


_PHASE_RULES: str = (
    "- Check for error popup/toast after every action.\n"
    "- Report any deviations from expected behavior.\n"
    "- Report any possible bugs.\n"
    "- Do NOT use workarounds. If an action fails or a feature does not work as expected, report it as a FAILURE — do not try a different mechanism to achieve the same goal.\n"
    "- Complete all steps of YOUR ASSIGNED PHASE. Do NOT continue into subsequent phases."
)


_REPLAY_PREFIX: str = """\
You are RE-EXECUTING a previously run browser automation phase. The summary below
describes what a prior agent did during this phase. Your job is to perform the SAME
task again — use the summary as a guide for WHAT TO DO, not as a report of completed
work. Adapt to the current browser state (element indices and timings may differ).

IMPORTANT:
- Do NOT skip steps because the summary says they succeeded before — you must
  actually perform every action yourself.
- Do NOT invent additional steps beyond what the summary describes (e.g. do not
  fill out forms or generate content unless the summary explicitly includes those
  actions).
- The "Outcome", "Observations", and "Key Facts" sections describe what happened
  last time — treat them as hints, not as your own results.
""" + _PHASE_RULES + """

=== PRIOR RUN SUMMARY (use as instructions) ===
"""


def build_augmented_task(
    original_task: str,
    prior_summaries: list[dict[str, str]],
    restore_fn: Callable[[str], str],
    cross_goal_observations: list[str | dict[str, str]] | None = None,
) -> str:
    """Prepend accumulated context to a phase's task instructions.

    Args:
        original_task: The raw task instructions for this phase.
        prior_summaries: Summaries of all preceding phases in the current run.
        restore_fn: Function to restore placeholders in stored text.
        cross_goal_observations: Optional observations from prior goal runs.

    Returns:
        The augmented task string with context sections prepended.
    """
    context_parts: list[str] = []

    if cross_goal_observations:
        context_parts.append("=== KNOWLEDGE FROM PRIOR SUCCESSFUL GOALS ===")
        for obs in cross_goal_observations:
            context_parts.append(f"- {_observation_text(obs)}")
        context_parts.append("")

    if prior_summaries:
        context_parts.append("=== CONTEXT FROM PRIOR PHASES (current run) ===")
        for s in prior_summaries:
            context_parts.append(f"\n-- Phase: {s['name']} ({s['outcome']}) --")
            context_parts.append(s["summary"])
        context_parts.append("")

    if not context_parts:
        return restore_fn(f"{_PHASE_RULES}\n\n{original_task}")

    context_parts.append("=== YOUR TASK (use the context above to inform your actions) ===\n")
    context_parts.append(_PHASE_RULES + "\n")
    task_text: str = "\n".join(context_parts) + original_task
    return restore_fn(task_text)


def _collect_screenshots(
    result: AgentHistoryList[Any],
    phase_name: str,
    run_dir: Path,
) -> list[ScreenshotRecord]:
    """Copy browser-use step screenshots into the run directory.

    browser-use saves screenshots to a temp directory as ``step_N.png``.
    This function copies them into ``run_dir/screenshots/`` with phase-prefixed
    names so they persist after the temp dir is cleaned up.

    Returns:
        A list of ``ScreenshotRecord`` for each copied screenshot.
    """
    screenshots_dir: Path = run_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    records: list[ScreenshotRecord] = []
    slug: str = phase_name_to_slug(phase_name)
    now_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        source_paths: list[str | None] = result.screenshot_paths()
    except Exception:
        return records

    for step_idx, src in enumerate(source_paths):
        if src is None:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue

        dest_name: str = f"{slug}_step_{step_idx:03d}.png"
        dest_path: Path = screenshots_dir / dest_name
        try:
            shutil.copy2(str(src_path), str(dest_path))
        except OSError:
            continue

        # Check if this step had an error
        error_msg: str | None = None
        if step_idx < len(result.history):
            for r in result.history[step_idx].result:
                if r.error:
                    error_msg = r.error
                    break

        records.append(ScreenshotRecord(
            path=dest_path,
            event_type="error" if error_msg else "step",
            phase_name=phase_name,
            step_number=step_idx,
            error_message=error_msg,
            timestamp=now_str,
        ))

    return records


async def run_phase(
    name: str,
    task: str,
    llm: ChatBrowserUse,
    browser: Browser,
    conversation_log: Path,
    event_log: Path,
    problem_log: Path,
    run_dir: Path,
) -> tuple[bool, AgentHistoryList[Any], list[ScreenshotRecord]]:
    """Run a single phase of the workflow using a browser automation agent.

    Args:
        name: Human-readable phase name.
        task: The full (possibly augmented) task instructions.
        llm: The ``ChatBrowserUse`` LLM instance for the agent.
        browser: The shared ``Browser`` instance.
        conversation_log: Path for the agent's conversation JSON.
        event_log: Path to the event log file.
        problem_log: Path to the problem log file.
        run_dir: Directory for this run's artifacts.

    Returns:
        A tuple of ``(success, result, screenshots)``.
    """
    log_event(event_log, f"PHASE START: {name}")

    assets_dir: Path = Path(__file__).resolve().parent / "assets"
    available_files: list[str] = [
        str(p) for p in assets_dir.iterdir() if p.is_file()
    ]

    agent: Agent[Any, Any] = Agent(
        task=task,
        llm=llm,
        browser=browser,
        save_conversation_path=str(conversation_log),
        max_failures=5,
        max_actions_per_step=5,
        available_file_paths=available_files,
    )

    result: AgentHistoryList[Any] = await agent.run(max_steps=50)
    success: bool = result.is_done() and result.is_successful()

    if not result.is_done() and len(result.history) >= 50:
        log_event(event_log, f"WARNING: Phase '{name}' exhausted step limit (50) without completing")
        log_problem(problem_log, f"STEP LIMIT: Phase '{name}' used all 50 steps without completing — "
                    "the agent may have been stuck searching for a non-existent UI element")

    final: str = result.final_result() or ""
    final_lines: str = "\n".join([f"  FINAL RESULT: {line}" for line in final.split("\n")])

    # Collect step screenshots from browser-use temp dir into run_dir/screenshots/
    phase_screenshots: list[ScreenshotRecord] = _collect_screenshots(result, name, run_dir)
    if phase_screenshots:
        log_event(event_log, f"Saved {len(phase_screenshots)} step screenshot(s) for phase '{name}'")

    if success:
        log_event(event_log, f"PHASE SUCCESS: {name}\n{final_lines}")
    else:
        log_event(event_log, f"PHASE FAILED: {name}\n{final_lines}")
        log_problem(problem_log, f"PHASE FAILED: {name}\n{final_lines}")
        # Take an explicit failure screenshot of the current state
        try:
            failure_name: str = f"failure_{name.replace(' ', '_')}.png"
            screenshot_path: str = str(run_dir / "screenshots" / failure_name)
            (run_dir / "screenshots").mkdir(exist_ok=True)
            page = await browser.get_current_page()
            await page.screenshot(screenshot_path)
            log_event(event_log, f"Failure screenshot saved to {screenshot_path}")
            phase_screenshots.append(ScreenshotRecord(
                path=Path(screenshot_path),
                event_type="phase_end",
                phase_name=name,
                error_message=final or "Phase failed",
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ))
        except Exception as e:
            log_event(event_log, f"Could not save failure screenshot: {e}")
            log_problem(problem_log, f"Could not save failure screenshot for {name}: {e}")

    for h in result.history:
        for r in h.result:
            if r.error:
                log_problem(problem_log, f"ACTION ERROR ({name}): {r.error}")

    history_text: str = extract_phase_history(result)
    history_truncated: str = history_text[:500] + "..." if len(history_text) > 500 else history_text
    history_truncated = "\n".join([f"  PHASE {name} HISTORY: {line}" for line in history_truncated.split("\n")])
    log_event(event_log, f"PHASE HISTORY ({name}):\n{history_truncated}")

    return success, result, phase_screenshots
