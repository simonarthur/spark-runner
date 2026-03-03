"""Phase execution: build_augmented_task and run_phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from browser_use import Agent, Browser, ChatBrowserUse
from browser_use.agent.views import AgentHistoryList

from sparky_runner.classification import _observation_text
from sparky_runner.log import log_event, log_problem
from sparky_runner.summarization import extract_phase_history


_PHASE_RULES: str = (
    "- Check for error popup/toast after every action.\n"
    "- Report any deviations from expected behavior.\n"
    "- Report any possible bugs.\n"
    "- Do not use workarounds unless absolutely necessary. Note all workarounds as problems.\n"
    "- Complete all steps of a process unless otherwise directed."
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
        return restore_fn(original_task)

    context_parts.append("=== YOUR TASK (use the context above to inform your actions) ===\n")
    task_text: str = "\n".join(context_parts) + original_task
    return restore_fn(task_text)


async def run_phase(
    name: str,
    task: str,
    llm: ChatBrowserUse,
    browser: Browser,
    conversation_log: Path,
    event_log: Path,
    problem_log: Path,
    run_dir: Path,
) -> tuple[bool, AgentHistoryList[Any]]:
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
        A tuple of ``(success, result)``.
    """
    log_event(event_log, f"PHASE START: {name}")

    agent: Agent[Any, Any] = Agent(
        task=task,
        llm=llm,
        browser=browser,
        save_conversation_path=str(conversation_log),
        max_failures=5,
        max_actions_per_step=5,
    )

    result: AgentHistoryList[Any] = await agent.run(max_steps=50)
    success: bool = result.is_done() and result.is_successful()

    if not result.is_done() and len(result.history) >= 50:
        log_event(event_log, f"WARNING: Phase '{name}' exhausted step limit (50) without completing")
        log_problem(problem_log, f"STEP LIMIT: Phase '{name}' used all 50 steps without completing — "
                    "the agent may have been stuck searching for a non-existent UI element")

    final: str = result.final_result() or ""
    final_lines: str = "\n".join([f"  FINAL RESULT: {line}" for line in final.split("\n")])

    if success:
        log_event(event_log, f"PHASE SUCCESS: {name}\n{final_lines}")
    else:
        log_event(event_log, f"PHASE FAILED: {name}\n{final_lines}")
        log_problem(problem_log, f"PHASE FAILED: {name}\n{final_lines}")
        try:
            screenshot_path: str = str(run_dir / f"failure_{name.replace(' ', '_')}.png")
            page = await browser.get_current_page()
            await page.screenshot(screenshot_path)
            log_event(event_log, f"Failure screenshot saved to {screenshot_path}")
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

    return success, result
