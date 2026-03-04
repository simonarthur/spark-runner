"""Phase summarization and task report generation."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from browser_use.agent.views import AgentHistoryList

from spark_runner.models import ModelConfig


def extract_phase_history(result: AgentHistoryList[Any]) -> str:
    """Extract a structured text log from the agent's step-by-step history.

    Args:
        result: The ``AgentHistoryList`` returned by ``Agent.run()``.

    Returns:
        A multi-line string with one section per step.
    """
    lines: list[str] = []
    for i, h in enumerate(result.history, 1):
        lines.append(f"--- Step {i} ---")
        if h.model_output:
            mo: Any = h.model_output
            if mo.evaluation_previous_goal:
                lines.append(f"  Eval: {mo.evaluation_previous_goal}")
            if mo.memory:
                lines.append(f"  Memory: {mo.memory}")
            if mo.next_goal:
                lines.append(f"  Next goal: {mo.next_goal}")
            for action in mo.action:
                lines.append(f"  Action: {action.model_dump(exclude_none=True)}")
        for r in h.result:
            if r.error:
                lines.append(f"  ERROR: {r.error}")
            if r.extracted_content:
                lines.append(f"  Extracted: {r.extracted_content}")
        if h.state and h.state.url:
            lines.append(f"  URL: {h.state.url}")
    return "\n".join(lines)


def summarize_phase(
    phase_name: str,
    phase_task: str,
    result: AgentHistoryList[Any],
    success: bool,
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> str:
    """Use an LLM to generate a structured summary of a completed phase.

    Args:
        phase_name: Human-readable name of the phase.
        phase_task: The original task instructions given to the agent.
        result: The ``AgentHistoryList`` from the agent's execution.
        success: Whether the phase completed successfully.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A structured text summary.
    """
    if model_config is None:
        model_config = ModelConfig(max_tokens=2048)

    history_text: str = extract_phase_history(result)
    final: str = result.final_result() or "(no final result)"
    errors: list[str] = [e for e in result.errors() if e]

    prompt: str = f"""You are analyzing the results of an automated browser testing phase.

Phase: {phase_name}
Outcome: {"SUCCESS" if success else "FAILED"}
Final result: {final}
{"Errors encountered: " + "; ".join(errors) if errors else "No errors."}

Original task instructions:
{phase_task}

Step-by-step agent history:
{history_text}

Produce a structured summary with these sections:
1. **Outcome**: One-line success/failure statement.
2. **Actions Taken**: List every distinct action in order (navigate, click, type, wait, etc.) with specifics.
3. **Sub-phases**: Break the phase into logical sub-phases (e.g. "Navigate to page", "Enter credentials", "Click submit", "Verify result"). For each sub-phase state what happened and whether it succeeded.
4. **Page state**: Describe the current state of the page (URL, what's visible, any modals/toasts).
5. **Observations**: Any unexpected behavior, warnings, retries, or deviations from the instructions. Wrap the content of this section in <OBSERVATIONS>...</OBSERVATIONS> tags.
6. **Key facts learned**: Concrete details about the UI (element names, selectors that worked, layout info) that would help a follow-up agent.
7. **Timing**: Approximate time taken or number of steps.

Be concise but specific. Use the actual element names and URLs from the history."""

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_task_report(
    task_name: str,
    prompt: str,
    summaries: list[dict[str, str]],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Use an LLM to generate the final task report.

    Args:
        task_name: The short identifier for this task run.
        prompt: The user's original task description.
        summaries: List of phase summary dicts.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A dict with ``main_task``, ``key_observations``, and ``subtasks``.
    """
    if model_config is None:
        model_config = ModelConfig()

    summaries_text: str = "\n\n".join(
        f"Phase: {s['name']} [{s['outcome']}]\n{s['summary']}"
        for s in summaries
    )

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""You are writing a task report for a browser automation workflow.

Task name: {task_name}
Original prompt: {prompt}

Phase execution summaries:
{summaries_text}

Return ONLY valid JSON with exactly these fields:
{{
  "main_task": "One-line description of the overall task",
  "key_observations": ["observation 1", "observation 2", ...]
}}

key_observations should be a list of strings — anything an LLM agent should know for future runs (UI quirks, timing, error patterns).
Do NOT include a subtask breakdown — that will be added separately."""}],
    )
    text: str = response.content[0].text.strip()
    match: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    report: dict[str, Any] = json.loads(text)

    report["subtasks"] = [
        {"subtask": i, "filename": s["filename"]}
        for i, s in enumerate(summaries, 1)
    ]

    return report
