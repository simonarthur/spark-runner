"""Task decomposition and naming via LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import anthropic

from sparky_runner.models import ModelConfig


def generate_task_name(
    prompt: str,
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> str:
    """Use an LLM to generate a short, filename-safe task name from the prompt.

    Args:
        prompt: The user's full task description.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A short hyphenated name like ``"login-test"``.
    """
    if model_config is None:
        model_config = ModelConfig(max_tokens=64)

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": (
            "Generate a short (2-8 word) descriptive name for this browser automation task. "
            "The name will be used as a filename, so use only lowercase letters, numbers, and hyphens. "
            "No spaces, no underscores, no special characters. Examples: 'write-tea-blog', 'login-test', 'scrape-products'.\n\n"
            f"Task: {prompt}\n\n"
            "Reply with ONLY the name, nothing else."
        )}],
    )
    name: str = response.content[0].text.strip().strip('"').strip("'")
    name = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    if len(name) > 80:
        name = name[:80].rstrip("-")
    return name or "unnamed-task"


def decompose_task(
    prompt: str,
    host: str,
    user_email: str,
    user_password: str,
    tasks_dir: Path,
    client: anthropic.Anthropic,
    restore_fn: Callable[[str], str],
    model_config: ModelConfig | None = None,
    knowledge_match: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to decompose a free-form task into ordered execution phases.

    Args:
        prompt: The user's task description.
        host: Base URL of the application under test.
        user_email: Login email credential.
        user_password: Login password credential.
        tasks_dir: Directory containing subtask files for reuse.
        client: Anthropic client for LLM calls.
        restore_fn: Function to restore placeholders in stored text.
        model_config: Model configuration.
        knowledge_match: Optional knowledge match result with reusable subtasks.

    Returns:
        A list of phase dicts with ``name`` and ``task`` keys.
    """
    if model_config is None:
        model_config = ModelConfig()

    reuse_section: str = ""
    if knowledge_match and knowledge_match.get("reusable_subtasks"):
        reuse_parts: list[str] = ["\n\nREUSABLE SUBTASKS FROM PRIOR RUNS:"]
        reuse_parts.append("The following subtasks have been successfully executed before and can be reused.")
        reuse_parts.append('For any phase that is fully covered by a reusable subtask, return "reuse": "filename.txt" INSTEAD of "task".')
        reuse_parts.append('Only reuse a subtask if it covers the ENTIRE phase — do not partially reuse.\n')
        for st in knowledge_match["reusable_subtasks"]:
            subtask_path: Path = tasks_dir / st["filename"]
            if subtask_path.exists():
                content: str = restore_fn(subtask_path.read_text())
                reuse_parts.append(f'  Filename: {st["filename"]}')
                reuse_parts.append(f'  Phase name: {st["phase_name"]}')
                reuse_parts.append(f'  Reason: {st["reason"]}')
                reuse_parts.append(f"  Content summary:\n    {content[:500]}...")
                reuse_parts.append("")
        reuse_section = "\n".join(reuse_parts)

    observations_section: str = ""
    if knowledge_match and knowledge_match.get("relevant_observations"):
        obs_parts: list[str] = [
            "\n\nOBSERVATIONS FROM PRIOR RUNS (these describe actual UI behavior and structure "
            "observed during previous executions — use them to write accurate instructions):"
        ]
        for obs in knowledge_match["relevant_observations"]:
            obs_parts.append(f"  - {obs}")
        observations_section = "\n".join(obs_parts)

    reuse_format: str = ""
    if reuse_section:
        reuse_format = '\nFor reused phases, use: {"name": "Phase Name", "reuse": "filename.txt"}'

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""You are a browser automation planner for SparkyAI ({host}).

Given a user's task description, decompose it into sequential phases that a browser automation agent will execute one at a time. Each phase shares the same browser session, so the next phase picks up where the previous one left off.

IMPORTANT RULES:
- The FIRST phase must ALWAYS be "Login" — navigate to {host} and log in with:
    Email: {user_email}
    Password: {user_password}
- Each phase should be a self-contained step with clear success criteria.
- Write the task instructions as explicit, step-by-step directions for an AI agent controlling a browser.
- Every phase must start with ALL of these paragraphs (copy them verbatim):
    "IMPORTANT: Before doing anything else, check whether this phase's goal has ALREADY been achieved by a prior phase (e.g. the expected UI state is already visible). If so, report success immediately without taking any actions."
    "Check for error popup/toast after every action. Report any deviations from expected behavior."
    "If you cannot find an expected UI element after reasonable exploration (scrolling, searching, checking alternate locations), do NOT keep retrying the same approaches. Report what you found, note the missing element, and mark the phase as done."
- Every phase's FINAL instruction (last numbered step) MUST be a STOP condition in this exact format:
    "STOP: Report success once <expected state>. Do NOT click any buttons, fill any fields, or proceed further — the next phase will handle that."
  Additionally, repeat the stop condition BEFORE the last action step so the agent sees it twice. For example, if the last action is "Click Next", the step before it should say "After clicking Next, STOP as soon as the new page is visible."
- For form fields, specify exact values to enter.
- For dropdowns/selectors that are custom (like Tone), instruct the agent to type to filter/search rather than scroll.
- For generation/loading steps, instruct the agent to poll every 5 seconds (NOT one long wait) up to a timeout.
- For navigation, instruct the agent to verify the page loaded by checking for expected elements.
- Do NOT assume specific UI element names, field labels, or page layouts you haven't seen.
  Describe the GOAL of each step and let the agent discover the actual UI.
    WRONG: "Click the 'Platform Selection' checkbox for Facebook, LinkedIn, and Google Ads"
    RIGHT: "If there is an option to select target platforms, select Facebook, LinkedIn, and Google Ads.
            If no platform selection is available, note this and proceed — the tool may auto-select platforms."
  Use phrases like "if available", "look for", "if present" for UI elements beyond login.
{reuse_section}{observations_section}

Return ONLY valid JSON — an array of objects with "name" and "task" keys.{reuse_format} Example:
[
  {{"name": "Login", "task": "Check for error popup/toast after every action. ..."}},
  {{"name": "Navigate to feature", "task": "Check for error popup/toast after every action. ..."}}
]
BASE_URL is {host}
User's task:
{prompt}"""}],
    )
    text: str = response.content[0].text.strip()
    match: re.Match[str] | None = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    phases: list[dict[str, str]] = json.loads(text)

    base_instructions: str = """
                    "- Log in if necessary.\n"
                    - Check for error popup/toast after every action.
                    - Report any deviations from expected behavior.
                    - Report any possible bugs.
                    - Report any unexpected failures.
                    - Complete all steps of YOUR ASSIGNED PHASE. Do NOT continue into subsequent phases.
    """
    for phase in phases:
        if "reuse" in phase:
            reuse_file: Path = tasks_dir / phase["reuse"]
            if reuse_file.exists():
                task_content: str = restore_fn(reuse_file.read_text())
                phase["task"] = (
                    "=== INSTRUCTIONS FROM PREVIOUSLY SUCCESSFUL EXECUTION ===\n"
                    f"{task_content}\n\n"
                    "=== YOUR TASK ===\n"
                    f"BASE_URL is {host}"
                    "- Repeat the procedure described above. Follow the same steps.\n"
                    "- Your goal is to TEST the given task.\n"
                    "- Repeat problems/workarounds noted in the procedure above in order to determine if they have been fixed.\n"
                    f"{base_instructions}\n"
                )
                print(f"  Reusing subtask '{phase['reuse']}' for phase '{phase['name']}'")
            else:
                print(f"  Warning: reuse file '{phase['reuse']}' not found, phase will need fresh instructions")
                phase["task"] = (
                    f"{base_instructions}\n"
                    "- Avoid workarounds. Your goal is to TEST the given task\n"
                    f"- Execute the '{phase['name']}' phase of the task."
                )
            del phase["reuse"]

    return phases
