"""
SparkyAI Browser Automation Runner

Orchestrates multi-phase browser automation workflows against a SparkyAI web
application. The runner decomposes a free-form user prompt into sequential
phases (e.g. Login -> Navigate -> Fill Form -> Verify), executes each phase
via a browser-controlling AI agent, and accumulates structured summaries that
provide context to subsequent phases.

Key capabilities:
  - Knowledge reuse: loads summaries from prior goal runs and uses an LLM to
    identify reusable subtasks, avoiding redundant work across runs.
  - Phase decomposition: an LLM planner breaks down the user's prompt into
    ordered, self-contained phases with explicit browser instructions.
  - Contextual augmentation: each phase receives summaries of all preceding
    phases plus cross-goal observations so the agent can adapt to the current
    browser state.
  - Reporting: after execution, generates a JSON task report with observations
    and subtask references for future knowledge reuse.

Usage:
  python sparky_runner.py                        # interactive prompt
  python sparky_runner.py -p "do something"      # inline prompt
  python sparky_runner.py goal_summaries/x.json   # replay a saved goal
  python sparky_runner.py --help                  # full usage info
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
import anthropic
from browser_use import Agent, Browser, ChatBrowserUse
from browser_use.agent.views import AgentHistoryList

# Load environment variables (expects ANTHROPIC_API_KEY at minimum)
load_dotenv()

# Credentials for the SparkyAI application under test, loaded from .env
USER_EMAIL: str = os.environ.get("USER_EMAIL", "")
USER_PASSWORD: str = os.environ.get("USER_PASSWORD", "")

# Anthropic client used for all LLM summarization / planning calls
# (separate from the browser agent's own LLM instance)
summary_client: anthropic.Anthropic = anthropic.Anthropic()

# Default URL for the SparkyAI web application under test
DEFAULT_HOST: str = "https://sparky-web-dev.vercel.app"

# Active host URL — set at runtime via CLI args, defaults to DEFAULT_HOST
HOST: str = ""

def _host_to_placeholder(text: str) -> str:
    """Replace the active HOST URL with {BASE_URL}, normalizing trailing slashes."""
    return text.replace(HOST.rstrip("/"), "{BASE_URL}")


def _placeholder_to_host(text: str) -> str:
    """Replace {BASE_URL} with the active HOST URL, normalizing trailing slashes."""
    return text.replace("{BASE_URL}", HOST.rstrip("/"))


def _credentials_to_placeholders(text: str) -> str:
    """Replace literal credentials with {USER_EMAIL}/{USER_PASSWORD} placeholders."""
    if USER_EMAIL:
        text = text.replace(USER_EMAIL, "{USER_EMAIL}")
    if USER_PASSWORD:
        text = text.replace(USER_PASSWORD, "{USER_PASSWORD}")
    return text


def _placeholders_to_credentials(text: str) -> str:
    """Replace {USER_EMAIL}/{USER_PASSWORD} placeholders with literal credentials."""
    if USER_EMAIL:
        text = text.replace("{USER_EMAIL}", USER_EMAIL)
    if USER_PASSWORD:
        text = text.replace("{USER_PASSWORD}", USER_PASSWORD)
    return text


def _sanitize_for_storage(text: str) -> str:
    """Replace host URL and credentials with placeholders for safe storage."""
    return _credentials_to_placeholders(_host_to_placeholder(text))


def _restore_from_storage(text: str) -> str:
    """Restore host URL and credentials from placeholders."""
    return _placeholders_to_credentials(_placeholder_to_host(text))


# Directory where individual phase subtask summaries are stored as .txt files.
# Each file contains the LLM-generated summary of what happened during that phase.
TASKS_DIR: Path = Path("tasks")
TASKS_DIR.mkdir(exist_ok=True)

# Directory where high-level goal summary JSON files are stored.
# Each file aggregates observations and subtask references for one complete run.
GOAL_SUMMARIES_DIR: Path = Path("goal_summaries")
GOAL_SUMMARIES_DIR.mkdir(exist_ok=True)

# Directory where per-run artifacts (event log, conversation log, etc.) are stored.
# Each run gets its own subdirectory: runs/{task_name}/{YYYY-MM-DD_HH-MM-SS}/
RUNS_DIR: Path = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# ── Configurable classification rules ────────────────────────────────────


@dataclass
class ClassificationRules:
    """Prioritized rules that guide LLM observation classification."""

    error_rules: list[str] = field(default_factory=list)
    warning_rules: list[str] = field(default_factory=list)


CLASSIFICATION_RULES_PATH: Path = Path("classification_rules.txt")


def load_classification_rules(path: Path = CLASSIFICATION_RULES_PATH) -> ClassificationRules:
    """Parse a classification rules file into a ``ClassificationRules`` object.

    The file uses ``[ERRORS]`` and ``[WARNINGS]`` section headers (case-insensitive).
    Blank lines and lines starting with ``#`` are ignored.  Lines that appear
    before any section header are silently skipped.

    Returns an empty ``ClassificationRules`` if the file is missing or unreadable.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ClassificationRules()

    rules = ClassificationRules()
    current_section: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower == "[errors]":
            current_section = rules.error_rules
        elif lower == "[warnings]":
            current_section = rules.warning_rules
        elif current_section is not None:
            current_section.append(line)

    return rules


# Load rules once at import time; tests can monkeypatch ``_classification_rules``.
_classification_rules: ClassificationRules = load_classification_rules()


def load_knowledge_index() -> list[dict[str, Any]]:
    """Read all goal summaries and their referenced task files into a knowledge index.

    Scans the GOAL_SUMMARIES_DIR for files matching the pattern ``*-task.json``,
    parses each one, and loads the full text content of every referenced subtask
    file from the TASKS_DIR. This builds a comprehensive in-memory index that
    the knowledge-matching LLM can search through.

    Returns:
        A list of dicts, each representing one prior goal with keys:
          - ``goal_file`` (str): filename of the goal summary JSON
          - ``main_task`` (str): one-line description of the goal
          - ``key_observations`` (list[str | dict]): learned facts from that run
          - ``subtasks`` (list[dict]): each with ``filename``, ``name``, and
            ``content`` (the full text of the subtask summary file)
    """
    index: list[dict[str, Any]] = []

    for goal_file in sorted(GOAL_SUMMARIES_DIR.glob("*-task.json")):
        # Attempt to parse the goal summary JSON; skip malformed files gracefully
        try:
            data: dict[str, Any] = json.loads(_restore_from_storage(goal_file.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: skipping malformed goal file {goal_file.name}: {e}")
            continue

        # Each entry represents one prior goal run:
        #   "goal_file": str — filename of the goal summary JSON (e.g. "login-test-task.json")
        #   "main_task": str — one-line description of what the goal accomplished
        #   "key_observations": list[str] — learned facts/insights from that run
        #   "subtasks": list[dict] — phase summaries loaded from referenced files
        entry: dict[str, Any] = {
            "goal_file": goal_file.name,
            "main_task": data.get("main_task", ""),
            "key_observations": data.get("key_observations", []),
            "subtasks": [],
        }

        # Load each referenced subtask file's content into the index entry
        for subtask_entry in data.get("subtasks", []):
            filename: str = subtask_entry.get("filename", "")
            subtask_path: Path = TASKS_DIR / filename
            if not subtask_path.exists():
                print(f"  Warning: subtask file not found {subtask_path}, skipping")
                continue
            try:
                content: str = _restore_from_storage(subtask_path.read_text())
            except OSError as e:
                print(f"  Warning: could not read {subtask_path}: {e}")
                continue
            # Derive a human-readable phase name from the filename
            # e.g. "login.txt" -> "Login", "fill-form.txt" -> "Fill Form"
            name: str = subtask_path.stem.replace("-", " ").title()
            # Each subtask entry within a goal:
            #   "filename": str — name of the subtask .txt file in TASKS_DIR
            #   "name": str — human-readable phase name derived from filename
            #   "content": str — full text of the subtask summary file
            entry["subtasks"].append({
                "filename": filename,
                "name": name,
                "content": content,
            })

        index.append(entry)
    return index


def find_relevant_knowledge(
    prompt: str,
    knowledge_index: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use an LLM to find reusable subtasks and observations from prior goals.

    Sends the full knowledge index (all prior goals, observations, and subtask
    contents) along with the new task prompt to an LLM. The LLM analyzes each
    existing subtask's content to determine which can be directly reused and
    which observations are relevant to the new task.

    Args:
        prompt: The user's new task description.
        knowledge_index: The full knowledge index as returned by
            ``load_knowledge_index()``.

    Returns:
        A dict with keys:
          - ``reusable_subtasks`` (list[dict]): subtasks that can be reused,
            each with ``filename``, ``phase_name``, and ``reason``.
          - ``relevant_observations`` (list[str]): observations from prior runs
            that apply to the new task.
          - ``coverage_notes`` (str): what the new task needs that isn't covered
            by existing knowledge.
    """
    if not knowledge_index:
        # Empty knowledge match result:
        #   "reusable_subtasks": list[dict] — subtasks that can be reused (empty here)
        #   "relevant_observations": list[str] — applicable observations (empty here)
        #   "coverage_notes": str — gaps not covered by existing knowledge (empty here)
        return {"reusable_subtasks": [], "relevant_observations": [], "coverage_notes": ""}

    # Log all goals and subtasks being searched
    total_subtasks: int = sum(len(g["subtasks"]) for g in knowledge_index)
    total_observations: int = sum(len(g["key_observations"]) for g in knowledge_index)
    print(f"  Searching {len(knowledge_index)} goal(s) with {total_subtasks} subtask(s) and {total_observations} observation(s)")
    for i, goal in enumerate(knowledge_index, 1):
        subtask_names: str = ", ".join(st["name"] for st in goal["subtasks"]) or "(none)"
        print(f"    Goal {i}: {goal['goal_file']}")
        print(f"      Task: {goal['main_task']}")
        print(f"      Subtasks: {subtask_names}")
        print(f"      Observations: {len(goal['key_observations'])}")

    # Build a compact text representation of all known goals for the LLM prompt.
    # Includes full subtask content so the LLM can verify relevance by content,
    # not just by name.
    index_text_parts: list[str] = []
    for i, goal in enumerate(knowledge_index):
        parts: list[str] = [f"Goal {i + 1}: {goal['goal_file']}"]
        parts.append(f"  Main task: {goal['main_task']}")
        if goal["key_observations"]:
            parts.append("  Observations:")
            for obs in goal["key_observations"]:
                parts.append(f"    - {_observation_text(obs)}")
        if goal["subtasks"]:
            parts.append("  Subtasks:")
            for st in goal["subtasks"]:
                parts.append(f"    [{st['filename']}] {st['name']}:")
                # Include full content so the LLM can verify relevance
                parts.append(f"      {st['content']}")
        index_text_parts.append("\n".join(parts))

    index_text: str = "\n\n".join(index_text_parts)

    print(f"  Sending {len(index_text)} chars to LLM for knowledge matching...")

    # Ask the LLM to match the new task against the knowledge index
    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"""You are analyzing a knowledge base of prior browser automation runs to find reusable components for a new task.

New task prompt:
{prompt}

Existing goals and their subtasks:
{index_text}

Analyze each existing subtask's CONTENT (not just its name) and determine:
1. Which subtasks can be directly reused for the new task (the steps described in the content must actually match what the new task needs)
2. Which observations from prior goals are relevant to the new task
3. What the new task needs that isn't covered by existing knowledge

Return ONLY valid JSON:
{{
    "reusable_subtasks": [
        {{"filename": "exact-filename.txt", "phase_name": "Phase Name", "reason": "Why this subtask is reusable"}}
    ],
    "relevant_observations": ["observation string 1", "observation string 2"],
    "coverage_notes": "What the new task needs that isn't covered by existing subtasks"
}}

IMPORTANT:
- Only include subtasks whose content genuinely matches what the new task needs — don't match on name alone.
- For observations, include any that would help the new task succeed (UI patterns, timing, form quirks, etc).
- If nothing is reusable, return empty arrays."""}],
    )

    # Extract JSON from the LLM response (may be wrapped in markdown fences)
    text: str = response.content[0].text.strip()
    try:
        match: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        result: dict[str, Any] = json.loads(text)
        # Ensure all expected keys are present with sensible defaults
        result.setdefault("reusable_subtasks", [])
        result.setdefault("relevant_observations", [])
        result.setdefault("coverage_notes", "")

        # Log detailed results of the knowledge match
        print(f"  Knowledge matching complete:")
        if result["reusable_subtasks"]:
            print(f"  Reusable subtasks ({len(result['reusable_subtasks'])}):")
            for st in result["reusable_subtasks"]:
                print(f"    - [{st.get('filename', '?')}] {st.get('phase_name', '?')}")
                print(f"      Reason: {st.get('reason', '(none)')}")
        else:
            print(f"  Reusable subtasks: none found")
        if result["relevant_observations"]:
            print(f"  Relevant observations ({len(result['relevant_observations'])}):")
            for obs in result["relevant_observations"]:
                print(f"    - {obs}")
        else:
            print(f"  Relevant observations: none found")
        if result["coverage_notes"]:
            print(f"  Coverage gaps: {result['coverage_notes']}")

        return result
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  Warning: could not parse knowledge match response: {e}")
        # Fallback empty knowledge match (same structure as successful result)
        return {"reusable_subtasks": [], "relevant_observations": [], "coverage_notes": ""}


def safe_write_path(path: Path) -> Path:
    """Return a non-conflicting file path, appending -2, -3, etc. if needed.

    Prevents accidental overwrites by checking if the target path already
    exists and, if so, appending an incrementing numeric suffix to the stem.

    Args:
        path: The desired file path.

    Returns:
        The original path if it doesn't exist, otherwise a suffixed variant
        like ``stem-2.ext``, ``stem-3.ext``, etc.
    """
    if not path.exists():
        return path
    stem: str = path.stem
    suffix: str = path.suffix
    parent: Path = path.parent
    i: int = 2
    while True:
        candidate: Path = parent / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def log_event(event_log: Path, msg: str) -> None:
    """Append a timestamped message to the event log file and print it.

    Each log entry is prefixed with an ISO-style timestamp in brackets.
    Messages are both printed to stdout and appended to the log file so
    there is a persistent record of the workflow execution.

    Args:
        event_log: Path to the event log text file.
        msg: The message to log.
    """
    ts: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line: str = f"[{ts}] {msg}"
    print(line)
    with open(event_log, "a") as f:
        f.write(line + "\n")


def log_problem(problem_log: Path, msg: str) -> None:
    """Append a timestamped problem entry to the problem log file.

    Unlike ``log_event``, this does **not** print to stdout — the
    corresponding ``log_event`` call already handles console output.
    The file is created lazily on first write so clean runs produce no
    problem log file.

    Args:
        problem_log: Path to the problem log text file.
        msg: The problem description to log.
    """
    ts: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line: str = f"[{ts}] {msg}"
    with open(problem_log, "a") as f:
        f.write(line + "\n")


def _extract_and_log_observations(
    summary: str, phase_name: str, problem_log: Path
) -> None:
    """Extract observations and sub-phase failures from a phase summary and log them.

    Performs two passes over the LLM-generated phase summary:

    1. **Observations**: Extracts ``<OBSERVATIONS>...</OBSERVATIONS>`` content.
    2. **Sub-phase failures**: Parses ``### Sub-phase N: ...`` sections and logs
       any with a non-SUCCESS status as ``ERROR`` entries in the problem log,
       with the observations block appended for full context.

    If there are no sub-phase failures but observations exist, the observations
    are logged as a standalone ``OBSERVATIONS`` entry.

    Args:
        summary: The full structured summary text returned by
            ``summarize_phase()``.
        phase_name: Human-readable phase name for the log entry prefix.
        problem_log: Path to the problem log file.
    """
    # --- Extract <OBSERVATIONS> block first ---
    observations: str = ""
    obs_match: re.Match[str] | None = re.search(
        r"<OBSERVATIONS>(.*?)</OBSERVATIONS>",
        summary,
        re.DOTALL | re.IGNORECASE,
    )
    if obs_match:
        observations = obs_match.group(1).strip()
        # Treat trivial content as empty
        if observations.lower() in ("none", "n/a", "none.", "n/a."):
            observations = ""

    # --- Detect sub-phase failures ---
    # The LLM uses varying heading formats for sub-phases, e.g.:
    #   ### Sub-phase 2: Search Bar Interaction
    #   ### Sub-phase A: Initial Search Attempt
    #   ### 3.1 Navigate to Landing Page
    # And varying status/result lines, e.g.:
    #   - **Status**: PARTIAL FAILURE
    #   **Status**: Failed
    #   - **Result**: ⚠️ Partial Success
    # We match any ### heading followed by a Status or Result line.
    _SUCCESS_INDICATORS: tuple[str, ...] = ("SUCCESS", "SUCCEED", "PASSED", "✅")
    has_errors: bool = False
    for sp_match in re.finditer(
        r"###\s*(?:Sub-phase\s+)?[\dA-Za-z.]+[:\s]+([^\n]+)\n"  # sub-phase title
        r".*?\*\*(?:Status|Result)\*\*:\s*([^\n]+)",              # status or result line (may be lines away)
        summary,
        re.DOTALL,
    ):
        sp_name: str = sp_match.group(1).strip()
        sp_status: str = sp_match.group(2).strip()
        sp_status_upper: str = sp_status.upper()
        is_success: bool = any(ind in sp_status_upper for ind in _SUCCESS_INDICATORS)
        if not is_success:
            error_msg: str = (
                f"ERROR ({phase_name} / {sp_name}): Sub-phase status: {sp_status}"
                f"\n  [diagnostic: status \"{sp_status}\" matched none of {_SUCCESS_INDICATORS}]"
            )
            if observations:
                error_msg += f"\n  Details:\n  {observations}"
            log_problem(problem_log, error_msg)
            has_errors = True

    # Log observations standalone only when no sub-phase errors consumed them
    if observations and not has_errors:
        log_problem(problem_log, f"OBSERVATIONS ({phase_name}): {observations}")


def generate_task_name(prompt: str) -> str:
    """Use an LLM to generate a short, filename-safe task name from the prompt.

    The generated name is sanitized to contain only lowercase letters, digits,
    and hyphens, making it safe for use in filenames and directory names.

    Args:
        prompt: The user's full task description.

    Returns:
        A short hyphenated name like ``"login-test"`` or ``"write-tea-blog"``.
        Falls back to ``"unnamed-task"`` if the LLM returns an empty result.
    """
    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=64,
        messages=[{"role": "user", "content": (
            "Generate a short (2-8 word) descriptive name for this browser automation task. "
            "The name will be used as a filename, so use only lowercase letters, numbers, and hyphens. "
            "No spaces, no underscores, no special characters. Examples: 'write-tea-blog', 'login-test', 'scrape-products'.\n\n"
            f"Task: {prompt}\n\n"
            "Reply with ONLY the name, nothing else."
        )}],
    )
    name: str = response.content[0].text.strip().strip('"').strip("'")
    # Sanitize: replace any non-allowed characters with hyphens, collapse runs
    name = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    # Truncate to 80 chars (well within filesystem limits even with timestamp subdir)
    if len(name) > 80:
        name = name[:80].rstrip("-")
    return name or "unnamed-task"


def decompose_task(
    prompt: str,
    knowledge_match: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to decompose a free-form task into ordered execution phases.

    The LLM acts as a browser automation planner: given the user's task and
    optionally a set of reusable subtasks from prior runs, it produces an
    ordered list of phases. Each phase has a name and detailed step-by-step
    browser instructions. The first phase is always "Login".

    If reusable subtasks are provided via ``knowledge_match``, the LLM may
    mark phases with ``"reuse": "filename.txt"`` instead of generating new
    instructions. These are post-processed to load the saved instructions.

    Args:
        prompt: The user's task description.
        knowledge_match: Optional knowledge match result from
            ``find_relevant_knowledge()``, containing reusable subtasks.

    Returns:
        A list of phase dicts, each with keys:
          - ``name`` (str): human-readable phase name (e.g. "Login")
          - ``task`` (str): detailed browser automation instructions
    """
    # Build a section describing reusable subtasks if any were matched
    reuse_section: str = ""
    if knowledge_match and knowledge_match.get("reusable_subtasks"):
        reuse_parts: list[str] = ["\n\nREUSABLE SUBTASKS FROM PRIOR RUNS:"]
        reuse_parts.append("The following subtasks have been successfully executed before and can be reused.")
        reuse_parts.append('For any phase that is fully covered by a reusable subtask, return "reuse": "filename.txt" INSTEAD of "task".')
        reuse_parts.append('Only reuse a subtask if it covers the ENTIRE phase — do not partially reuse.\n')
        for st in knowledge_match["reusable_subtasks"]:
            # Load the actual subtask content so the LLM can verify it matches
            subtask_path: Path = TASKS_DIR / st["filename"]
            if subtask_path.exists():
                content: str = _restore_from_storage(subtask_path.read_text())
                reuse_parts.append(f'  Filename: {st["filename"]}')
                reuse_parts.append(f'  Phase name: {st["phase_name"]}')
                reuse_parts.append(f'  Reason: {st["reason"]}')
                reuse_parts.append(f"  Content summary:\n    {content[:500]}...")
                reuse_parts.append("")
        reuse_section = "\n".join(reuse_parts)

    # Add format guidance for reuse if the section is non-empty
    reuse_format: str = ""
    if reuse_section:
        reuse_format = '\nFor reused phases, use: {"name": "Phase Name", "reuse": "filename.txt"}'

    # Ask the LLM to produce the phase decomposition as a JSON array
    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"""You are a browser automation planner for SparkyAI ({HOST}).

Given a user's task description, decompose it into sequential phases that a browser automation agent will execute one at a time. Each phase shares the same browser session, so the next phase picks up where the previous one left off.

IMPORTANT RULES:
- The FIRST phase must ALWAYS be "Login" — navigate to {HOST} and log in with:
    Email: {USER_EMAIL}
    Password: {USER_PASSWORD}
- Each phase should be a self-contained step with clear success criteria.
- Every phase's instructions MUST end with an explicit STOP condition that tells the agent exactly what state to stop at and to NOT proceed further. For example: "Report success once the Step 2 interface is visible. Do NOT click any buttons or continue to the next step." This prevents the agent from overshooting into work that belongs to a later phase.
- Write the task instructions as explicit, step-by-step directions for an AI agent controlling a browser.
- Every phase must start with: "Check for error popup/toast after every action. Report any deviations from expected behavior."
- For form fields, specify exact values to enter.
- For dropdowns/selectors that are custom (like Tone), instruct the agent to type to filter/search rather than scroll.
- For generation/loading steps, instruct the agent to poll every 5 seconds (NOT one long wait) up to a timeout.
- For navigation, instruct the agent to verify the page loaded by checking for expected elements.
{reuse_section}

Return ONLY valid JSON — an array of objects with "name" and "task" keys.{reuse_format} Example:
[
  {{"name": "Login", "task": "Check for error popup/toast after every action. ..."}},
  {{"name": "Navigate to feature", "task": "Check for error popup/toast after every action. ..."}}
]
BASE_URL is {HOST}
User's task:
{prompt}"""}],
    )
    text: str = response.content[0].text.strip()
    # Extract the JSON array from the response (handles markdown code fences)
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
                    - Complete all steps of a process unles otherwise directed.
    """
    # Post-process: replace "reuse" markers with the actual saved task content.
    # When a phase has "reuse" instead of "task", load the referenced file and
    # wrap it with instructions to repeat the previously successful procedure.
    for phase in phases:
        if "reuse" in phase:
            reuse_file: Path = TASKS_DIR / phase["reuse"]
            if reuse_file.exists():
                task_content: str = _restore_from_storage(reuse_file.read_text())
                phase["task"] = (
                    "=== INSTRUCTIONS FROM PREVIOUSLY SUCCESSFUL EXECUTION ===\n"
                    f"{task_content}\n\n"
                    "=== YOUR TASK ===\n"
                    f"BASE_URL is {HOST}"
                    "- Repeat the procedure described above. Follow the same steps.\n"
                    "- Your goal is to TEST the given task.\n"
                    "- Repeat problems/workarounds noted in the procedure above in order to determine if they have been fixed.\n"
                    f"{base_instructions}\n"

                )
                print(f"  Reusing subtask '{phase['reuse']}' for phase '{phase['name']}'")
            else:
                print(f"  Warning: reuse file '{phase['reuse']}' not found, phase will need fresh instructions")
                # Generate a basic fallback instruction since the file is missing
                phase["task"] = (
                    f"{base_instructions}\n"
                    "- Avoid workarounds. Your goal is to TEST the given task\n"
                    f"- Execute the '{phase['name']}' phase of the task."
                )
            del phase["reuse"]

    return phases


def extract_phase_history(result: AgentHistoryList[Any]) -> str:
    """Extract a structured text log from the agent's step-by-step history.

    Iterates over every step in the agent's execution history and formats
    the key information (evaluations, memory, goals, actions, errors,
    extracted content, and URLs) into a human-readable text log. This log
    is then fed to the summarization LLM.

    Args:
        result: The ``AgentHistoryList`` returned by ``Agent.run()``,
            containing the full execution trace.

    Returns:
        A multi-line string with one section per step, showing evaluations,
        memory notes, goals, actions taken, errors, extracted content, and
        the browser URL at each step.
    """
    lines: list[str] = []
    for i, h in enumerate(result.history, 1):
        lines.append(f"--- Step {i} ---")
        if h.model_output:
            mo: Any = h.model_output
            # The agent's self-evaluation of whether the previous goal succeeded
            if mo.evaluation_previous_goal:
                lines.append(f"  Eval: {mo.evaluation_previous_goal}")
            # Working memory the agent chose to persist across steps
            if mo.memory:
                lines.append(f"  Memory: {mo.memory}")
            # What the agent planned to do next
            if mo.next_goal:
                lines.append(f"  Next goal: {mo.next_goal}")
            # Each individual action the agent took (click, type, navigate, etc.)
            for action in mo.action:
                lines.append(f"  Action: {action.model_dump(exclude_none=True)}")
        # Results from executing actions — may include errors or extracted data
        for r in h.result:
            if r.error:
                lines.append(f"  ERROR: {r.error}")
            if r.extracted_content:
                lines.append(f"  Extracted: {r.extracted_content}")
        # Current browser URL after this step completed
        if h.state and h.state.url:
            lines.append(f"  URL: {h.state.url}")
    return "\n".join(lines)


def summarize_phase(
    phase_name: str,
    phase_task: str,
    result: AgentHistoryList[Any],
    success: bool,
) -> str:
    """Use an LLM to generate a structured summary of a completed phase.

    Produces a detailed summary including outcome, actions taken, logical
    sub-phases, page state, observations, key facts learned, and timing.
    This summary serves two purposes: (1) providing context to subsequent
    phases in the same run, and (2) persisting as a subtask file for future
    knowledge reuse.

    Args:
        phase_name: Human-readable name of the phase (e.g. "Login").
        phase_task: The original task instructions given to the agent.
        result: The ``AgentHistoryList`` from the agent's execution.
        success: Whether the phase completed successfully.

    Returns:
        A structured text summary with sections for outcome, actions,
        sub-phases, page state, observations, key facts, and timing.
    """
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

    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_task_report(
    task_name: str,
    prompt: str,
    summaries: list[dict[str, str]],
) -> dict[str, Any]:
    """Use an LLM to generate the final task report as a JSON-serializable dict.

    Collects all phase summaries and asks an LLM to produce a high-level
    report with a one-line task description and a list of key observations.
    The subtask breakdown (phase filenames and numbers) is appended
    programmatically after the LLM response.

    Args:
        task_name: The short identifier for this task run.
        prompt: The user's original task description.
        summaries: List of phase summary dicts, each with ``name``,
            ``outcome``, ``summary``, and ``filename`` keys.

    Returns:
        A dict with keys:
          - ``main_task`` (str): one-line description of the overall task.
          - ``key_observations`` (list[str]): things an LLM agent should know
            for future runs (UI quirks, timing issues, error patterns, etc.).
            These are plain strings as returned by the LLM; classification
            into error/warning is done separately by ``classify_observations()``.
          - ``subtasks`` (list[dict]): numbered references to subtask files,
            each with ``subtask`` (int) and ``filename`` (str).
    """
    # Format all phase summaries into a single text block for the LLM
    summaries_text: str = "\n\n".join(
        f"Phase: {s['name']} [{s['outcome']}]\n{s['summary']}"
        for s in summaries
    )

    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
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
    # Extract the JSON object (may be wrapped in markdown code fences)
    match: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    report: dict[str, Any] = json.loads(text)

    # Append the subtask breakdown referencing each phase's saved summary file.
    # Each subtask entry:
    #   "subtask": int — 1-based phase number
    #   "filename": str — name of the subtask summary file in TASKS_DIR
    report["subtasks"] = [
        {"subtask": i, "filename": s["filename"]}
        for i, s in enumerate(summaries, 1)
    ]

    return report


def _observation_text(obs: str | dict[str, str]) -> str:
    """Extract the text from an observation, handling both old and new formats.

    Old format: plain string.
    New format: dict with ``text`` and ``severity`` keys.

    Args:
        obs: An observation in either string or dict format.

    Returns:
        The observation text as a plain string.
    """
    if isinstance(obs, dict):
        return obs.get("text", "")
    return obs


def _build_rules_prompt_section(rules: ClassificationRules) -> str:
    """Format classification rules into a prompt block for the LLM.

    Returns an empty string when no rules exist, keeping the prompt unchanged.
    """
    if not rules.error_rules and not rules.warning_rules:
        return ""

    parts: list[str] = [
        "\n**PRIORITY CLASSIFICATION RULES** (these override the general criteria above):\n",
    ]

    if rules.error_rules:
        parts.append("The following should ALWAYS be classified as \"error\":")
        for i, rule in enumerate(rules.error_rules, 1):
            parts.append(f"  {i}. {rule}")
        parts.append("")

    if rules.warning_rules:
        parts.append("The following should ALWAYS be classified as \"warning\":")
        for i, rule in enumerate(rules.warning_rules, 1):
            parts.append(f"  {i}. {rule}")
        parts.append("")

    return "\n".join(parts)


def classify_observations(
    prompt: str,
    observations: list[str | dict[str, str]],
    rules: ClassificationRules | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to classify each observation as an error or a warning.

    An **error** means application functionality is broken — the feature under
    test did not work, even if a workaround achieved the broader goal.  A
    **warning** means the environment or UI changed from the original
    instructions, but the feature under test still worked correctly.

    Args:
        prompt: The original task description that produced these observations.
        observations: List of observation strings (or dicts from a prior
            classification) to classify.
        rules: Optional classification rules to guide the LLM.  Defaults to
            the module-level ``_classification_rules`` loaded at startup.

    Returns:
        A list of dicts, each with ``text`` (str) and ``severity``
        (``"error"`` or ``"warning"``) keys, in the same order as the input.
    """
    if rules is None:
        rules = _classification_rules

    # Normalise to plain strings for the LLM prompt
    obs_texts: list[str] = [_observation_text(o) for o in observations]

    if not obs_texts:
        return []

    rules_section: str = _build_rules_prompt_section(rules)

    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"""You are a QA analyst classifying observations from a browser automation test run.

Classify each observation as "error" or "warning" using these criteria:

**error**: Application functionality is broken. The feature under test did not
work as specified, even if a workaround achieved the broader goal. If a
workaround was used, consider whether it tests the SAME feature as the original
step — if the workaround uses a DIFFERENT mechanism (e.g. browsing categories
instead of using search), the original feature is still broken → "error".

**warning**: The environment or UI changed from the original instructions, but
the feature under test worked correctly. Examples: element indices shifted,
domain/URL changed, layout rearranged, extra fields present, cosmetic
differences.

For each observation, reason through:
1. Was the underlying feature/functionality working correctly?
2. Was a workaround needed that bypasses the feature under test?
3. Is this just an environmental or documentation difference?
{rules_section}
Original task instructions:
{prompt}

Observations:
{json.dumps(obs_texts, indent=2)}

Return ONLY a valid JSON array with one entry per observation, in the same order:
[
  {{"text": "observation text here", "severity": "error"}},
  {{"text": "observation text here", "severity": "warning"}}
]"""}],
    )
    text: str = response.content[0].text.strip()
    match: re.Match[str] | None = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        classified: list[dict[str, str]] = json.loads(text)
        # Validate and normalise each entry
        result: list[dict[str, str]] = []
        for i, entry in enumerate(classified):
            obs_text: str = entry.get("text", obs_texts[i] if i < len(obs_texts) else "")
            severity: str = entry.get("severity", "warning")
            if severity not in ("error", "warning"):
                severity = "warning"
            result.append({"text": obs_text, "severity": severity})
        return result
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        print(f"  Warning: could not parse classification response: {e}")
        # Fall back to marking everything as warning
        return [{"text": t, "severity": "warning"} for t in obs_texts]


def merge_observations(
    existing: list[str | dict[str, str]],
    new: list[str | dict[str, str]],
) -> list[dict[str, str]]:
    """Use an LLM to merge and de-duplicate two lists of observations.

    When replaying a goal, this merges the new run's observations with the
    previously saved ones. The LLM keeps the more detailed or recent version
    when two observations say the same thing in different words, while
    preserving all unique information. Severity classifications are preserved
    where available.

    Args:
        existing: Observations from the previously saved goal summary
            (plain strings or dicts with ``text`` and ``severity``).
        new: Observations from the current run (plain strings or dicts).

    Returns:
        A merged, de-duplicated list of observation dicts with ``text`` and
        ``severity`` keys.
    """
    # Build a severity lookup from both lists so we can restore classifications
    severity_map: dict[str, str] = {}
    for obs in list(existing) + list(new):
        if isinstance(obs, dict):
            severity_map[obs.get("text", "")] = obs.get("severity", "warning")

    # Normalise to plain strings for the merge LLM call
    existing_texts: list[str] = [_observation_text(o) for o in existing]
    new_texts: list[str] = [_observation_text(o) for o in new]

    response: anthropic.types.Message = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"""You have two lists of observations from browser automation runs.
Merge them into a single de-duplicated list. If two observations say the same thing in different words, keep the more detailed or recent one. Preserve all unique information.

Existing observations:
{json.dumps(existing_texts, indent=2)}

New observations:
{json.dumps(new_texts, indent=2)}

Return ONLY a valid JSON array of strings — the merged, de-duplicated observations."""}],
    )
    text: str = response.content[0].text.strip()
    match: re.Match[str] | None = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    merged_texts: list[str] = json.loads(text)

    # Restore severity from the lookup; default to "warning" for new entries
    return [
        {"text": t, "severity": severity_map.get(t, "warning")}
        for t in merged_texts
    ]


def build_augmented_task(
    original_task: str,
    prior_summaries: list[dict[str, str]],
    cross_goal_observations: list[str | dict[str, str]] | None = None,
) -> str:
    """Prepend accumulated context to a phase's task instructions.

    Builds an augmented version of the original task by prepending:
      1. Cross-goal observations (knowledge from prior successful goal runs)
      2. Prior phase summaries from the current run (so the agent knows what
         has already happened in the browser session)

    This contextual augmentation allows each phase's agent to understand the
    current browser state and leverage lessons learned from past executions.

    Args:
        original_task: The raw task instructions for this phase.
        prior_summaries: Summaries of all preceding phases in the current run,
            each with ``name``, ``outcome``, and ``summary`` keys.
        cross_goal_observations: Optional list of observations (strings or
            dicts with ``text``/``severity``) from prior goal runs.

    Returns:
        The augmented task string with context sections prepended, and any
        ``{BASE_URL}`` placeholders replaced with the active HOST value.
    """
    context_parts: list[str] = []

    # Add knowledge from prior successful goals (cross-run learning)
    if cross_goal_observations:
        context_parts.append("=== KNOWLEDGE FROM PRIOR SUCCESSFUL GOALS ===")
        for obs in cross_goal_observations:
            context_parts.append(f"- {_observation_text(obs)}")
        context_parts.append("")

    # Add summaries from earlier phases in the current run (intra-run context)
    if prior_summaries:
        context_parts.append("=== CONTEXT FROM PRIOR PHASES (current run) ===")
        for s in prior_summaries:
            context_parts.append(f"\n-- Phase: {s['name']} ({s['outcome']}) --")
            context_parts.append(s["summary"])
        context_parts.append("")

    # If no context was accumulated, just return the original task with URL substitution
    if not context_parts:
        return _restore_from_storage(original_task)

    context_parts.append("=== YOUR TASK (use the context above to inform your actions) ===\n")
    task_text: str = "\n".join(context_parts) + original_task
    return _restore_from_storage(task_text)


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

    Creates a new ``Agent`` instance for this phase (sharing the same browser
    session and LLM), runs it for up to 50 steps, and logs the outcome. On
    failure, attempts to capture a screenshot for debugging.

    Args:
        name: Human-readable phase name (e.g. "Login").
        task: The full (possibly augmented) task instructions for this phase.
        llm: The ``ChatBrowserUse`` LLM instance for the agent.
        browser: The shared ``Browser`` instance (persistent across phases).
        conversation_log: Path where the agent saves its conversation JSON.
        event_log: Path to the event log file for timestamped logging.
        problem_log: Path to the problem log file for recording failures.
        run_dir: Directory for this run's artifacts (screenshots, etc.).

    Returns:
        A tuple of ``(success, result)`` where ``success`` is True if the
        phase completed and was marked successful by the agent, and ``result``
        is the full ``AgentHistoryList`` execution trace.
    """
    log_event(event_log, f"PHASE START: {name}")

    # Create a fresh agent for this phase — the browser session carries over
    # from the previous phase, so the agent picks up where the last one left off
    agent: Agent[Any, Any] = Agent(
        task=task,
        llm=llm,
        browser=browser,
        save_conversation_path=str(conversation_log),
        max_failures=5,        # Allow up to 5 consecutive action failures before giving up
        max_actions_per_step=5, # Limit actions per LLM turn to prevent runaway loops
    )

    result: AgentHistoryList[Any] = await agent.run(max_steps=50)
    success: bool = result.is_done() and result.is_successful()

    # Format the agent's final result for logging (indent each line)
    final: str = result.final_result() or ""
    final_lines: str = "\n".join([f"  FINAL RESULT: {line}" for line in final.split("\n")])

    if success:
        log_event(event_log, f"PHASE SUCCESS: {name}\n{final_lines}")
    else:
        log_event(event_log, f"PHASE FAILED: {name}\n{final_lines}")
        log_problem(problem_log, f"PHASE FAILED: {name}\n{final_lines}")
        # Attempt to capture a screenshot of the failure state for debugging
        try:
            screenshot_path: str = str(run_dir / f"failure_{name.replace(' ', '_')}.png")
            page = await browser.get_current_page()
            await page.screenshot(screenshot_path)
            log_event(event_log, f"Failure screenshot saved to {screenshot_path}")
        except Exception as e:
            log_event(event_log, f"Could not save failure screenshot: {e}")
            log_problem(problem_log, f"Could not save failure screenshot for {name}: {e}")

    # Log per-step action errors to the problem log
    for h in result.history:
        for r in h.result:
            if r.error:
                log_problem(problem_log, f"ACTION ERROR ({name}): {r.error}")

    # Log a truncated version of the step-by-step history for the event log
    history_text: str = extract_phase_history(result)
    history_truncated: str = history_text[:500] + "..." if len(history_text) > 500 else history_text
    history_truncated = "\n".join([f"  PHASE {name} HISTORY: {line}" for line in history_truncated.split("\n")])
    log_event(event_log, f"PHASE HISTORY ({name}):\n{history_truncated}")

    return success, result


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


def load_goal_summary(goal_path: Path) -> tuple[str, str, list[dict[str, str]]]:
    """Load a goal summary JSON and reconstruct the prompt, task name, and phases.

    Used when replaying a previously saved goal. Reads the goal summary file,
    extracts the main task description, and loads each referenced subtask file
    to reconstruct the full phase list. Each subtask's summary text is prefixed
    with a replay instruction so the agent treats it as actions to re-execute
    rather than a report of already-completed work.

    Args:
        goal_path: Path to the goal summary JSON file (e.g.
            ``goal_summaries/login-test-task.json``).

    Returns:
        A tuple of ``(prompt, task_name, phases)`` where:
          - ``prompt`` is the original task description
          - ``task_name`` is derived from the filename (minus the ``-task`` suffix)
          - ``phases`` is a list of dicts with ``name`` and ``task`` keys

    Raises:
        FileNotFoundError: If a referenced subtask file does not exist.
        json.JSONDecodeError: If the goal summary JSON is malformed.
    """
    goal_data: dict[str, Any] = json.loads(_restore_from_storage(goal_path.read_text()))
    prompt: str = goal_data["main_task"]

    phases: list[dict[str, str]] = []
    for entry in goal_data["subtasks"]:
        subtask_path: Path = TASKS_DIR / entry["filename"]
        if not subtask_path.exists():
            raise FileNotFoundError(f"Subtask file not found: {subtask_path}")
        task_content: str = _restore_from_storage(subtask_path.read_text())
        # Derive a human-readable phase name from the filename
        # e.g. "login.txt" -> "Login", "fill-form.txt" -> "Fill Form"
        name: str = subtask_path.stem.replace("-", " ").title()
        # Phase dict:
        #   "name": str — human-readable phase name (e.g. "Login", "Fill Form")
        #   "task": str — prior summary wrapped with replay instructions so the
        #                  agent re-executes the actions instead of treating
        #                  them as already-completed work
        phases.append({"name": name, "task": _REPLAY_PREFIX + task_content})

    # Strip the "-task" suffix from the filename to get the task name
    task_name: str = goal_path.stem.removesuffix("-task")
    return prompt, task_name, phases


# Usage text displayed when --help is passed. Defined as a module-level constant
# so it can reference the HOST variable via f-string.
def list_goals() -> None:
    """Print a summary of all existing goals from the goal summaries directory."""
    goal_files: list[Path] = sorted(GOAL_SUMMARIES_DIR.glob("*-task.json"))
    if not goal_files:
        print("No goals found.")
        return

    print(f"Found {len(goal_files)} goal(s):\n")
    for goal_file in goal_files:
        try:
            data: dict[str, Any] = json.loads(_restore_from_storage(goal_file.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"  {goal_file.name}  (unreadable)")
            continue
        main_task: str = data.get("main_task", "(no description)")
        num_subtasks: int = len(data.get("subtasks", []))
        observations: list[str | dict[str, str]] = data.get("key_observations", [])
        num_observations: int = len(observations)
        num_errors: int = sum(
            1 for o in observations
            if isinstance(o, dict) and o.get("severity") == "error"
        )
        num_warnings: int = sum(
            1 for o in observations
            if isinstance(o, dict) and o.get("severity") == "warning"
        )
        num_unclassified: int = num_observations - num_errors - num_warnings
        print(f"  {goal_file.name}")
        print(f"    Task: {main_task}")
        severity_parts: list[str] = []
        if num_errors:
            severity_parts.append(f"{num_errors} errors")
        if num_warnings:
            severity_parts.append(f"{num_warnings} warnings")
        if num_unclassified:
            severity_parts.append(f"{num_unclassified} unclassified")
        severity_str: str = f" ({', '.join(severity_parts)})" if severity_parts else ""
        print(f"    Subtasks: {num_subtasks}  Observations: {num_observations}{severity_str}")
        print()


USAGE: str = f"""\
usage: sparky_runner.py [options] [goal_summary.json]

SparkyAI Browser Automation Runner

positional arguments:
  goal_summary.json       Replay a previously saved goal summary file

options:
  -h, --help              Show this help message and exit
  -p, --prompt PROMPT     Task prompt (skips interactive input)
  -u, --url URL           Base URL (default: {HOST})
  --no-update-summary     Don't update the goal summary file on replay
  --no-update-tasks       Don't overwrite existing task files on replay
  --no-knowledge-reuse    Don't reuse knowledge from prior goals
  --auto-close            Close the browser automatically on finish
  --list-goals            List all existing goals and exit
  --classify-existing     Classify observations in all existing goal summaries
  --find-orphans          List task files not referenced by any goal summary
  --clean-orphans         Delete task files not referenced by any goal summary
"""


def _get_orphan_tasks() -> list[str]:
    """Return sorted list of task filenames not referenced by any goal summary.

    Scans every ``*-task.json`` file in GOAL_SUMMARIES_DIR, collects the set of
    referenced subtask filenames, then compares against the actual files on disk
    in TASKS_DIR.
    """
    task_files: set[str] = {
        f.name for f in TASKS_DIR.iterdir() if f.is_file()
    }

    referenced: set[str] = set()
    for goal_file in GOAL_SUMMARIES_DIR.glob("*-task.json"):
        try:
            data: dict[str, Any] = json.loads(goal_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for subtask in data.get("subtasks", []):
            filename: str = subtask.get("filename", "")
            if filename:
                referenced.add(filename)

    return sorted(task_files - referenced)


def find_orphan_tasks() -> None:
    """Print task files in TASKS_DIR that are not referenced by any goal summary."""
    orphans: list[str] = _get_orphan_tasks()
    if not orphans:
        print("No orphan task files found.")
        return

    print(f"Found {len(orphans)} orphan task file(s) not referenced in any goal summary:\n")
    for name in orphans:
        print(f"  {name}")


def clean_orphan_tasks() -> None:
    """Delete task files in TASKS_DIR that are not referenced by any goal summary.

    Lists the orphans first, then asks for confirmation before deleting.
    """
    orphans: list[str] = _get_orphan_tasks()
    if not orphans:
        print("No orphan task files to clean.")
        return

    print(f"Found {len(orphans)} orphan task file(s) to delete:\n")
    for name in orphans:
        print(f"  {name}")

    answer: str = input(f"\nDelete {len(orphans)} file(s)? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    for name in orphans:
        (TASKS_DIR / name).unlink()
        print(f"  Deleted {name}")
    print(f"\nDeleted {len(orphans)} orphan task file(s).")


def classify_existing_goals() -> None:
    """Retroactively classify observations in all existing goal summaries.

    Reads each goal summary JSON, runs ``classify_observations()`` on its
    ``key_observations`` list, and writes the classified results back.
    Skips goals whose observations are already classified (all dicts with
    ``severity`` keys).
    """
    goal_files: list[Path] = sorted(GOAL_SUMMARIES_DIR.glob("*-task.json"))
    if not goal_files:
        print("No goals found.")
        return

    print(f"Classifying observations in {len(goal_files)} goal(s)...\n")
    for goal_file in goal_files:
        try:
            data: dict[str, Any] = json.loads(goal_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {goal_file.name}: {e}")
            continue

        observations: list[str | dict[str, str]] = data.get("key_observations", [])
        if not observations:
            print(f"  {goal_file.name}: no observations, skipping")
            continue

        # Skip if already fully classified
        if all(isinstance(o, dict) and "severity" in o for o in observations):
            print(f"  {goal_file.name}: already classified, skipping")
            continue

        prompt: str = data.get("main_task", "")
        print(f"  {goal_file.name}: classifying {len(observations)} observations...")
        classified: list[dict[str, str]] = classify_observations(prompt, observations)
        data["key_observations"] = classified
        goal_file.write_text(json.dumps(data, indent=2))

        num_errors: int = sum(1 for o in classified if o["severity"] == "error")
        num_warnings: int = len(classified) - num_errors
        print(f"    -> {num_errors} errors, {num_warnings} warnings")

    print("\nDone.")


# ── Helpers extracted from main() for testability ────────────────────────


def phase_name_to_slug(name: str) -> str:
    """Convert a human-readable phase name to a filename-safe slug.

    >>> phase_name_to_slug("Fill Form")
    'fill-form'
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def make_run_dir(runs_dir: Path, task_name: str) -> Path:
    """Create and return a timestamped run directory under *runs_dir*/*task_name*."""
    run_timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir: Path = runs_dir / task_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_phase_plan(phases: list[dict[str, str]]) -> list[str]:
    """Format the decomposition plan as a list of log-ready lines."""
    lines: list[str] = [f"PHASE PLAN ({len(phases)} phases):"]
    for i, p in enumerate(phases, 1):
        lines.append(f"  Phase {i}: {p['name']}")
        task_preview: str = p.get("task", "(reuse — see above)")[:2000]
        lines.append(f"    Instructions: {task_preview}")
    return lines


def format_knowledge_match(knowledge_match: dict[str, Any]) -> list[str]:
    """Format knowledge-match info as a list of log-ready lines."""
    lines: list[str] = []
    reusable: list[dict[str, str]] = knowledge_match.get("reusable_subtasks", [])
    if reusable:
        lines.append(f"REUSABLE SUBTASKS ({len(reusable)}):")
        for st in reusable:
            lines.append(f"  - {st['filename']} ({st['phase_name']}): {st['reason']}")
    obs: list[str] = knowledge_match.get("relevant_observations", [])
    if obs:
        lines.append(f"RELEVANT OBSERVATIONS FROM PRIOR GOALS ({len(obs)}):")
        for o in obs:
            lines.append(f"  - {o}")
    coverage: str = knowledge_match.get("coverage_notes", "")
    if coverage:
        lines.append(f"COVERAGE NOTES: {coverage}")
    return lines


async def main() -> None:
    """Entry point: parse CLI arguments, orchestrate the full workflow.

    The main function handles:
      1. CLI argument parsing (prompt, URL, flags, goal file)
      2. Loading the knowledge index from prior runs
      3. Either replaying a saved goal or decomposing a new prompt into phases
      4. Executing each phase sequentially with contextual augmentation
      5. Summarizing each phase and saving subtask files
      6. Generating and saving the final task report
      7. Merging observations into existing goal summaries on replay
    """
    # --- Parse arguments manually (no argparse dependency) ---
    args: list[str] = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(USAGE)
        return

    if "--list-goals" in args:
        list_goals()
        return

    if "--classify-existing" in args:
        classify_existing_goals()
        return

    if "--find-orphans" in args:
        find_orphan_tasks()
        return

    if "--clean-orphans" in args:
        clean_orphan_tasks()
        return

    # Boolean flags — remove from args list after detection
    no_update_summary: bool = "--no-update-summary" in args
    if no_update_summary:
        args.remove("--no-update-summary")
    no_update_tasks: bool = "--no-update-tasks" in args
    if no_update_tasks:
        args.remove("--no-update-tasks")
    no_knowledge_reuse: bool = "--no-knowledge-reuse" in args
    if no_knowledge_reuse:
        args.remove("--no-knowledge-reuse")
    auto_close: bool = "--auto-close" in args
    if auto_close:
        args.remove("--auto-close")

    # Optional --prompt / -p flag with a value
    prompt_arg: str | None = None
    for flag in ("-p", "--prompt"):
        if flag in args:
            idx: int = args.index(flag)
            if idx + 1 >= len(args):
                print(f"Error: {flag} requires a value")
                return
            prompt_arg = args[idx + 1]
            del args[idx:idx + 2]
            break

    # Optional --url / -u flag to override the target host
    global HOST
    HOST = DEFAULT_HOST
    for flag in ("-u", "--url"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 >= len(args):
                print(f"Error: {flag} requires a value")
                return
            HOST = args[idx + 1]
            del args[idx:idx + 2]
            break

    print("=" * 60)
    print("  SparkyAI Browser Automation")
    print(f"  Target: {HOST}")
    print("=" * 60)
    print()

    # --- Load knowledge index from all prior goal summaries ---
    knowledge_match: dict[str, Any] | None = None
    if not no_knowledge_reuse:
        print("Loading knowledge from prior goals...")
        knowledge_index: list[dict[str, Any]] = load_knowledge_index()
        print(f"  Loaded {len(knowledge_index)} prior goal(s)")
    else:
        knowledge_index = []
        print("Knowledge reuse disabled (--no-knowledge-reuse)")

    # --- Determine the task: either replay from a goal file or prompt ---
    goal_path: Path | None = None
    if args:
        # Positional argument: path to a goal summary JSON to replay
        goal_path = Path(args[0])
        if not goal_path.exists():
            print(f"Goal summary file not found: {goal_path}")
            return
        print(f"Loading goal from: {goal_path}")
        prompt: str
        task_name: str
        phases: list[dict[str, str]]
        prompt, task_name, phases = load_goal_summary(goal_path)
        print(f"Task: {prompt}")
        print(f"Task name: {task_name}")

        # For replays, find relevant knowledge but exclude the goal being
        # replayed to avoid self-referencing
        if knowledge_index:
            replay_file: str = goal_path.name
            filtered_index: list[dict[str, Any]] = [
                g for g in knowledge_index if g["goal_file"] != replay_file
            ]
            if filtered_index:
                print("\nFinding relevant knowledge from other goals...")
                knowledge_match = find_relevant_knowledge(prompt, filtered_index)
            else:
                print("  No other goals to learn from")
    else:
        # No goal file provided — prompt the user interactively or use -p value
        prompt = prompt_arg or input("Enter your task: ").strip()
        if not prompt:
            print("No task provided. Exiting.")
            return
        print()
        print("Generating task name...")
        task_name = generate_task_name(prompt)
        print(f"Task name: {task_name}")

        # For new goals, search all prior goals for relevant knowledge
        if knowledge_index:
            print("\nFinding relevant knowledge from prior goals...")
            knowledge_match = find_relevant_knowledge(prompt, knowledge_index)

        # Decompose the user's prompt into sequential execution phases
        print()
        print("Decomposing task into phases...")
        phases = decompose_task(prompt, knowledge_match=knowledge_match)

    print(f"Planned {len(phases)} phases:")
    for i, p in enumerate(phases, 1):
        print(f"  {i}. {p['name']}")
    print()

    # Create a unique run directory using task name and timestamp
    run_dir: Path = make_run_dir(RUNS_DIR, task_name)

    event_log: Path = run_dir / "event_log.txt"
    conversation_log: Path = run_dir / "conversation_log.json"
    summaries_path: Path = run_dir / "phase_summaries.json"
    problem_log: Path = run_dir / "problem_log.txt"
    report_path: Path = GOAL_SUMMARIES_DIR / f"{task_name}-task.json"

    # --- Execute the multi-phase workflow ---
    event_log.write_text("")  # Initialize the event log file
    log_event(event_log, "=" * 60)
    log_event(event_log, f"WORKFLOW START: {task_name}")
    log_event(event_log, f"Run directory: {run_dir}")
    log_event(event_log, f"Prompt: {prompt}")
    log_event(event_log, f"Target: {HOST}")
    log_event(event_log, "=" * 60)

    # Log the decomposition plan so we can trace phase selection decisions
    for line in format_phase_plan(phases):
        log_event(event_log, line)

    if knowledge_match:
        for line in format_knowledge_match(knowledge_match):
            log_event(event_log, line)

    # Create a visible (non-headless) browser that stays open between phases
    browser: Browser = Browser(headless=False, keep_alive=True)
    llm: ChatBrowserUse = ChatBrowserUse()

    # Accumulates summary records for all executed phases (for final reporting)
    all_summaries: list[dict[str, str]] = []

    try:
        # Prior summaries grow as each phase completes, providing context
        # to subsequent phases via build_augmented_task()
        prior_summaries: list[dict[str, str]] = []

        for phase in phases:
            # Gather cross-goal observations if knowledge matching found any
            cross_obs: list[str] | None = (
                knowledge_match["relevant_observations"] if knowledge_match else None
            )
            # Build the augmented task with all accumulated context
            augmented_task: str = build_augmented_task(
                phase["task"], prior_summaries, cross_goal_observations=cross_obs
            )
            # Log a truncated version of the augmented task for debugging
            augmented_task_truncated: str = (
                augmented_task[:1000] + "..." if len(augmented_task) > 1000 else augmented_task
            )
            log_event(event_log, f"Task for '{phase['name']}':\n{augmented_task_truncated}")

            # Execute the phase — the browser session persists across phases
            success: bool
            result: AgentHistoryList[Any]
            success, result = await run_phase(
                phase["name"], augmented_task, llm, browser,
                conversation_log, event_log, problem_log, run_dir,
            )

            # Summarize what the agent did and learned during this phase
            log_event(event_log, f"Summarizing phase '{phase['name']}'...")
            summary: str = summarize_phase(
                phase["name"], phase["task"], result, success
            )
            _extract_and_log_observations(summary, phase["name"], problem_log)
            # Create a filename-safe slug from the phase name for the subtask file
            phase_slug: str = phase_name_to_slug(phase["name"])
            subtask_path: Path = TASKS_DIR / f"{phase_slug}.txt"
            if goal_path and not no_update_tasks:
                # On replay, overwrite the existing subtask file in place
                subtask_path.write_text(_sanitize_for_storage(summary))
            else:
                # For new runs, use safe_write_path to avoid overwriting
                if not no_update_tasks:
                    subtask_path = safe_write_path(subtask_path)
                    subtask_path.write_text(_sanitize_for_storage(summary))
            log_event(event_log, f"Subtask summary saved to {subtask_path}")

            # Record the phase outcome for both inter-phase context and final report.
            # Phase record dict:
            #   "name": str — human-readable phase name (e.g. "Login")
            #   "outcome": str — "SUCCESS" or "FAILED"
            #   "summary": str — LLM-generated structured summary of the phase execution
            #   "filename": str — name of the saved subtask summary file in TASKS_DIR
            phase_record: dict[str, str] = {
                "name": phase["name"],
                "outcome": "SUCCESS" if success else "FAILED",
                "summary": summary,
                "filename": subtask_path.name,
            }
            prior_summaries.append(phase_record)
            all_summaries.append(phase_record)

            # Log a truncated summary for the event log
            summary_truncated: str = summary[:500] + "..." if len(summary) > 500 else summary
            summary_truncated = "\n".join(
                [f"  PHASE {phase['name']} SUMMARY:\n{line}" for line in summary_truncated.split("\n")]
            )
            log_event(event_log, f"PHASE SUMMARY ({phase['name']}):\n{summary_truncated}")

            # Abort the workflow if any phase fails — later phases depend on
            # the browser state left by earlier ones
            if not success:
                log_event(event_log, f"ABORTING: phase '{phase['name']}' failed.")
                break
        else:
            # The for/else block: this runs only if the loop completed without break,
            # meaning all phases succeeded
            log_event(event_log, "ALL PHASES COMPLETED SUCCESSFULLY.")

    except Exception as e:
        # Catch unexpected errors (network issues, library crashes, etc.)
        log_event(event_log, f"UNEXPECTED ERROR: {e}")
        log_problem(problem_log, f"UNEXPECTED ERROR: {e}")
        try:
            page = await browser.get_current_page()
            unexpected_screenshot: str = str(run_dir / "failure_unexpected.png")
            await page.screenshot(unexpected_screenshot)
            log_event(event_log, f"Failure screenshot saved to {unexpected_screenshot}")
        except Exception as screenshot_err:
            log_problem(problem_log, f"Could not save unexpected-error screenshot: {screenshot_err}")
    finally:
        # Always save outputs and clean up, regardless of success or failure
        log_event(event_log, "WORKFLOW END")
        log_event(event_log, f"Conversation log saved to {conversation_log}")
        if problem_log.exists():
            log_event(event_log, f"Problem log saved to {problem_log}")

        # Save the raw phase summaries as a JSON array for external analysis
        summaries_path.write_text(_sanitize_for_storage(json.dumps(all_summaries, indent=2)))
        log_event(event_log, f"Phase summaries saved to {summaries_path}")

        # Generate and save the final task report (goal summary)
        if all_summaries:

            if goal_path and not no_update_summary:
                # Replay mode: merge new observations into the existing goal summary
                print(f"\nGenerating task report...")
                report: dict[str, Any] = generate_task_report(task_name, prompt, all_summaries)
                # Load the existing goal summary to merge observations
                existing_data: dict[str, Any] = json.loads(goal_path.read_text())
                existing_obs: list[str | dict[str, str]] = existing_data.get("key_observations", [])
                new_obs: list[str] = report.get("key_observations", [])
                print("Merging observations with existing goal summary...")
                merged_obs: list[dict[str, str]] = merge_observations(existing_obs, new_obs)
                # Classify the merged observations as errors or warnings
                print("Classifying observations...")
                classified_obs: list[dict[str, str]] = classify_observations(prompt, merged_obs)
                existing_data["key_observations"] = classified_obs
                existing_data["subtasks"] = report["subtasks"]
                goal_path.write_text(_sanitize_for_storage(json.dumps(existing_data, indent=2)))
                num_errors: int = sum(1 for o in classified_obs if o["severity"] == "error")
                num_warnings: int = len(classified_obs) - num_errors
                log_event(event_log, f"Updated goal summary: {goal_path} ({num_errors} errors, {num_warnings} warnings)")
                print(f"Updated goal summary: {goal_path} ({num_errors} errors, {num_warnings} warnings)")
            else:
                # New goal (not a replay) — generate and save a fresh report
                if not no_update_summary:
                    print(f"\nGenerating task report...")
                    report: dict[str, Any] = generate_task_report(task_name, prompt, all_summaries)
                    # Classify observations as errors or warnings
                    print("Classifying observations...")
                    classified_obs = classify_observations(prompt, report.get("key_observations", []))
                    report["key_observations"] = classified_obs
                    report_path = safe_write_path(report_path)
                    report_path.write_text(_sanitize_for_storage(json.dumps(report, indent=2)))
                    num_errors = sum(1 for o in classified_obs if o["severity"] == "error")
                    num_warnings = len(classified_obs) - num_errors
                    log_event(event_log, f"Task report saved to {report_path} ({num_errors} errors, {num_warnings} warnings)")
                    print(f"Task report saved to {report_path} ({num_errors} errors, {num_warnings} warnings)")
        # Unless --auto-close was passed, wait for user confirmation before
        # closing the browser so they can inspect the final state
        if not auto_close:
            input("\nPress Enter to close the browser...")
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
