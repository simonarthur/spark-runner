"""REPL-style interactive mode for Spark Runner."""

from __future__ import annotations

import asyncio
import dataclasses
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from spark_runner.models import SparkConfig


# ── Toolbar state ─────────────────────────────────────────────────────


@dataclass
class _ToolbarState:
    """Mutable state driving the prompt_toolkit bottom toolbar."""

    goal_count: int = 0
    last_run_name: str = ""
    last_run_status: str = ""  # "PASS", "FAIL", or ""


_toolbar_state = _ToolbarState()


def _bottom_toolbar() -> HTML:
    """Return styled text for the prompt_toolkit bottom toolbar."""
    parts: list[str] = []
    if _toolbar_state.goal_count:
        parts.append(f"{_toolbar_state.goal_count} goal(s)")
    if _toolbar_state.last_run_name:
        status = _toolbar_state.last_run_status
        style = "green" if status == "PASS" else "red"
        parts.append(
            f'Last: {_toolbar_state.last_run_name} '
            f'<style fg="{style}">{status}</style>'
        )
    if not parts:
        parts.append("Ready")
    return HTML("  |  ".join(parts))


# ── Commands ─────────────────────────────────────────────────────────

COMMANDS: dict[str, str] = {
    "goals": "List all goals (--unrun, --failed)",
    "show": "Show goal detail: show <goal>",
    "run": "Run goal(s): run <goal> ... (--unrun, --failed, --no-update-summary, --no-update-tasks, --no-knowledge-reuse, --regenerate-tasks, --hints, --reset-errors)",
    "delete": "Delete a goal: delete <goal>",
    "results": "List runs, or show detail: results [task/timestamp]",
    "errors": "Show runs with errors",
    "classify": "Classify observations in all goals",
    "orphans": "List orphan tasks (--clean to remove)",
    "hint": "Add hint: hint <goal> [<phase>] -- <text>",
    "hints": "List hints: hints <goal>",
    "unhint": "Remove hint: unhint <goal> <index>",
    "reset": "Reset phase for fresh decomposition: reset <goal> <phase>",
    "resets": "List reset phases: resets <goal>",
    "unreset": "Undo reset: unreset <goal> <phase>",
    "help": "Show available commands",
    "quit": "Exit the REPL",
}

# Commands that accept goal names as arguments
_GOAL_ARG_COMMANDS = {"show", "run", "delete", "hints", "unhint", "resets"}
# Commands that accept run paths as arguments
_RUN_ARG_COMMANDS = {"results"}


# ── Completer ────────────────────────────────────────────────────────


def _list_goal_names(config: SparkConfig) -> list[str]:
    """Return sorted goal names (without -task.json) from goal_summaries_dir."""
    gs_dir = config.goal_summaries_dir
    if gs_dir is None or not gs_dir.exists():
        return []
    return sorted(
        f.stem.removesuffix("-task")
        for f in gs_dir.glob("*-task.json")
    )


def _list_run_paths(config: SparkConfig) -> list[str]:
    """Return available run paths as ``task/timestamp`` strings."""
    runs_dir = config.runs_dir
    if runs_dir is None or not runs_dir.exists():
        return []
    paths: list[str] = []
    for task_dir in sorted(runs_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                paths.append(f"{task_dir.name}/{run_dir.name}")
    return paths


class SparkCompleter(Completer):
    """Context-aware tab completer for the Spark Runner REPL."""

    def __init__(self, config: SparkConfig) -> None:
        self._config = config

    def get_completions(
        self, document: Document, complete_event: Any,
    ) -> Any:
        text = document.text_before_cursor
        parts = text.split()

        # Complete command name
        if len(parts) == 0 or (len(parts) == 1 and not text.endswith(" ")):
            prefix = parts[0] if parts else ""
            for cmd, help_text in COMMANDS.items():
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix), display_meta=help_text)
            return

        cmd = parts[0]
        # What the user is currently typing for the argument
        current = parts[-1] if not text.endswith(" ") else ""

        # Complete flags
        if current.startswith("-"):
            flags: list[str] = []
            if cmd == "goals":
                flags = ["--unrun", "--failed"]
            elif cmd == "run":
                flags = ["--unrun", "--failed", "--no-update-summary", "--no-update-tasks", "--no-knowledge-reuse", "--regenerate-tasks", "--hints", "--reset-errors"]
            elif cmd == "orphans":
                flags = ["--clean"]
            for flag in flags:
                if flag.startswith(current):
                    yield Completion(flag, start_position=-len(current))
            return

        # Special: 'reset'/'unreset' have goal completion then phase completion
        if cmd in ("reset", "unreset"):
            at_goal_pos = (
                (len(parts) == 1 and text.endswith(" "))
                or (len(parts) == 2 and not text.endswith(" "))
            )
            if at_goal_pos:
                for name in _list_goal_names(self._config):
                    if name.startswith(current):
                        yield Completion(name, start_position=-len(current))
                return

            # Phase name position (after goal)
            goal_name = parts[1]
            from spark_runner.goals import get_phase_names

            gs_dir = self._config.goal_summaries_dir
            phases: list[str] = []
            if gs_dir is not None:
                goal_path = gs_dir / f"{goal_name}-task.json"
                phases = get_phase_names(goal_path)

            # Build typed prefix for the phase
            if text.endswith(" "):
                phase_parts = parts[2:]
            else:
                phase_parts = parts[2:-1]
            phase_typed = " ".join(phase_parts)
            partial = current if len(parts) > 2 and not text.endswith(" ") else ""
            full_prefix = (
                (phase_typed + " " + partial).strip() if partial else phase_typed
            )

            raw_len = len(full_prefix)
            if text.endswith(" ") and full_prefix:
                raw_len += 1
            start_pos = -raw_len

            for phase in phases:
                if not full_prefix or phase.lower().startswith(
                    full_prefix.lower()
                ):
                    yield Completion(phase, start_position=start_pos)
            return

        # Special: 'hint' has goal completion then phase completion
        if cmd == "hint":
            if "--" in parts:
                # After separator — no completion
                return

            at_goal_pos = (
                (len(parts) == 1 and text.endswith(" "))
                or (len(parts) == 2 and not text.endswith(" "))
            )
            if at_goal_pos:
                for name in _list_goal_names(self._config):
                    if name.startswith(current):
                        yield Completion(name, start_position=-len(current))
                return

            # Phase name position (after goal, before --)
            goal_name = parts[1]
            from spark_runner.goals import get_phase_names

            gs_dir = self._config.goal_summaries_dir
            phases: list[str] = []
            if gs_dir is not None:
                goal_path = gs_dir / f"{goal_name}-task.json"
                phases = get_phase_names(goal_path)

            # Build typed prefix for the phase
            if text.endswith(" "):
                phase_parts = parts[2:]
            else:
                phase_parts = parts[2:-1]
            phase_typed = " ".join(phase_parts)
            partial = current if len(parts) > 2 and not text.endswith(" ") else ""
            full_prefix = (
                (phase_typed + " " + partial).strip() if partial else phase_typed
            )

            # Calculate start_position to replace entire typed phase text
            raw_len = len(full_prefix)
            if text.endswith(" ") and full_prefix:
                raw_len += 1  # account for trailing space
            start_pos = -raw_len

            exact = any(p.lower() == full_prefix.lower() for p in phases)

            if exact:
                # Phase fully typed — suggest separator
                yield Completion("--", start_position=0)
            else:
                for phase in phases:
                    if not full_prefix or phase.lower().startswith(
                        full_prefix.lower()
                    ):
                        yield Completion(phase, start_position=start_pos)
                # Suggest -- for goal-level hint
                if not full_prefix:
                    yield Completion("--", start_position=0)
            return

        # Complete goal names
        if cmd in _GOAL_ARG_COMMANDS:
            already_used = set(parts[1:]) if text.endswith(" ") else set(parts[1:-1])
            for name in _list_goal_names(self._config):
                if name in already_used:
                    continue
                if name.startswith(current):
                    yield Completion(name, start_position=-len(current))
            return

        # Complete run paths
        if cmd in _RUN_ARG_COMMANDS:
            for rp in _list_run_paths(self._config):
                if rp.startswith(current):
                    yield Completion(rp, start_position=-len(current))
            return


# ── Parsing ──────────────────────────────────────────────────────────


def parse_command(line: str) -> tuple[str, list[str]]:
    """Split a REPL input line into (command, args).

    Returns ``("", [])`` for blank input.
    """
    line = line.strip()
    if not line:
        return "", []
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    return tokens[0], tokens[1:]


# ── Dispatch ─────────────────────────────────────────────────────────


def dispatch(
    cmd: str,
    args: list[str],
    config: SparkConfig,
    restore_fn: Callable[[str], str],
) -> bool:
    """Execute a REPL command. Returns False to quit, True to continue."""
    if cmd in ("quit", "exit"):
        return False

    if cmd == "help":
        _handle_help()
    elif cmd == "goals":
        _handle_goals(args, config, restore_fn)
    elif cmd == "show":
        _handle_show(args, config, restore_fn)
    elif cmd == "run":
        _handle_run(args, config)
    elif cmd == "delete":
        _handle_delete(args, config)
    elif cmd == "results":
        _handle_results(args, config)
    elif cmd == "errors":
        _handle_errors(args, config)
    elif cmd == "classify":
        _handle_classify(config)
    elif cmd == "hint":
        _handle_hint(args, config)
    elif cmd == "hints":
        _handle_hints(args, config)
    elif cmd == "unhint":
        _handle_unhint(args, config)
    elif cmd == "reset":
        _handle_reset(args, config)
    elif cmd == "resets":
        _handle_resets(args, config)
    elif cmd == "unreset":
        _handle_unreset(args, config)
    elif cmd == "orphans":
        _handle_orphans(args, config)
    else:
        print(f"Unknown command: {cmd}")
        print("Type 'help' for available commands.")

    return True


def _handle_help() -> None:
    print("\nAvailable commands:\n")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:12s} {desc}")
    print()


def _handle_goals(
    args: list[str], config: SparkConfig, restore_fn: Callable[[str], str],
) -> None:
    from spark_runner.goals import list_goals

    assert config.goal_summaries_dir is not None
    filter_unrun = "--unrun" in args
    filter_failed = "--failed" in args
    list_goals(
        config.goal_summaries_dir, restore_fn, config.runs_dir,
        filter_unrun=filter_unrun, filter_failed=filter_failed,
    )


def _handle_show(
    args: list[str], config: SparkConfig, restore_fn: Callable[[str], str],
) -> None:
    if not args:
        print("Usage: show <goal-name>")
        return

    from spark_runner.goals import show_goal_detail

    assert config.goal_summaries_dir is not None
    show_goal_detail(config.goal_summaries_dir, args[0], restore_fn)


async def _phase_failure_callback(phase_name: str, error_summary: str) -> str | None:
    """Prompt the user for a hint when a phase fails during interactive mode.

    Args:
        phase_name: Name of the failed phase.
        error_summary: Short description of the failure.

    Returns:
        The hint text if the user provided one, or ``None`` to stop.
    """
    print(f"\n  Phase \"{phase_name}\" failed.")
    print(f"  Error: {error_summary[:200]}")
    print()
    try:
        hint: str = await asyncio.to_thread(
            input, "  Enter a hint to save and retry, or press Enter to stop: "
        )
    except (EOFError, KeyboardInterrupt):
        return None
    return hint.strip() or None


def _handle_run(args: list[str], config: SparkConfig) -> None:
    from spark_runner.models import TaskSpec

    filter_unrun = "--unrun" in args
    filter_failed = "--failed" in args
    no_update_summary = "--no-update-summary" in args
    no_update_tasks = "--no-update-tasks" in args
    no_knowledge_reuse = "--no-knowledge-reuse" in args
    regenerate_tasks = "--regenerate-tasks" in args
    enable_hints = "--hints" in args
    reset_errors = "--reset-errors" in args
    goal_names = [a for a in args if not a.startswith("--")]

    tasks: list[TaskSpec] = []

    # Resolve named goals
    if goal_names and config.goal_summaries_dir is not None:
        for name in goal_names:
            goal_path = config.goal_summaries_dir / f"{name}-task.json"
            if not goal_path.exists():
                print(f"Goal not found: {name}")
                return
            tasks.append(TaskSpec(goal_path=goal_path))

    # Discover filtered goals
    if (filter_unrun or filter_failed) and config.goal_summaries_dir is not None:
        from spark_runner.goals import get_last_run_info

        seen = {t.goal_path for t in tasks if t.goal_path}
        for gf in sorted(config.goal_summaries_dir.glob("*-task.json")):
            if gf in seen:
                continue
            task_name = gf.stem.removesuffix("-task")
            run_info = get_last_run_info(config.runs_dir, task_name)
            if filter_unrun and run_info is None:
                tasks.append(TaskSpec(goal_path=gf))
            elif filter_failed and run_info is not None and run_info[1] == "errors":
                tasks.append(TaskSpec(goal_path=gf))

    if not tasks:
        print("No goals specified. Usage: run <goal> [goal...] or run --unrun/--failed")
        return

    # Auto-reset errored phases from the last run if requested
    if reset_errors and config.runs_dir is not None:
        from spark_runner.goals import reset_errored_phases

        for task_spec in tasks:
            if task_spec.goal_path is not None:
                reset_errored_phases(task_spec.goal_path, config.runs_dir)

    # Build per-run config with flag overrides
    run_config = config
    overrides: dict[str, bool] = {}
    if no_update_summary:
        overrides["update_summary"] = False
    if no_update_tasks:
        overrides["update_tasks"] = False
    if no_knowledge_reuse:
        overrides["knowledge_reuse"] = False
    if regenerate_tasks:
        overrides["regenerate_tasks"] = True
    if overrides:
        run_config = dataclasses.replace(config, **overrides)

    callback = _phase_failure_callback if enable_hints else None

    print(f"Running {len(tasks)} goal(s)...\n")
    from spark_runner.models import RunResult
    from spark_runner.orchestrator import run_multiple, run_single

    if len(tasks) == 1:
        result: RunResult = asyncio.run(
            run_single(tasks[0], run_config, on_phase_failure=callback)
        )
        _toolbar_state.last_run_name = result.task_name
        _toolbar_state.last_run_status = "PASS" if result.all_phases_succeeded else "FAIL"
    else:
        # Auto-close browsers between goals so the user isn't prompted
        multi_config = dataclasses.replace(run_config, auto_close=True)
        results: list[RunResult] = asyncio.run(
            run_multiple(tasks, multi_config, on_phase_failure=callback)
        )
        if results:
            last = results[-1]
            _toolbar_state.last_run_name = last.task_name
            _toolbar_state.last_run_status = "PASS" if last.all_phases_succeeded else "FAIL"


def _handle_delete(args: list[str], config: SparkConfig) -> None:
    if not args:
        print("Usage: delete <goal-name>")
        return

    from spark_runner.goals import delete_goal

    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None
    delete_goal(config.goal_summaries_dir, config.tasks_dir, args[0])


def _handle_results(args: list[str], config: SparkConfig) -> None:
    from spark_runner.results import format_run_detail, format_run_summary, get_run_detail, list_runs

    assert config.runs_dir is not None

    if args:
        # Show detail for a specific run
        run_path = config.runs_dir / args[0]
        if not run_path.is_dir():
            print(f"Run not found: {args[0]}")
            return
        detail = get_run_detail(run_path)
        print(format_run_detail(detail))
        return

    runs = list_runs(config.runs_dir)
    if not runs:
        print("No runs found.")
        return
    print(f"Found {len(runs)} run(s):\n")
    for r in runs:
        print(format_run_summary(r))


def _handle_errors(args: list[str], config: SparkConfig) -> None:
    from spark_runner.results import format_run_summary, list_runs

    assert config.runs_dir is not None
    runs = [r for r in list_runs(config.runs_dir) if r.has_errors]
    if not runs:
        print("No runs with errors found.")
        return
    print(f"Found {len(runs)} run(s) with errors:\n")
    for r in runs:
        print(format_run_summary(r))


def _handle_classify(config: SparkConfig) -> None:
    import anthropic as anth

    from spark_runner.classification import classify_observations
    from spark_runner.goals import classify_existing_goals

    client = anth.Anthropic()
    model_config = config.get_model("classification")

    assert config.goal_summaries_dir is not None
    classify_existing_goals(
        config.goal_summaries_dir,
        lambda prompt, obs: classify_observations(prompt, obs, client, model_config),
    )


def _handle_hint(args: list[str], config: SparkConfig) -> None:
    """Add a hint: ``hint <goal> [<phase>] -- <text>``."""
    if "--" not in args:
        print("Usage: hint <goal> [<phase>] -- <text>")
        return

    sep_idx: int = args.index("--")
    before: list[str] = args[:sep_idx]
    after: list[str] = args[sep_idx + 1:]

    if len(before) < 1 or not after:
        print("Usage: hint <goal> [<phase>] -- <text>")
        return

    goal_name: str = before[0]
    phase_name: str = " ".join(before[1:]) if len(before) >= 2 else ""
    text: str = " ".join(after)

    assert config.goal_summaries_dir is not None
    goal_path: Path = config.goal_summaries_dir / f"{goal_name}-task.json"
    if not goal_path.exists():
        print(f"Goal not found: {goal_name}")
        return

    from spark_runner.goals import get_phase_names, save_hint

    # Validate phase name if provided
    if phase_name:
        valid_phases: list[str] = get_phase_names(goal_path)
        matched: list[str] = [
            p for p in valid_phases if p.lower() == phase_name.lower()
        ]
        if not matched:
            print(f"Unknown phase: \"{phase_name}\"")
            if valid_phases:
                print(f"Available phases: {', '.join(valid_phases)}")
            else:
                print("This goal has no subtask phases defined.")
            return
        phase_name = matched[0]  # use canonical case

    save_hint(goal_path, phase_name, text)
    if phase_name:
        print(f"Hint saved for phase \"{phase_name}\" in goal \"{goal_name}\".")
    else:
        print(f"Goal-level hint saved for \"{goal_name}\".")


def _handle_hints(args: list[str], config: SparkConfig) -> None:
    """List hints for a goal: ``hints <goal>``."""
    if not args:
        print("Usage: hints <goal>")
        return

    goal_name: str = args[0]
    assert config.goal_summaries_dir is not None
    goal_path: Path = config.goal_summaries_dir / f"{goal_name}-task.json"
    if not goal_path.exists():
        print(f"Goal not found: {goal_name}")
        return

    from spark_runner.goals import load_hints

    hints: list[dict[str, str]] = load_hints(goal_path)
    if not hints:
        print(f"No hints for goal \"{goal_name}\".")
        return

    print(f"Hints for \"{goal_name}\":\n")
    for i, h in enumerate(hints):
        label: str = h["phase"] if h["phase"] else "Goal"
        print(f"  {i}. [{label}] {h['text']}")
    print()


def _handle_unhint(args: list[str], config: SparkConfig) -> None:
    """Remove a hint by index: ``unhint <goal> <index>``."""
    if len(args) < 2:
        print("Usage: unhint <goal> <index>")
        return

    goal_name: str = args[0]
    try:
        index: int = int(args[1])
    except ValueError:
        print(f"Invalid index: {args[1]}")
        return

    assert config.goal_summaries_dir is not None
    goal_path: Path = config.goal_summaries_dir / f"{goal_name}-task.json"
    if not goal_path.exists():
        print(f"Goal not found: {goal_name}")
        return

    from spark_runner.goals import remove_hint

    if remove_hint(goal_path, index):
        print(f"Hint {index} removed from goal \"{goal_name}\".")
    else:
        print(f"Invalid hint index: {index}")


def _handle_reset(args: list[str], config: SparkConfig) -> None:
    """Mark a phase for fresh decomposition: ``reset <goal> <phase>``."""
    if len(args) < 2:
        print("Usage: reset <goal> <phase>")
        return

    goal_name: str = args[0]
    phase_name: str = " ".join(args[1:])

    assert config.goal_summaries_dir is not None
    goal_path: Path = config.goal_summaries_dir / f"{goal_name}-task.json"
    if not goal_path.exists():
        print(f"Goal not found: {goal_name}")
        return

    from spark_runner.goals import reset_phase

    if reset_phase(goal_path, phase_name):
        print(f"Phase \"{phase_name}\" marked for fresh decomposition in goal \"{goal_name}\".")
    else:
        from spark_runner.goals import get_phase_names

        valid_phases: list[str] = get_phase_names(goal_path)
        print(f"Unknown phase: \"{phase_name}\"")
        if valid_phases:
            print(f"Available phases: {', '.join(valid_phases)}")
        else:
            print("This goal has no subtask phases defined.")


def _handle_resets(args: list[str], config: SparkConfig) -> None:
    """List reset phases for a goal: ``resets <goal>``."""
    if not args:
        print("Usage: resets <goal>")
        return

    goal_name: str = args[0]
    assert config.goal_summaries_dir is not None
    goal_path: Path = config.goal_summaries_dir / f"{goal_name}-task.json"
    if not goal_path.exists():
        print(f"Goal not found: {goal_name}")
        return

    from spark_runner.goals import get_reset_phases

    reset_phases: list[str] = get_reset_phases(goal_path)
    if not reset_phases:
        print(f"No phases reset for goal \"{goal_name}\".")
        return

    print(f"Reset phases for \"{goal_name}\":\n")
    for rp in reset_phases:
        print(f"  - {rp}")
    print()


def _handle_unreset(args: list[str], config: SparkConfig) -> None:
    """Remove a phase from the reset list: ``unreset <goal> <phase>``."""
    if len(args) < 2:
        print("Usage: unreset <goal> <phase>")
        return

    goal_name: str = args[0]
    phase_name: str = " ".join(args[1:])

    assert config.goal_summaries_dir is not None
    goal_path: Path = config.goal_summaries_dir / f"{goal_name}-task.json"
    if not goal_path.exists():
        print(f"Goal not found: {goal_name}")
        return

    from spark_runner.goals import unreset_phase

    if unreset_phase(goal_path, phase_name):
        print(f"Phase \"{phase_name}\" unreset in goal \"{goal_name}\".")
    else:
        from spark_runner.goals import get_reset_phases

        current: list[str] = get_reset_phases(goal_path)
        print(f"Phase \"{phase_name}\" is not in the reset list.")
        if current:
            print(f"Currently reset: {', '.join(current)}")
        else:
            print("No phases are currently reset.")


def _handle_orphans(args: list[str], config: SparkConfig) -> None:
    from spark_runner.storage import clean_orphan_tasks, find_orphan_tasks

    assert config.tasks_dir is not None
    assert config.goal_summaries_dir is not None
    if "--clean" in args:
        clean_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)
    else:
        find_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)


# ── Main loop ────────────────────────────────────────────────────────


def _refresh_toolbar_goal_count(config: SparkConfig) -> None:
    """Update the toolbar goal count from the goal summaries directory."""
    gs_dir = config.goal_summaries_dir
    if gs_dir is not None and gs_dir.exists():
        _toolbar_state.goal_count = len(list(gs_dir.glob("*-task.json")))
    else:
        _toolbar_state.goal_count = 0


def interactive_loop(config: SparkConfig) -> None:
    """Run the interactive REPL."""
    from spark_runner.orchestrator import _make_restore_fn

    restore_fn = _make_restore_fn(config)
    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None

    _refresh_toolbar_goal_count(config)

    history_path = config.data_dir / ".repl_history"
    session: PromptSession[str] = PromptSession(
        completer=SparkCompleter(config),
        history=FileHistory(str(history_path)),
        bottom_toolbar=_bottom_toolbar,
    )

    print("Spark Runner interactive mode. Type 'help' for commands, Ctrl-D to exit.\n")

    while True:
        try:
            line = session.prompt("spark> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        cmd, args = parse_command(line)
        if not cmd:
            continue

        try:
            if not dispatch(cmd, args, config, restore_fn):
                print("Goodbye.")
                break
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")

        # Refresh toolbar state after every command.
        _refresh_toolbar_goal_count(config)
