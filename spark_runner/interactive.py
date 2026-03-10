"""REPL-style interactive mode for Spark Runner."""

from __future__ import annotations

import asyncio
import dataclasses
import shlex
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

from spark_runner.models import SparkConfig


# ── Commands ─────────────────────────────────────────────────────────

COMMANDS: dict[str, str] = {
    "goals": "List all goals (--unrun, --failed)",
    "show": "Show goal detail: show <goal>",
    "run": "Run goal(s): run <goal> ... (--unrun, --failed, --no-update-summary, --no-update-tasks, --no-knowledge-reuse)",
    "delete": "Delete a goal: delete <goal>",
    "results": "List runs, or show detail: results [task/timestamp]",
    "errors": "Show runs with errors",
    "classify": "Classify observations in all goals",
    "orphans": "List orphan tasks (--clean to remove)",
    "help": "Show available commands",
    "quit": "Exit the REPL",
}

# Commands that accept goal names as arguments
_GOAL_ARG_COMMANDS = {"show", "run", "delete"}
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
                flags = ["--unrun", "--failed", "--no-update-summary", "--no-update-tasks", "--no-knowledge-reuse"]
            elif cmd == "orphans":
                flags = ["--clean"]
            for flag in flags:
                if flag.startswith(current):
                    yield Completion(flag, start_position=-len(current))
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


def _handle_run(args: list[str], config: SparkConfig) -> None:
    from spark_runner.models import TaskSpec

    filter_unrun = "--unrun" in args
    filter_failed = "--failed" in args
    no_update_summary = "--no-update-summary" in args
    no_update_tasks = "--no-update-tasks" in args
    no_knowledge_reuse = "--no-knowledge-reuse" in args
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

    # Build per-run config with flag overrides
    run_config = config
    overrides: dict[str, bool] = {}
    if no_update_summary:
        overrides["update_summary"] = False
    if no_update_tasks:
        overrides["update_tasks"] = False
    if no_knowledge_reuse:
        overrides["knowledge_reuse"] = False
    if overrides:
        run_config = dataclasses.replace(config, **overrides)

    print(f"Running {len(tasks)} goal(s)...\n")
    from spark_runner.orchestrator import run_multiple, run_single

    if len(tasks) == 1:
        asyncio.run(run_single(tasks[0], run_config))
    else:
        asyncio.run(run_multiple(tasks, run_config))


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


def _handle_orphans(args: list[str], config: SparkConfig) -> None:
    from spark_runner.storage import clean_orphan_tasks, find_orphan_tasks

    assert config.tasks_dir is not None
    assert config.goal_summaries_dir is not None
    if "--clean" in args:
        clean_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)
    else:
        find_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)


# ── Main loop ────────────────────────────────────────────────────────


def interactive_loop(config: SparkConfig) -> None:
    """Run the interactive REPL."""
    from spark_runner.orchestrator import _make_restore_fn

    restore_fn = _make_restore_fn(config)
    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None

    history_path = config.data_dir / ".repl_history"
    session: PromptSession[str] = PromptSession(
        completer=SparkCompleter(config),
        history=FileHistory(str(history_path)),
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
