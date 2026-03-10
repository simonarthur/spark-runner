"""Interactive terminal UI for browsing and acting on goals."""

from __future__ import annotations

import asyncio
from typing import Any

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import radiolist_dialog

from spark_runner.goals import get_goal_summaries, show_goal_detail, delete_goal
from spark_runner.models import GoalInfo, SparkConfig, TaskSpec


def _format_goal_entry(goal: GoalInfo) -> str:
    """Format a goal as a single menu line with status indicator and description.

    Format: ``name  [status]  description``
    """
    if goal.last_run_status == "ok":
        status = "[ok]"
    elif goal.last_run_status == "errors":
        status = "[errors]"
    elif goal.last_run_status is not None:
        status = f"[{goal.last_run_status}]"
    else:
        status = "[never run]"

    desc = goal.main_task
    if len(desc) > 60:
        desc = desc[:57] + "..."

    return f"{goal.name}  {status}  {desc}"


def _show_action_menu(goals: list[GoalInfo]) -> str | None:
    """Display the main action menu and return the chosen action string.

    Returns one of ``"run"``, ``"show"``, ``"delete"``, ``"refresh"``,
    or ``None`` if the user presses Escape / cancels.
    """
    count = len(goals)
    values: list[tuple[str, str]] = [
        ("run", f"Run a goal ({count} available)"),
        ("show", "Show goal details"),
        ("delete", "Delete a goal"),
        ("refresh", "Refresh goal list"),
        ("quit", "Quit"),
    ]
    result: str | None = radiolist_dialog(
        title=HTML("<b>Spark Runner Interactive</b>"),
        text=f"{count} goal(s) found. Choose an action:",
        values=values,
    ).run()
    if result == "quit":
        return None
    return result


def _show_goal_selector(
    goals: list[GoalInfo], title: str = "Select a goal",
) -> GoalInfo | None:
    """Display a goal selection menu and return the chosen :class:`GoalInfo`.

    Returns ``None`` if the list is empty or the user cancels.
    """
    if not goals:
        print("No goals available.")
        return None

    values: list[tuple[int, str]] = [
        (i, _format_goal_entry(g)) for i, g in enumerate(goals)
    ]
    result: int | None = radiolist_dialog(
        title=HTML(f"<b>{title}</b>"),
        text="Use arrow keys to navigate, Enter to select, Esc to cancel.",
        values=values,
    ).run()
    if result is None:
        return None
    return goals[result]


def interactive_loop(config: SparkConfig) -> None:
    """Main interactive loop: fetch goals, show menus, dispatch actions."""
    from spark_runner.orchestrator import _make_restore_fn

    restore_fn = _make_restore_fn(config)
    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None

    while True:
        goals = get_goal_summaries(
            config.goal_summaries_dir, restore_fn, config.runs_dir,
        )

        action = _show_action_menu(goals)
        if action is None:
            print("Goodbye.")
            break

        if action == "refresh":
            continue

        try:
            if action == "run":
                _handle_run(goals, config)
            elif action == "show":
                _handle_show(goals, config, restore_fn)
            elif action == "delete":
                _handle_delete(goals, config)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


def _handle_run(
    goals: list[GoalInfo], config: SparkConfig,
) -> None:
    """Prompt for a goal and run it."""
    goal = _show_goal_selector(goals, title="Select a goal to run")
    if goal is None:
        return

    from spark_runner.orchestrator import run_single

    task = TaskSpec(goal_path=goal.file_path)
    print(f"\nRunning goal: {goal.name}\n")
    asyncio.run(run_single(task, config))


def _handle_show(
    goals: list[GoalInfo],
    config: SparkConfig,
    restore_fn: Any,
) -> None:
    """Prompt for a goal and show its details."""
    goal = _show_goal_selector(goals, title="Select a goal to view")
    if goal is None:
        return

    assert config.goal_summaries_dir is not None
    print()
    show_goal_detail(config.goal_summaries_dir, goal.name, restore_fn)
    input("\nPress Enter to continue...")


def _handle_delete(
    goals: list[GoalInfo], config: SparkConfig,
) -> None:
    """Prompt for a goal and delete it."""
    goal = _show_goal_selector(goals, title="Select a goal to delete")
    if goal is None:
        return

    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None
    print()
    delete_goal(config.goal_summaries_dir, config.tasks_dir, goal.name)
