"""Click-based CLI with subcommands."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from sparky_runner.config import build_config
from sparky_runner.models import SparkyConfig, TaskSpec


def _parse_model_overrides(values: tuple[str, ...]) -> dict[str, str]:
    """Parse ``PURPOSE=MODEL_ID`` strings into a dict."""
    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise click.BadParameter(f"Expected PURPOSE=MODEL_ID, got: {v}")
        purpose, model_id = v.split("=", 1)
        result[purpose.strip()] = model_id.strip()
    return result


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """SparkyAI Browser Automation Runner."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _validate_url(
    ctx: click.Context, param: click.Parameter, value: str | None,
) -> str | None:
    """Reject values that look like another flag (e.g. ``-u -p`` with no URL)."""
    if value is not None and value.startswith("-"):
        raise click.BadParameter(
            f"Got '{value}' which looks like a flag, not a URL. "
            "Please provide a URL value after -u/--url.",
            param_hint="'-u' / '--url'",
        )
    return value


@cli.command()
@click.option("-p", "--prompt", "prompts", multiple=True, help="Task prompt(s)")
@click.option("-u", "--url", "base_url", default=None, callback=_validate_url,
              help="Base URL")
@click.option("--no-update-summary", is_flag=True, help="Don't update goal summaries")
@click.option("--no-update-tasks", is_flag=True, help="Don't overwrite task files")
@click.option("--no-knowledge-reuse", is_flag=True, help="Don't reuse prior knowledge")
@click.option("--auto-close", is_flag=True, help="Close browser automatically")
@click.option("--headless", is_flag=True, help="Run browser in headless mode")
@click.option("--credential-profile", default=None, help="Credential profile name")
@click.option("--model", "model_strs", multiple=True, help="PURPOSE=MODEL_ID override")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Config file path")
@click.option("--shared-session", is_flag=True, help="Share browser session across tasks")
@click.option("--parallel", type=int, default=1, help="Parallel execution count")
@click.argument("goal_files", nargs=-1, type=click.Path())
def run(
    prompts: tuple[str, ...],
    base_url: str | None,
    no_update_summary: bool,
    no_update_tasks: bool,
    no_knowledge_reuse: bool,
    auto_close: bool,
    headless: bool,
    credential_profile: str | None,
    model_strs: tuple[str, ...],
    data_dir: str | None,
    config_path: str | None,
    shared_session: bool,
    parallel: int,
    goal_files: tuple[str, ...],
) -> None:
    """Run browser automation tasks."""
    model_overrides = _parse_model_overrides(model_strs) if model_strs else None

    config: SparkyConfig = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
        base_url=base_url,
        credential_profile=credential_profile,
        model_overrides=model_overrides,
        headless=headless,
        auto_close=auto_close,
        update_summary=not no_update_summary,
        update_tasks=not no_update_tasks,
        knowledge_reuse=not no_knowledge_reuse,
    )
    config.ensure_dirs()

    # Build task specs
    tasks: list[TaskSpec] = []
    for p in prompts:
        tasks.append(TaskSpec(
            prompt=p,
            credential_profile=credential_profile or "default",
        ))
    for gf in goal_files:
        gf_path = Path(gf)
        if not gf_path.exists():
            raise click.BadParameter(f"Goal file not found: {gf}", param_hint="'GOAL_FILES'")
        tasks.append(TaskSpec(
            goal_path=gf_path,
            credential_profile=credential_profile or "default",
        ))

    if not tasks:
        # Interactive mode
        prompt_text = click.prompt("Enter your task", type=str)
        tasks.append(TaskSpec(
            prompt=prompt_text,
            credential_profile=credential_profile or "default",
        ))

    print("=" * 60)
    print("  SparkyAI Browser Automation")
    print(f"  Target: {config.base_url}")
    print("=" * 60)
    print()

    from sparky_runner.orchestrator import run_multiple, run_single

    if len(tasks) == 1 and not shared_session:
        asyncio.run(run_single(tasks[0], config))
    else:
        asyncio.run(run_multiple(tasks, config, shared_session=shared_session, parallel=parallel))


# ── Goals subcommands ─────────────────────────────────────────────────


@cli.group()
def goals() -> None:
    """Manage goals."""


@goals.command("list")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def goals_list(data_dir: str | None, config_path: str | None) -> None:
    """List all existing goals."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.goals import list_goals
    from sparky_runner.orchestrator import _make_restore_fn

    assert config.goal_summaries_dir is not None
    list_goals(config.goal_summaries_dir, _make_restore_fn(config))


@goals.command("show")
@click.argument("goal_name")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def goals_show(goal_name: str, data_dir: str | None, config_path: str | None) -> None:
    """Show details for a specific goal."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.goals import show_goal_detail
    from sparky_runner.orchestrator import _make_restore_fn

    assert config.goal_summaries_dir is not None
    show_goal_detail(config.goal_summaries_dir, goal_name, _make_restore_fn(config))


@goals.command("delete")
@click.argument("goal_name")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def goals_delete(
    goal_name: str, force: bool, data_dir: str | None, config_path: str | None
) -> None:
    """Delete a goal and its unreferenced task files."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.goals import delete_goal

    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None
    delete_goal(config.goal_summaries_dir, config.tasks_dir, goal_name, force=force)


@goals.command("classify")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def goals_classify(data_dir: str | None, config_path: str | None) -> None:
    """Classify observations in all existing goal summaries."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    import anthropic as anth
    from sparky_runner.classification import classify_observations
    from sparky_runner.goals import classify_existing_goals

    client = anth.Anthropic()
    model_config = config.get_model("classification")

    assert config.goal_summaries_dir is not None
    classify_existing_goals(
        config.goal_summaries_dir,
        lambda prompt, obs: classify_observations(prompt, obs, client, model_config),
    )


@goals.command("orphans")
@click.option("--clean", is_flag=True, help="Delete orphan task files")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def goals_orphans(clean: bool, data_dir: str | None, config_path: str | None) -> None:
    """List or clean orphan task files."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.storage import clean_orphan_tasks, find_orphan_tasks

    assert config.tasks_dir is not None
    assert config.goal_summaries_dir is not None
    if clean:
        clean_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)
    else:
        find_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)


# ── Results subcommands ───────────────────────────────────────────────


@cli.group()
def results() -> None:
    """View and manage run results."""


@results.command("list")
@click.option("--task", "task_name", default=None, help="Filter by task name")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def results_list(task_name: str | None, data_dir: str | None, config_path: str | None) -> None:
    """List all runs."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.results import format_run_summary, list_runs

    assert config.runs_dir is not None
    runs = list_runs(config.runs_dir, task_name)
    if not runs:
        print("No runs found.")
        return
    print(f"Found {len(runs)} run(s):\n")
    for r in runs:
        print(format_run_summary(r))


@results.command("show")
@click.argument("run_path", type=click.Path(exists=True))
def results_show(run_path: str) -> None:
    """Show full detail for a run."""
    from sparky_runner.results import format_run_detail, get_run_detail

    detail = get_run_detail(Path(run_path))
    print(format_run_detail(detail))


@results.command("errors")
@click.option("--task", "task_name", default=None, help="Filter by task name")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def results_errors(task_name: str | None, data_dir: str | None, config_path: str | None) -> None:
    """Show only runs with errors."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.results import format_run_summary, list_runs

    assert config.runs_dir is not None
    runs = [r for r in list_runs(config.runs_dir, task_name) if r.has_errors]
    if not runs:
        print("No runs with errors found.")
        return
    print(f"Found {len(runs)} run(s) with errors:\n")
    for r in runs:
        print(format_run_summary(r))


@results.command("screenshots")
@click.argument("run_path", type=click.Path(exists=True))
def results_screenshots(run_path: str) -> None:
    """List screenshots for a run."""
    from sparky_runner.results import get_run_detail

    detail = get_run_detail(Path(run_path))
    all_ss = list(detail.screenshots)
    for phase in detail.phases:
        all_ss.extend(phase.screenshots)

    if not all_ss:
        print("No screenshots found for this run.")
        return

    print(f"Screenshots ({len(all_ss)}):")
    for ss in all_ss:
        parts = [f"  {ss.path.name}"]
        if ss.event_type:
            parts.append(f"[{ss.event_type}]")
        if ss.phase_name:
            parts.append(f"phase: {ss.phase_name}")
        print(" ".join(parts))


# ── Generate goals ────────────────────────────────────────────────────


@cli.command("generate-goals")
@click.argument("source_path", type=click.Path())
@click.option("--branch", default="main", help="Git branch for repo URLs")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def generate_goals_cmd(
    source_path: str,
    branch: str,
    output_dir: str | None,
    data_dir: str | None,
    config_path: str | None,
) -> None:
    """Generate test goals from frontend source code."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.goal_generator import generate_goals_from_source

    out = Path(output_dir) if output_dir else config.goal_summaries_dir
    assert out is not None
    asyncio.run(generate_goals_from_source(Path(source_path), out, config, branch))


# ── Record trajectory ─────────────────────────────────────────────────


@cli.command("record")
@click.option("--url", default=None, help="Starting URL")
@click.option("--data-dir", type=click.Path(), default=None,
              help="Sparky Runner home directory (tasks, goal summaries, runs). Default: ~/sparky_runner")
@click.option("--config", "config_path", type=click.Path(), default=None)
def record_cmd(
    url: str | None,
    data_dir: str | None,
    config_path: str | None,
) -> None:
    """Record a user demonstration and generate a goal."""
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from sparky_runner.trajectory_recorder import record_and_generate_goal

    base = url or config.base_url
    asyncio.run(record_and_generate_goal(base, config))


# ── Legacy CLI compatibility ──────────────────────────────────────────


def legacy_main() -> None:
    """Backward-compatible entry point for ``python sparky_runner.py`` usage.

    Translates the old-style sys.argv flags into Click-compatible invocations.
    """
    args: list[str] = sys.argv[1:]

    # Handle legacy flags that map to subcommands
    if "--list-goals" in args:
        sys.argv = [sys.argv[0], "goals", "list"]
        cli()
        return
    if "--classify-existing" in args:
        sys.argv = [sys.argv[0], "goals", "classify"]
        cli()
        return
    if "--find-orphans" in args:
        sys.argv = [sys.argv[0], "goals", "orphans"]
        cli()
        return
    if "--clean-orphans" in args:
        sys.argv = [sys.argv[0], "goals", "orphans", "--clean"]
        cli()
        return

    # Default: treat as "run" subcommand
    new_args: list[str] = [sys.argv[0], "run"] + args
    sys.argv = new_args
    cli()
