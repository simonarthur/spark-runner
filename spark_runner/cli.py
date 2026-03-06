"""Click-based CLI with subcommands."""

from __future__ import annotations

import asyncio
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import click
from click.shell_completion import CompletionItem

from spark_runner.config import build_config, resolve_config_path, run_setup_wizard
from spark_runner.models import SparkConfig, TaskSpec


def _is_interactive() -> bool:
    """Return True when stdin is attached to a terminal."""
    return sys.stdin.isatty()


def _parse_model_overrides(values: tuple[str, ...]) -> dict[str, str]:
    """Parse ``PURPOSE=MODEL_ID`` strings into a dict."""
    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise click.BadParameter(f"Expected PURPOSE=MODEL_ID, got: {v}")
        purpose, model_id = v.split("=", 1)
        result[purpose.strip()] = model_id.strip()
    return result


def _resolve_goal_file(name: str, goal_summaries_dir: Path | None) -> Path:
    """Resolve a goal file argument to an existing path.

    Tries in order:
    1. Literal path (absolute or relative to cwd)
    2. Name inside ``goal_summaries_dir`` (with or without ``.json``)
    """
    literal = Path(name)
    if literal.exists():
        return literal

    if goal_summaries_dir is not None:
        # Try as-is inside goal_summaries_dir
        in_dir = goal_summaries_dir / name
        if in_dir.exists():
            return in_dir
        # Try appending .json
        with_ext = goal_summaries_dir / f"{name}.json"
        if with_ext.exists():
            return with_ext

    # Collect available goal names for suggestions
    hint = ""
    if goal_summaries_dir is not None and goal_summaries_dir.exists():
        available = [
            f.name for f in sorted(goal_summaries_dir.iterdir())
            if f.is_file() and f.suffix == ".json"
        ]
        stems = [
            f.stem for f in sorted(goal_summaries_dir.iterdir())
            if f.is_file() and f.suffix == ".json"
        ]
        candidates = available + stems
        close = get_close_matches(name, candidates, n=3, cutoff=0.6)
        if close:
            # Deduplicate (e.g. "foo.json" and "foo" might both match)
            seen: set[str] = set()
            unique: list[str] = []
            for c in close:
                key = c.removesuffix(".json")
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            listing = "\n".join(f"  {c}" for c in unique)
            hint = f"\n\nDid you mean:\n{listing}"
        elif available:
            listing = "\n".join(f"  {n}" for n in available)
            hint = f"\n\nAvailable goals:\n{listing}"

    raise click.BadParameter(
        f"Goal file not found: {name}{hint}",
        param_hint="'GOAL_FILES'",
    )


def _resolve_run_path(name: str | None, runs_dir: Path | None) -> Path:
    """Resolve a run path argument to an existing directory.

    Tries in order:
    1. If *name* is ``None``, list available runs and exit with an error.
    2. Literal path (absolute or relative to cwd).
    3. Relative path inside ``runs_dir``.

    When no match is found, lists available runs in the error message.
    """
    available: list[str] = []
    if runs_dir is not None and runs_dir.exists():
        for task_dir in sorted(runs_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for run_dir in sorted(task_dir.iterdir(), reverse=True):
                if run_dir.is_dir():
                    available.append(f"{task_dir.name}/{run_dir.name}")

    if name is None:
        if available:
            listing = "\n".join(f"  {r}" for r in available)
            raise click.UsageError(
                f"Missing argument 'RUN_PATH'.\n\nAvailable runs:\n{listing}"
            )
        raise click.UsageError("Missing argument 'RUN_PATH'. No runs found.")

    # Try literal path
    literal = Path(name)
    if literal.exists() and literal.is_dir():
        return literal

    # Try inside runs_dir
    if runs_dir is not None:
        in_dir = runs_dir / name
        if in_dir.exists() and in_dir.is_dir():
            return in_dir

    hint = ""
    if available:
        listing = "\n".join(f"  {r}" for r in available)
        hint = f"\n\nAvailable runs:\n{listing}"
    raise click.BadParameter(
        f"Run not found: {name}{hint}",
        param_hint="'RUN_PATH'",
    )


def _file_mtime_label(path: Path) -> str:
    """Return a human-readable modification time for a path."""
    from datetime import datetime

    try:
        mtime = path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _build_config_for_completion(ctx: click.Context) -> SparkConfig:
    """Build a config from whatever context is available during tab completion."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    return build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )


def _complete_run_path(
    ctx: click.Context, param: click.Parameter, incomplete: str,
) -> list[CompletionItem]:
    """Tab-complete run paths (task/timestamp) from runs_dir."""
    try:
        config = _build_config_for_completion(ctx)
        runs_dir = config.runs_dir
    except Exception:
        return []

    if runs_dir is None or not runs_dir.exists():
        return []

    items: list[CompletionItem] = []
    for task_dir in sorted(runs_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                name = f"{task_dir.name}/{run_dir.name}"
                if name.startswith(incomplete):
                    items.append(CompletionItem(
                        name, help=_file_mtime_label(run_dir),
                    ))
    return items


def _complete_goal_file(
    ctx: click.Context, param: click.Parameter, incomplete: str,
) -> list[CompletionItem]:
    """Tab-complete goal names from goal_summaries_dir."""
    try:
        config = _build_config_for_completion(ctx)
        gs_dir = config.goal_summaries_dir
    except Exception:
        return []

    if gs_dir is None or not gs_dir.exists():
        return []

    items: list[CompletionItem] = []
    for f in sorted(gs_dir.iterdir()):
        if f.is_file() and f.suffix == ".json":
            mtime = _file_mtime_label(f)
            # Offer both with and without .json
            if f.stem.startswith(incomplete):
                items.append(CompletionItem(
                    f.stem, help=mtime,
                ))
            if f.name.startswith(incomplete):
                items.append(CompletionItem(
                    f.name, help=mtime,
                ))
    return items


def _complete_goal_name(
    ctx: click.Context, param: click.Parameter, incomplete: str,
) -> list[CompletionItem]:
    """Tab-complete goal names (without .json) from goal_summaries_dir."""
    try:
        config = _build_config_for_completion(ctx)
        gs_dir = config.goal_summaries_dir
    except Exception:
        return []

    if gs_dir is None or not gs_dir.exists():
        return []

    return [
        CompletionItem(f.stem, help=_file_mtime_label(f))
        for f in sorted(gs_dir.iterdir())
        if f.is_file() and f.suffix == ".json" and f.stem.startswith(incomplete)
    ]


def _get_data_dir(ctx: click.Context) -> str | None:
    """Walk up the context chain to find --data-dir."""
    while ctx is not None:
        if ctx.params.get("data_dir") is not None:
            return ctx.params["data_dir"]
        ctx = ctx.parent  # type: ignore[assignment]
    return None


def _get_config_path(ctx: click.Context) -> str | None:
    """Walk up the context chain to find --config."""
    while ctx is not None:
        if ctx.params.get("config_path") is not None:
            return ctx.params["config_path"]
        ctx = ctx.parent  # type: ignore[assignment]
    return None


@click.group(invoke_without_command=True)
@click.option("--data-dir", type=click.Path(), default=None,
              envvar="SPARK_RUNNER_DATA_DIR",
              help="Spark Runner home directory (tasks, goal summaries, runs). Default: ~/spark_runner")
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Config file path")
@click.pass_context
def cli(ctx: click.Context, data_dir: str | None, config_path: str | None) -> None:
    """Spark Runner – Browser Automation."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["config_path"] = config_path
    if ctx.invoked_subcommand is None:
        resolved_cfg = resolve_config_path(
            config_path=Path(config_path) if config_path else None,
            data_dir=Path(data_dir) if data_dir else None,
        )
        if not resolved_cfg.exists():
            if click.confirm(
                "spark-runner configuration not found. Set up spark-runner now?",
                default=True,
            ):
                ctx.invoke(init, force=False)
                return
        click.echo(ctx.get_help())


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite existing config without asking")
@click.pass_context
def init(ctx: click.Context, force: bool) -> None:
    """Set up Spark Runner configuration interactively."""
    data_dir = _get_data_dir(ctx)
    config_path_str = _get_config_path(ctx)
    config_path = resolve_config_path(
        config_path=Path(config_path_str) if config_path_str else None,
        data_dir=Path(data_dir) if data_dir else None,
    )

    if config_path.exists() and not force:
        if not click.confirm(
            f"Config already exists at {config_path}. Overwrite?"
        ):
            click.echo("Aborted.")
            return

    run_setup_wizard(config_path)


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
              envvar="SPARK_RUNNER_BASE_URL", help="Base URL")
@click.option("--no-update-summary", is_flag=True, help="Don't update goal summaries")
@click.option("--no-update-tasks", is_flag=True, help="Don't overwrite task files")
@click.option("--no-knowledge-reuse", is_flag=True, help="Don't reuse prior knowledge")
@click.option("--regenerate-tasks", is_flag=True, help="Re-decompose goal into fresh task files")
@click.option("--auto-close", is_flag=True, help="Close browser automatically")
@click.option("--headless", is_flag=True, help="Run browser in headless mode")
@click.option("--credential-profile", default=None, help="Credential profile name")
@click.option("--model", "model_strs", multiple=True, help="PURPOSE=MODEL_ID override")
@click.option("--shared-session", is_flag=True, help="Share browser session across tasks")
@click.option("--parallel", type=int, default=1, help="Parallel execution count")
@click.option("--env", "env_name", default=None, help="Environment profile name (from config)")
@click.option("--force-unsafe", is_flag=True, help="Override goal safety checks")
@click.option("--unrun", is_flag=True, help="Run goals that have never been run")
@click.option("--failed", is_flag=True, help="Run goals whose last run had errors")
@click.argument("goal_files", nargs=-1, type=click.Path(), shell_complete=_complete_goal_file)
@click.pass_context
def run(
    ctx: click.Context,
    prompts: tuple[str, ...],
    base_url: str | None,
    no_update_summary: bool,
    no_update_tasks: bool,
    no_knowledge_reuse: bool,
    regenerate_tasks: bool,
    auto_close: bool,
    headless: bool,
    credential_profile: str | None,
    model_strs: tuple[str, ...],
    shared_session: bool,
    parallel: int,
    env_name: str | None,
    force_unsafe: bool,
    unrun: bool,
    failed: bool,
    goal_files: tuple[str, ...],
) -> None:
    """Run browser automation tasks."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    model_overrides = _parse_model_overrides(model_strs) if model_strs else None

    # First-run detection: offer setup wizard if config doesn't exist
    resolved_cfg = resolve_config_path(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    if not resolved_cfg.exists():
        if _is_interactive():
            click.echo("No config found. Running first-time setup...\n")
            run_setup_wizard(resolved_cfg)
            click.echo()
        else:
            click.echo(
                "No config found. Run 'spark-runner init' to set up, "
                "or continuing with defaults."
            )

    try:
        config: SparkConfig = build_config(
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
            regenerate_tasks=regenerate_tasks,
            env=env_name,
            force_unsafe=force_unsafe,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    config.ensure_dirs()

    # Discover goals matching --unrun / --failed filters
    filtered_goal_paths: list[Path] = []
    if (unrun or failed) and config.goal_summaries_dir is not None:
        from spark_runner.goals import get_last_run_info

        for gf_path in sorted(config.goal_summaries_dir.glob("*-task.json")):
            task_name = gf_path.stem.removesuffix("-task")
            run_info = get_last_run_info(config.runs_dir, task_name)
            if unrun and run_info is None:
                filtered_goal_paths.append(gf_path)
            elif failed and run_info is not None and run_info[1] == "errors":
                filtered_goal_paths.append(gf_path)

    # Build task specs
    tasks: list[TaskSpec] = []
    for p in prompts:
        tasks.append(TaskSpec(
            prompt=p,
            credential_profile=credential_profile or "default",
        ))

    # Collect goal paths from explicit args + filters, deduplicated
    seen_goals: set[Path] = set()
    all_goal_paths: list[Path] = []
    for gf in goal_files:
        gf_path = _resolve_goal_file(gf, config.goal_summaries_dir)
        if gf_path not in seen_goals:
            seen_goals.add(gf_path)
            all_goal_paths.append(gf_path)
    for gf_path in filtered_goal_paths:
        if gf_path not in seen_goals:
            seen_goals.add(gf_path)
            all_goal_paths.append(gf_path)

    for gf_path in all_goal_paths:
        tasks.append(TaskSpec(
            goal_path=gf_path,
            credential_profile=credential_profile or "default",
        ))

    if not tasks and (unrun or failed):
        raise click.ClickException("No matching goals found.")

    if not tasks:
        # Interactive mode
        prompt_text = click.prompt("Enter your task", type=str)
        tasks.append(TaskSpec(
            prompt=prompt_text,
            credential_profile=credential_profile or "default",
        ))

    # ── Safety gate: check goals before execution ──
    if config.active_environment is not None:
        from spark_runner.safety import check_goal_allowed, load_goal_safety

        blocked_names: list[str] = []
        allowed_tasks: list[TaskSpec] = []
        for t in tasks:
            if t.goal_path is not None:
                safety = load_goal_safety(t.goal_path)
                allowed, reason = check_goal_allowed(safety, config)
                if not allowed:
                    goal_label = t.goal_path.stem
                    print(
                        f"BLOCKED: Goal '{goal_label}' cannot run in "
                        f"environment '{config.active_environment}'."
                    )
                    if reason:
                        print(f"  Reason: {reason}")
                    blocked_names.append(goal_label)
                    continue
            allowed_tasks.append(t)
        tasks = allowed_tasks
        if blocked_names and not tasks:
            raise click.ClickException(
                f"All goals were blocked in environment '{config.active_environment}'."
            )

    print("=" * 60)
    print("  Spark Runner – Browser Automation")
    print(f"  Target: {config.base_url}")
    if config.active_environment:
        print(f"  Environment: {config.active_environment}")
    print("=" * 60)
    print()

    from spark_runner.orchestrator import run_multiple, run_single

    if len(tasks) == 1 and not shared_session:
        asyncio.run(run_single(tasks[0], config))
    else:
        asyncio.run(run_multiple(tasks, config, shared_session=shared_session, parallel=parallel))


# ── Goals subcommands ─────────────────────────────────────────────────


@cli.group()
@click.pass_context
def goals(ctx: click.Context) -> None:
    """Manage goals."""


@goals.command("list")
@click.option("--unrun", is_flag=True, help="Show only goals that have never been run")
@click.option("--failed", is_flag=True, help="Show only goals whose last run had errors")
@click.pass_context
def goals_list(ctx: click.Context, unrun: bool, failed: bool) -> None:
    """List all existing goals."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.goals import list_goals
    from spark_runner.orchestrator import _make_restore_fn

    assert config.goal_summaries_dir is not None
    list_goals(
        config.goal_summaries_dir, _make_restore_fn(config), config.runs_dir,
        filter_unrun=unrun, filter_failed=failed,
    )


@goals.command("show")
@click.argument("goal_name", shell_complete=_complete_goal_name)
@click.pass_context
def goals_show(ctx: click.Context, goal_name: str) -> None:
    """Show details for a specific goal."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.goals import show_goal_detail
    from spark_runner.orchestrator import _make_restore_fn

    assert config.goal_summaries_dir is not None
    show_goal_detail(config.goal_summaries_dir, goal_name, _make_restore_fn(config))


@goals.command("delete")
@click.argument("goal_name", shell_complete=_complete_goal_name)
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def goals_delete(ctx: click.Context, goal_name: str, force: bool) -> None:
    """Delete a goal and its unreferenced task files."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.goals import delete_goal

    assert config.goal_summaries_dir is not None
    assert config.tasks_dir is not None
    delete_goal(config.goal_summaries_dir, config.tasks_dir, goal_name, force=force)


@goals.command("classify")
@click.pass_context
def goals_classify(ctx: click.Context) -> None:
    """Classify observations in all existing goal summaries."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
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


@goals.command("orphans")
@click.option("--clean", is_flag=True, help="Delete orphan task files")
@click.pass_context
def goals_orphans(ctx: click.Context, clean: bool) -> None:
    """List or clean orphan task files."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.storage import clean_orphan_tasks, find_orphan_tasks

    assert config.tasks_dir is not None
    assert config.goal_summaries_dir is not None
    if clean:
        clean_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)
    else:
        find_orphan_tasks(config.tasks_dir, config.goal_summaries_dir)


# ── Results subcommands ───────────────────────────────────────────────


@cli.group()
@click.pass_context
def results(ctx: click.Context) -> None:
    """View and manage run results."""


@results.command("list")
@click.option("--task", "task_name", default=None, help="Filter by task name")
@click.pass_context
def results_list(ctx: click.Context, task_name: str | None) -> None:
    """List all runs."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.results import format_run_summary, list_runs

    assert config.runs_dir is not None
    runs = list_runs(config.runs_dir, task_name)
    if not runs:
        print("No runs found.")
        return
    print(f"Found {len(runs)} run(s):\n")
    for r in runs:
        print(format_run_summary(r))


@results.command("show")
@click.argument("run_path", required=False, default=None, shell_complete=_complete_run_path)
@click.pass_context
def results_show(ctx: click.Context, run_path: str | None) -> None:
    """Show full detail for a run."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    resolved = _resolve_run_path(run_path, config.runs_dir)
    from spark_runner.results import format_run_detail, get_run_detail

    detail = get_run_detail(resolved)
    print(format_run_detail(detail))


@results.command("errors")
@click.option("--task", "task_name", default=None, help="Filter by task name")
@click.pass_context
def results_errors(ctx: click.Context, task_name: str | None) -> None:
    """Show only runs with errors."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.results import format_run_summary, list_runs

    assert config.runs_dir is not None
    runs = [r for r in list_runs(config.runs_dir, task_name) if r.has_errors]
    if not runs:
        print("No runs with errors found.")
        return
    print(f"Found {len(runs)} run(s) with errors:\n")
    for r in runs:
        print(format_run_summary(r))


@results.command("screenshots")
@click.argument("run_path", required=False, default=None, shell_complete=_complete_run_path)
@click.pass_context
def results_screenshots(ctx: click.Context, run_path: str | None) -> None:
    """List screenshots for a run."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    resolved = _resolve_run_path(run_path, config.runs_dir)
    from spark_runner.results import get_run_detail

    detail = get_run_detail(resolved)
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


@results.command("report")
@click.argument("run_path", required=False, default=None, shell_complete=_complete_run_path)
@click.option(
    "--all",
    "regenerate_all",
    is_flag=True,
    default=False,
    help="Regenerate reports for all runs.",
)
@click.pass_context
def results_report(
    ctx: click.Context, run_path: str | None, regenerate_all: bool
) -> None:
    """Generate or regenerate HTML report for a run."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )

    from spark_runner.report import generate_report

    if regenerate_all:
        from spark_runner.results import list_runs

        runs_dir = config.runs_dir
        summaries = list_runs(runs_dir)
        for summary in summaries:
            index_path = generate_report(summary.run_dir)
            print(f"  {summary.task_name}/{summary.timestamp} -> {index_path}")
        print(f"Regenerated reports for {len(summaries)} run(s).")
        return

    resolved = _resolve_run_path(run_path, config.runs_dir)
    index_path = generate_report(resolved)
    print(f"HTML report generated: {index_path}")


# ── Generate goals ────────────────────────────────────────────────────


@cli.command("generate-goals")
@click.argument("source_path", type=click.Path())
@click.option("--branch", default="main", help="Git branch for repo URLs")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory")
@click.pass_context
def generate_goals_cmd(
    ctx: click.Context,
    source_path: str,
    branch: str,
    output_dir: str | None,
) -> None:
    """Generate test goals from frontend source code."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.goal_generator import generate_goals_from_source

    out = Path(output_dir) if output_dir else config.goal_summaries_dir
    assert out is not None
    asyncio.run(generate_goals_from_source(Path(source_path), out, config, branch))


# ── Record trajectory ─────────────────────────────────────────────────


@cli.command("record")
@click.option("--url", default=None, help="Starting URL")
@click.pass_context
def record_cmd(
    ctx: click.Context,
    url: str | None,
) -> None:
    """Record a user demonstration and generate a goal."""
    data_dir = _get_data_dir(ctx)
    config_path = _get_config_path(ctx)
    config = build_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=Path(data_dir) if data_dir else None,
    )
    from spark_runner.trajectory_recorder import record_and_generate_goal

    base = url or config.base_url
    asyncio.run(record_and_generate_goal(base, config))


# ── Legacy CLI compatibility ──────────────────────────────────────────


def legacy_main() -> None:
    """Backward-compatible entry point for ``python spark_runner.py`` usage.

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
