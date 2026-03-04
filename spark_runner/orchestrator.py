"""Workflow orchestration: the main execution logic extracted from the original main()."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from browser_use import Browser, ChatBrowserUse
from browser_use.agent.views import AgentHistoryList

from spark_runner.classification import classify_observations
from spark_runner.decomposition import decompose_task, generate_task_name
from spark_runner.execution import build_augmented_task, run_phase
from spark_runner.llm_trace import save_llm_conversation
from spark_runner.observation_routing import route_observations_to_phases
from spark_runner.goals import load_goal_summary
from spark_runner.knowledge import find_relevant_knowledge, load_knowledge_index
from spark_runner.log import attach_agent_log_handler, detach_agent_log_handler, log_event, log_problem
from spark_runner.models import (
    CredentialProfile,
    ModelConfig,
    PhaseResult,
    RunResult,
    ScreenshotRecord,
    SparkConfig,
    TaskSpec,
)
from spark_runner.observations import _extract_and_log_observations, merge_observations
from spark_runner.placeholders import restore_from_storage, restore_host_only, sanitize_for_storage
from spark_runner.report import generate_report
from spark_runner.results import write_run_metadata
from spark_runner.storage import make_run_dir, phase_name_to_slug, safe_write_path
from spark_runner.summarization import (
    generate_task_report,
    summarize_phase,
)


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


def _make_restore_fn(config: SparkConfig) -> Any:
    """Create a restore_from_storage closure using config credentials."""
    cred: CredentialProfile = config.active_credentials
    def _restore(text: str) -> str:
        return restore_from_storage(text, config.base_url, cred.email, cred.password)
    return _restore


def _make_host_only_restore_fn(config: SparkConfig) -> Any:
    """Create a restore closure that only resolves {BASE_URL}, leaving credential placeholders."""
    def _restore(text: str) -> str:
        return restore_host_only(text, config.base_url)
    return _restore


def _make_sanitize_fn(config: SparkConfig) -> Any:
    """Create a sanitize_for_storage closure using config credentials."""
    cred: CredentialProfile = config.active_credentials
    def _sanitize(text: str) -> str:
        return sanitize_for_storage(text, config.base_url, cred.email, cred.password)
    return _sanitize


async def run_single(
    task: TaskSpec,
    config: SparkConfig,
    client: anthropic.Anthropic | None = None,
    browser: Browser | None = None,
) -> RunResult:
    """Run a single task (prompt or goal file) and return the result.

    Args:
        task: The task specification.
        config: Configuration.
        client: Optional pre-configured Anthropic client.
        browser: Optional pre-configured browser (for shared sessions).

    Returns:
        A ``RunResult`` with all phase outcomes and screenshots.
    """
    if client is None:
        client = anthropic.Anthropic()

    # Use the task-specific credential profile if specified
    if task.credential_profile != config.active_credential_profile:
        config.active_credential_profile = task.credential_profile

    cred: CredentialProfile = config.active_credentials
    restore_fn = _make_restore_fn(config)
    host_restore_fn = _make_host_only_restore_fn(config)
    sanitize_fn = _make_sanitize_fn(config)

    assert config.tasks_dir is not None
    assert config.goal_summaries_dir is not None
    assert config.runs_dir is not None

    # --- Load knowledge index ---
    knowledge_match: dict[str, Any] | None = None
    if config.knowledge_reuse:
        print("Loading knowledge from prior goals...")
        knowledge_index: list[dict[str, Any]] = load_knowledge_index(
            config.goal_summaries_dir, config.tasks_dir, host_restore_fn
        )
        print(f"  Loaded {len(knowledge_index)} prior goal(s)")
    else:
        knowledge_index = []
        print("Knowledge reuse disabled")

    # --- Determine task identity ---
    goal_path: Path | None = task.goal_path
    prompt: str
    task_name: str
    phases: list[dict[str, str]]
    naming_response: anthropic.types.Message | None = None
    naming_msgs: list[dict[str, Any]] | None = None
    km_index: list[dict[str, Any]]

    if goal_path is not None:
        if not goal_path.exists():
            print(f"Goal summary file not found: {goal_path}")
            return RunResult()
        print(f"Loading goal from: {goal_path}")
        prompt, task_name, phases = load_goal_summary(
            goal_path, config.tasks_dir, host_restore_fn
        )
        print(f"Task: {prompt}")
        print(f"Task name: {task_name}")
        km_index = [g for g in knowledge_index if g["goal_file"] != goal_path.name]
    else:
        prompt = task.prompt or ""
        if not prompt:
            print("No task provided.")
            return RunResult()
        print()
        print("Generating task name...")
        # Call generate_task_name manually to capture the response for tracing
        naming_model = config.get_model("task_naming") or ModelConfig(max_tokens=64)
        naming_msgs = [{"role": "user", "content": (
            "Generate a short (2-8 word) descriptive name for this browser automation task. "
            "The name will be used as a filename, so use only lowercase letters, numbers, and hyphens. "
            "No spaces, no underscores, no special characters. Examples: 'write-tea-blog', 'login-test', 'scrape-products'.\n\n"
            f"Task: {prompt}\n\n"
            "Reply with ONLY the name, nothing else."
        )}]
        task_name = generate_task_name(
            prompt, client, config.get_model("task_naming")
        )
        print(f"Task name: {task_name}")
        phases = []
        km_index = knowledge_index

    # --- Pipeline steps tracking ---
    pipeline_steps: list[dict[str, Any]] = []
    pipeline_steps.append({
        "name": "Goal Source",
        "step_type": "goal_source",
        "status": "completed",
        "summary": f"Goal file: {goal_path.name}" if goal_path else "CLI prompt",
        "conversation_file": None,
    })

    # --- Create run_dir early ---
    run_dir: Path = make_run_dir(config.runs_dir, task_name)

    # Save generate_task_name conversation if it happened (prompt mode only)
    if naming_msgs is not None:
        # Re-create the response to save it — we need to call the API again for the trace.
        # Instead, save a synthetic trace from the task_name we got.
        naming_model_cfg = config.get_model("task_naming") or ModelConfig(max_tokens=64)
        # We already called the API in generate_task_name but can't capture the response
        # from that function. Save the prompt messages with a note.
        import json as _json
        _trace: dict[str, Any] = {
            "step": "task_naming",
            "model": naming_model_cfg.model,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "messages": naming_msgs,
            "response_text": task_name,
            "stop_reason": "end_turn",
            "input_tokens": 0,
            "output_tokens": 0,
        }
        (run_dir / "llm_task_naming.json").write_text(_json.dumps(_trace, indent=2))
        pipeline_steps.append({
            "name": "Task Naming",
            "step_type": "task_naming",
            "status": "completed",
            "summary": f"Generated name: {task_name}",
            "conversation_file": "llm_task_naming.json",
        })

    # --- Knowledge matching (run_dir available) ---
    if km_index:
        print("\nFinding relevant knowledge from prior goals...")
        knowledge_match = find_relevant_knowledge(
            prompt, km_index, client,
            config.get_model("knowledge_matching"),
            run_dir=run_dir,
        )
        reusable = knowledge_match.get("reusable_subtasks", [])
        obs = knowledge_match.get("relevant_observations", [])
        pipeline_steps.append({
            "name": "Knowledge Matching",
            "step_type": "knowledge_matching",
            "status": "completed",
            "summary": f"{len(reusable)} reusable subtask(s), {len(obs)} observation(s)",
            "conversation_file": "llm_knowledge_matching.json",
        })
    elif knowledge_index:
        print("  No other goals to learn from")

    # --- Decomposition (run_dir available) ---
    if not phases:
        print()
        print("Decomposing task into phases...")
        phases = decompose_task(
            prompt, config.base_url,
            config.tasks_dir, client, host_restore_fn,
            config.get_model("task_decomposition"),
            knowledge_match=knowledge_match,
            run_dir=run_dir,
        )
    pipeline_steps.append({
        "name": "Task Decomposition",
        "step_type": "task_decomposition",
        "status": "completed",
        "summary": f"{len(phases)} phases planned",
        "conversation_file": "llm_task_decomposition.json" if (run_dir / "llm_task_decomposition.json").exists() else None,
    })

    print(f"Planned {len(phases)} phases:")
    for i, p in enumerate(phases, 1):
        print(f"  {i}. {p['name']}")
    print()

    # Route cross-goal observations to relevant phases
    routed_observations: dict[str, list[str | dict[str, str]]] = {}
    if knowledge_match and knowledge_match.get("relevant_observations"):
        print("Routing observations to phases...")
        routed_observations = route_observations_to_phases(
            knowledge_match["relevant_observations"],
            phases,
            client,
            config.get_model("observation_routing"),
            run_dir=run_dir,
        )
        for phase_name, obs_list in routed_observations.items():
            print(f"  {phase_name}: {len(obs_list)} observation(s)")
        pipeline_steps.append({
            "name": "Observation Routing",
            "step_type": "observation_routing",
            "status": "completed",
            "summary": f"{len(routed_observations)} phase(s) received observations",
            "conversation_file": "llm_observation_routing.json",
        })

    agent_log_handler = attach_agent_log_handler(run_dir)

    event_log: Path = run_dir / "event_log.txt"
    conversation_log: Path = run_dir / "conversation_log.json"
    summaries_path: Path = run_dir / "phase_summaries.json"
    problem_log: Path = run_dir / "problem_log.txt"
    report_path: Path = config.goal_summaries_dir / f"{task_name}-task.json"

    # --- Execute ---
    event_log.write_text("")
    log_event(event_log, "=" * 60)
    log_event(event_log, f"WORKFLOW START: {task_name}")
    log_event(event_log, f"Run directory: {run_dir}")
    log_event(event_log, f"Prompt: {prompt}")
    log_event(event_log, f"Target: {config.base_url}")
    log_event(event_log, "=" * 60)

    for line in format_phase_plan(phases):
        log_event(event_log, line)

    if knowledge_match:
        for line in format_knowledge_match(knowledge_match):
            log_event(event_log, line)

    if routed_observations:
        log_event(event_log, f"OBSERVATION ROUTING ({len(routed_observations)} phase(s) received observations):")
        for phase_name, obs_list in routed_observations.items():
            log_event(event_log, f"  {phase_name}: {len(obs_list)} observation(s)")

    own_browser: bool = browser is None
    if browser is None:
        browser = Browser(headless=config.headless, keep_alive=True)
    llm: ChatBrowserUse = ChatBrowserUse()

    all_summaries: list[dict[str, str]] = []
    phase_results: list[PhaseResult] = []
    all_screenshots: list[ScreenshotRecord] = []

    try:
        prior_summaries: list[dict[str, str]] = []

        for phase in phases:
            cross_obs: list[str | dict[str, str]] | None = (
                routed_observations.get(phase["name"]) or None
            )
            augmented_task: str = build_augmented_task(
                phase["task"], prior_summaries, restore_fn,
                cross_goal_observations=cross_obs,
            )
            augmented_task_truncated: str = (
                augmented_task[:1000] + "..." if len(augmented_task) > 1000 else augmented_task
            )
            log_event(event_log, f"Task for '{phase['name']}':\n{augmented_task_truncated}")

            success: bool
            result: AgentHistoryList[Any]
            phase_screenshots: list[ScreenshotRecord]
            success, result, phase_screenshots = await run_phase(
                phase["name"], augmented_task, llm, browser,
                conversation_log, event_log, problem_log, run_dir,
            )
            all_screenshots.extend(phase_screenshots)

            log_event(event_log, f"Summarizing phase '{phase['name']}'...")
            summary: str = summarize_phase(
                phase["name"], phase["task"], result, success,
                client, config.get_model("summarization"),
                run_dir=run_dir,
            )
            _extract_and_log_observations(summary, phase["name"], event_log, problem_log, success=success)

            phase_slug: str = phase_name_to_slug(phase["name"])
            subtask_path: Path = config.tasks_dir / f"{phase_slug}.txt"
            now_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            dated_summary: str = f"<!-- updated: {now_str} -->\n{sanitize_fn(summary)}"
            if goal_path and config.update_tasks:
                subtask_path.write_text(dated_summary)
            elif config.update_tasks:
                subtask_path = safe_write_path(subtask_path)
                subtask_path.write_text(dated_summary)
            log_event(event_log, f"Subtask summary saved to {subtask_path}")

            phase_record: dict[str, str] = {
                "name": phase["name"],
                "outcome": "SUCCESS" if success else "FAILED",
                "summary": summary,
                "filename": subtask_path.name,
            }
            prior_summaries.append(phase_record)
            all_summaries.append(phase_record)
            phase_results.append(PhaseResult(
                name=phase["name"],
                outcome="SUCCESS" if success else "FAILED",
                summary=summary,
                filename=subtask_path.name,
                screenshots=phase_screenshots,
            ))

            phase_slug_for_step: str = phase_name_to_slug(phase["name"])
            pipeline_steps.append({
                "name": f"Phase: {phase['name']}",
                "step_type": "phase_execution",
                "status": "completed" if success else "failed",
                "summary": "SUCCESS" if success else "FAILED",
                "conversation_file": None,
                "phase_slug": phase_slug_for_step,
            })
            summarize_file = f"llm_summarize_{phase_slug_for_step}.json"
            if (run_dir / summarize_file).exists():
                pipeline_steps.append({
                    "name": f"Summarize: {phase['name']}",
                    "step_type": "summarization",
                    "status": "completed",
                    "summary": f"Phase {'succeeded' if success else 'failed'}",
                    "conversation_file": summarize_file,
                })

            summary_truncated: str = summary[:500] + "..." if len(summary) > 500 else summary
            summary_truncated = "\n".join(
                [f"  PHASE {phase['name']} SUMMARY:\n{line}" for line in summary_truncated.split("\n")]
            )
            log_event(event_log, f"PHASE SUMMARY ({phase['name']}):\n{summary_truncated}")

            if not success:
                log_event(event_log, f"ABORTING: phase '{phase['name']}' failed.")
                break
        else:
            log_event(event_log, "ALL PHASES COMPLETED SUCCESSFULLY.")

    except Exception as e:
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
        detach_agent_log_handler(agent_log_handler)
        log_event(event_log, "WORKFLOW END")
        log_event(event_log, f"Conversation log saved to {conversation_log}")
        if problem_log.exists():
            log_event(event_log, f"Problem log saved to {problem_log}")

        summaries_path.write_text(sanitize_fn(json.dumps(all_summaries, indent=2)))
        log_event(event_log, f"Phase summaries saved to {summaries_path}")

        if all_summaries:
            if goal_path and config.update_summary:
                print("\nGenerating task report...")
                report: dict[str, Any] = generate_task_report(
                    task_name, prompt, all_summaries, client,
                    config.get_model("summarization"),
                    run_dir=run_dir,
                )
                pipeline_steps.append({
                    "name": "Task Report",
                    "step_type": "task_report",
                    "status": "completed",
                    "summary": "Generated task report",
                    "conversation_file": "llm_task_report.json",
                })
                existing_data: dict[str, Any] = json.loads(goal_path.read_text())
                existing_obs: list[str | dict[str, str]] = existing_data.get("key_observations", [])
                new_obs: list[str] = report.get("key_observations", [])
                print("Merging observations with existing goal summary...")
                merged_obs: list[dict[str, str]] = merge_observations(
                    existing_obs, new_obs, client,
                    config.get_model("classification"),
                    run_dir=run_dir,
                )
                pipeline_steps.append({
                    "name": "Merge Observations",
                    "step_type": "merge_observations",
                    "status": "completed",
                    "summary": f"{len(merged_obs)} merged observation(s)",
                    "conversation_file": "llm_merge_observations.json",
                })
                print("Classifying observations...")
                classified_obs: list[dict[str, str]] = classify_observations(
                    prompt, merged_obs, client,
                    config.get_model("classification"),
                    run_dir=run_dir,
                )
                pipeline_steps.append({
                    "name": "Classify Observations",
                    "step_type": "classify_observations",
                    "status": "completed",
                    "summary": f"{len(classified_obs)} classified observation(s)",
                    "conversation_file": "llm_classify_observations.json",
                })
                existing_data["key_observations"] = classified_obs
                existing_data["subtasks"] = report["subtasks"]
                existing_data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                goal_path.write_text(sanitize_fn(json.dumps(existing_data, indent=2)))
                num_errors: int = sum(1 for o in classified_obs if o["severity"] == "error")
                num_warnings: int = len(classified_obs) - num_errors
                log_event(event_log, f"Updated goal summary: {goal_path} ({num_errors} errors, {num_warnings} warnings)")
                print(f"Updated goal summary: {goal_path} ({num_errors} errors, {num_warnings} warnings)")
            elif config.update_summary:
                print("\nGenerating task report...")
                report = generate_task_report(
                    task_name, prompt, all_summaries, client,
                    config.get_model("summarization"),
                    run_dir=run_dir,
                )
                pipeline_steps.append({
                    "name": "Task Report",
                    "step_type": "task_report",
                    "status": "completed",
                    "summary": "Generated task report",
                    "conversation_file": "llm_task_report.json",
                })
                print("Classifying observations...")
                classified_obs = classify_observations(
                    prompt, report.get("key_observations", []), client,
                    config.get_model("classification"),
                    run_dir=run_dir,
                )
                pipeline_steps.append({
                    "name": "Classify Observations",
                    "step_type": "classify_observations",
                    "status": "completed",
                    "summary": f"{len(classified_obs)} classified observation(s)",
                    "conversation_file": "llm_classify_observations.json",
                })
                report["key_observations"] = classified_obs
                now_iso: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                report["created_at"] = now_iso
                report["updated_at"] = now_iso
                report_path = safe_write_path(report_path)
                report_path.write_text(sanitize_fn(json.dumps(report, indent=2)))
                num_errors = sum(1 for o in classified_obs if o["severity"] == "error")
                num_warnings = len(classified_obs) - num_errors
                log_event(event_log, f"Task report saved to {report_path} ({num_errors} errors, {num_warnings} warnings)")
                print(f"Task report saved to {report_path} ({num_errors} errors, {num_warnings} warnings)")

        # Save pipeline data
        (run_dir / "pipeline.json").write_text(json.dumps(pipeline_steps, indent=2))

        # Write run metadata
        write_run_metadata(
            run_dir=run_dir,
            task_name=task_name,
            prompt=prompt,
            base_url=config.base_url,
            credential_profile=config.active_credential_profile,
            phases=[
                {
                    "name": pr.name,
                    "outcome": pr.outcome,
                    "screenshots": [
                        {
                            "path": str(s.path.relative_to(run_dir)),
                            "event_type": s.event_type,
                            "step_number": s.step_number,
                            "timestamp": s.timestamp,
                        }
                        for s in pr.screenshots
                    ],
                }
                for pr in phase_results
            ],
            screenshots=all_screenshots,
        )

        try:
            report_index = generate_report(run_dir)
            print(f"\nHTML report: {report_index}")
            log_event(event_log, f"HTML report generated: {report_index}")
        except Exception as report_err:
            log_event(event_log, f"WARNING: Could not generate HTML report: {report_err}")

        if own_browser:
            if not config.auto_close:
                input("\nPress Enter to close the browser...")
            await browser.stop()

    return RunResult(
        task_name=task_name,
        phases=phase_results,
        screenshots=all_screenshots,
        run_dir=run_dir,
    )


async def run_multiple(
    tasks: list[TaskSpec],
    config: SparkConfig,
    shared_session: bool = False,
    parallel: int = 1,
) -> list[RunResult]:
    """Run multiple tasks sequentially or in parallel.

    Args:
        tasks: List of task specifications.
        config: Configuration.
        shared_session: If True, keep the same browser across tasks.
        parallel: Number of concurrent browser sessions (future use).

    Returns:
        A list of ``RunResult`` for each task.
    """
    client: anthropic.Anthropic = anthropic.Anthropic()
    results: list[RunResult] = []

    browser: Browser | None = None
    if shared_session:
        browser = Browser(headless=config.headless, keep_alive=True)

    try:
        if parallel > 1 and not shared_session:
            # Concurrent execution (each with own browser)
            async def _run_one(t: TaskSpec) -> RunResult:
                return await run_single(t, config, client)

            results = await asyncio.gather(*[_run_one(t) for t in tasks])
            results = list(results)
        else:
            # Sequential execution
            for task in tasks:
                result = await run_single(task, config, client, browser)
                results.append(result)
    finally:
        if shared_session and browser is not None:
            if not config.auto_close:
                input("\nPress Enter to close the browser...")
            await browser.stop()

    # Print aggregated summary
    print("\n" + "=" * 60)
    print("  Run Summary")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        status = "PASS" if result.all_phases_succeeded else "FAIL"
        print(f"  {i}. {result.task_name}: {status} ({len(result.phases)} phases)")
    total_pass = sum(1 for r in results if r.all_phases_succeeded)
    print(f"\n  {total_pass}/{len(results)} tasks passed")

    return results
