"""Tests for HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

from spark_runner.report import (
    _html_escape,
    _markdown_to_html,
    generate_report,
    generate_runs_index,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic run directories
# ---------------------------------------------------------------------------

def _make_run_dir(
    tmp_path: Path,
    *,
    task_name: str = "login-test",
    prompt: str = "Test login flow",
    phases: list[dict[str, object]] | None = None,
    event_log_content: str = "",
    problem_log_content: str = "",
    phase_summaries: list[dict[str, str]] | None = None,
    screenshots: list[str] | None = None,
    conversation_files: dict[str, object] | None = None,
    pipeline: list[dict[str, object]] | None = None,
    llm_files: dict[str, object] | None = None,
    agent_log_content: str = "",
) -> Path:
    """Create a synthetic run directory under *tmp_path*."""
    run_dir = tmp_path / task_name / "2025-01-01T00-00-00"
    run_dir.mkdir(parents=True)

    if phases is None:
        phases = [
            {
                "name": "Phase 1: Login",
                "outcome": "SUCCESS",
                "screenshots": [],
            }
        ]

    metadata = {
        "task_name": task_name,
        "prompt": prompt,
        "timestamp": "2025-01-01T00:00:00",
        "base_url": "https://example.com",
        "credential_profile": "default",
        "phases": phases,
        "screenshots": [],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    if event_log_content:
        (run_dir / "event_log.txt").write_text(event_log_content)

    if problem_log_content:
        (run_dir / "problem_log.txt").write_text(problem_log_content)

    if phase_summaries is not None:
        (run_dir / "phase_summaries.json").write_text(json.dumps(phase_summaries, indent=2))

    if screenshots:
        ss_dir = run_dir / "screenshots"
        ss_dir.mkdir()
        for name in screenshots:
            (ss_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    if conversation_files:
        conv_dir = run_dir / "conversation_log.json"
        conv_dir.mkdir()
        for fname, data in conversation_files.items():
            if isinstance(data, str):
                # Write raw text (for .txt conversation files)
                (conv_dir / fname).write_text(data)
            else:
                (conv_dir / fname).write_text(json.dumps(data))

    if pipeline is not None:
        (run_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2))

    if llm_files:
        for fname, data in llm_files.items():
            (run_dir / fname).write_text(json.dumps(data))

    if agent_log_content:
        (run_dir / "agent_log.txt").write_text(agent_log_content)

    return run_dir


_EXPECTED_FILES = [
    "index.html",
    "pipeline.html",
    "phases.html",
    "events.html",
    "problems.html",
    "conversations.html",
    "agent_log.html",
    "screenshots.html",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """Test that generate_report creates the expected files and content."""

    def test_creates_all_html_files(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = generate_report(run_dir)

        assert result == run_dir / "report" / "index.html"
        assert result.exists()
        for fname in _EXPECTED_FILES:
            assert (run_dir / "report" / fname).exists(), f"Missing {fname}"

    def test_index_contains_task_name_and_phase_table(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            task_name="my-task",
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "FAILED", "screenshots": []},
            ],
        )
        generate_report(run_dir)
        index_html = (run_dir / "report" / "index.html").read_text()

        assert "my-task" in index_html
        assert "Login" in index_html
        assert "Search" in index_html
        assert "SUCCESS" in index_html
        assert "FAILED" in index_html

    def test_phases_page_renders_markdown_summaries(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            phases=[{"name": "Setup", "outcome": "SUCCESS", "screenshots": []}],
            phase_summaries=[
                {"name": "Setup", "summary": "# Heading\n\n**Bold text** here.\n\n- Item one\n- Item two"},
            ],
        )
        generate_report(run_dir)
        phases_html = (run_dir / "report" / "phases.html").read_text()

        assert "<h2>" in phases_html
        assert "<strong>Bold text</strong>" in phases_html
        assert "<li>Item one</li>" in phases_html

    def test_phases_page_links_to_conversations_and_agent_log(self, tmp_path: Path) -> None:
        """Each phase on the Phases page should link to Conversations and Agent Log."""
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "SUCCESS", "screenshots": []},
            ],
        )
        generate_report(run_dir)
        phases_html = (run_dir / "report" / "phases.html").read_text()

        assert 'conversations.html#phase-1' in phases_html
        assert 'agent_log.html#phase-1' in phases_html
        assert 'conversations.html#phase-2' in phases_html
        assert 'agent_log.html#phase-2' in phases_html

    def test_events_page_contains_log_content(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            event_log_content="[2025-01-01 00:00:00] WORKFLOW START: login-test\n",
        )
        generate_report(run_dir)
        events_html = (run_dir / "report" / "events.html").read_text()

        assert "WORKFLOW START" in events_html
        assert "log-line-ts" in events_html  # timestamp highlight class

    def test_problems_page_shows_content_when_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            problem_log_content="[2025-01-01 00:01:00] ERROR: login failed\n",
        )
        generate_report(run_dir)
        problems_html = (run_dir / "report" / "problems.html").read_text()

        assert "login failed" in problems_html

    def test_problems_page_shows_green_when_empty(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, problem_log_content="")
        generate_report(run_dir)
        problems_html = (run_dir / "report" / "problems.html").read_text()

        assert "No problems recorded" in problems_html
        assert "green-ok" in problems_html

    def test_screenshots_page_references_relative_paths(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {
                    "name": "Login",
                    "outcome": "SUCCESS",
                    "screenshots": [
                        {
                            "path": "screenshots/step_01.png",
                            "event_type": "step",
                            "timestamp": "2025-01-01T00:00:01",
                        }
                    ],
                }
            ],
            screenshots=["step_01.png"],
        )
        generate_report(run_dir)
        ss_html = (run_dir / "report" / "screenshots.html").read_text()

        assert "../screenshots/step_01.png" in ss_html

    def test_screenshots_page_shows_missing_placeholder(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {
                    "name": "Login",
                    "outcome": "SUCCESS",
                    "screenshots": [
                        {
                            "path": "screenshots/gone.png",
                            "event_type": "step",
                            "timestamp": "",
                        }
                    ],
                }
            ],
        )
        generate_report(run_dir)
        ss_html = (run_dir / "report" / "screenshots.html").read_text()

        assert "missing" in ss_html.lower()

    def test_conversations_page_groups_by_uuid(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "SUCCESS", "screenshots": []},
            ],
            conversation_files={
                "00000001-0000-0000-0000-000000000001.json": [
                    {"role": "user", "content": "Please log in"},
                    {"role": "assistant", "content": "Done"},
                ],
                "00000002-0000-0000-0000-000000000002.json": [
                    {"role": "user", "content": "Search for items"},
                ],
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        # Phase names should appear as section headings
        assert "Login" in conv_html
        assert "Search" in conv_html
        assert "Please log in" in conv_html
        assert "Search for items" in conv_html

    def test_conversations_includes_full_long_messages(self, tmp_path: Path) -> None:
        long_msg = "x" * 15_000
        run_dir = _make_run_dir(
            tmp_path,
            conversation_files={
                "aaaa.json": [{"role": "assistant", "content": long_msg}],
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        # Full content should be present (no truncation in HTML report)
        assert _html_escape(long_msg) in conv_html
        assert "truncated" not in conv_html.lower()

    def test_handles_missing_metadata(self, tmp_path: Path) -> None:
        """Report generation should not crash on a bare run directory."""
        run_dir = tmp_path / "bare-run" / "2025-01-01T00-00-00"
        run_dir.mkdir(parents=True)
        # No metadata file at all
        result = generate_report(run_dir)
        assert result.exists()

    def test_handles_empty_run(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            phases=[],
            phase_summaries=[],
        )
        result = generate_report(run_dir)
        assert result.exists()
        index_html = (run_dir / "report" / "index.html").read_text()
        assert "No phase data available" in index_html

    def test_existing_report_dir_is_overwritten(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        report_dir = run_dir / "report"
        report_dir.mkdir()
        (report_dir / "old_file.html").write_text("old")

        generate_report(run_dir)

        # New files should exist
        assert (report_dir / "index.html").exists()
        # Old file should still be there (we don't delete unknown files)
        assert (report_dir / "old_file.html").exists()

    def test_txt_conversations_grouped_by_uuid(self, tmp_path: Path) -> None:
        """Browser-use writes conversation_{UUID}_{step}.txt files."""
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "SUCCESS", "screenshots": []},
            ],
            conversation_files={
                # UUID-A: two snapshots — only _2 should be used
                "conversation_aaaa-bbbb_1.txt": " system \nSystem prompt\n user \nStep 1",
                "conversation_aaaa-bbbb_2.txt": " system \nSystem prompt\n user \nStep 1\n assistant \nResponse 1\n user \nStep 2",
                # UUID-B: one snapshot
                "conversation_cccc-dddd_1.txt": " user \nSearch query\n assistant \nSearch result",
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        # Phase names as headings
        assert "Login" in conv_html
        assert "Search" in conv_html
        # Content from the latest snapshot of UUID-A
        assert "Step 2" in conv_html
        assert "Search result" in conv_html

    def test_txt_conversations_includes_full_long_messages(self, tmp_path: Path) -> None:
        long_body = "x" * 15_000
        run_dir = _make_run_dir(
            tmp_path,
            conversation_files={
                "conversation_aaaa_1.txt": f" assistant \n{long_body}",
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        # Full content should be present (no truncation in HTML report)
        assert long_body in conv_html
        assert "truncated" not in conv_html.lower()

    def test_txt_preferred_over_json(self, tmp_path: Path) -> None:
        """When both .txt and .json files exist, .txt should be used."""
        run_dir = _make_run_dir(
            tmp_path,
            conversation_files={
                "conversation_aaaa_1.txt": " user \nFrom txt file",
                "legacy.json": [{"role": "user", "content": "From json file"}],
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        assert "From txt file" in conv_html
        # JSON content should NOT appear when .txt files are present
        assert "From json file" not in conv_html

    def test_task_level_screenshots_mapped_to_phases(self, tmp_path: Path) -> None:
        """Screenshots in the top-level array should be matched to phases by filename prefix."""
        run_dir = tmp_path / "my-task" / "2025-01-01T00-00-00"
        run_dir.mkdir(parents=True)
        ss_dir = run_dir / "screenshots"
        ss_dir.mkdir()
        # Create screenshot files that match phase slugs
        (ss_dir / "login_step_001.png").write_bytes(b"\x89PNG")
        (ss_dir / "login_step_002.png").write_bytes(b"\x89PNG")
        (ss_dir / "fill-form_step_000.png").write_bytes(b"\x89PNG")

        metadata = {
            "task_name": "my-task",
            "prompt": "Test",
            "timestamp": "2025-01-01T00:00:00",
            "base_url": "https://example.com",
            "credential_profile": "default",
            "phases": [
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Fill Form", "outcome": "SUCCESS", "screenshots": []},
            ],
            "screenshots": [
                {"path": "screenshots/login_step_001.png", "event_type": "step", "timestamp": ""},
                {"path": "screenshots/login_step_002.png", "event_type": "step", "timestamp": ""},
                {"path": "screenshots/fill-form_step_000.png", "event_type": "step", "timestamp": ""},
            ],
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

        generate_report(run_dir)

        # Phases page should show thumbnails for both phases
        phases_html = (run_dir / "report" / "phases.html").read_text()
        assert "../screenshots/login_step_001.png" in phases_html
        assert "../screenshots/login_step_002.png" in phases_html
        assert "../screenshots/fill-form_step_000.png" in phases_html

        # Screenshots page should organize by phase
        ss_html = (run_dir / "report" / "screenshots.html").read_text()
        assert "Login" in ss_html
        assert "Fill Form" in ss_html
        assert "../screenshots/login_step_001.png" in ss_html

        # Index page should show correct counts
        index_html = (run_dir / "report" / "index.html").read_text()
        # Login has 2 screenshots, Fill Form has 1
        assert ">2<" in index_html
        assert ">1<" in index_html

    def test_no_screenshots_no_conversations(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            phases=[{"name": "Step1", "outcome": "SUCCESS", "screenshots": []}],
        )
        generate_report(run_dir)
        ss_html = (run_dir / "report" / "screenshots.html").read_text()
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        assert "No screenshots found" in ss_html
        assert "No conversation logs found" in conv_html


class TestPipelineRendering:
    """Tests for pipeline timeline and pipeline page."""

    def test_overview_renders_pipeline_steps(self, tmp_path: Path) -> None:
        pipeline = [
            {
                "name": "Goal Source",
                "step_type": "goal_source",
                "status": "completed",
                "summary": "CLI prompt",
                "conversation_file": None,
            },
            {
                "name": "Task Decomposition",
                "step_type": "task_decomposition",
                "status": "completed",
                "summary": "3 phases planned",
                "conversation_file": "llm_task_decomposition.json",
            },
            {
                "name": "Phase: Login",
                "step_type": "phase_execution",
                "status": "completed",
                "summary": "SUCCESS",
            },
            {
                "name": "Phase: Search",
                "step_type": "phase_execution",
                "status": "failed",
                "summary": "FAILED",
            },
        ]
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "FAILED", "screenshots": []},
            ],
            pipeline=pipeline,
        )
        generate_report(run_dir)
        index_html = (run_dir / "report" / "index.html").read_text()

        assert "Pipeline" in index_html
        assert "Goal Source" in index_html
        assert "Task Decomposition" in index_html
        assert "3 phases planned" in index_html
        assert "pipeline-timeline" in index_html
        assert "view conversation" in index_html

    def test_overview_fallback_without_pipeline(self, tmp_path: Path) -> None:
        """Old runs without pipeline.json should still render the phase table."""
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
            ],
        )
        generate_report(run_dir)
        index_html = (run_dir / "report" / "index.html").read_text()

        # Should fall back to the old-style phase table
        assert "Login" in index_html
        assert "SUCCESS" in index_html
        # Should NOT have pipeline timeline div in the body
        assert '<div class="pipeline-timeline">' not in index_html

    def test_pipeline_page_renders_conversations(self, tmp_path: Path) -> None:
        llm_data = {
            "step": "knowledge_matching",
            "model": "claude-sonnet-4-5-20250929",
            "timestamp": "2026-03-04T10:15:30Z",
            "messages": [{"role": "user", "content": "Find relevant knowledge"}],
            "response_text": "Here is the matched knowledge",
            "stop_reason": "end_turn",
            "input_tokens": 1234,
            "output_tokens": 567,
        }
        run_dir = _make_run_dir(
            tmp_path,
            llm_files={"llm_knowledge_matching.json": llm_data},
        )
        generate_report(run_dir)
        pipeline_html = (run_dir / "report" / "pipeline.html").read_text()

        assert "knowledge_matching" in pipeline_html
        assert "claude-sonnet-4-5-20250929" in pipeline_html
        assert "1,234" in pipeline_html  # formatted input tokens
        assert "567" in pipeline_html
        assert "Find relevant knowledge" in pipeline_html
        assert "Here is the matched knowledge" in pipeline_html

    def test_pipeline_page_no_conversations(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        generate_report(run_dir)
        pipeline_html = (run_dir / "report" / "pipeline.html").read_text()

        assert "No conversation data available" in pipeline_html

    def test_pipeline_page_in_nav(self, tmp_path: Path) -> None:
        from spark_runner.report import _nav
        nav_html = _nav("index.html", has_problems=True)
        assert "pipeline.html" in nav_html
        assert "Pipeline" in nav_html

    def test_pipeline_page_no_truncation(self, tmp_path: Path) -> None:
        """Pipeline page should show full content without truncation."""
        long_prompt = "y" * 15_000
        long_response = "z" * 15_000
        llm_data = {
            "step": "test_step",
            "model": "test-model",
            "timestamp": "2026-03-04T10:00:00Z",
            "messages": [{"role": "user", "content": long_prompt}],
            "response_text": long_response,
            "stop_reason": "end_turn",
            "input_tokens": 100,
            "output_tokens": 200,
        }
        run_dir = _make_run_dir(
            tmp_path,
            llm_files={"llm_test_step.json": llm_data},
        )
        generate_report(run_dir)
        pipeline_html = (run_dir / "report" / "pipeline.html").read_text()

        assert _html_escape(long_prompt) in pipeline_html
        assert _html_escape(long_response) in pipeline_html
        assert "truncated" not in pipeline_html.lower()

    def test_phases_loaded_step_in_timeline(self, tmp_path: Path) -> None:
        """When phases are loaded from goal file, pipeline shows 'Phases Loaded'."""
        pipeline = [
            {
                "name": "Goal Source",
                "step_type": "goal_source",
                "status": "completed",
                "summary": "Goal file: my-goal.json",
                "conversation_file": None,
            },
            {
                "name": "Phases Loaded",
                "step_type": "phases_loaded",
                "status": "completed",
                "summary": "3 phases from goal file",
                "conversation_file": None,
            },
        ]
        run_dir = _make_run_dir(
            tmp_path,
            phases=[{"name": "Login", "outcome": "SUCCESS", "screenshots": []}],
            pipeline=pipeline,
        )
        generate_report(run_dir)
        index_html = (run_dir / "report" / "index.html").read_text()

        assert "Phases Loaded" in index_html
        assert "3 phases from goal file" in index_html

    def test_phase_execution_links_to_conversations(self, tmp_path: Path) -> None:
        """Phase execution steps should link to the conversations page."""
        pipeline = [
            {
                "name": "Phase: Login",
                "step_type": "phase_execution",
                "status": "completed",
                "summary": "SUCCESS",
                "conversation_file": None,
                "phase_slug": "login",
            },
        ]
        run_dir = _make_run_dir(
            tmp_path,
            phases=[{"name": "Login", "outcome": "SUCCESS", "screenshots": []}],
            pipeline=pipeline,
        )
        generate_report(run_dir)
        index_html = (run_dir / "report" / "index.html").read_text()

        assert 'conversations.html#phase-1' in index_html
        assert "agent conversation" in index_html
        assert 'agent_log.html#phase-1' in index_html
        assert "agent log" in index_html
        # Phase name should link to the Phases page
        assert 'phases.html#phase-1' in index_html

    def test_phase_execution_name_links_to_phases_page(self, tmp_path: Path) -> None:
        """Phase execution step headers should link to the Phases page."""
        pipeline = [
            {
                "name": "Phase: Login",
                "step_type": "phase_execution",
                "status": "completed",
                "summary": "SUCCESS",
                "conversation_file": None,
                "phase_slug": "login",
            },
        ]
        run_dir = _make_run_dir(
            tmp_path,
            phases=[{"name": "Login", "outcome": "SUCCESS", "screenshots": []}],
            pipeline=pipeline,
        )
        generate_report(run_dir)
        index_html = (run_dir / "report" / "index.html").read_text()

        assert '<a href="phases.html#phase-1">Phase: Login</a>' in index_html

    def test_conversations_page_has_anchor_ids(self, tmp_path: Path) -> None:
        """Conversations page headings should have anchor IDs for linking."""
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "SUCCESS", "screenshots": []},
            ],
            conversation_files={
                "conversation_aaaa_1.txt": " user \nHello\n assistant \nHi",
                "conversation_bbbb_1.txt": " user \nSearch\n assistant \nResults",
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        assert 'id="phase-1"' in conv_html
        assert 'id="phase-2"' in conv_html

    def test_conversations_page_links_back_to_phases(self, tmp_path: Path) -> None:
        """Conversation phase headings should link back to the Phases page."""
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
            ],
            conversation_files={
                "conversation_aaaa_1.txt": " user \nHello\n assistant \nHi",
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        assert 'phases.html#phase-1' in conv_html
        assert "view phase" in conv_html


class TestHtmlEscape:
    def test_escapes_angle_brackets(self) -> None:
        assert _html_escape("<script>") == "&lt;script&gt;"

    def test_escapes_ampersand(self) -> None:
        assert _html_escape("a & b") == "a &amp; b"

    def test_escapes_quotes(self) -> None:
        assert _html_escape('"hello"') == "&quot;hello&quot;"


class TestMarkdownToHtml:
    def test_heading_levels(self) -> None:
        result = _markdown_to_html("# One\n## Two\n### Three")
        assert "<h2>One</h2>" in result
        assert "<h3>Two</h3>" in result
        assert "<h4>Three</h4>" in result

    def test_bold(self) -> None:
        result = _markdown_to_html("Some **bold** text")
        assert "<strong>bold</strong>" in result

    def test_list_items(self) -> None:
        result = _markdown_to_html("- First\n- Second")
        assert "<ul>" in result
        assert "<li>First</li>" in result
        assert "<li>Second</li>" in result

    def test_observations_block(self) -> None:
        result = _markdown_to_html("<OBSERVATIONS>\nSomething noted\n</OBSERVATIONS>")
        assert 'class="obs-block"' in result
        assert "Something noted" in result

    def test_paragraphs(self) -> None:
        result = _markdown_to_html("A paragraph of text.")
        assert "<p>A paragraph of text.</p>" in result

    def test_html_is_escaped(self) -> None:
        result = _markdown_to_html("Contains <script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_nav_links_all_pages(self) -> None:
        """Verify all 8 pages appear in the nav bar."""
        from spark_runner.report import _nav
        nav_html = _nav("index.html", has_problems=True)
        assert "index.html" in nav_html
        assert "pipeline.html" in nav_html
        assert "phases.html" in nav_html
        assert "events.html" in nav_html
        assert "problems.html" in nav_html
        assert "conversations.html" in nav_html
        assert "agent_log.html" in nav_html
        assert "screenshots.html" in nav_html

    def test_nav_links_to_runs_index(self) -> None:
        """Nav bar should include a link back to the master Run Reports index."""
        from spark_runner.report import _nav
        nav_html = _nav("index.html", has_problems=True)
        assert "../../../index.html" in nav_html
        assert "Run Reports" in nav_html

    def test_nav_disables_problems_when_empty(self) -> None:
        from spark_runner.report import _nav
        nav_html = _nav("index.html", has_problems=False)
        assert "disabled" in nav_html


class TestAgentLogPage:
    """Tests for the Agent Log HTML page."""

    def test_no_agent_log_file(self, tmp_path: Path) -> None:
        """When agent_log.txt is missing, show a fallback message."""
        run_dir = _make_run_dir(tmp_path)
        generate_report(run_dir)
        html = (run_dir / "report" / "agent_log.html").read_text()

        assert "No agent log found" in html

    def test_renders_step_markers(self, tmp_path: Path) -> None:
        """📍 Step lines should be highlighted with the agent-step class."""
        log = (
            "2026-01-01 00:00:00,000 INFO     [Agent] Starting a browser-use agent with version 0.12.1\n"
            "2026-01-01 00:00:01,000 INFO     [Agent] 📍 Step 1:\n"
            "2026-01-01 00:00:02,000 INFO     [Agent]   🧠 Memory: remembering things\n"
        )
        run_dir = _make_run_dir(tmp_path, agent_log_content=log)
        generate_report(run_dir)
        html = (run_dir / "report" / "agent_log.html").read_text()

        assert "agent-step" in html
        assert "agent-memory" in html

    def test_strips_ansi_codes(self, tmp_path: Path) -> None:
        """ANSI escape sequences should be removed from the output."""
        log = (
            "2026-01-01 00:00:00,000 INFO     [Agent] Starting a browser-use agent\n"
            "2026-01-01 00:00:01,000 INFO     [Agent]   \x1b[34m▶️  navigate\x1b[0m\n"
        )
        run_dir = _make_run_dir(tmp_path, agent_log_content=log)
        generate_report(run_dir)
        html = (run_dir / "report" / "agent_log.html").read_text()

        assert "\x1b[" not in html
        assert "navigate" in html

    def test_splits_by_phase(self, tmp_path: Path) -> None:
        """Multiple agent sessions should produce separate <h2> sections."""
        log = (
            "2026-01-01 00:00:00,000 INFO     [Agent] Starting a browser-use agent\n"
            "2026-01-01 00:00:01,000 INFO     [Agent] 📍 Step 1:\n"
            "2026-01-01 00:00:10,000 INFO     [Agent] Starting a browser-use agent\n"
            "2026-01-01 00:00:11,000 INFO     [Agent] 📍 Step 1:\n"
        )
        run_dir = _make_run_dir(
            tmp_path,
            phases=[
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "SUCCESS", "screenshots": []},
            ],
            agent_log_content=log,
        )
        generate_report(run_dir)
        html = (run_dir / "report" / "agent_log.html").read_text()

        assert 'id="phase-1"' in html
        assert "Login</h2>" in html
        assert 'id="phase-2"' in html
        assert "Search</h2>" in html

    def test_phase_headings_have_anchor_ids(self, tmp_path: Path) -> None:
        """Agent log phase headings should have id attributes for deep-linking."""
        log = (
            "2026-01-01 00:00:00,000 INFO     [Agent] Starting a browser-use agent\n"
            "2026-01-01 00:00:01,000 INFO     [Agent] 📍 Step 1:\n"
        )
        run_dir = _make_run_dir(tmp_path, agent_log_content=log)
        generate_report(run_dir)
        html = (run_dir / "report" / "agent_log.html").read_text()

        assert '<h2 id="phase-1">' in html

    def test_bubus_lines_collapsed(self, tmp_path: Path) -> None:
        """[bubus] lines should be inside a <details> block."""
        log = (
            "2026-01-01 00:00:00,000 INFO     [Agent] Starting a browser-use agent\n"
            "2026-01-01 00:00:01,000 INFO     [Agent] 📍 Step 1:\n"
            "2026-01-01 00:00:01,500 INFO     [bubus] dispatch(SomeEvent)\n"
        )
        run_dir = _make_run_dir(tmp_path, agent_log_content=log)
        generate_report(run_dir)
        html = (run_dir / "report" / "agent_log.html").read_text()

        assert "<details>" in html
        assert "Show event bus details" in html
        assert "dispatch(SomeEvent)" in html


# ---------------------------------------------------------------------------
# Runs index tests
# ---------------------------------------------------------------------------

def _make_runs_dir(
    tmp_path: Path,
    runs: list[dict[str, object]] | None = None,
) -> Path:
    """Create a synthetic runs directory with multiple run subdirs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    if runs is None:
        runs = [
            {
                "task_name": "login-test",
                "timestamp": "2025-06-01T10-00-00",
                "prompt": "Test login flow end to end",
                "phases": [
                    {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                ],
            },
            {
                "task_name": "search-test",
                "timestamp": "2025-06-02T14-30-00",
                "prompt": "Search for products and verify results",
                "phases": [
                    {"name": "Search", "outcome": "FAILED", "screenshots": []},
                ],
            },
        ]

    for run in runs:
        task_name = str(run["task_name"])
        timestamp = str(run["timestamp"])
        run_dir = runs_dir / task_name / timestamp
        run_dir.mkdir(parents=True)
        metadata = {
            "task_name": task_name,
            "prompt": run.get("prompt", ""),
            "timestamp": timestamp.replace("T", " ").replace("-", ":", 2),
            "base_url": "https://example.com",
            "credential_profile": "default",
            "phases": run.get("phases", []),
            "screenshots": [],
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

    return runs_dir


class TestGenerateRunsIndex:
    """Test the runs-level index.html generation."""

    def test_creates_index_html(self, tmp_path: Path) -> None:
        runs_dir = _make_runs_dir(tmp_path)
        result = generate_runs_index(runs_dir)
        assert result == runs_dir / "index.html"
        assert result.exists()

    def test_lists_all_runs(self, tmp_path: Path) -> None:
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert "login-test" in html
        assert "search-test" in html
        assert "Test login flow end to end" in html
        assert "Search for products and verify results" in html

    def test_shows_status_badges(self, tmp_path: Path) -> None:
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert "badge-success" in html
        assert "badge-fail" in html
        assert "OK" in html
        assert "FAIL" in html

    def test_links_to_existing_reports(self, tmp_path: Path) -> None:
        runs_dir = _make_runs_dir(tmp_path)
        # Create a report dir for one of the runs
        report_dir = runs_dir / "login-test" / "2025-06-01T10-00-00" / "report"
        report_dir.mkdir()
        (report_dir / "index.html").write_text("<html>report</html>")

        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert "login-test/2025-06-01T10-00-00/report/index.html" in html
        # search-test has no report, should not be a link
        assert "search-test/2025-06-02T14-30-00/report/index.html" not in html

    def test_empty_runs_dir(self, tmp_path: Path) -> None:
        runs_dir = _make_runs_dir(tmp_path, runs=[])
        result = generate_runs_index(runs_dir)
        html = result.read_text()

        assert "No runs found" in html
        assert "0 run(s)" in html

    def test_shows_run_count(self, tmp_path: Path) -> None:
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert "2 run(s)" in html

    def test_truncates_long_goals(self, tmp_path: Path) -> None:
        long_goal = "x" * 200
        runs_dir = _make_runs_dir(tmp_path, runs=[
            {
                "task_name": "long-goal",
                "timestamp": "2025-01-01T00-00-00",
                "prompt": long_goal,
                "phases": [{"name": "Step", "outcome": "SUCCESS", "screenshots": []}],
            },
        ])
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        # The display text should be truncated (117 chars + "...")
        assert "..." in html
        # Full goal should still be in the title attribute for hover
        assert f'title="{long_goal}"' in html

    def test_sortable_headers(self, tmp_path: Path) -> None:
        """Table headers should have the sortable class and sort-arrow spans."""
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert 'class="sortable"' in html
        assert '<span class="sort-arrow">' in html
        for col in ("Task", "Goal", "Run Datetime", "Status"):
            assert f">{col}<" in html

    def test_data_sort_value_attributes(self, tmp_path: Path) -> None:
        """Each <td> should carry a data-sort-value attribute."""
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert 'data-sort-value="login-test"' in html
        assert 'data-sort-value="search-test"' in html
        # Datetime values
        assert 'data-sort-value="2025-06-01' in html
        assert 'data-sort-value="2025-06-02' in html

    def test_status_sort_values(self, tmp_path: Path) -> None:
        """Status cells should use '0' for FAIL and '1' for OK."""
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert 'data-sort-value="0"' in html  # FAIL
        assert 'data-sort-value="1"' in html  # OK

    def test_script_tag_present(self, tmp_path: Path) -> None:
        """The generated HTML should contain an inline <script> for sorting."""
        runs_dir = _make_runs_dir(tmp_path)
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert "<script>" in html
        assert "sort" in html.lower()

    def test_empty_runs_no_sortable_headers(self, tmp_path: Path) -> None:
        """When there are no runs, the table (and sortable headers) should not appear."""
        runs_dir = _make_runs_dir(tmp_path, runs=[])
        generate_runs_index(runs_dir)
        html = (runs_dir / "index.html").read_text()

        assert '<th class="sortable">' not in html
        assert "<thead>" not in html

    def test_generate_report_creates_runs_index(self, tmp_path: Path) -> None:
        """generate_report should also create a runs-level index."""
        runs_dir = _make_runs_dir(tmp_path)
        run_dir = runs_dir / "login-test" / "2025-06-01T10-00-00"
        generate_report(run_dir)

        assert (runs_dir / "index.html").exists()
        html = (runs_dir / "index.html").read_text()
        assert "login-test" in html
