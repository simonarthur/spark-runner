"""Tests for HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

from sparky_runner.report import (
    _html_escape,
    _markdown_to_html,
    generate_report,
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

    return run_dir


_EXPECTED_FILES = [
    "index.html",
    "phases.html",
    "events.html",
    "problems.html",
    "conversations.html",
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

    def test_conversations_truncates_long_messages(self, tmp_path: Path) -> None:
        long_msg = "x" * 15_000
        run_dir = _make_run_dir(
            tmp_path,
            conversation_files={
                "aaaa.json": [{"role": "assistant", "content": long_msg}],
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        assert "truncated" in conv_html.lower()
        # The full 15k string should NOT appear
        assert long_msg not in conv_html

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

    def test_txt_conversations_truncates_long_messages(self, tmp_path: Path) -> None:
        long_body = "x" * 15_000
        run_dir = _make_run_dir(
            tmp_path,
            conversation_files={
                "conversation_aaaa_1.txt": f" assistant \n{long_body}",
            },
        )
        generate_report(run_dir)
        conv_html = (run_dir / "report" / "conversations.html").read_text()

        assert "truncated" in conv_html.lower()
        assert long_body not in conv_html

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
        """Verify all 6 pages appear in the nav bar."""
        from sparky_runner.report import _nav
        nav_html = _nav("index.html", has_problems=True)
        assert "index.html" in nav_html
        assert "phases.html" in nav_html
        assert "events.html" in nav_html
        assert "problems.html" in nav_html
        assert "conversations.html" in nav_html
        assert "screenshots.html" in nav_html

    def test_nav_disables_problems_when_empty(self) -> None:
        from sparky_runner.report import _nav
        nav_html = _nav("index.html", has_problems=False)
        assert "disabled" in nav_html
