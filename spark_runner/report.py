"""HTML report generation for spark_runner runs.

Produces a ``report/`` directory with interlinked HTML pages that can be
opened directly from the local filesystem (no server required).  All CSS
is inlined — there are no external dependencies.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from spark_runner.models import ScreenshotRecord
from spark_runner.results import PhaseDetail, RunDetail, RunSummary, get_run_detail, list_runs
from spark_runner.storage import phase_name_to_slug


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _html_escape(text: str) -> str:
    """Escape ``<``, ``>``, ``&``, and ``"`` in user content."""
    return html.escape(text, quote=True)


def _css() -> str:
    """Return the inline CSS used by every page."""
    return """\
:root {
  --bg: #fafafa;
  --fg: #1a1a1a;
  --accent: #2563eb;
  --border: #d1d5db;
  --success: #16a34a;
  --fail: #dc2626;
  --muted: #6b7280;
  --pre-bg: #f3f4f6;
  --card-bg: #fff;
  --nav-bg: #1e293b;
  --nav-fg: #e2e8f0;
  --highlight-bg: #fef9c3;
}
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 0;
  line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 1rem 2rem 3rem; }
nav {
  background: var(--nav-bg);
  padding: 0.6rem 2rem;
  display: flex;
  gap: 1.2rem;
  flex-wrap: wrap;
  align-items: center;
}
nav a {
  color: var(--nav-fg);
  text-decoration: none;
  font-size: 0.95rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
}
nav a:hover { background: rgba(255,255,255,0.12); }
nav a.active { background: var(--accent); color: #fff; font-weight: 600; }
nav a.disabled { opacity: 0.4; pointer-events: none; }
h1 { margin-top: 1.5rem; font-size: 1.6rem; }
h2 { margin-top: 2rem; font-size: 1.3rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
h3 { margin-top: 1.5rem; font-size: 1.1rem; }
h4 { margin-top: 1rem; font-size: 1rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }
th { background: var(--pre-bg); font-weight: 600; }
.badge {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 4px;
  font-size: 0.85rem;
  font-weight: 600;
  color: #fff;
}
.badge-success { background: var(--success); }
.badge-fail { background: var(--fail); }
pre {
  background: var(--pre-bg);
  padding: 1rem;
  overflow-x: auto;
  border-radius: 6px;
  font-size: 0.88rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
.log-line-ts { color: var(--accent); font-weight: 600; }
.obs-block {
  background: var(--highlight-bg);
  border-left: 4px solid #eab308;
  padding: 0.75rem 1rem;
  margin: 0.75rem 0;
  border-radius: 0 6px 6px 0;
}
.thumb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.75rem;
  margin: 1rem 0;
}
.thumb-grid a { display: block; }
.thumb-grid img {
  width: 100%;
  border-radius: 6px;
  border: 1px solid var(--border);
  transition: transform 0.15s;
}
.thumb-grid img:hover { transform: scale(1.03); }
.thumb-grid .missing {
  width: 100%;
  aspect-ratio: 16/9;
  background: var(--pre-bg);
  border: 2px dashed var(--border);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  font-size: 0.85rem;
}
.thumb-caption { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; text-align: center; }
details { margin: 0.75rem 0; }
details summary {
  cursor: pointer;
  padding: 0.5rem 0.75rem;
  background: var(--pre-bg);
  border-radius: 6px;
  font-weight: 500;
}
details[open] summary { border-radius: 6px 6px 0 0; }
details .detail-body {
  border: 1px solid var(--border);
  border-top: none;
  padding: 0.75rem 1rem;
  border-radius: 0 0 6px 6px;
}
.msg { margin: 0.5rem 0; padding: 0.5rem 0.75rem; border-radius: 6px; }
.msg-user { background: #dbeafe; }
.msg-assistant { background: #f0fdf4; }
.msg-system { background: #fef3c7; }
.msg-tool { background: #f3e8ff; }
.msg-role { font-weight: 600; font-size: 0.85rem; text-transform: uppercase; color: var(--muted); }
.green-ok { color: var(--success); font-weight: 600; }
"""


_PAGE_NAMES: list[tuple[str, str]] = [
    ("index.html", "Overview"),
    ("phases.html", "Phases"),
    ("events.html", "Events"),
    ("problems.html", "Problems"),
    ("conversations.html", "Conversations"),
    ("screenshots.html", "Screenshots"),
]


def _nav(active: str, has_problems: bool) -> str:
    """Build the horizontal nav bar."""
    parts: list[str] = []
    for filename, label in _PAGE_NAMES:
        classes: list[str] = []
        if filename == active:
            classes.append("active")
        if filename == "problems.html" and not has_problems:
            classes.append("disabled")
        cls = f' class="{" ".join(classes)}"' if classes else ""
        parts.append(f'<a href="{filename}"{cls}>{_html_escape(label)}</a>')
    return "<nav>" + "".join(parts) + "</nav>"


def _page(title: str, nav_html: str, body: str) -> str:
    """Wrap *body* in a full HTML document with inline CSS and nav."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{_html_escape(title)}</title>\n"
        f"<style>{_css()}</style>\n"
        "</head>\n<body>\n"
        f"{nav_html}\n"
        f"<div class=\"container\">\n{body}\n</div>\n"
        "</body>\n</html>\n"
    )


def _markdown_to_html(text: str) -> str:
    """Lightweight Markdown → HTML for phase summaries.

    Supports headings (``#`` → ``<h2>``, ``##`` → ``<h3>``, ``###`` → ``<h4>``),
    ``**bold**``, ``- list items``, ``<OBSERVATIONS>`` blocks as highlighted
    divs, and remaining text as ``<p>`` paragraphs.
    """
    lines = text.splitlines()
    out: list[str] = []
    in_obs = False
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Observation blocks
        if stripped.upper().startswith("<OBSERVATIONS>") or stripped.upper().startswith("&LT;OBSERVATIONS&GT;"):
            in_obs = True
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append('<div class="obs-block">')
            # If there is text on the same line after the tag, include it
            inner = re.sub(r"(?i)</?observations>", "", stripped).strip()
            if inner:
                out.append(f"<p>{_inline_format(inner)}</p>")
            continue
        if stripped.upper().startswith("</OBSERVATIONS>") or stripped.upper().startswith("&LT;/OBSERVATIONS&GT;"):
            in_obs = False
            out.append("</div>")
            continue

        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", stripped)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(m.group(1)) + 1  # # → h2, ## → h3, ### → h4
            out.append(f"<h{level}>{_inline_format(m.group(2))}</h{level}>")
            continue

        # List items
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_format(stripped[2:])}</li>")
            continue

        # Blank line closes list
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue

        # Paragraph
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_inline_format(stripped)}</p>")

    if in_list:
        out.append("</ul>")
    if in_obs:
        out.append("</div>")

    return "\n".join(out)


def _inline_format(text: str) -> str:
    """Apply inline formatting: ``**bold**`` and HTML-escape."""
    escaped = _html_escape(text)
    # Bold
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _outcome_badge(outcome: str) -> str:
    """Return an HTML badge span for a phase outcome."""
    cls = "badge-success" if outcome == "SUCCESS" else "badge-fail"
    return f'<span class="badge {cls}">{_html_escape(outcome)}</span>'


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------

def _generate_index_page(
    run_dir: Path,
    detail: RunDetail,
    ss_map: dict[str, list[ScreenshotRecord]],
) -> str:
    """Generate the Overview / index page."""
    nav_html = _nav("index.html", _has_problems(run_dir))

    rows: list[str] = []
    meta_fields: list[tuple[str, str]] = [
        ("Task", detail.task_name or "(unknown)"),
        ("Prompt", detail.prompt or "(unknown)"),
        ("Timestamp", detail.timestamp or "(unknown)"),
        ("Base URL", detail.base_url or "(unknown)"),
        ("Credential Profile", detail.credential_profile or "default"),
        ("Run Directory", str(detail.run_dir)),
    ]
    for label, value in meta_fields:
        rows.append(f"<tr><th>{_html_escape(label)}</th><td>{_html_escape(value)}</td></tr>")
    meta_table = f"<table>{''.join(rows)}</table>"

    # Phase summary table
    phase_rows = ""
    if detail.phases:
        phase_rows_list: list[str] = []
        for i, phase in enumerate(detail.phases, 1):
            badge = _outcome_badge(phase.outcome)
            name_link = f'<a href="phases.html#phase-{i}">{_html_escape(phase.name)}</a>'
            ss_count = len(ss_map.get(phase.name, []))
            phase_rows_list.append(
                f"<tr><td>{i}</td><td>{name_link}</td><td>{badge}</td><td>{ss_count}</td></tr>"
            )
        phase_rows = (
            "<h2>Phases</h2>\n"
            "<table><tr><th>#</th><th>Name</th><th>Outcome</th><th>Screenshots</th></tr>\n"
            + "\n".join(phase_rows_list)
            + "\n</table>"
        )
    else:
        phase_rows = "<h2>Phases</h2>\n<p>No phase data available.</p>"

    body = f"<h1>Run Report: {_html_escape(detail.task_name or '(unknown)')}</h1>\n{meta_table}\n{phase_rows}"
    return _page(f"Report: {detail.task_name}", nav_html, body)


def _generate_phases_page(
    run_dir: Path,
    detail: RunDetail,
    phase_summaries: list[dict[str, Any]],
    ss_map: dict[str, list[ScreenshotRecord]],
) -> str:
    """Generate the Phases page with rendered markdown summaries."""
    nav_html = _nav("phases.html", _has_problems(run_dir))
    sections: list[str] = []

    for i, phase in enumerate(detail.phases, 1):
        badge = _outcome_badge(phase.outcome)
        heading = f'<h2 id="phase-{i}">Phase {i}: {_html_escape(phase.name)} {badge}</h2>'

        # Find matching summary
        summary_md = ""
        for ps in phase_summaries:
            if ps.get("name") == phase.name:
                summary_md = ps.get("summary", "")
                break
        if summary_md:
            rendered = _markdown_to_html(summary_md)
        else:
            rendered = "<p><em>No summary available.</em></p>"

        # Screenshot thumbnails (from combined per-phase + task-level mapping)
        phase_screenshots = ss_map.get(phase.name, [])
        thumbs = _render_thumbnail_grid(phase_screenshots) if phase_screenshots else ""

        sections.append(f"{heading}\n{rendered}\n{thumbs}")

    if not sections:
        sections.append("<p>No phases recorded.</p>")

    body = "<h1>Phases</h1>\n" + "\n".join(sections)
    return _page("Phases", nav_html, body)


def _generate_events_page(run_dir: Path) -> str:
    """Generate the Events page from ``event_log.txt``."""
    nav_html = _nav("events.html", _has_problems(run_dir))

    event_log = run_dir / "event_log.txt"
    if event_log.exists():
        raw = event_log.read_text()
        highlighted = _highlight_timestamps(raw)
        content = f"<pre>{highlighted}</pre>"
    else:
        content = "<p>No event log found.</p>"

    body = f"<h1>Event Log</h1>\n{content}"
    return _page("Events", nav_html, body)


def _generate_problems_page(
    run_dir: Path,
    detail: RunDetail,
    ss_map: dict[str, list[ScreenshotRecord]],
) -> str:
    """Generate the Problems page from ``problem_log.txt``."""
    nav_html = _nav("problems.html", _has_problems(run_dir))

    problem_log = run_dir / "problem_log.txt"
    if problem_log.exists() and problem_log.stat().st_size > 0:
        raw = problem_log.read_text()
        highlighted = _highlight_timestamps(raw)
        log_section = f"<pre>{highlighted}</pre>"
    else:
        log_section = '<p class="green-ok">No problems recorded.</p>'

    # Screenshots from failed phases
    ss_sections: list[str] = []
    for phase in detail.phases:
        if phase.outcome == "SUCCESS":
            continue
        phase_screenshots = ss_map.get(phase.name, [])
        if phase_screenshots:
            grid = _render_thumbnail_grid(phase_screenshots)
            ss_sections.append(
                f"<h2>{_html_escape(phase.name)}</h2>\n{grid}"
            )

    ss_html = "\n".join(ss_sections) if ss_sections else ""

    body = f"<h1>Problem Log</h1>\n{log_section}\n{ss_html}"
    return _page("Problems", nav_html, body)


def _generate_conversations_page(run_dir: Path, detail: RunDetail) -> str:
    """Generate the Conversations page from ``conversation_log.json/`` directory.

    The browser-use library writes conversation files as plain-text ``.txt``
    files named ``conversation_{UUID}_{step}.txt``.  Multiple files share the
    same UUID — each is a growing snapshot, so we only need the highest-
    numbered file per UUID.  The text format uses ``" role "`` header lines
    (e.g. ``" system "``, ``" user "``, ``" assistant "``) to delimit messages.

    Legacy JSON conversations are also supported.
    """
    nav_html = _nav("conversations.html", _has_problems(run_dir))
    sections: list[str] = []

    conv_dir = run_dir / "conversation_log.json"

    # Collect the latest file per UUID (or all JSON files)
    conv_files: list[Path] = _collect_conversation_files(conv_dir)

    if not conv_files:
        body = "<h1>Conversations</h1>\n<p>No conversation logs found.</p>"
        return _page("Conversations", nav_html, body)

    # Map files to phases (v7 UUIDs sort chronologically → 1:1 with phases)
    phase_names: list[str] = [p.name for p in detail.phases]

    for idx, cf in enumerate(conv_files):
        phase_label = phase_names[idx] if idx < len(phase_names) else f"Conversation {idx + 1}"

        if cf.suffix == ".txt":
            turns = _parse_txt_conversation(cf)
        else:
            turns = _parse_json_conversation(cf)

        sections.append(
            f"<h2>{_html_escape(phase_label)}</h2>\n"
            f"<p><em>File: {_html_escape(cf.name)}</em></p>\n"
            + "\n".join(turns)
        )

    body = "<h1>Conversations</h1>\n" + "\n".join(sections)
    return _page("Conversations", nav_html, body)


def _generate_screenshots_page(
    run_dir: Path,
    detail: RunDetail,
    ss_map: dict[str, list[ScreenshotRecord]],
) -> str:
    """Generate the Screenshots gallery page."""
    nav_html = _nav("screenshots.html", _has_problems(run_dir))
    sections: list[str] = []

    for i, phase in enumerate(detail.phases, 1):
        phase_screenshots = ss_map.get(phase.name, [])
        if not phase_screenshots:
            continue
        grid = _render_screenshot_grid(phase_screenshots)
        sections.append(
            f'<h2>Phase {i}: {_html_escape(phase.name)}</h2>\n{grid}'
        )

    # Unmatched task-level screenshots
    unmatched = ss_map.get("_unmatched", [])
    if unmatched:
        grid = _render_screenshot_grid(unmatched)
        sections.append(f'<h2>Other Screenshots</h2>\n{grid}')

    if not sections:
        sections.append("<p>No screenshots found.</p>")

    body = "<h1>Screenshots</h1>\n" + "\n".join(sections)
    return _page("Screenshots", nav_html, body)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _render_thumbnail_grid(screenshots: list[ScreenshotRecord]) -> str:
    """Render a compact thumbnail grid linking each image to the full-size file."""
    items: list[str] = []
    for ss in screenshots:
        rel = f"../screenshots/{ss.path.name}"
        caption = ss.event_type or ss.path.name
        if ss.path.exists():
            items.append(
                f'<div><a href="{_html_escape(rel)}" target="_blank"><img src="{_html_escape(rel)}" '
                f'alt="{_html_escape(ss.path.name)}"></a>'
                f'<div class="thumb-caption">{_html_escape(caption)}</div></div>'
            )
        else:
            items.append(
                f'<div><div class="missing">missing: {_html_escape(ss.path.name)}</div>'
                f'<div class="thumb-caption">{_html_escape(caption)}</div></div>'
            )
    return f'<h3>Screenshots</h3>\n<div class="thumb-grid">{"".join(items)}</div>'


def _render_screenshot_grid(screenshots: list[ScreenshotRecord]) -> str:
    """Render a full-size screenshot gallery grid."""
    items: list[str] = []
    for ss in screenshots:
        rel = f"../screenshots/{ss.path.name}"
        caption_parts: list[str] = [ss.path.name]
        if ss.event_type:
            caption_parts.append(ss.event_type)
        if ss.timestamp:
            caption_parts.append(ss.timestamp)
        caption = " | ".join(caption_parts)

        if ss.path.exists():
            items.append(
                f'<div><a href="{_html_escape(rel)}" target="_blank">'
                f'<img src="{_html_escape(rel)}" alt="{_html_escape(ss.path.name)}"></a>'
                f'<div class="thumb-caption">{_html_escape(caption)}</div></div>'
            )
        else:
            items.append(
                f'<div><div class="missing">missing: {_html_escape(ss.path.name)}</div>'
                f'<div class="thumb-caption">{_html_escape(caption)}</div></div>'
            )
    return f'<div class="thumb-grid">{"".join(items)}</div>'


def _map_screenshots_to_phases(
    detail: RunDetail,
) -> dict[str, list[ScreenshotRecord]]:
    """Build a mapping from phase name → screenshots.

    Screenshots stored in ``phase.screenshots`` are used directly.  Task-level
    screenshots (``detail.screenshots``) whose filenames start with the phase
    slug (e.g. ``login_step_001.png`` for phase "Login") are also matched.

    Returns a dict keyed by phase name.
    """
    result: dict[str, list[ScreenshotRecord]] = {}

    # Collect explicit per-phase screenshots
    for phase in detail.phases:
        result[phase.name] = list(phase.screenshots)

    # Match task-level screenshots by filename prefix
    if detail.screenshots:
        # Build slug → phase name lookup (longest slug first to avoid prefix collisions)
        slug_map: list[tuple[str, str]] = sorted(
            [(phase_name_to_slug(p.name), p.name) for p in detail.phases],
            key=lambda t: len(t[0]),
            reverse=True,
        )
        matched: set[str] = set()
        for ss in detail.screenshots:
            fname = ss.path.name
            for slug, phase_name in slug_map:
                if fname.startswith(slug + "_"):
                    result[phase_name].append(ss)
                    matched.add(fname)
                    break
        # Leftovers go into a special key
        unmatched = [ss for ss in detail.screenshots if ss.path.name not in matched]
        if unmatched:
            result["_unmatched"] = unmatched

    return result


def _has_problems(run_dir: Path) -> bool:
    """Return ``True`` if the problem log exists and is non-empty."""
    p = run_dir / "problem_log.txt"
    return p.exists() and p.stat().st_size > 0


def _highlight_timestamps(text: str) -> str:
    """HTML-escape *text* and wrap ``[YYYY-MM-DD HH:MM:SS]`` in a span."""
    escaped = _html_escape(text)
    return re.sub(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",
        r'<span class="log-line-ts">[\1]</span>',
        escaped,
    )


def _extract_message_content(msg: dict[str, Any]) -> str:
    """Extract displayable text from a conversation message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use: {block.get('name', '?')}]")
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result: {block.get('content', '')!s:.200}]")
                else:
                    parts.append(json.dumps(block, indent=2)[:500])
        return "\n".join(parts)
    return str(content)


# -- Conversation file helpers -----------------------------------------------

_CONV_FILENAME_RE = re.compile(r"^conversation_(.+?)_(\d+)\.txt$")
"""Matches ``conversation_{UUID}_{step}.txt`` and captures UUID + step."""


def _collect_conversation_files(conv_dir: Path) -> list[Path]:
    """Return one file per conversation UUID, choosing the highest step number.

    The browser-use library writes ``conversation_{UUID}_{N}.txt`` where each
    successive *N* is a growing snapshot.  We only need the last one per UUID.
    Legacy ``.json`` files are also included.
    """
    if conv_dir.is_file():
        return [conv_dir]
    if not conv_dir.is_dir():
        return []

    # Group .txt files by UUID, keeping the highest step number
    uuid_best: dict[str, tuple[int, Path]] = {}
    json_files: list[Path] = []

    for f in conv_dir.iterdir():
        if f.suffix == ".json":
            json_files.append(f)
            continue
        m = _CONV_FILENAME_RE.match(f.name)
        if m:
            uuid = m.group(1)
            step = int(m.group(2))
            prev = uuid_best.get(uuid)
            if prev is None or step > prev[0]:
                uuid_best[uuid] = (step, f)

    # Sort by UUID (v7 UUIDs sort chronologically)
    txt_files = [path for _, (_, path) in sorted(uuid_best.items())]

    # Prefer .txt files if present, fall back to .json
    return txt_files if txt_files else sorted(json_files)


def _parse_txt_conversation(path: Path) -> list[str]:
    """Parse a browser-use plain-text conversation file into HTML turns.

    The format uses lines like ``" system "``, ``" user "``, ``" assistant "``
    to delimit messages.
    """
    try:
        raw = path.read_text()
    except OSError:
        return [f"<p><em>Could not read: {_html_escape(path.name)}</em></p>"]

    # Split into (role, content) pairs
    role_re = re.compile(r"^ (system|user|assistant|tool) $", re.MULTILINE)
    parts = role_re.split(raw)

    # parts[0] is text before the first role marker (usually empty)
    # then alternating: role, content, role, content, ...
    turns: list[str] = []
    i = 1  # skip leading text
    turn_num = 0
    while i + 1 < len(parts):
        role = parts[i].strip()
        content = parts[i + 1].strip()
        i += 2
        turn_num += 1

        if len(content) > 10_000:
            content = content[:10_000] + "\n... (truncated)"

        role_cls = f"msg-{role}" if role in ("user", "assistant", "system", "tool") else "msg-system"
        turns.append(
            f'<details><summary><span class="msg-role">{_html_escape(role)}</span> '
            f"(turn {turn_num})</summary>"
            f'<div class="detail-body {role_cls}"><pre>{_html_escape(content)}</pre></div></details>'
        )

    if not turns:
        # Fall back: show the whole file as a single block
        content = raw
        if len(content) > 10_000:
            content = content[:10_000] + "\n... (truncated)"
        turns.append(f"<pre>{_html_escape(content)}</pre>")

    return turns


def _parse_json_conversation(path: Path) -> list[str]:
    """Parse a JSON conversation file into HTML turns."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return [f"<p><em>Could not parse: {_html_escape(path.name)}</em></p>"]

    messages: list[dict[str, Any]] = data if isinstance(data, list) else data.get("messages", [data])

    turns: list[str] = []
    for mi, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = _extract_message_content(msg)
        if len(content) > 10_000:
            content = content[:10_000] + "\n... (truncated)"
        role_cls = f"msg-{role}" if role in ("user", "assistant", "system", "tool") else "msg-system"
        turns.append(
            f'<details><summary><span class="msg-role">{_html_escape(role)}</span> '
            f"(turn {mi + 1})</summary>"
            f'<div class="detail-body {role_cls}"><pre>{_html_escape(content)}</pre></div></details>'
        )
    return turns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_report(run_dir: Path) -> Path:
    """Generate an HTML report for a completed run.

    Creates a ``report/`` sub-directory inside *run_dir* with six
    interlinked HTML pages.

    Args:
        run_dir: Path to the run directory containing ``run_metadata.json``
                 and other artifacts.

    Returns:
        Path to ``report/index.html``.
    """
    detail: RunDetail = get_run_detail(run_dir)

    # Load phase summaries (contains markdown not present in RunDetail)
    phase_summaries: list[dict[str, Any]] = []
    summaries_path = run_dir / "phase_summaries.json"
    if summaries_path.exists():
        try:
            phase_summaries = json.loads(summaries_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Map screenshots to phases (handles both per-phase and task-level)
    ss_map = _map_screenshots_to_phases(detail)

    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)

    pages: dict[str, str] = {
        "index.html": _generate_index_page(run_dir, detail, ss_map),
        "phases.html": _generate_phases_page(run_dir, detail, phase_summaries, ss_map),
        "events.html": _generate_events_page(run_dir),
        "problems.html": _generate_problems_page(run_dir, detail, ss_map),
        "conversations.html": _generate_conversations_page(run_dir, detail),
        "screenshots.html": _generate_screenshots_page(run_dir, detail, ss_map),
    }

    for filename, content in pages.items():
        (report_dir / filename).write_text(content)

    # Generate the runs-level index (best-effort; don't fail the run report)
    try:
        runs_dir = run_dir.parent.parent
        if runs_dir.is_dir():
            generate_runs_index(runs_dir)
    except Exception:
        pass

    return report_dir / "index.html"


def _runs_index_css() -> str:
    """Return inline CSS for the runs index page."""
    return """\
:root {
  --bg: #fafafa;
  --fg: #1a1a1a;
  --accent: #2563eb;
  --border: #d1d5db;
  --success: #16a34a;
  --fail: #dc2626;
  --muted: #6b7280;
  --pre-bg: #f3f4f6;
  --card-bg: #fff;
}
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 0;
  line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.5rem; }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }
th { background: var(--pre-bg); font-weight: 600; font-size: 0.9rem; }
td { font-size: 0.9rem; }
.badge {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 4px;
  font-size: 0.82rem;
  font-weight: 600;
  color: #fff;
}
.badge-success { background: var(--success); }
.badge-fail { background: var(--fail); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.goal-cell { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
"""


def generate_runs_index(runs_dir: Path) -> Path:
    """Generate an ``index.html`` listing all runs across all tasks.

    The page shows each run's goal (prompt), datetime, and status
    (success / failure), with links to individual run reports.

    Args:
        runs_dir: The top-level runs directory (contains task subdirs).

    Returns:
        Path to the generated ``index.html``.
    """
    summaries: list[RunSummary] = list_runs(runs_dir)

    rows: list[str] = []
    for run in summaries:
        status_label = "FAIL" if run.has_errors else "OK"
        badge_cls = "badge-fail" if run.has_errors else "badge-success"
        badge = f'<span class="badge {badge_cls}">{_html_escape(status_label)}</span>'

        # Build link to the individual run report
        report_index = run.run_dir / "report" / "index.html"
        if report_index.exists():
            rel_path = report_index.relative_to(runs_dir)
            name_cell = (
                f'<a href="{_html_escape(str(rel_path))}">'
                f'{_html_escape(run.task_name)}</a>'
            )
        else:
            name_cell = _html_escape(run.task_name)

        goal = run.prompt or "(no goal)"
        # Truncate long goals for the table display
        goal_display = goal if len(goal) <= 120 else goal[:117] + "..."

        rows.append(
            f"<tr>"
            f"<td>{name_cell}</td>"
            f'<td class="goal-cell" title="{_html_escape(goal)}">{_html_escape(goal_display)}</td>'
            f"<td>{_html_escape(run.timestamp)}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )

    if rows:
        table = (
            "<table>\n"
            "<tr><th>Task</th><th>Goal</th><th>Run Datetime</th><th>Status</th></tr>\n"
            + "\n".join(rows)
            + "\n</table>"
        )
    else:
        table = "<p>No runs found.</p>"

    body = (
        f"<h1>Run Reports</h1>\n"
        f'<p class="subtitle">{len(summaries)} run(s) found</p>\n'
        f"{table}"
    )

    page_html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>All Run Reports</title>\n"
        f"<style>{_runs_index_css()}</style>\n"
        "</head>\n<body>\n"
        f"<div class=\"container\">\n{body}\n</div>\n"
        "</body>\n</html>\n"
    )

    index_path = runs_dir / "index.html"
    index_path.write_text(page_html)
    return index_path
