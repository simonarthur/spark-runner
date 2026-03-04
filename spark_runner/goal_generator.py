"""Automated goal generation from frontend source code."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from spark_runner.models import ModelConfig, SparkConfig


@dataclass
class SourceFileInfo:
    """Information about a scanned source file."""

    path: Path = field(default_factory=lambda: Path())
    file_type: str = ""  # "tsx", "jsx", "vue", "svelte", "html"
    content: str = ""


@dataclass
class FeatureDescription:
    """A testable feature extracted from source code."""

    name: str = ""
    description: str = ""
    routes: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)


_FRONTEND_EXTENSIONS: set[str] = {".tsx", ".jsx", ".vue", ".svelte", ".html"}


def scan_frontend_source(source_path: Path) -> list[SourceFileInfo]:
    """Scan a directory for frontend source files.

    Args:
        source_path: Root directory to scan.

    Returns:
        A list of ``SourceFileInfo`` for each frontend file found.
    """
    files: list[SourceFileInfo] = []
    if source_path.is_file():
        if source_path.suffix in _FRONTEND_EXTENSIONS:
            files.append(SourceFileInfo(
                path=source_path,
                file_type=source_path.suffix.lstrip("."),
                content=source_path.read_text(errors="ignore"),
            ))
        return files

    for p in sorted(source_path.rglob("*")):
        if p.is_file() and p.suffix in _FRONTEND_EXTENSIONS:
            # Skip node_modules, dist, build directories
            parts = p.relative_to(source_path).parts
            if any(part in ("node_modules", "dist", "build", ".next") for part in parts):
                continue
            try:
                content = p.read_text(errors="ignore")
            except OSError:
                continue
            files.append(SourceFileInfo(
                path=p,
                file_type=p.suffix.lstrip("."),
                content=content,
            ))
    return files


def extract_testable_features(
    files: list[SourceFileInfo],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> list[FeatureDescription]:
    """Use an LLM to extract testable features from source files.

    Args:
        files: Scanned source files.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A list of testable feature descriptions.
    """
    if model_config is None:
        model_config = ModelConfig()

    if not files:
        return []

    # Build a compact representation of the source files
    file_summaries: list[str] = []
    for f in files[:50]:  # Limit to avoid token overflow
        # Truncate large files
        content = f.content[:3000] if len(f.content) > 3000 else f.content
        file_summaries.append(f"=== {f.path} ({f.file_type}) ===\n{content}\n")

    all_content = "\n".join(file_summaries)

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""Analyze these frontend source files and identify testable features that a browser automation tool could verify.

For each feature, identify:
1. Feature name (short, descriptive)
2. What it does
3. Routes/pages involved
4. Forms and inputs
5. User interactions to test

Source files:
{all_content}

Return ONLY valid JSON — an array of feature objects:
[
  {{
    "name": "Feature Name",
    "description": "What the feature does",
    "routes": ["/page1", "/page2"],
    "forms": ["login form", "search form"],
    "interactions": ["click button", "fill form", "navigate"]
  }}
]"""}],
    )

    raw_text: str = response.content[0].text.strip()
    print(f"  LLM response length: {len(raw_text)} chars")
    print(f"  LLM response (first 500 chars): {raw_text[:500]}")
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        text: str = match.group(0)
        print(f"  Extracted JSON array length: {len(text)} chars")
    else:
        print("  Warning: no JSON array found in response")
        print(f"  Full response:\n{raw_text}")
        return []
    try:
        raw_features: list[dict[str, Any]] = json.loads(text)
        return [
            FeatureDescription(
                name=f.get("name", ""),
                description=f.get("description", ""),
                routes=f.get("routes", []),
                forms=f.get("forms", []),
                interactions=f.get("interactions", []),
            )
            for f in raw_features
        ]
    except (json.JSONDecodeError, AttributeError) as exc:
        print(f"Warning: could not parse feature extraction response: {exc}")
        print(f"  Extracted text (first 500 chars): {text[:500]}")
        return []


def generate_goals_from_features(
    features: list[FeatureDescription],
    output_dir: Path,
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> list[Path]:
    """Generate goal JSON files from extracted features.

    Args:
        features: Extracted feature descriptions.
        output_dir: Directory to write goal files.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        Paths to the generated goal files.
    """
    if model_config is None:
        model_config = ModelConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    goal_paths: list[Path] = []

    for feature in features:
        # Generate a prompt for this feature
        response: anthropic.types.Message = client.messages.create(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            messages=[{"role": "user", "content": f"""Generate a browser automation test goal for this feature:

Feature: {feature.name}
Description: {feature.description}
Routes: {', '.join(feature.routes) if feature.routes else 'N/A'}
Forms: {', '.join(feature.forms) if feature.forms else 'N/A'}
Interactions: {', '.join(feature.interactions) if feature.interactions else 'N/A'}

Return ONLY valid JSON with this structure:
{{
  "main_task": "One-line description of what to test",
  "key_observations": [],
  "subtasks": []
}}"""}],
        )

        text: str = response.content[0].text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            goal_data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            print(f"  Warning: could not generate goal for '{feature.name}'")
            continue

        # Add timestamps
        now_iso: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        goal_data["created_at"] = now_iso
        goal_data["updated_at"] = now_iso

        # Create filename from feature name
        slug = re.sub(r"[^a-z0-9]+", "-", feature.name.lower()).strip("-")
        goal_path = output_dir / f"{slug}-task.json"
        goal_path.write_text(json.dumps(goal_data, indent=2))
        goal_paths.append(goal_path)
        print(f"  Generated goal: {goal_path.name}")

    return goal_paths


async def generate_goals_from_source(
    source_path: Path,
    output_dir: Path,
    config: SparkConfig,
    branch: str = "main",
) -> list[Path]:
    """End-to-end: scan source, extract features, generate goals.

    Args:
        source_path: Path to source directory or a git URL.
        output_dir: Directory for generated goal files.
        config: Configuration.
        branch: Git branch for repo URLs.

    Returns:
        Paths to generated goal files.
    """
    client: anthropic.Anthropic = anthropic.Anthropic()
    model_config = config.get_model("task_decomposition")

    # Check if source_path looks like a URL
    source_str = str(source_path)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        source_path = clone_and_scan_repo(source_str, branch)

    print(f"Scanning source: {source_path}")
    files = scan_frontend_source(source_path)
    print(f"  Found {len(files)} frontend file(s)")

    if not files:
        print("  No frontend files found.")
        return []

    print("Extracting testable features...")
    features = extract_testable_features(files, client, model_config)
    print(f"  Found {len(features)} testable feature(s)")

    if not features:
        print("  No testable features found.")
        return []

    print(f"Generating goals to {output_dir}...")
    return generate_goals_from_features(features, output_dir, client, model_config)


def clone_and_scan_repo(repo_url: str, branch: str = "main") -> Path:
    """Clone a git repo to a temp directory and return the path.

    Args:
        repo_url: URL of the git repository.
        branch: Branch to clone.

    Returns:
        Path to the cloned directory.
    """
    import subprocess
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="spark_goals_"))
    print(f"  Cloning {repo_url} (branch: {branch}) to {tmpdir}...")
    subprocess.run(
        ["git", "clone", "--depth=1", f"--branch={branch}", repo_url, str(tmpdir)],
        check=True,
        capture_output=True,
    )
    return tmpdir
