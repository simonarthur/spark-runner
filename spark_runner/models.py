"""Pure dataclasses used across the spark_runner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ClassificationRules:
    """Prioritized rules that guide LLM observation classification."""

    error_rules: list[str] = field(default_factory=list)
    warning_rules: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model usage."""

    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.0


@dataclass
class CredentialProfile:
    """A named set of credentials for the application under test."""

    email: str = ""
    password: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class EnvironmentProfile:
    """An environment with its own base_url and credential pool."""

    name: str = ""
    base_url: str = ""
    is_production: bool = False
    credentials: dict[str, CredentialProfile] = field(default_factory=dict)


@dataclass
class GoalSafety:
    """Safety metadata for a goal."""

    blocked_in_production: bool = False
    allowed_environments: list[str] = field(default_factory=list)
    risk_level: str = ""
    reason: str = ""


@dataclass
class SparkConfig:
    """Central configuration for spark_runner."""

    data_dir: Path = field(default_factory=lambda: Path.home() / "spark_runner")
    tasks_dir: Path | None = None
    goal_summaries_dir: Path | None = None
    runs_dir: Path | None = None
    base_url: str = "https://sparky-web-dev.vercel.app"
    credentials: dict[str, CredentialProfile] = field(default_factory=dict)
    active_credential_profile: str = "default"
    models: dict[str, ModelConfig] = field(default_factory=dict)
    environments: dict[str, EnvironmentProfile] = field(default_factory=dict)
    active_environment: str | None = None
    force_unsafe: bool = False

    # API keys
    anthropic_api_key: str = ""
    use_browseruse_llm: bool = False
    browseruse_api_key: str = ""

    # Behavior flags
    update_summary: bool = True
    update_tasks: bool = True
    knowledge_reuse: bool = True
    regenerate_tasks: bool = False
    auto_close: bool = False
    headless: bool = False
    classification_rules_path: Path = field(
        default_factory=lambda: Path("classification_rules.txt")
    )

    def __post_init__(self) -> None:
        if self.tasks_dir is None:
            self.tasks_dir = self.data_dir / "tasks"
        if self.goal_summaries_dir is None:
            self.goal_summaries_dir = self.data_dir / "goal_summaries"
        if self.runs_dir is None:
            self.runs_dir = self.data_dir / "runs"
        if not self.models:
            self.models = {
                # NOTE: browser_control is reserved but not yet wired into the
                # execution flow.  The orchestrator currently creates
                # ChatBrowserUse() without consulting this config.
                "browser_control": ModelConfig(model="claude-sonnet-4-5-20250929"),
                # Used by orchestrator for phase-result summaries
                "summarization": ModelConfig(
                    model="claude-sonnet-4-5-20250929", max_tokens=2048
                ),
                # Used by orchestrator to break a goal into phases
                "task_decomposition": ModelConfig(
                    model="claude-sonnet-4-5-20250929", max_tokens=16384
                ),
                # Used by orchestrator to classify observations
                "classification": ModelConfig(model="claude-sonnet-4-5-20250929"),
                # Used by orchestrator to find relevant prior knowledge
                "knowledge_matching": ModelConfig(model="claude-sonnet-4-5-20250929"),
                # Used by orchestrator to generate short task names
                "task_naming": ModelConfig(
                    model="claude-sonnet-4-5-20250929", max_tokens=64
                ),
            }
        if not self.credentials:
            self.credentials = {"default": CredentialProfile()}

    @property
    def active_credentials(self) -> CredentialProfile:
        """Return the currently active credential profile."""
        return self.credentials.get(
            self.active_credential_profile, CredentialProfile()
        )

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        assert self.tasks_dir is not None
        assert self.goal_summaries_dir is not None
        assert self.runs_dir is not None
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.goal_summaries_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, purpose: str) -> ModelConfig:
        """Return the model config for a given purpose, falling back to defaults."""
        return self.models.get(purpose, ModelConfig())


@dataclass
class ScreenshotRecord:
    """Metadata for a captured screenshot."""

    path: Path = field(default_factory=lambda: Path())
    event_type: str = ""  # "error" | "phase_end" | "task_end" | "step"
    phase_name: str = ""
    step_number: int | None = None
    error_message: str | None = None
    timestamp: str = ""


@dataclass
class PhaseResult:
    """Result of a single phase execution."""

    name: str = ""
    outcome: str = ""  # "SUCCESS" | "FAILED"
    summary: str = ""
    filename: str = ""
    screenshots: list[ScreenshotRecord] = field(default_factory=list)


@dataclass
class RunResult:
    """Aggregated result of a full task run."""

    task_name: str = ""
    phases: list[PhaseResult] = field(default_factory=list)
    screenshots: list[ScreenshotRecord] = field(default_factory=list)
    run_dir: Path = field(default_factory=lambda: Path())

    @property
    def all_phases_succeeded(self) -> bool:
        return all(p.outcome == "SUCCESS" for p in self.phases)

    @property
    def has_errors(self) -> bool:
        return any(p.outcome != "SUCCESS" for p in self.phases)

    @property
    def error_observations(self) -> list[str]:
        return [
            s.error_message
            for p in self.phases
            for s in p.screenshots
            if s.error_message
        ]

    @property
    def warning_observations(self) -> list[str]:
        return []


@dataclass
class TaskSpec:
    """Specification for a single task to execute."""

    prompt: str | None = None
    goal_path: Path | None = None
    credential_profile: str = "default"
