# SPARK RUNNER

*"Who watches the website? We do. Tirelessly. Without blinking."*

---

There exists, in every web application, a silent covenant between the developer and the user. A promise that the button shall do what the button purports to do. That the form shall submit. That the login shall, against all entropy, *log in*. Spark Runner is the enforcer of that covenant — an autonomous browser agent that decomposes your intentions into phases, executes them through a real browser, and reports back with the unflinching honesty of a coroner's inquest.

It is built on [Browser Use](https://github.com/browser-use/browser-use) and [Claude](https://www.anthropic.com/claude). It learns from its past. It does not forget.

## Installation

```bash
pip install spark-runner
```

Or, if you prefer to work from the source — as one should, to truly understand the machinery:

```bash
git clone https://github.com/simonarthur/spark-runner.git
cd spark-runner
pip install -e .
```

**Requirements:** Python 3.11 or later. An Anthropic API key. The courage to see what your application actually does.

## First Light

```bash
spark-runner init
```

A wizard — not the robed kind, though no less powerful — will guide you through configuration. It will ask for your data directory, your base URL, your credentials, your API keys. It will offer to store your secrets as environment variable references rather than plaintext, because some things should not be written down.

When the wizard is finished, it will suggest:

```bash
spark-runner generate-goals /path/to/your/repo
```

This scans your frontend source code — your `.tsx`, your `.jsx`, your `.vue`, your `.svelte` — and extracts testable features. Each becomes a goal. Each goal, a contract to be verified.

## The Execution

At the heart of Spark Runner lies a cycle, elegant in its brutality:

1. **Decomposition.** A goal is broken into phases — discrete, ordered steps that a browser must perform.
2. **Execution.** Each phase is carried out by an autonomous agent piloting a real browser. Playwright underneath. Claude at the helm.
3. **Summarisation.** What happened is distilled into structured prose. Observations are extracted — errors, warnings, the quiet failures that pass unnoticed by human eyes.
4. **Classification.** Each observation is weighed and judged. Error or warning. Signal or noise.
5. **Knowledge.** Everything learned is stored. The next run inherits from the last. The system remembers what worked, what broke, and why.

```bash
# Run a single task from a prompt
spark-runner run -p "Log in, navigate to settings, change the display name"

# Run from goal files
spark-runner run login-task.json checkout-task.json

# Run in parallel, headless, in a specific environment
spark-runner run --parallel 3 --headless --env staging *.json
```

## Configuration

All configuration lives in `config.yaml`, placed inside your data directory (default: `~/spark_runner`).

```yaml
general:
  data_dir: ~/spark_runner
  base_url: https://your-app.example.com
  use_browseruse_llm: false

api_keys:
  anthropic: $ANTHROPIC_API_KEY
  browseruse: $BROWSER_USE_API_KEY

credentials:
  default:
    email: $SPARK_RUNNER_DEFAULT_EMAIL
    password: $SPARK_RUNNER_DEFAULT_PASSWORD
  admin:
    email: admin@example.com
    password: $SPARK_RUNNER_ADMIN_PASSWORD

environments:
  staging:
    base_url: https://staging.example.com
    credentials:
      default:
        email: $SPARK_RUNNER_STAGING_DEFAULT_EMAIL
        password: $SPARK_RUNNER_STAGING_DEFAULT_PASSWORD
  production:
    base_url: https://app.example.com
    is_production: true

models:
  task_decomposition:
    model: claude-sonnet-4-5-20250929
    max_tokens: 16384
  summarization:
    model: claude-sonnet-4-5-20250929
    max_tokens: 2048
```

Credential values support `$VAR` and `${VAR}` syntax. If the variable is set in the environment, it resolves. If not, the literal string is kept — a visible scar reminding you to set it.

## Commands

### Running Tasks

```bash
spark-runner run [OPTIONS] [GOAL_FILES...]

  -p, --prompt TEXT             Task prompt (repeatable)
  -u, --url TEXT                Override base URL
  --env TEXT                    Select environment profile
  --credential-profile TEXT     Select credential profile
  --headless                    No visible browser
  --auto-close                  Close browser when done
  --shared-session              Share browser across tasks
  --parallel N                  Parallel execution count
  --model PURPOSE=MODEL_ID      Override a model (repeatable)
  --regenerate-tasks            Re-decompose into fresh phases
  --no-knowledge-reuse          Ignore prior runs
  --no-update-summary           Don't update goal summaries
  --no-update-tasks             Don't overwrite task files
  --force-unsafe                Override production safety checks
  --unrun                       Only run goals never executed
  --failed                      Only run goals with prior errors
```

### Generating Goals

```bash
spark-runner generate-goals SOURCE_PATH [--branch main] [--output-dir DIR]
```

Accepts a local directory or a git repository URL. Scans frontend source files, extracts features, produces goal files.

### Recording Demonstrations

```bash
spark-runner record [--url URL]
```

Opens a browser. You demonstrate. The system watches, records your actions — clicks, keystrokes, navigations — and when you press Ctrl+C, it transmutes the recording into a structured goal.

### Managing Goals

```bash
spark-runner goals list [--unrun] [--failed]
spark-runner goals show GOAL_NAME
spark-runner goals delete GOAL_NAME [--force]
spark-runner goals classify
spark-runner goals orphans [--clean]
```

### Viewing Results

```bash
spark-runner results list [--task NAME]
spark-runner results show RUN_PATH
spark-runner results errors [--task NAME]
spark-runner results screenshots RUN_PATH
spark-runner results report RUN_PATH [--all]
```

Reports are self-contained HTML — screenshots, phase timelines, event logs, observations — viewable in any browser without a server.

## Knowledge Reuse

This is not a stateless tool. Each run deposits knowledge: which subtasks worked, which observations arose, what the system learned about your application. On subsequent runs, Spark Runner searches this accumulated knowledge for reusable subtasks and relevant observations. Prior failures inform future attempts. Prior successes are not re-derived from nothing.

The knowledge index lives in your `goal_summaries/` and `tasks/` directories. Disable it with `--no-knowledge-reuse` if you prefer amnesia.

## Environments & Safety

Goals may declare safety metadata:

```json
{
  "safety": {
    "blocked_in_production": true,
    "allowed_environments": ["staging", "development"],
    "risk_level": "high",
    "reason": "Creates test data"
  }
}
```

A goal marked `blocked_in_production` will refuse to run in a production environment. This is not a suggestion. Override with `--force-unsafe` if you are certain — truly certain — of what you are doing.

## LLM Models

Six model slots, each configurable independently:

| Purpose | Default | Max Tokens | Role |
|---|---|---|---|
| `task_decomposition` | claude-sonnet-4-5 | 16,384 | Breaking goals into phases |
| `summarization` | claude-sonnet-4-5 | 2,048 | Phase result summaries |
| `classification` | claude-sonnet-4-5 | 4,096 | Observation classification |
| `knowledge_matching` | claude-sonnet-4-5 | 4,096 | Finding prior knowledge |
| `task_naming` | claude-sonnet-4-5 | 64 | Short names for tasks |
| `browser_control` | claude-sonnet-4-5 | 4,096 | Reserved for future use |

Override per run:

```bash
spark-runner run --model task_decomposition=claude-opus-4-6 goal.json
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `SPARK_RUNNER_DATA_DIR` | Data directory |
| `SPARK_RUNNER_CONFIG` | Config file path |
| `SPARK_RUNNER_BASE_URL` | Base URL override |
| `ANTHROPIC_API_KEY` | Claude API key |
| `BROWSER_USE_API_KEY` | BrowserUse cloud key |

## Data Directory Structure

```
~/spark_runner/
├── config.yaml
├── tasks/                    # Phase instruction files
├── goal_summaries/           # Goal metadata (JSON)
└── runs/
    └── task-name/
        └── 2025-03-06T12-34-56Z/
            ├── metadata.json
            ├── event_log.txt
            ├── problem_log.txt
            ├── conversation_log.json
            ├── phase_summaries.json
            ├── llm_*.json            # Full LLM traces
            ├── *.png                 # Screenshots
            ├── goal/                 # Goal snapshot
            └── report/               # HTML report
                ├── index.html
                ├── goal.html
                ├── phases.html
                └── events.html
```

Every LLM call is traced. Every screenshot is kept. Every event is logged. The past is not hidden here — it is *preserved*.

## License

MIT

---

*"The accumulated filth of all their broken forms and unhandled exceptions will foam up about their waists and all the developers will look up and shout 'Does it work?' ... and Spark Runner will look down and whisper 'Here is the report.'"*
