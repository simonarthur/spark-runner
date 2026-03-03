# Browser Testing Project

The purpose of this is to test website functionality within the browser.

## Project Setup

- Python 3.13+ with virtual environment in `.venv`
- Key dependencies: `browser-use`, `playwright`, `python-dotenv`
- API keys are stored in `.env` (not committed to version control)
- Chromium is installed via Playwright at `~/.cache/ms-playwright/`

## How to activate the environment

```bash
source .venv/bin/activate
```

## How to run scripts

```bash
source .venv/bin/activate && python login.py
```

## browser-use Library Reference

- Docs: https://docs.browser-use.com
- GitHub: https://github.com/browser-use/browser-use
- Requires Python 3.11+
- Uses Playwright under the hood for browser automation

### Key imports

```python
from browser_use import Agent, Browser, ChatAnthropic
```

### Browser configuration

- `Browser(headless=False)` — opens a visible browser window
- `Browser(headless=True)` — runs headless (no window)
- `Browser(headless=False, window_size={'width': 1000, 'height': 700})` — custom size

### Supported LLM providers

| Provider | Class | Env Variable |
|---|---|---|
| Anthropic | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `ChatOpenAI` | `OPENAI_API_KEY` |
| Google | `ChatGoogle` | `GOOGLE_API_KEY` |
| Browser Use | `ChatBrowserUse` | `BROWSER_USE_API_KEY` |
| Ollama | `ChatOllama` | (none — local) |

All classes are imported directly from `browser_use`.

### Agent task tips

- Be explicit and step-by-step in the `task` string (e.g. "Go to URL, find the field, type X, click Y")
- Include the full URL including protocol (`http://` or `https://`)
- Specify exact credentials and field labels when doing login tasks
- Use `temperature=0.0` for deterministic, repeatable actions
- `agent.run()` returns a history object; call `.final_result()` for the outcome

### Key Agent parameters

- `max_failures=5` — retries per step before giving up
- `max_actions_per_step=5` — concurrent actions per step
- `save_conversation_path="conversation_log.json"` — save full agent log for debugging
- `agent.run(max_steps=50)` — max steps is a param on `run()`, not the constructor
- `result.is_done()` and `result.is_successful()` — check outcome
- Use `await browser.stop()` to close (NOT `browser.close()`)

### Script pattern

Script is in `login.py`. It uses a **phased approach** — each workflow step runs as a
separate agent sharing the same browser. This provides:
- Clear failure isolation (know exactly which phase broke)
- Automatic screenshot capture on failure
- Conversation log saved for debugging
- Poll-based waiting (check every 5s) instead of fixed long waits

```python
TASKS = [
    {"name": "Phase 1", "task": "..."},
    {"name": "Phase 2", "task": "..."},
]

async def run_phase(name, task, llm, browser):
    agent = Agent(
        task=task, llm=llm, browser=browser,
        save_conversation_path="conversation_log.json",
        max_failures=5,
    )
    result = await agent.run(max_steps=50)
    if not (result.is_done() and result.is_successful()):
        page = await browser.get_current_page()
        await page.screenshot(path=f"failure_{name}.png")
    return result.is_done() and result.is_successful()
```

## Guidelines for AI prompts in this project

- Always specify the target URL, credentials, and expected outcome
- Tests should be robust
- Mention `browser-use` and that a real (non-headless) browser should be used
- Reference this PROMPT.md so the AI has project context
- Ensure `.env` has the required API key before running
- Check for and report errors AT EACH STEP, including error popup/toast
- Report on all deviations from each step

### Robustness patterns

- **Break workflows into phases** — each phase is a separate Agent sharing one browser
- **Poll, don't fixed-wait** — "check every 5 seconds up to 60s" beats "wait 60 seconds"
- **Validate after each action** — "verify no error toast AND loading indicator is visible"
- **Type to filter dropdowns** — never rely on scrolling, always type to search/filter
- **Screenshot on failure** — capture page state for debugging when a phase fails
- **Save conversation logs** — use `save_conversation_path` for post-mortem analysis
