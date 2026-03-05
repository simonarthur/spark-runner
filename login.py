import asyncio
import json
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from browser_use import Agent, Browser, ChatAnthropic, ChatBrowserUse

load_dotenv()

summary_client = anthropic.Anthropic()
#HOST = "http://localhost:3000"
HOST = "https://sparky-web-staging.vercel.app/"
CONVERSATION_LOG = Path("conversation_log.json")

TASKS = [
    {
        "name": "Login",
        "task": (
            "Check for error popup/toast after every action. "
            "Report any deviations from expected behavior.\n"
            f"Go to {HOST} and log in with the following credentials:\n"
            "  Email: chroma@mindspring.com\n"
            "  Password: bolts5151\n"
            "Click the login/sign-in button and wait for the page to load after login.\n"
            "Verify you are logged in by confirming the dashboard/home page is visible."
        ),
    },
    {
        "name": "Fill blog form",
        "task": (
            "Check for error popup/toast after every action. "
            "Report any deviations from expected behavior.\n"
            "Click the SparkyExpress 'Write a Blog' button and wait for the page to load.\n"
            "Enter the topic 'The Future of AI' into the 'Topic' field.\n"
            "Enter the Keywords 'AI, Future, Technology' into the 'Keywords' field.\n"
            "Enter 2 as the Number of Sections.\n"
            "Enter the Target Audience 'Tech Enthusiasts' into the 'Target Audience' field.\n"
            "For the Tone field: click on the Tone dropdown/selector to open it. "
            "Do NOT scroll. Instead, look for the option text and click directly on 'Professional' to select it. "
            "Then click the Tone field again and click directly on 'Informative' to select it. "
            "If the options are not visible, try typing the option name to filter/search for it.\n"
            "Verify all fields are filled correctly before proceeding."
        ),
    },
    {
        "name": "Generate outline",
        "task": (
            "Check for error popup/toast after every action. "
            "Report any deviations from expected behavior.\n"
            "Click the 'Generate' button.\n"
            "Verify no error toast appeared AND that a loading indicator is visible.\n"
            "Check every 5 seconds until the outline content appears on the page, "
            "up to a maximum of 60 seconds. Do NOT wait in one long block — "
            "check the page after each 5-second wait.\n"
            "Once the outline is visible and fully loaded, click the 'Next' button to proceed. "
            "If no 'Next' button is visible, scroll down to find it. "
            "The Next button may only become enabled after the outline finishes generating."
        ),
    },
    {
        "name": "Generate blog body",
        "task": (
            "Check for error popup/toast after every action. "
            "Report any deviations from expected behavior.\n"
            "Click the 'Generate' button to generate the blog body.\n"
            "Verify no error toast appeared AND that a loading indicator is visible.\n"
            "Check every 5 seconds until the blog body content appears on the page, "
            "up to a maximum of 60 seconds. Do NOT wait in one long block — "
            "check the page after each 5-second wait.\n"
            "Once the blog body is fully visible, report the blog title and confirm completion."
        ),
    },
]


def extract_phase_history(result) -> str:
    """Extract a structured text log from agent history for LLM summarization."""
    lines = []
    for i, h in enumerate(result.history, 1):
        lines.append(f"--- Step {i} ---")
        if h.model_output:
            mo = h.model_output
            if mo.evaluation_previous_goal:
                lines.append(f"  Eval: {mo.evaluation_previous_goal}")
            if mo.memory:
                lines.append(f"  Memory: {mo.memory}")
            if mo.next_goal:
                lines.append(f"  Next goal: {mo.next_goal}")
            for action in mo.action:
                lines.append(f"  Action: {action.model_dump(exclude_none=True)}")
        for r in h.result:
            if r.error:
                lines.append(f"  ERROR: {r.error}")
            if r.extracted_content:
                lines.append(f"  Extracted: {r.extracted_content}")
        if h.state and h.state.url:
            lines.append(f"  URL: {h.state.url}")
    return "\n".join(lines)


def summarize_phase(phase_name: str, phase_task: str, result, success: bool) -> str:
    """Use an LLM to summarize what happened during a phase."""
    history_text = extract_phase_history(result)
    final = result.final_result() or "(no final result)"
    errors = [e for e in result.errors() if e]

    prompt = f"""You are analyzing the results of an automated browser testing phase.

Phase: {phase_name}
Outcome: {"SUCCESS" if success else "FAILED"}
Final result: {final}
{"Errors encountered: " + "; ".join(errors) if errors else "No errors."}

Original task instructions:
{phase_task}

Step-by-step agent history:
{history_text}

Produce a structured summary with these sections:
1. **Outcome**: One-line success/failure statement.
2. **Sub-phases**: Break the phase into logical sub-phases (e.g. "Navigate to page", "Enter credentials", "Click submit", "Verify result"). For each sub-phase state what happened and whether it succeeded.
3. **Page state**: Describe the current state of the page (URL, what's visible, any modals/toasts).
4. **Observations**: Any unexpected behavior, warnings, retries, or deviations from the instructions.
5. **Key facts learned**: Concrete details about the UI (element names, selectors that worked, layout info) that would help a follow-up agent.

Be concise but specific. Use the actual element names and URLs from the history."""

    response = summary_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response.content[0].text
    return summary


def build_augmented_task(original_task: str, prior_summaries: list[dict]) -> str:
    """Prepend accumulated phase summaries as context for the next phase's task."""
    if not prior_summaries:
        return original_task

    context_parts = ["=== CONTEXT FROM PRIOR PHASES ==="]
    for s in prior_summaries:
        context_parts.append(f"\n-- Phase: {s['name']} ({s['outcome']}) --")
        context_parts.append(s["summary"])
    context_parts.append("\n=== YOUR TASK (use the context above to inform your actions) ===\n")

    return "\n".join(context_parts) + original_task


async def run_phase(name, task, llm, browser):
    """Run a single phase of the workflow. Returns (success, result)."""
    print(f"\n{'='*60}")
    print(f"  Phase: {name}")
    print(f"{'='*60}")

    assets_dir = Path(__file__).resolve().parent / "spark_runner" / "assets"
    available_files = [
        str(p) for p in assets_dir.iterdir() if p.is_file()
    ]

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        save_conversation_path=str(CONVERSATION_LOG),
        max_failures=5,
        max_actions_per_step=5,
        use_judge=False,
        available_file_paths=available_files,
    )

    result = await agent.run(max_steps=50)
    success = result.is_done() and result.is_successful()

    if success:
        print(f"  Phase '{name}' succeeded: {result.final_result()}")
    else:
        print(f"  Phase '{name}' FAILED: {result.final_result()}")
        # Take a screenshot for debugging
        try:
            screenshot_path = f"failure_{name.replace(' ', '_')}.png"
            page = await browser.get_current_page()
            await page.screenshot(screenshot_path)
            print(f"  Failure screenshot saved to {screenshot_path}")
        except Exception as e:
            print(f"  Could not save failure screenshot: {e}")

    return success, result


async def main():
    user_data_dir = Path(tempfile.mkdtemp(prefix="browser-use-user-data-dir-"))
    prefs_dir = user_data_dir / "Default"
    prefs_dir.mkdir(parents=True, exist_ok=True)
    (prefs_dir / "Preferences").write_text(json.dumps({
        "credentials_enable_service": False,
        "profile": {"password_manager_enabled": False},
    }))
    browser = Browser(
        headless=False,
        keep_alive=True,
        user_data_dir=str(user_data_dir),
    )
    #llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.0)
    llm = ChatBrowserUse()

    try:
        prior_summaries = []
        for phase in TASKS:
            augmented_task = build_augmented_task(phase["task"], prior_summaries)
            print(f"\n  Task (with context):\n{augmented_task}\n")

            success, result = await run_phase(
                phase["name"], augmented_task, llm, browser
            )

            # Summarize what was learned in this phase
            print(f"\n  Summarizing phase '{phase['name']}'...")
            summary = summarize_phase(
                phase["name"], phase["task"], result, success
            )
            prior_summaries.append({
                "name": phase["name"],
                "outcome": "SUCCESS" if success else "FAILED",
                "summary": summary,
            })
            print(f"\n{'─'*60}")
            print(f"  Phase summary: {phase['name']}")
            print(f"{'─'*60}")
            print(summary)
            print(f"{'─'*60}")

            if not success:
                print(f"\nAborting: phase '{phase['name']}' failed.")
                break
        else:
            print("\nAll phases completed successfully.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        try:
            page = await browser.get_current_page()
            await page.screenshot("failure_unexpected.png")
            print("Failure screenshot saved to failure_unexpected.png")
        except Exception:
            pass
    finally:
        print(f"\nConversation log saved to {CONVERSATION_LOG}")
        input("Press Enter to close the browser...")
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
