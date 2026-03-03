"""One-time script to replace plaintext credentials with placeholders in existing files."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

USER_EMAIL: str = os.environ.get("USER_EMAIL", "")
USER_PASSWORD: str = os.environ.get("USER_PASSWORD", "")

# Known password variants that have appeared in files due to typos or old values
PASSWORD_VARIANTS: list[str] = [
    USER_PASSWORD,  # bolts5151
    "bolts5451",
    "bolts5551",
    "8ECV7hat!",
]


def sanitize_file(path: Path) -> bool:
    """Replace credentials with placeholders in a single file. Returns True if modified."""
    original: str = path.read_text()
    text: str = original

    if USER_EMAIL:
        text = text.replace(USER_EMAIL, "{USER_EMAIL}")

    for variant in PASSWORD_VARIANTS:
        if variant:
            text = text.replace(variant, "{USER_PASSWORD}")

    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    tasks_dir = Path("tasks")
    goals_dir = Path("goal_summaries")

    modified: int = 0
    scanned: int = 0

    for pattern, directory in [("*.txt", tasks_dir), ("*.json", goals_dir)]:
        if not directory.exists():
            continue
        for filepath in sorted(directory.glob(pattern)):
            scanned += 1
            if sanitize_file(filepath):
                print(f"  Sanitized: {filepath}")
                modified += 1

    print(f"\nDone. Scanned {scanned} files, modified {modified}.")


if __name__ == "__main__":
    main()
