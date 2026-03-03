"""SparkyAI Browser Automation Runner — thin wrapper.

This file exists for backward compatibility so that
``python sparky_runner.py -p "..."`` still works.

All functionality lives in the ``sparky_runner/`` package.
"""

from sparky_runner.cli import legacy_main  # noqa: F401

if __name__ == "__main__":
    legacy_main()
