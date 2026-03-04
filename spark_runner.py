"""Spark Runner – Browser Automation — thin wrapper.

This file exists for backward compatibility so that
``python spark_runner.py -p "..."`` still works.

All functionality lives in the ``spark_runner/`` package.
"""

from spark_runner.cli import legacy_main  # noqa: F401

if __name__ == "__main__":
    legacy_main()
