"""Tests for the agent log file-handler helpers in spark_runner.log."""

from __future__ import annotations

import logging
from pathlib import Path

from spark_runner.log import (
    _AGENT_LOG_LOGGERS,
    attach_agent_log_handler,
    detach_agent_log_handler,
)


class TestAttachAgentLogHandler:
    """Verify attach/detach lifecycle for the agent log file handler."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        handler = attach_agent_log_handler(tmp_path)
        try:
            assert (tmp_path / "agent_log.txt").exists()
        finally:
            detach_agent_log_handler(handler)

    def test_handler_attached_to_expected_loggers(self, tmp_path: Path) -> None:
        handler = attach_agent_log_handler(tmp_path)
        try:
            for name in _AGENT_LOG_LOGGERS:
                assert handler in logging.getLogger(name).handlers
        finally:
            detach_agent_log_handler(handler)

    def test_handler_removed_after_detach(self, tmp_path: Path) -> None:
        handler = attach_agent_log_handler(tmp_path)
        detach_agent_log_handler(handler)
        for name in _AGENT_LOG_LOGGERS:
            assert handler not in logging.getLogger(name).handlers

    def test_log_messages_written_to_file(self, tmp_path: Path) -> None:
        handler = attach_agent_log_handler(tmp_path)
        try:
            logging.getLogger("browser_use").info("step 1: click button")
            logging.getLogger("bubus").info("event bus msg")
        finally:
            detach_agent_log_handler(handler)

        content = (tmp_path / "agent_log.txt").read_text()
        assert "step 1: click button" in content
        assert "event bus msg" in content

    def test_unrelated_logger_not_captured(self, tmp_path: Path) -> None:
        handler = attach_agent_log_handler(tmp_path)
        try:
            logging.getLogger("unrelated").warning("should not appear")
        finally:
            detach_agent_log_handler(handler)

        content = (tmp_path / "agent_log.txt").read_text()
        assert "should not appear" not in content

    def test_format_includes_logger_name(self, tmp_path: Path) -> None:
        handler = attach_agent_log_handler(tmp_path)
        try:
            logging.getLogger("browser_use").info("memory update")
        finally:
            detach_agent_log_handler(handler)

        content = (tmp_path / "agent_log.txt").read_text()
        assert "[browser_use]" in content
        assert "memory update" in content
