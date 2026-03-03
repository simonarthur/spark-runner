"""Tests for observation routing to phases."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import sparky_runner
from tests.conftest import make_llm_response


class TestRouteObservationsToPhases:
    """Tests for route_observations_to_phases."""

    def test_empty_observations_returns_empty(self) -> None:
        result = sparky_runner.route_observations_to_phases(
            [], [{"name": "Login", "task": "Log in"}]
        )
        assert result == {}

    def test_empty_phases_returns_empty(self) -> None:
        result = sparky_runner.route_observations_to_phases(
            ["some observation"], []
        )
        assert result == {}

    def test_routes_observations_to_correct_phases(
        self, mock_summary_client: MagicMock
    ) -> None:
        routing_json = json.dumps({"0": [0], "1": [1]})
        mock_summary_client.messages.create.return_value = make_llm_response(
            routing_json
        )

        observations = ["Login requires MFA", "Step 2 needs checkbox selection"]
        phases = [
            {"name": "Login", "task": "Log in to the app"},
            {"name": "Create Campaign", "task": "Fill out campaign form"},
        ]

        result = sparky_runner.route_observations_to_phases(observations, phases)
        assert result["Login"] == ["Login requires MFA"]
        assert result["Create Campaign"] == ["Step 2 needs checkbox selection"]

    def test_observation_routed_to_multiple_phases(
        self, mock_summary_client: MagicMock
    ) -> None:
        routing_json = json.dumps({"0": [0, 1]})
        mock_summary_client.messages.create.return_value = make_llm_response(
            routing_json
        )

        observations = ["Error toasts appear at top-right"]
        phases = [
            {"name": "Login", "task": "Log in"},
            {"name": "Submit Form", "task": "Submit the form"},
        ]

        result = sparky_runner.route_observations_to_phases(observations, phases)
        assert result["Login"] == ["Error toasts appear at top-right"]
        assert result["Submit Form"] == ["Error toasts appear at top-right"]

    def test_unrouted_observation_excluded(
        self, mock_summary_client: MagicMock
    ) -> None:
        # Observation 1 is not in the routing — should be absent from result
        routing_json = json.dumps({"0": [0]})
        mock_summary_client.messages.create.return_value = make_llm_response(
            routing_json
        )

        observations = ["Login tip", "Irrelevant observation"]
        phases = [
            {"name": "Login", "task": "Log in"},
            {"name": "Navigate", "task": "Go to dashboard"},
        ]

        result = sparky_runner.route_observations_to_phases(observations, phases)
        assert result == {"Login": ["Login tip"]}
        assert "Navigate" not in result

    def test_parse_error_falls_back_to_all(
        self, mock_summary_client: MagicMock
    ) -> None:
        mock_summary_client.messages.create.return_value = make_llm_response(
            "I cannot parse this as JSON sorry"
        )

        observations = ["obs1", "obs2"]
        phases = [
            {"name": "Login", "task": "Log in"},
            {"name": "Navigate", "task": "Go to dashboard"},
        ]

        result = sparky_runner.route_observations_to_phases(observations, phases)
        # Fallback: all observations go to all phases
        assert result["Login"] == ["obs1", "obs2"]
        assert result["Navigate"] == ["obs1", "obs2"]

    def test_dict_format_observations_preserved(
        self, mock_summary_client: MagicMock
    ) -> None:
        routing_json = json.dumps({"0": [0], "1": [1]})
        mock_summary_client.messages.create.return_value = make_llm_response(
            routing_json
        )

        observations: list[str | dict[str, str]] = [
            {"text": "Login needs MFA", "severity": "error"},
            {"text": "Form has date picker", "severity": "warning"},
        ]
        phases = [
            {"name": "Login", "task": "Log in"},
            {"name": "Fill Form", "task": "Fill out form"},
        ]

        result = sparky_runner.route_observations_to_phases(observations, phases)
        assert result["Login"] == [{"text": "Login needs MFA", "severity": "error"}]
        assert result["Fill Form"] == [{"text": "Form has date picker", "severity": "warning"}]

    def test_markdown_wrapped_json(
        self, mock_summary_client: MagicMock
    ) -> None:
        inner = json.dumps({"0": [1]})
        mock_summary_client.messages.create.return_value = make_llm_response(
            f"```json\n{inner}\n```"
        )

        observations = ["Use polling for generation"]
        phases = [
            {"name": "Login", "task": "Log in"},
            {"name": "Generate", "task": "Generate content"},
        ]

        result = sparky_runner.route_observations_to_phases(observations, phases)
        assert "Login" not in result
        assert result["Generate"] == ["Use polling for generation"]
