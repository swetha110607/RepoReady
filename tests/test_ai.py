"""Unit tests for AI analyzer (Google Gemma integration via Ollama API) using mocked HTTP responses."""

import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.ai_analyzer import AIAnalyzer, AIAnalyzerError
from src.config import DEFAULT_AI_TIMEOUT


@pytest.fixture
def mock_findings():
    return [
        {
            "id": "DOC-001",
            "title": "Missing README file",
            "severity": "Critical",
            "explanation": "No README file found.",
            "recommendation": "Add README.md.",
            "category": "Docs",
        },
        {
            "id": "LIC-001",
            "title": "No license",
            "severity": "Critical",
            "explanation": "No open-source license detected.",
            "recommendation": "Add LICENSE.",
            "category": "Setup",
        },
    ]


@pytest.fixture
def mock_metadata():
    return {
        "name": "sample-repo",
        "owner": "test-maintainer",
        "language": "Python",
        "description": "A demo repository for testing.",
    }


def test_default_timeout_configuration():
    """Verify that default timeout is set to 180s."""
    analyzer = AIAnalyzer()
    assert analyzer.timeout == DEFAULT_AI_TIMEOUT
    assert analyzer.timeout == 180


def test_custom_timeout_configuration():
    """Verify that timeout can be overridden."""
    analyzer = AIAnalyzer(timeout=60)
    assert analyzer.timeout == 60


@patch("urllib.request.urlopen")
def test_generate_insights_success(mock_urlopen, mock_metadata, mock_findings):
    """Verify successful insight generation from mocked Ollama response."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "model": "gemma3:1b",
        "response": "### Priority Fixes\n1. Add a README.md immediately.\n2. Include an open source LICENSE.",
        "done": True,
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    analyzer = AIAnalyzer()
    insights = analyzer.generate_readiness_insights(
        metadata=mock_metadata,
        score=65,
        max_score=100,
        findings=mock_findings,
        ecosystems=["Python"],
    )

    assert "Priority Fixes" in insights
    assert "README.md" in insights


@patch("urllib.request.urlopen")
def test_generate_insights_empty_response(mock_urlopen, mock_metadata, mock_findings):
    """Verify handling when Ollama returns empty response text."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "model": "gemma3:1b",
        "response": "   ",
        "done": True,
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    analyzer = AIAnalyzer()
    with pytest.raises(AIAnalyzerError) as exc_info:
        analyzer.generate_readiness_insights(
            metadata=mock_metadata,
            score=65,
            max_score=100,
            findings=mock_findings,
            ecosystems=["Python"],
        )

    assert "empty response" in exc_info.value.user_message.lower()


@patch("urllib.request.urlopen")
def test_generate_insights_missing_model_404(mock_urlopen, mock_metadata, mock_findings):
    """Verify handling when specified model is not downloaded in Ollama (HTTP 404)."""
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="model not found",
        hdrs={},
        fp=None,
    )

    analyzer = AIAnalyzer(model="gemma3:1b")
    with pytest.raises(AIAnalyzerError) as exc_info:
        analyzer.generate_readiness_insights(
            metadata=mock_metadata,
            score=50,
            max_score=100,
            findings=mock_findings,
            ecosystems=["Python"],
        )

    assert exc_info.value.status_code == 404
    assert "not installed" in exc_info.value.user_message.lower()
    assert "ollama pull" in exc_info.value.troubleshooting_hint.lower()


@patch("urllib.request.urlopen")
def test_generate_insights_connection_failure(mock_urlopen, mock_metadata, mock_findings):
    """Verify graceful handling when Ollama daemon is not running."""
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    analyzer = AIAnalyzer()
    with pytest.raises(AIAnalyzerError) as exc_info:
        analyzer.generate_readiness_insights(
            metadata=mock_metadata,
            score=50,
            max_score=100,
            findings=mock_findings,
            ecosystems=["Python"],
        )

    assert "could not connect to ollama" in exc_info.value.user_message.lower()
    assert "ollama serve" in exc_info.value.troubleshooting_hint.lower()


@patch("urllib.request.urlopen")
def test_generate_insights_connection_timeout(mock_urlopen, mock_metadata, mock_findings):
    """Verify graceful handling when connection socket times out."""
    mock_urlopen.side_effect = urllib.error.URLError(socket.timeout("connection timed out"))

    analyzer = AIAnalyzer(timeout=180)
    with pytest.raises(AIAnalyzerError) as exc_info:
        analyzer.generate_readiness_insights(
            metadata=mock_metadata,
            score=50,
            max_score=100,
            findings=mock_findings,
            ecosystems=["Python"],
        )

    assert "connection timed out" in exc_info.value.user_message.lower()


@patch("urllib.request.urlopen")
def test_generate_insights_read_timeout(mock_urlopen, mock_metadata, mock_findings):
    """Verify handling of read/generation timeouts."""
    mock_urlopen.side_effect = TimeoutError("Timed out")

    analyzer = AIAnalyzer(timeout=180)
    with pytest.raises(AIAnalyzerError) as exc_info:
        analyzer.generate_readiness_insights(
            metadata=mock_metadata,
            score=50,
            max_score=100,
            findings=mock_findings,
            ecosystems=["Python"],
        )

    assert "timeout" in exc_info.value.user_message.lower()
    assert "180" in exc_info.value.user_message


@patch("urllib.request.urlopen")
def test_generate_insights_malformed_json(mock_urlopen, mock_metadata, mock_findings):
    """Verify handling of invalid JSON response from Ollama API."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"{ invalid json"
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    analyzer = AIAnalyzer()
    with pytest.raises(AIAnalyzerError) as exc_info:
        analyzer.generate_readiness_insights(
            metadata=mock_metadata,
            score=50,
            max_score=100,
            findings=mock_findings,
            ecosystems=["Python"],
        )

    assert "invalid response format" in exc_info.value.user_message.lower()


@patch("urllib.request.urlopen")
def test_check_availability_success(mock_urlopen):
    """Verify Ollama availability check returns installed status."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [{"name": "gemma3:1b"}, {"name": "llama3:latest"}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    analyzer = AIAnalyzer(model="gemma3:1b")
    status = analyzer.check_availability()

    assert status["available"] is True
    assert status["model_installed"] is True
    assert "gemma3:1b" in status["available_models"]


@patch("urllib.request.urlopen")
def test_check_availability_offline(mock_urlopen):
    """Verify Ollama availability check handles offline daemon safely without raising."""
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    analyzer = AIAnalyzer()
    status = analyzer.check_availability()

    assert status["available"] is False
    assert status["model_installed"] is False
    assert "error" in status
