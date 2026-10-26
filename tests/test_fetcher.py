"""Unit tests for GitHub fetching logic, URL safety, and error handling with mocked responses."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.github_fetcher import (
    GitHubFetcher,
    GitHubFetchError,
    is_safe_to_inspect,
    is_secret_bearing_file,
)


def test_is_secret_bearing_file():
    """Verify that secret-bearing and sensitive files are flagged."""
    assert is_secret_bearing_file(".env")
    assert is_secret_bearing_file(".env.local")
    assert is_secret_bearing_file(".env.production")
    assert is_secret_bearing_file("config/id_rsa")
    assert is_secret_bearing_file("certs/server.key")
    assert is_secret_bearing_file("secrets.json")
    assert is_secret_bearing_file("aws_credentials.txt")
    assert is_secret_bearing_file(".git/config")

    # Safe files
    assert not is_secret_bearing_file("README.md")
    assert not is_secret_bearing_file(".env.example")
    assert not is_secret_bearing_file("src/main.py")


def test_is_safe_to_inspect():
    """Verify that only explicitly permitted small text files can be inspected."""
    assert is_safe_to_inspect("README.md")
    assert is_safe_to_inspect("CONTRIBUTING.md")
    assert is_safe_to_inspect("LICENSE")
    assert is_safe_to_inspect(".env.example")
    assert is_safe_to_inspect("pyproject.toml")
    assert is_safe_to_inspect("package.json")

    # Sensitive or code files must not be inspected directly
    assert not is_safe_to_inspect(".env")
    assert not is_safe_to_inspect("app.py")
    assert not is_safe_to_inspect("server.key")
    assert not is_safe_to_inspect("random_binary.exe")


def test_github_fetcher_url_validation_rejection():
    """Verify fetcher rejects invalid and untrusted URLs before making any network calls."""
    fetcher = GitHubFetcher()

    with pytest.raises(GitHubFetchError) as exc_info:
        fetcher.fetch_repository_bundle("https://gitlab.com/owner/repo")
    assert "Invalid" in exc_info.value.user_message

    with pytest.raises(GitHubFetchError):
        fetcher.fetch_repository_bundle("ftp://github.com/owner/repo")

    with pytest.raises(GitHubFetchError):
        fetcher.fetch_repository_bundle("not-a-valid-url")


@patch("urllib.request.urlopen")
def test_fetch_repo_details_success(mock_urlopen):
    """Verify repository metadata parsing from API."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "name": "repoready",
        "description": "Readiness analyzer",
        "default_branch": "main",
        "language": "Python",
        "stargazers_count": 42,
        "forks_count": 5,
        "open_issues_count": 2,
        "license": {"key": "mit", "name": "MIT License"},
    }).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    fetcher = GitHubFetcher()
    details = fetcher.fetch_repo_details("testowner", "repoready")

    assert details["name"] == "repoready"
    assert details["default_branch"] == "main"
    assert details["language"] == "Python"
    assert details["license"]["key"] == "mit"


@patch("urllib.request.urlopen")
def test_fetch_404_error_handling(mock_urlopen):
    """Verify clean user error on repository 404."""
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://api.github.com/repos/owner/missing",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )

    fetcher = GitHubFetcher()
    with pytest.raises(GitHubFetchError) as exc_info:
        fetcher.fetch_repo_details("owner", "missing")
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.user_message.lower()


@patch("urllib.request.urlopen")
def test_fetch_rate_limit_error_handling(mock_urlopen):
    """Verify handling of rate limit (403/429) errors."""
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://api.github.com/repos/owner/repo",
        code=403,
        msg="rate limit exceeded",
        hdrs={"X-RateLimit-Reset": "1700000000"},
        fp=None,
    )

    fetcher = GitHubFetcher()
    with pytest.raises(GitHubFetchError) as exc_info:
        fetcher.fetch_repo_details("owner", "repo")
    assert exc_info.value.status_code == 403
    assert "rate limit" in exc_info.value.user_message.lower()


@patch("urllib.request.urlopen")
def test_fetch_malformed_json_handling(mock_urlopen):
    """Verify handling of malformed API JSON payload."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"{ invalid json content"
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    fetcher = GitHubFetcher()
    with pytest.raises(GitHubFetchError) as exc_info:
        fetcher.fetch_repo_details("owner", "repo")
    assert "unexpected response format" in exc_info.value.user_message.lower()


@patch("urllib.request.urlopen")
def test_fetch_tree_filters_secret_files(mock_urlopen):
    """Verify that file tree discovery strips out secret files."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "tree": [
            {"path": "README.md", "type": "blob", "size": 1024},
            {"path": ".env", "type": "blob", "size": 128},
            {"path": ".env.local", "type": "blob", "size": 128},
            {"path": "secrets/key.pem", "type": "blob", "size": 512},
            {"path": "src/app.py", "type": "blob", "size": 2048},
        ]
    }).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    fetcher = GitHubFetcher()
    tree = fetcher.fetch_file_tree("owner", "repo", "main")
    paths = [item["path"] for item in tree]

    assert "README.md" in paths
    assert "src/app.py" in paths
    assert ".env" not in paths
    assert ".env.local" not in paths
    assert "secrets/key.pem" not in paths
