"""Automated tests for RepoReady."""

from src import __version__
from src.config import APP_NAME
from src.utils import extract_repo_info, is_valid_github_url


def test_package_metadata():
    """Verify package metadata is accessible."""
    assert __version__ == "0.1.0"
    assert APP_NAME == "RepoReady"


def test_is_valid_github_url():
    """Verify GitHub URL validator handles valid and invalid inputs."""
    assert is_valid_github_url("https://github.com/psf/requests")
    assert is_valid_github_url("http://github.com/streamlit/streamlit")
    assert is_valid_github_url("https://www.github.com/fastapi/fastapi")
    assert is_valid_github_url("https://github.com/pallets/flask.git")

    assert not is_valid_github_url("")
    assert not is_valid_github_url("https://gitlab.com/user/repo")
    assert not is_valid_github_url("not-a-url")
    assert not is_valid_github_url("https://github.com")


def test_extract_repo_info():
    """Verify repo and owner parsing."""
    info = extract_repo_info("https://github.com/psf/requests")
    assert info == {"owner": "psf", "repo": "requests"}

    info_git = extract_repo_info("https://github.com/psf/requests.git")
    assert info_git == {"owner": "psf", "repo": "requests"}

    assert extract_repo_info("invalid_url") is None
