"""Utilities for validating and parsing repository inputs."""

import re

# Basic regex for GitHub public repo URL pattern: https://github.com/{owner}/{repo}
GITHUB_REPO_URL_REGEX = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)(?:/.*)?$",
    re.IGNORECASE,
)


def is_valid_github_url(url: str) -> bool:
    """Check if the provided string is a valid GitHub repository URL format."""
    if not url or not isinstance(url, str):
        return False
    return bool(GITHUB_REPO_URL_REGEX.match(url.strip()))


def extract_repo_info(url: str) -> dict[str, str] | None:
    """Extract owner and repo name from a valid GitHub URL."""
    if not url or not isinstance(url, str):
        return None
    match = GITHUB_REPO_URL_REGEX.match(url.strip())
    if not match:
        return None
    owner = match.group("owner")
    repo = match.group("repo")
    # Clean possible .git suffix
    if repo.endswith(".git"):
        repo = repo[:-4]
    return {"owner": owner, "repo": repo}
