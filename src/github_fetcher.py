"""GitHub API fetcher module for RepoReady.

Fetches public repository metadata, file trees, and selected safe text files
using GitHub REST API v3 without requiring a personal access token.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set

from src.config import GITHUB_API_BASE_URL, MAX_FILE_SIZE_BYTES, REQUEST_TIMEOUT_SECONDS, USER_AGENT
from src.utils import extract_repo_info, is_valid_github_url

logger = logging.getLogger(__name__)

# Disallowed files that must NEVER be fetched, inspected, or displayed
DISALLOWED_SECRET_PATTERNS = [
    re.compile(r"(^|/)\.env(\.(?!example|sample|template).+)?$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]*id_rsa[^/]*$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]*\.pem$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]*\.key$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]*credentials.*$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]*secret.*$", re.IGNORECASE),
    re.compile(r"(^|/)\.git/.*$", re.IGNORECASE),
]

# Whitelist of safe filenames/patterns permitted for small text content inspection
SAFE_INSPECTABLE_FILENAMES: Set[str] = {
    # Documentation
    "readme", "readme.md", "readme.rst", "readme.txt",
    "contributing", "contributing.md", "contributing.rst",
    "license", "license.md", "license.txt", "copying",
    "code_of_conduct", "code_of_conduct.md",
    # Safe env templates
    ".env.example", ".env.sample", ".env.template",
    # Manifests and configs (for verifying setup / test instructions)
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "pom.xml", "build.gradle", "cargo.toml", "go.mod"
}


class GitHubFetchError(Exception):
    """Custom exception raised for GitHub API fetching errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, user_message: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.user_message = user_message or message


def is_secret_bearing_file(path: str) -> bool:
    """Check if the given relative file path matches patterns of secret-bearing files."""
    norm_path = path.replace("\\", "/").strip()
    return any(pattern.search(norm_path) for pattern in DISALLOWED_SECRET_PATTERNS)


def is_safe_to_inspect(path: str) -> bool:
    """Check if a file path is safe to fetch and within allowed inspectable files."""
    if is_secret_bearing_file(path):
        return False
    norm_path = path.replace("\\", "/").strip().lower()
    filename = norm_path.split("/")[-1]
    return filename in SAFE_INSPECTABLE_FILENAMES


class GitHubFetcher:
    """Fetches repository metadata and structure from GitHub REST API."""

    def __init__(
        self,
        base_url: str = GITHUB_API_BASE_URL,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        user_agent: str = USER_AGENT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent

    def _make_request(self, endpoint: str) -> Any:
        """Make an HTTP GET request to GitHub REST API with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/vnd.github.v3+json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise GitHubFetchError(
                    f"Repository or resource not found (404) at {url}",
                    status_code=404,
                    user_message="Repository not found. Please ensure it is a public GitHub repository.",
                ) from e
            elif e.code in (403, 429):
                # Rate limit or access forbidden
                rate_limit_reset = e.headers.get("X-RateLimit-Reset")
                msg = "GitHub API rate limit exceeded or access forbidden. Please try again shortly."
                raise GitHubFetchError(
                    f"GitHub API returned {e.code}: {e.reason}",
                    status_code=e.code,
                    user_message=msg,
                ) from e
            else:
                raise GitHubFetchError(
                    f"GitHub API HTTP error {e.code}: {e.reason}",
                    status_code=e.code,
                    user_message=f"GitHub API error ({e.code}). Please try again later.",
                ) from e
        except urllib.error.URLError as e:
            raise GitHubFetchError(
                f"Network connection failed: {e.reason}",
                user_message="Unable to connect to GitHub. Please check your internet connection.",
            ) from e
        except TimeoutError as e:
            raise GitHubFetchError(
                "Request timed out while connecting to GitHub",
                user_message="Connection timed out while contacting GitHub API.",
            ) from e
        except json.JSONDecodeError as e:
            raise GitHubFetchError(
                "Received malformed JSON from GitHub API",
                user_message="Received an unexpected response format from GitHub.",
            ) from e

    def fetch_repo_details(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetch general repository metadata."""
        data = self._make_request(f"repos/{owner}/{repo}")
        return {
            "name": data.get("name", repo),
            "owner": owner,
            "full_name": data.get("full_name", f"{owner}/{repo}"),
            "description": data.get("description") or "No description provided.",
            "default_branch": data.get("default_branch", "main"),
            "language": data.get("language") or "Not specified",
            "stars_count": data.get("stargazers_count", 0),
            "forks_count": data.get("forks_count", 0),
            "open_issues_count": data.get("open_issues_count", 0),
            "license": data.get("license") or {},
            "archived": data.get("archived", False),
            "fork": data.get("fork", False),
            "topics": data.get("topics", []),
        }

    def fetch_file_tree(self, owner: str, repo: str, default_branch: str) -> List[Dict[str, Any]]:
        """Fetch git tree recursively using the Git Trees API."""
        endpoint = f"repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        try:
            data = self._make_request(endpoint)
            tree = data.get("tree", [])
            # Return files/folders list with path, type (blob/tree), and size
            return [
                {
                    "path": item.get("path", ""),
                    "type": item.get("type", "blob"),
                    "size": item.get("size", 0),
                }
                for item in tree
                if not is_secret_bearing_file(item.get("path", ""))
            ]
        except GitHubFetchError as e:
            # If recursive tree fails (e.g. empty repository), bubble up or handle
            logger.warning("Failed to fetch file tree for %s/%s: %s", owner, repo, e)
            raise

    def fetch_file_content(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        """Fetch and decode text content of a safe file. Enforces safety and size checks."""
        if not is_safe_to_inspect(path):
            logger.warning("Attempted to fetch disallowed or unsafe file path: %s", path)
            return None

        safe_path = urllib.parse.quote(path)
        endpoint = f"repos/{owner}/{repo}/contents/{safe_path}?ref={ref}"
        try:
            data = self._make_request(endpoint)
            # Check file size limit
            file_size = data.get("size", 0)
            if file_size > MAX_FILE_SIZE_BYTES:
                logger.info("File %s exceeds max inspectable size (%d bytes)", path, file_size)
                return None

            content_b64 = data.get("content")
            encoding = data.get("encoding")
            if content_b64 and encoding == "base64":
                raw_bytes = base64.b64decode(content_b64)
                return raw_bytes.decode("utf-8", errors="replace")
            elif "download_url" in data and not content_b64:
                # Content might be empty or direct
                return None
            return None
        except Exception as e:
            logger.info("Could not fetch content for %s: %s", path, e)
            return None

    def fetch_repository_bundle(self, url: str) -> Dict[str, Any]:
        """Fetch complete bundle (metadata, file tree, and key text files) for a GitHub URL."""
        if not is_valid_github_url(url):
            raise GitHubFetchError(
                "Invalid GitHub repository URL",
                user_message="Invalid repository URL. Please provide a link formatted like https://github.com/owner/repository",
            )

        info = extract_repo_info(url)
        if not info:
            raise GitHubFetchError(
                "Could not parse repository owner and name",
                user_message="Could not parse GitHub repository owner and name.",
            )

        owner, repo = info["owner"], info["repo"]

        # 1. Repo metadata
        metadata = self.fetch_repo_details(owner, repo)
        branch = metadata["default_branch"]

        # 2. File tree
        file_tree = self.fetch_file_tree(owner, repo, branch)

        # 3. Fetch small safe inspectable files (like README, CONTRIBUTING, .env.example)
        inspected_files: Dict[str, str] = {}
        for item in file_tree:
            path = item["path"]
            if item["type"] == "blob" and is_safe_to_inspect(path):
                # Only inspect root/near-root files or specific names
                content = self.fetch_file_content(owner, repo, path, branch)
                if content is not None:
                    inspected_files[path] = content

        return {
            "metadata": metadata,
            "file_tree": file_tree,
            "files": inspected_files,
        }
