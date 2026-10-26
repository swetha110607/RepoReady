"""Configuration and constants for RepoReady."""

import os

APP_NAME = "RepoReady"
APP_TAGLINE = "Open-Source Repository Readiness Analyzer"
APP_DESCRIPTION = (
    "RepoReady helps open-source maintainers and contributors inspect repositories "
    "to identify setup problems, missing documentation, testing gaps, and contribution-readiness issues."
)
DEFAULT_PLACEHOLDER_URL = "https://github.com/psf/requests"

# Network & GitHub API configuration
GITHUB_API_BASE_URL = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = "RepoReady-Analyzer/0.1.0"
MAX_FILE_SIZE_BYTES = 50 * 1024  # 50 KB max per inspected text file

# Ollama / Gemma AI configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
AI_ANALYSIS_VERSION = "2.0.0"  # Increment to invalidate prior cached prompt outputs
# Allow configuration via OLLAMA_TIMEOUT or fallback to AI_REQUEST_TIMEOUT_SECONDS (default 180s)
DEFAULT_AI_TIMEOUT = 180
raw_timeout = os.environ.get("OLLAMA_TIMEOUT") or os.environ.get("AI_REQUEST_TIMEOUT_SECONDS")
try:
    AI_REQUEST_TIMEOUT_SECONDS = int(raw_timeout) if raw_timeout else DEFAULT_AI_TIMEOUT
except ValueError:
    AI_REQUEST_TIMEOUT_SECONDS = DEFAULT_AI_TIMEOUT

# Readiness Scoring Weights (Total max score = 100)
WEIGHT_README = 20
WEIGHT_LICENSE = 15
WEIGHT_TESTS = 20
WEIGHT_CI = 15
WEIGHT_CONTRIBUTING = 15
WEIGHT_MANIFEST = 15
TOTAL_POSSIBLE_SCORE = (
    WEIGHT_README
    + WEIGHT_LICENSE
    + WEIGHT_TESTS
    + WEIGHT_CI
    + WEIGHT_CONTRIBUTING
    + WEIGHT_MANIFEST
)  # 100
