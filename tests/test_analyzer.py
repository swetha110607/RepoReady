"""Unit tests for deterministic static analysis checks, ecosystem detection, and scoring boundaries."""

import pytest

from src.config import TOTAL_POSSIBLE_SCORE
from src.static_analyzer import StaticAnalyzer


def create_mock_bundle(paths, files=None, metadata=None):
    """Helper to build a mock repo bundle for analyzer tests."""
    return {
        "metadata": metadata or {"language": "Python", "license": {"key": "mit"}},
        "file_tree": [{"path": p, "type": "blob", "size": 100} for p in paths],
        "files": files or {},
    }


def test_perfect_python_repository():
    """Verify that a fully equipped Python repo receives full score and minimal findings."""
    paths = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        ".github/workflows/ci.yml",
        ".github/issue_template/bug_report.md",
        "pyproject.toml",
        "tests/test_main.py",
        "src/main.py",
    ]
    files = {
        "README.md": "# Sample Project\n\n## Getting Started\nInstall dependencies via pip:\n```bash\npip install .\n```\n\n## Usage\nRun the app:\n```bash\npython main.py\n```\n\n## Testing\nRun tests using pytest.",
    }
    bundle = create_mock_bundle(paths, files)
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    assert results["score"] == TOTAL_POSSIBLE_SCORE
    assert results["score"] == 100
    assert len(results["findings"]) == 0
    assert "Python" in results["ecosystems"]


def test_missing_readme_finding():
    """Verify missing README detection and score deduction."""
    paths = [
        "LICENSE",
        "CONTRIBUTING.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "tests/test_main.py",
    ]
    bundle = create_mock_bundle(paths)
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    finding_ids = [f["id"] for f in results["findings"]]
    assert "DOC-001" in finding_ids
    assert results["score"] < 100


def test_missing_license_finding():
    """Verify missing LICENSE detection."""
    paths = [
        "README.md",
        "CONTRIBUTING.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "tests/test_main.py",
    ]
    bundle = create_mock_bundle(paths, metadata={"language": "Python", "license": {}})
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    finding_ids = [f["id"] for f in results["findings"]]
    assert "LIC-001" in finding_ids


def test_missing_tests_finding():
    """Verify missing tests detection."""
    paths = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "src/app.py",
    ]
    files = {"README.md": "# Sample\n\n## Setup\npip install .\n\n## Usage\nRun app."}
    bundle = create_mock_bundle(paths, files)
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    finding_ids = [f["id"] for f in results["findings"]]
    assert "TEST-001" in finding_ids


def test_missing_ci_finding():
    """Verify missing CI detection."""
    paths = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "pyproject.toml",
        "tests/test_sample.py",
    ]
    files = {"README.md": "# Sample\n\n## Setup\npip install .\n\n## Usage\nRun app."}
    bundle = create_mock_bundle(paths, files)
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    finding_ids = [f["id"] for f in results["findings"]]
    assert "CI-001" in finding_ids


def test_javascript_ecosystem_detection():
    """Verify detection of JavaScript/TypeScript ecosystem with package.json and jest tests."""
    paths = [
        "package.json",
        "README.md",
        "LICENSE",
        ".github/workflows/test.yaml",
        "src/index.ts",
        "__tests__/index.test.ts",
    ]
    files = {"README.md": "# JS App\n\n## Setup\nnpm install\n\n## Usage\nnpm start"}
    bundle = create_mock_bundle(paths, files, metadata={"language": "TypeScript", "license": {"key": "mit"}})
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    assert "JavaScript/TypeScript" in results["ecosystems"]
    # Should find manifests and tests
    finding_ids = [f["id"] for f in results["findings"]]
    assert "TEST-001" not in finding_ids
    assert "MNF-001" not in finding_ids


def test_safe_env_example_recommendation():
    """Verify that a repo referencing env vars in README without .env.example produces a recommendation."""
    paths = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "tests/test_all.py",
    ]
    files = {
        "README.md": "# App\n\n## Setup\nSet your API_KEY environment variable in your terminal before running."
    }
    bundle = create_mock_bundle(paths, files)
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    finding_ids = [f["id"] for f in results["findings"]]
    assert "ENV-001" in finding_ids


def test_score_boundary_conditions():
    """Verify that empty/barren repository score does not fall below zero."""
    bundle = create_mock_bundle([], metadata={})
    analyzer = StaticAnalyzer(bundle)
    results = analyzer.analyze()

    assert results["score"] >= 0
    assert results["score"] <= 100
    assert results["severity_counts"]["Critical"] >= 1
