"""Deterministic static repository readiness analyzer for RepoReady.

Performs rule-based, ecosystem-aware inspections on repository trees and safe
text files to identify onboarding gaps, missing docs, testing signals, and configuration issues.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

from src.config import (
    TOTAL_POSSIBLE_SCORE,
    WEIGHT_CI,
    WEIGHT_CONTRIBUTING,
    WEIGHT_LICENSE,
    WEIGHT_MANIFEST,
    WEIGHT_README,
    WEIGHT_TESTS,
)


@dataclass
class Finding:
    """Standardized finding item emitted by static analysis."""

    id: str
    title: str
    severity: str  # Critical, High, Medium, Low, Info
    explanation: str
    evidence: str
    recommendation: str
    category: str  # Setup, Docs, Testing, Readiness

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Ecosystem definition with common manifests, test patterns, and descriptions
ECOSYSTEM_MANIFESTS = {
    "Python": {
        "manifests": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile", "poetry.lock"],
        "test_dirs": ["tests", "test"],
        "test_file_patterns": [r"test_.*\.py$", r".*_test\.py$"],
    },
    "JavaScript/TypeScript": {
        "manifests": ["package.json"],
        "test_dirs": ["__tests__", "test", "tests", "spec"],
        "test_file_patterns": [r".*\.(test|spec)\.(js|jsx|ts|tsx)$"],
    },
    "Java/Kotlin": {
        "manifests": ["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"],
        "test_dirs": ["src/test"],
        "test_file_patterns": [r".*Test\.java$", r".*Test\.kt$", r".*Tests\.java$"],
    },
    "Rust": {
        "manifests": ["Cargo.toml"],
        "test_dirs": ["tests"],
        "test_file_patterns": [r"tests/.*\.rs$"],
    },
    "Go": {
        "manifests": ["go.mod"],
        "test_dirs": [],
        "test_file_patterns": [r".*_test\.go$"],
    },
    "Ruby": {
        "manifests": ["Gemfile"],
        "test_dirs": ["spec", "test"],
        "test_file_patterns": [r".*_spec\.rb$", r"test_.*\.rb$"],
    },
    "PHP": {
        "manifests": ["composer.json"],
        "test_dirs": ["tests", "test"],
        "test_file_patterns": [r".*Test\.php$"],
    },
}


class StaticAnalyzer:
    """Deterministic analyzer executing heuristic checks across repository tree and files."""

    def __init__(self, bundle: Dict[str, Any]):
        self.metadata = bundle.get("metadata", {})
        self.file_tree = bundle.get("file_tree", [])
        self.files = bundle.get("files", {})
        self.file_paths = [item["path"].replace("\\", "/") for item in self.file_tree]
        self.file_paths_lower = [p.lower() for p in self.file_paths]
        self.root_files_lower = [
            p.lower() for p in self.file_paths if "/" not in p.strip("/")
        ]

    def detect_ecosystems(self) -> List[str]:
        """Detect primary languages/ecosystems from manifests and GitHub metadata."""
        detected: Set[str] = set()

        # Check primary language reported by GitHub
        primary_lang = self.metadata.get("language")
        if primary_lang:
            for eco_name in ECOSYSTEM_MANIFESTS:
                if eco_name.lower().startswith(primary_lang.lower()) or primary_lang.lower() in eco_name.lower():
                    detected.add(eco_name)

        # Check manifest files in repo tree
        for eco_name, info in ECOSYSTEM_MANIFESTS.items():
            for manifest in info["manifests"]:
                if any(p == manifest.lower() or p.endswith("/" + manifest.lower()) for p in self.file_paths_lower):
                    detected.add(eco_name)

        return sorted(list(detected)) if detected else ["General"]

    def _check_readme(self) -> tuple[Optional[Finding], int]:
        """Check for README presence, size, and essential setup/contributor sections."""
        readme_path = None
        for path in self.file_paths:
            base = path.split("/")[-1].lower()
            if base in ["readme.md", "readme.rst", "readme.txt", "readme"]:
                readme_path = path
                break

        if not readme_path:
            return (
                Finding(
                    id="DOC-001",
                    title="Missing README file",
                    severity="Critical",
                    explanation="A README is the primary entry point for users and contributors to understand the project.",
                    evidence="No README (.md, .rst, .txt) file found in repository.",
                    recommendation="Add a comprehensive README.md explaining the project purpose, setup, usage, and quickstart steps.",
                    category="Docs",
                ),
                0,
            )

        content = self.files.get(readme_path, "")
        if not content.strip():
            return (
                Finding(
                    id="DOC-002",
                    title="Empty README file",
                    severity="High",
                    explanation="An empty or near-empty README provides no onboarding guidance to users or contributors.",
                    evidence=f"Found {readme_path} but its content is empty or unreadable.",
                    recommendation="Populate README.md with project description, installation instructions, and examples.",
                    category="Docs",
                ),
                WEIGHT_README // 4,
            )

        # Check for installation / usage / testing instructions
        lower_content = content.lower()
        has_install = any(kw in lower_content for kw in ["install", "getting started", "setup", "quickstart", "prerequisites"])
        has_usage = any(kw in lower_content for kw in ["usage", "how to use", "example", "quick start"])
        has_test_mention = any(kw in lower_content for kw in ["pytest", "npm test", "cargo test", "testing", "run tests"])

        missing_sections = []
        if not has_install:
            missing_sections.append("installation/setup")
        if not has_usage:
            missing_sections.append("usage/examples")

        if missing_sections:
            return (
                Finding(
                    id="DOC-003",
                    title="Incomplete setup instructions in README",
                    severity="Medium",
                    explanation="The README appears to lack explicit sections for installation, setup, or usage.",
                    evidence=f"Missing keywords for: {', '.join(missing_sections)} in {readme_path}.",
                    recommendation=f"Add clear '{missing_sections[0].title()}' guidance so developers know how to run the project.",
                    category="Docs",
                ),
                WEIGHT_README // 2,
            )

        return (None, WEIGHT_README)

    def _check_license(self) -> tuple[Optional[Finding], int]:
        """Check for open-source license file or license metadata."""
        meta_license = self.metadata.get("license")
        has_meta_license = bool(meta_license and meta_license.get("key") and meta_license.get("key") != "other")

        license_path = None
        for path in self.file_paths:
            base = path.split("/")[-1].lower()
            if base in ["license", "license.md", "license.txt", "copying", "license.mit", "license.apache"]:
                license_path = path
                break

        if not license_path and not has_meta_license:
            return (
                Finding(
                    id="LIC-001",
                    title="No open-source license detected",
                    severity="Critical",
                    explanation="Without an open-source license, others cannot legally use, modify, or distribute your work.",
                    evidence="No LICENSE file found in repository and GitHub repository metadata specifies no recognized license.",
                    recommendation="Add a standard open-source license (such as MIT, Apache 2.0, or GPL-3.0) via a LICENSE file.",
                    category="Setup",
                ),
                0,
            )

        return (None, WEIGHT_LICENSE)

    def _check_contributing(self) -> tuple[Optional[Finding], int]:
        """Check for CONTRIBUTING guidelines, Code of Conduct, and issue templates."""
        has_contributing = False
        for path in self.file_paths_lower:
            base = path.split("/")[-1]
            if base in ["contributing.md", "contributing.rst", "contributing.txt", "contributing"]:
                has_contributing = True
                break
            if ".github/contributing" in path:
                has_contributing = True
                break

        has_coc = any(
            p.split("/")[-1] in ["code_of_conduct.md", "code_of_conduct.rst", "code_of_conduct.txt", "code_of_conduct"]
            or ".github/code_of_conduct" in p
            for p in self.file_paths_lower
        )

        has_templates = any(
            ".github/issue_template" in p or ".github/pull_request_template" in p
            for p in self.file_paths_lower
        )

        if not has_contributing:
            return (
                Finding(
                    id="CONTR-001",
                    title="Missing CONTRIBUTING guidelines",
                    severity="Medium",
                    explanation="A CONTRIBUTING guide sets expectations and helps new contributors submit pull requests efficiently.",
                    evidence="No CONTRIBUTING.md or .github/CONTRIBUTING.md found.",
                    recommendation="Add a CONTRIBUTING.md outlining branch guidelines, coding conventions, and PR workflows.",
                    category="Readiness",
                ),
                WEIGHT_CONTRIBUTING // 3 if (has_coc or has_templates) else 0,
            )

        # Bonus completeness check for templates/conduct
        if not has_coc and not has_templates:
            return (
                Finding(
                    id="CONTR-002",
                    title="Missing community health files (Code of Conduct / Templates)",
                    severity="Low",
                    explanation="Issue templates and a Code of Conduct foster a welcoming, organized contributor community.",
                    evidence="Found CONTRIBUTING.md, but no CODE_OF_CONDUCT.md or GitHub issue templates were detected.",
                    recommendation="Consider adding a CODE_OF_CONDUCT.md and GitHub Issue/PR templates in .github/.",
                    category="Readiness",
                ),
                int(WEIGHT_CONTRIBUTING * 0.75),
            )

        return (None, WEIGHT_CONTRIBUTING)

    def _check_ci(self) -> tuple[Optional[Finding], int]:
        """Check for GitHub Actions workflows or other CI configurations."""
        has_github_actions = any(
            p.startswith(".github/workflows/") and (p.endswith(".yml") or p.endswith(".yaml"))
            for p in self.file_paths_lower
        )

        has_other_ci = any(
            p in [".travis.yml", "circle.yml", ".circleci/config.yml", "appveyor.yml", ".gitlab-ci.yml", "azure-pipelines.yml"]
            for p in self.file_paths_lower
        )

        if not has_github_actions and not has_other_ci:
            return (
                Finding(
                    id="CI-001",
                    title="No Continuous Integration (CI) configuration detected",
                    severity="Medium",
                    explanation="CI pipelines automate testing, linting, and build validation for every pull request.",
                    evidence="No workflows found under .github/workflows/ and no third-party CI config files detected.",
                    recommendation="Set up a GitHub Actions workflow in .github/workflows/ci.yml to test pull requests automatically.",
                    category="Testing",
                ),
                0,
            )

        return (None, WEIGHT_CI)

    def _check_tests(self, ecosystems: List[str]) -> tuple[Optional[Finding], int]:
        """Check for test directories, test files, or ecosystem test setups."""
        test_file_matches: List[str] = []
        test_dir_matches: List[str] = []

        # Check directories
        for path in self.file_paths_lower:
            parts = path.split("/")
            if any(part in ["tests", "test", "spec", "__tests__"] for part in parts[:-1]):
                test_dir_matches.append(path)

        # Ecosystem specific regexes
        regexes: List[re.Pattern] = []
        for eco in ecosystems:
            if eco in ECOSYSTEM_MANIFESTS:
                for pat in ECOSYSTEM_MANIFESTS[eco]["test_file_patterns"]:
                    regexes.append(re.compile(pat, re.IGNORECASE))

        # Default fallback patterns
        regexes.append(re.compile(r"test_.*|.*_test\..*|.*\.spec\..*|.*\.test\..*", re.IGNORECASE))

        for path in self.file_paths:
            base = path.split("/")[-1]
            if any(rgx.search(base) for rgx in regexes):
                test_file_matches.append(path)

        if not test_file_matches and not test_dir_matches:
            return (
                Finding(
                    id="TEST-001",
                    title="No automated tests or test directories detected",
                    severity="High",
                    explanation="Automated test suites enable contributors to verify changes without introducing regressions.",
                    evidence="No test files or recognized test directories (e.g., tests/, __tests__/) found in repository tree.",
                    recommendation="Add unit tests in a tests/ directory and configure a test runner (e.g., pytest, jest, etc.).",
                    category="Testing",
                ),
                0,
            )

        return (None, WEIGHT_TESTS)

    def _check_manifests_and_env(self, ecosystems: List[str]) -> tuple[List[Finding], int]:
        """Check dependency manifests, package structure, and safe environment documentation."""
        findings: List[Finding] = []
        score = WEIGHT_MANIFEST

        found_manifests = []
        for eco in ecosystems:
            if eco in ECOSYSTEM_MANIFESTS:
                for mf in ECOSYSTEM_MANIFESTS[eco]["manifests"]:
                    if any(p == mf.lower() or p.endswith("/" + mf.lower()) for p in self.file_paths_lower):
                        found_manifests.append(mf)

        if not found_manifests and ecosystems != ["General"]:
            findings.append(
                Finding(
                    id="MNF-001",
                    title="No standard dependency manifest found",
                    severity="Medium",
                    explanation=f"Detected ecosystem ({', '.join(ecosystems)}), but no standard package manifest was identified.",
                    evidence=f"Expected one of: {', '.join([m for e in ecosystems if e in ECOSYSTEM_MANIFESTS for m in ECOSYSTEM_MANIFESTS[e]['manifests']])}",
                    recommendation="Add a standard package manifest (e.g. pyproject.toml, package.json) to declare dependencies and build rules.",
                    category="Setup",
                )
            )
            score = 0

        # Safe environment example check (informative/readiness)
        has_env_example = any(
            p.split("/")[-1] in [".env.example", ".env.sample", ".env.template"]
            for p in self.file_paths_lower
        )
        # Check if project seems to mention env vars in README without example
        readme_content = ""
        for path, content in self.files.items():
            if "readme" in path.lower():
                readme_content = content.lower()
                break

        if ("env" in readme_content or "api_key" in readme_content or "secret" in readme_content) and not has_env_example:
            findings.append(
                Finding(
                    id="ENV-001",
                    title="No safe .env.example template detected",
                    severity="Low",
                    explanation="The project mentions environment variables or configuration keys, but does not provide a safe template.",
                    evidence="README references environment variables, but no .env.example or .env.sample file exists.",
                    recommendation="Add a safe .env.example template with dummy values to help new developers configure their environment.",
                    category="Setup",
                )
            )

        return findings, score

    def analyze(self) -> Dict[str, Any]:
        """Run all static checks and compute deterministic readiness score."""
        ecosystems = self.detect_ecosystems()
        findings: List[Finding] = []
        category_scores: Dict[str, Dict[str, int]] = {}
        total_score = 0

        # 1. README check (Docs)
        f_readme, s_readme = self._check_readme()
        if f_readme:
            findings.append(f_readme)
        total_score += s_readme

        # 2. LICENSE check (Setup)
        f_license, s_license = self._check_license()
        if f_license:
            findings.append(f_license)
        total_score += s_license

        # 3. CONTRIBUTING check (Readiness)
        f_contr, s_contr = self._check_contributing()
        if f_contr:
            findings.append(f_contr)
        total_score += s_contr

        # 4. CI check (Testing)
        f_ci, s_ci = self._check_ci()
        if f_ci:
            findings.append(f_ci)
        total_score += s_ci

        # 5. Tests check (Testing)
        f_tests, s_tests = self._check_tests(ecosystems)
        if f_tests:
            findings.append(f_tests)
        total_score += s_tests

        # 6. Manifests & Env (Setup)
        f_mnf, s_mnf = self._check_manifests_and_env(ecosystems)
        findings.extend(f_mnf)
        total_score += s_mnf

        # Category rollups for developer cockpit
        category_scores = {
            "Setup": {
                "score": s_license + s_mnf,
                "max": WEIGHT_LICENSE + WEIGHT_MANIFEST,
            },
            "Docs": {
                "score": s_readme,
                "max": WEIGHT_README,
            },
            "Testing": {
                "score": s_ci + s_tests,
                "max": WEIGHT_CI + WEIGHT_TESTS,
            },
            "Readiness": {
                "score": s_contr,
                "max": WEIGHT_CONTRIBUTING,
            },
        }

        # Ensure score boundaries
        final_score = max(0, min(TOTAL_POSSIBLE_SCORE, total_score))

        # Severity breakdown
        severity_counts = {
            "Critical": 0,
            "High": 0,
            "Medium": 0,
            "Low": 0,
            "Info": 0,
        }
        for f in findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        return {
            "score": final_score,
            "max_score": TOTAL_POSSIBLE_SCORE,
            "ecosystems": ecosystems,
            "total_files_analyzed": len(self.file_tree),
            "findings": [f.to_dict() for f in findings],
            "severity_counts": severity_counts,
            "category_scores": category_scores,
        }
