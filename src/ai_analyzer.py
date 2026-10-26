"""AI analyzer module for RepoReady integrating Google Gemma via Ollama REST API.

Provides explanatory insights, prioritized recommendations, and contributor onboarding
guidance without altering deterministic analysis findings, scores, or severity metrics.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from src.config import (
    AI_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    USER_AGENT,
)

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    "Critical": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4,
    "Info": 5,
}


class AIAnalyzerError(Exception):
    """Custom exception raised for Ollama / Gemma AI communication failures."""

    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        status_code: Optional[int] = None,
        troubleshooting_hint: Optional[str] = None,
        details: Optional[str] = None,
    ):
        super().__init__(message)
        self.user_message = user_message or message
        self.status_code = status_code
        self.troubleshooting_hint = troubleshooting_hint
        self.details = details


class AIAnalyzer:
    """Communicates with local Ollama service to obtain Gemma AI-powered readiness insights."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = AI_REQUEST_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_availability(self) -> Dict[str, Any]:
        """Check if Ollama service is reachable and list downloaded models."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                model_found = any(
                    self.model == m or self.model in m or m.startswith(self.model)
                    for m in models
                )
                return {
                    "available": True,
                    "model_installed": model_found,
                    "available_models": models,
                }
        except urllib.error.URLError as e:
            return {
                "available": False,
                "model_installed": False,
                "error": f"Ollama not running: {e.reason}",
            }
        except Exception as e:
            return {
                "available": False,
                "model_installed": False,
                "error": str(e),
            }

    def _build_sanitized_prompt(
        self,
        metadata: Dict[str, Any],
        score: int,
        max_score: int,
        findings: List[Dict[str, Any]],
        ecosystems: List[str],
    ) -> str:
        """Construct a concise, prioritized prompt without excessive boilerplate."""
        safe_name = str(metadata.get("name", "Unknown"))[:50]
        safe_owner = str(metadata.get("owner", "Unknown"))[:50]
        safe_desc = str(metadata.get("description", "No description"))[:180]
        safe_lang = str(metadata.get("language", "Not specified"))[:30]

        # Prioritize findings by severity, taking at most top 8
        sorted_findings = sorted(
            findings,
            key=lambda x: SEVERITY_ORDER.get(x.get("severity", "Info"), 99),
        )[:8]

        findings_summary: List[str] = []
        for i, f in enumerate(sorted_findings, start=1):
            title = str(f.get("title", "Unknown"))[:80]
            rec = str(f.get("recommendation", ""))[:120]
            findings_summary.append(
                f"- [{f.get('severity', 'Info')}] {title} -> {rec}"
            )

        findings_block = (
            "\n".join(findings_summary)
            if findings_summary
            else "All deterministic static checks passed with no gaps."
        )

        prompt = f"""Review this repository's onboarding readiness:
Repository: {safe_owner}/{safe_name}
Language: {safe_lang} ({', '.join(ecosystems)})
Readiness Score: {score}/{max_score}
Description: {safe_desc}

Diagnostic Findings:
{findings_block}

Provide a concise, practical analysis:
1. Critical Issues: Explain the top priorities in 2-3 sentences.
2. Contributor Onboarding: Recommend exact first steps for new contributors.
3. Roadmap: List 2-3 prioritized maintainer fixes.
Keep the response structured, direct, and under 250 words."""
        return prompt

    def generate_readiness_insights(
        self,
        metadata: Dict[str, Any],
        score: int,
        max_score: int,
        findings: List[Dict[str, Any]],
        ecosystems: List[str],
    ) -> str:
        """Call Ollama /api/generate with sanitized input to produce Gemma readiness insights."""
        prompt = self._build_sanitized_prompt(metadata, score, max_score, findings, ecosystems)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                response_text = resp_json.get("response", "").strip()
                if not response_text:
                    raise AIAnalyzerError(
                        "Ollama returned an empty response",
                        user_message="Gemma generated an empty response. Please retry.",
                        details="The /api/generate endpoint succeeded but returned no text in 'response'.",
                    )
                return response_text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise AIAnalyzerError(
                    f"Model '{self.model}' not found in Ollama (HTTP 404)",
                    user_message=f"Model '{self.model}' is not installed in your local Ollama instance.",
                    status_code=404,
                    troubleshooting_hint=f"Run `ollama run {self.model}` or `ollama pull {self.model}` in your terminal.",
                ) from e
            else:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                raise AIAnalyzerError(
                    f"Ollama HTTP error {e.code}: {e.reason}",
                    user_message=f"Ollama server returned HTTP {e.code}.",
                    status_code=e.code,
                    details=err_body,
                ) from e
        except urllib.error.URLError as e:
            # Check specifically for connection timeout vs connection refused
            if isinstance(e.reason, socket.timeout):
                raise AIAnalyzerError(
                    f"Connection timed out after {self.timeout}s while connecting to Ollama",
                    user_message=f"Connection timed out ({self.timeout}s) while trying to reach Ollama.",
                    troubleshooting_hint="Ensure Ollama is running and not overloaded.",
                ) from e
            raise AIAnalyzerError(
                f"Cannot connect to Ollama at {self.base_url}: {e.reason}",
                user_message="Could not connect to Ollama. Make sure Ollama is installed and running locally.",
                troubleshooting_hint="Start Ollama by running `ollama serve` or opening the Ollama application.",
                details=str(e.reason),
            ) from e
        except TimeoutError as e:
            # Read timeout while waiting for model generation
            raise AIAnalyzerError(
                f"Ollama response generation timed out after {self.timeout}s",
                user_message=f"Ollama took longer than {self.timeout} seconds to complete synthesis (timeout).",
                troubleshooting_hint=f"Increase OLLAMA_TIMEOUT (current: {self.timeout}s) or run `ollama pull {self.model}` to ensure weights are cached.",
            ) from e
        except json.JSONDecodeError as e:
            raise AIAnalyzerError(
                "Malformed JSON received from Ollama API",
                user_message="Received an invalid response format from the Ollama service.",
                details=f"JSON decode failure: {e}",
            ) from e
        except Exception as e:
            if isinstance(e, AIAnalyzerError):
                raise
            raise AIAnalyzerError(
                f"Unexpected error calling Ollama: {e}",
                user_message="An unexpected error occurred while communicating with Gemma AI.",
                details=str(e),
            ) from e
