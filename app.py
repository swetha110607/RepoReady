"""RepoReady - Streamlit Application.

An open-source repository readiness analyzer that deterministically evaluates
public GitHub repositories for setup, documentation, testing, and contribution readiness,
with optional Google Gemma AI insights via local Ollama.
"""

from __future__ import annotations

import streamlit as st

from src.ai_analyzer import AIAnalyzer, AIAnalyzerError
from src.config import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_TAGLINE,
    DEFAULT_PLACEHOLDER_URL,
    OLLAMA_MODEL,
    TOTAL_POSSIBLE_SCORE,
)
from src.github_fetcher import GitHubFetcher, GitHubFetchError
from src.static_analyzer import StaticAnalyzer
from src.utils import extract_repo_info, is_valid_github_url

SEVERITY_COLORS = {
    "Critical": "🚨 Critical",
    "High": "⚠️ High",
    "Medium": "🟡 Medium",
    "Low": "🔵 Low",
    "Info": "ℹ️ Info",
}


def render_score_badge(score: int, max_score: int):
    """Render a visual gauge or colored badge for the score."""
    pct = (score / max_score) * 100 if max_score > 0 else 0
    if pct >= 80:
        color = "#28a745"
        label = "Strong Readiness"
    elif pct >= 50:
        color = "#ffc107"
        label = "Moderate Readiness"
    else:
        color = "#dc3545"
        label = "Action Recommended"

    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 16px; border-left: 6px solid {color}; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #333;">Readiness Score: <span style="color: {color};">{score} / {max_score}</span></h2>
            <p style="margin: 4px 0 0 0; color: #555; font-weight: 500;">Status: {label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title=f"{APP_NAME} - {APP_TAGLINE}",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Header section
    st.title(f"🚀 {APP_NAME}")
    st.subheader(APP_TAGLINE)
    st.write(APP_DESCRIPTION)

    # Transparency disclaimer
    st.caption(
        "ℹ️ **Note on Scoring:** This score measures repository onboarding, documentation, "
        "and community readiness signals. It is not a measure of code quality, popularity, or security."
    )

    st.markdown("---")

    # Features / Pillars overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 📋 Setup")
        st.caption("Verify licenses, manifests, and project configurations.")
    with col2:
        st.markdown("### 📖 Docs")
        st.caption("Inspect README depth, installation guides, and instructions.")
    with col3:
        st.markdown("### 🧪 Testing")
        st.caption("Detect automated test suites and continuous integration pipelines.")
    with col4:
        st.markdown("### 🤝 Readiness")
        st.caption("Check contributing guidelines, issue templates, and community health.")

    st.markdown("---")

    # Input section
    st.markdown("#### Analyze a Public GitHub Repository")
    st.write(
        "Enter the URL of any public GitHub repository to evaluate its open-source readiness."
    )

    repo_url = st.text_input(
        label="GitHub Repository URL",
        value=st.session_state.get("last_url", ""),
        placeholder=DEFAULT_PLACEHOLDER_URL,
        help="Paste a public GitHub repo link, e.g., https://github.com/psf/requests",
    )

    analyze_clicked = st.button("Check Readiness", type="primary", use_container_width=True)

    if analyze_clicked:
        clean_url = repo_url.strip()
        if not clean_url:
            st.warning("Please enter a GitHub repository URL to proceed.")
            return

        if not is_valid_github_url(clean_url):
            st.error("Please provide a valid GitHub repository URL (e.g., https://github.com/owner/repository).")
            return

        with st.spinner("Fetching repository metadata and running static analysis..."):
            fetcher = GitHubFetcher()
            try:
                bundle = fetcher.fetch_repository_bundle(clean_url)
                analyzer = StaticAnalyzer(bundle)
                results = analyzer.analyze()
                # Store in session state so user can interact with AI without re-fetching
                st.session_state["analysis_bundle"] = bundle
                st.session_state["analysis_results"] = results
                st.session_state["last_url"] = clean_url
                st.session_state["ai_insights"] = None
            except GitHubFetchError as e:
                st.error(f"**Analysis Error:** {e.user_message}")
                return
            except Exception:
                st.error("An unexpected error occurred while analyzing the repository. Please verify the URL and try again.")
                return

    # Render analysis results if available in session_state
    if "analysis_results" in st.session_state and st.session_state["analysis_results"]:
        results = st.session_state["analysis_results"]
        bundle = st.session_state["analysis_bundle"]
        metadata = bundle["metadata"]
        repo_info = extract_repo_info(st.session_state.get("last_url", ""))

        st.markdown("---")
        st.markdown(f"### Results for `{repo_info['owner']}/{repo_info['repo']}`")

        # Metadata banner
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.metric("Primary Language", metadata.get("language") or "Not specified")
        with meta_col2:
            st.metric("Stars", f"{metadata.get('stars_count', 0):,}")
        with meta_col3:
            st.metric("Detected Ecosystems", ", ".join(results.get("ecosystems", ["General"])))
        with meta_col4:
            st.metric("Files Inspected", results.get("total_files_analyzed", 0))

        # Description
        st.write(f"**Description:** {metadata.get('description')}")

        st.markdown("---")

        # Readiness Score Section
        render_score_badge(results["score"], results["max_score"])

        # Summary Counts by Severity
        st.markdown("#### Findings Summary")
        sev_cols = st.columns(5)
        severities = ["Critical", "High", "Medium", "Low", "Info"]
        for i, sev in enumerate(severities):
            count = results["severity_counts"].get(sev, 0)
            with sev_cols[i]:
                st.metric(label=SEVERITY_COLORS[sev], value=count)

        st.markdown("---")

        # Detailed Deterministic Findings Section
        st.markdown("#### Detailed Findings & Recommendations")
        findings = results.get("findings", [])

        if not findings:
            st.success("🎉 **No readiness gaps detected!** This repository demonstrates strong onboarding setup, test coverage signals, and documentation.")
            st.caption("*(Note: While all automated readiness signals passed, manual review is always encouraged.)*")
        else:
            for sev in severities:
                sev_findings = [f for f in findings if f.get("severity") == sev]
                if sev_findings:
                    st.markdown(f"##### {SEVERITY_COLORS[sev]} Findings ({len(sev_findings)})")
                    for finding in sev_findings:
                        with st.expander(f"**[{finding['id']}]** {finding['title']}"):
                            st.write(f"**Category:** `{finding['category']}`")
                            st.write(f"**Explanation:** {finding['explanation']}")
                            st.info(f"**Evidence:** {finding['evidence']}")
                            st.success(f"💡 **Recommendation:** {finding['recommendation']}")

        st.markdown("---")

        # AI Insights Section (Google Gemma via Ollama)
        st.markdown("### 🤖 AI Readiness Insights (Google Gemma)")
        st.caption(
            "Powered by local **Gemma** via **Ollama**. Gemma provides plain-English summaries, "
            "prioritized fixes, and onboarding suggestions. *The deterministic findings above remain the source of truth.*"
        )

        ai_analyzer = AIAnalyzer()

        # Check Ollama service availability
        col_ai_btn, col_ai_status = st.columns([1, 2])
        with col_ai_btn:
            generate_ai_clicked = st.button("✨ Generate AI Insights with Gemma", type="secondary")

        if generate_ai_clicked:
            availability = ai_analyzer.check_availability()
            if not availability.get("available"):
                st.warning(
                    f"⚠️ **Ollama is not running locally.**\n\n"
                    f"To enable AI insights:\n"
                    f"1. Install Ollama from [ollama.com](https://ollama.com).\n"
                    f"2. Start the service (`ollama serve`).\n"
                    f"3. Run `ollama pull {OLLAMA_MODEL}`.\n\n"
                    f"*The static analyzer above remains fully operational without AI.*"
                )
            elif not availability.get("model_installed"):
                st.warning(
                    f"⚠️ **Model '{OLLAMA_MODEL}' is not found in your Ollama installation.**\n\n"
                    f"Run the following command in your terminal to download it:\n"
                    f"```bash\nollama pull {OLLAMA_MODEL}\n```"
                )
            else:
                with st.spinner(f"Generating AI insights using local model '{OLLAMA_MODEL}'..."):
                    try:
                        insights = ai_analyzer.generate_readiness_insights(
                            metadata=metadata,
                            score=results["score"],
                            max_score=results["max_score"],
                            findings=findings,
                            ecosystems=results.get("ecosystems", ["General"]),
                        )
                        st.session_state["ai_insights"] = insights
                    except AIAnalyzerError as e:
                        st.error(f"**AI Insight Error:** {e.user_message}")
                        if e.troubleshooting_hint:
                            st.info(f"💡 {e.troubleshooting_hint}")
                    except Exception:
                        st.error("Failed to generate AI insights due to an unexpected error.")

        # Render saved AI insights
        if st.session_state.get("ai_insights"):
            st.markdown("#### 📝 AI-Generated Recommendations")
            st.info("⚠️ **AI Notice:** These recommendations are generated by Google Gemma to assist maintainers and contributors. Deterministic analysis findings and readiness scores are not modified.")
            st.markdown(st.session_state["ai_insights"])

    st.markdown("---")
    st.caption("RepoReady is 100% free, open-source, and does not require API keys or paid accounts.")


if __name__ == "__main__":
    main()
