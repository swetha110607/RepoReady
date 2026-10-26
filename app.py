"""RepoReady - Developer Cockpit & Streamlit Application.

An open-source repository readiness analyzer that deterministically evaluates
public GitHub repositories for setup, documentation, testing, and contribution readiness,
with optional Google Gemma AI insights via local Ollama.
"""

from __future__ import annotations

import streamlit as st

from src.ai_analyzer import AIAnalyzer, AIAnalyzerError, compute_repo_cache_key
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

SEVERITY_STYLES = {
    "Critical": {
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.4)",
        "badge_color": "#f87171",
        "icon": "🔴",
    },
    "High": {
        "badge_bg": "rgba(249, 115, 22, 0.15)",
        "badge_border": "rgba(249, 115, 22, 0.4)",
        "badge_color": "#fb923c",
        "icon": "🟠",
    },
    "Medium": {
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.4)",
        "badge_color": "#fbbf24",
        "icon": "🟡",
    },
    "Low": {
        "badge_bg": "rgba(59, 130, 246, 0.15)",
        "badge_border": "rgba(59, 130, 246, 0.4)",
        "badge_color": "#60a5fa",
        "icon": "🔵",
    },
    "Info": {
        "badge_bg": "rgba(100, 116, 139, 0.15)",
        "badge_border": "rgba(100, 116, 139, 0.4)",
        "badge_color": "#94a3b8",
        "icon": "⚪",
    },
}

CUSTOM_CSS = """
<style>
/* Global Dark Canvas adjustments */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Header & Typography */
.cockpit-header {
    border-bottom: 1px solid #21262d;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
}
.cockpit-title {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f0f6fc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.cockpit-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
    margin-top: 0.35rem;
    margin-bottom: 0;
}
.cockpit-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    vertical-align: middle;
}

/* Cards & Panels */
.cockpit-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* Pillar Pills */
.pillar-card {
    background: #11161d;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    height: 100%;
}
.pillar-title {
    color: #e6edf3;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.pillar-desc {
    color: #8b949e;
    font-size: 0.78rem;
    line-height: 1.35;
    margin: 0;
}

/* Readiness Health Scoreboard */
.health-scoreboard {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    margin-bottom: 1.25rem;
}
.health-score-val {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
}
.health-score-max {
    font-size: 1.15rem;
    font-weight: 500;
    color: #8b949e;
}

/* Compact Severity Bar */
.severity-chip {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    text-align: center;
}
.severity-chip-count {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.1;
}
.severity-chip-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 0.2rem;
}

/* Category Progress Bars */
.cat-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
}
.cat-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 600;
    color: #c9d1d9;
    margin-bottom: 0.35rem;
}
.cat-bar-bg {
    background: #21262d;
    border-radius: 9999px;
    height: 6px;
    overflow: hidden;
}
.cat-bar-fill {
    background: #10b981;
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}

/* Findings styling */
.finding-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left-width: 4px;
    border-radius: 6px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
}
.finding-id {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.75rem;
    color: #8b949e;
    font-weight: 600;
}
.finding-category-tag {
    font-size: 0.72rem;
    padding: 0.15rem 0.45rem;
    background: #1f242c;
    border: 1px solid #30363d;
    color: #c9d1d9;
    border-radius: 4px;
    margin-left: 0.4rem;
}

/* Streamlit button refinements */
div.stButton > button[kind="primary"] {
    background-color: #10b981 !important;
    border-color: #059669 !important;
    color: #042f2e !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.25rem !important;
    transition: all 0.15s ease-in-out !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #059669 !important;
    border-color: #047857 !important;
    color: #ffffff !important;
}
div.stButton > button[kind="secondary"] {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #10b981 !important;
    color: #34d399 !important;
}

/* Subtle Streamlit input field adjustments */
div[data-baseweb="input"] {
    background-color: #0d1117 !important;
    border-color: #30363d !important;
    border-radius: 6px !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #10b981 !important;
}
</style>
"""


def render_cockpit_header():
    """Render top brand header and concise status badge."""
    st.markdown(
        f"""
        <div class="cockpit-header">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <h1 class="cockpit-title">
                    <span>⚡ {APP_NAME}</span>
                    <span class="cockpit-badge">v0.1.0</span>
                </h1>
                <span style="font-family: ui-monospace, monospace; font-size: 0.78rem; color: #8b949e; background: #161b22; padding: 0.25rem 0.6rem; border-radius: 4px; border: 1px solid #30363d;">
                    Engine: Deterministic Static Analyzer
                </span>
            </div>
            <p class="cockpit-subtitle">{APP_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pillars():
    """Render 4 core inspection pillars in subtle dark panels."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="pillar-card">
                <div class="pillar-title">⚙️ Setup & Manifest</div>
                <p class="pillar-desc">Verifies licenses, project manifests, and configuration integrity.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="pillar-card">
                <div class="pillar-title">📄 Documentation</div>
                <p class="pillar-desc">Checks README depth, installation guides, and quickstart commands.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="pillar-card">
                <div class="pillar-title">🧪 Testing & CI</div>
                <p class="pillar-desc">Audits test directories, test suites, and GitHub Actions workflows.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """<div class="pillar-card">
                <div class="pillar-title">🤝 Contribution</div>
                <p class="pillar-desc">Evaluates CONTRIBUTING guidelines, CoC, and issue/PR templates.</p>
            </div>""",
            unsafe_allow_html=True,
        )


def render_health_scoreboard(score: int, max_score: int):
    """Render score gauge with emerald accents and status indicators."""
    pct = (score / max_score) * 100 if max_score > 0 else 0
    if pct >= 80:
        accent_color = "#10b981"
        status_label = "Production Ready"
        status_desc = "Strong onboarding hygiene, comprehensive docs, and testing signals."
    elif pct >= 50:
        accent_color = "#f59e0b"
        status_label = "Moderate Readiness"
        status_desc = "Core foundations detected, but notable contributor or testing gaps exist."
    else:
        accent_color = "#ef4444"
        status_label = "Action Required"
        status_desc = "Critical setup or documentation items are missing before onboarding."

    st.markdown(
        f"""
        <div class="health-scoreboard" style="border-left: 5px solid {accent_color};">
            <div>
                <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #8b949e; margin-bottom: 0.2rem;">
                    Repository Health Index
                </div>
                <div style="display: flex; align-items: baseline; gap: 0.35rem;">
                    <span class="health-score-val" style="color: {accent_color};">{score}</span>
                    <span class="health-score-max">/ {max_score}</span>
                </div>
                <div style="color: #c9d1d9; font-size: 0.85rem; margin-top: 0.4rem;">
                    <strong>Status:</strong> <span style="color: {accent_color};">{status_label}</span> — {status_desc}
                </div>
            </div>
            <div style="text-align: right; min-width: 140px;">
                <div style="font-size: 0.72rem; color: #8b949e; margin-bottom: 0.3rem;">Onboarding Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f0f6fc;">{pct:.0f}%</div>
                <div style="background: #21262d; border-radius: 9999px; height: 6px; width: 100%; margin-top: 0.3rem; overflow: hidden;">
                    <div style="background: {accent_color}; width: {pct}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_severity_summary(severity_counts: dict[str, int]):
    """Render compact horizontal severity cards."""
    severities = ["Critical", "High", "Medium", "Low", "Info"]
    cols = st.columns(5)
    for i, sev in enumerate(severities):
        cfg = SEVERITY_STYLES[sev]
        count = severity_counts.get(sev, 0)
        with cols[i]:
            st.markdown(
                f"""
                <div class="severity-chip" style="border-top: 2px solid {cfg['badge_color']};">
                    <div class="severity-chip-count" style="color: {cfg['badge_color']};">{count}</div>
                    <div class="severity-chip-label" style="color: #8b949e;">{cfg['icon']} {sev}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_category_bars(category_scores: dict[str, dict[str, int]]):
    """Render compact progress bars for the 4 diagnostic categories."""
    if not category_scores:
        return
    cols = st.columns(4)
    cat_names = [("Setup", "Setup & Config"), ("Docs", "Documentation"), ("Testing", "Testing & CI"), ("Readiness", "Contributor Ops")]
    for i, (cat_key, cat_label) in enumerate(cat_names):
        data = category_scores.get(cat_key, {"score": 0, "max": 1})
        s = data["score"]
        m = max(1, data["max"])
        p = min(100, int((s / m) * 100))
        color = "#10b981" if p >= 75 else ("#f59e0b" if p >= 40 else "#ef4444")
        with cols[i]:
            st.markdown(
                f"""
                <div class="cat-item">
                    <div class="cat-header">
                        <span>{cat_label}</span>
                        <span style="color: {color};">{s}/{m} pts</span>
                    </div>
                    <div class="cat-bar-bg">
                        <div class="cat-bar-fill" style="width: {p}%; background: {color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(
        page_title=f"{APP_NAME}",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject dark developer theme styles
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header section
    render_cockpit_header()

    # Four Pillars of Analysis
    render_pillars()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Cockpit Target Input
    st.markdown(
        """
        <div style="font-size: 0.88rem; font-weight: 600; color: #c9d1d9; margin-bottom: 0.4rem;">
            Repository Target URL
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        repo_url = st.text_input(
            label="GitHub Repository URL",
            label_visibility="collapsed",
            value=st.session_state.get("last_url", ""),
            placeholder=DEFAULT_PLACEHOLDER_URL,
            help="Public GitHub URL (e.g. https://github.com/psf/requests)",
        )
    with col_btn:
        analyze_clicked = st.button("Inspect Repo", type="primary", use_container_width=True)

    if analyze_clicked:
        clean_url = repo_url.strip()
        if not clean_url:
            st.warning("Please provide a GitHub repository URL.")
            return

        if not is_valid_github_url(clean_url):
            st.error("Invalid URL structure. Expected format: `https://github.com/owner/repository`")
            return

        with st.spinner("Executing static readiness telemetry..."):
            fetcher = GitHubFetcher()
            try:
                bundle = fetcher.fetch_repository_bundle(clean_url)
                analyzer = StaticAnalyzer(bundle)
                results = analyzer.analyze()
                st.session_state["analysis_bundle"] = bundle
                st.session_state["analysis_results"] = results
                st.session_state["last_url"] = clean_url
                st.session_state["ai_insights"] = None
                st.session_state["ai_insights_cache_key"] = None
            except GitHubFetchError as e:
                st.error(f"**Diagnostic Error:** {e.user_message}")
                return
            except Exception:
                st.error("An unexpected error occurred during repository analysis. Please check URL permissions.")
                return

    # Render Active Cockpit Results
    if "analysis_results" in st.session_state and st.session_state["analysis_results"]:
        results = st.session_state["analysis_results"]
        bundle = st.session_state["analysis_bundle"]
        metadata = bundle["metadata"]
        repo_info = extract_repo_info(st.session_state.get("last_url", ""))

        st.markdown("<hr style='border-color: #21262d; margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Target Repo Metadata Strip
        meta_left, meta_right = st.columns([3, 1])
        with meta_left:
            st.markdown(
                f"""
                <div style="display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.25rem;">
                    <span style="font-size: 1.25rem; font-weight: 700; color: #f0f6fc;">
                        {repo_info['owner']} / <span style="color: #34d399;">{repo_info['repo']}</span>
                    </span>
                    <span style="font-size: 0.75rem; background: #21262d; color: #8b949e; padding: 0.1rem 0.45rem; border-radius: 4px;">
                        Branch: {metadata.get('default_branch', 'main')}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #8b949e; line-height: 1.4;">
                    {metadata.get('description') or 'No description provided.'}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with meta_right:
            st.markdown(
                f"""
                <div style="background: #11161d; border: 1px solid #21262d; border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.78rem; color: #8b949e;">
                    <div>⭐ <strong>Stars:</strong> {metadata.get('stars_count', 0):,}</div>
                    <div>📦 <strong>Files Scanned:</strong> {results.get('total_files_analyzed', 0):,}</div>
                    <div>🌐 <strong>Ecosystem:</strong> <span style="color: #e6edf3;">{', '.join(results.get('ecosystems', ['General']))}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # 1. Health Scoreboard
        render_health_scoreboard(results["score"], results["max_score"])

        # 2. Category Performance Bars
        render_category_bars(results.get("category_scores", {}))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        # 3. Severity Breakdown Chips
        render_severity_summary(results["severity_counts"])

        st.markdown("<hr style='border-color: #21262d; margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # 4. Detailed Diagnostic Findings
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
                <h3 style="font-size: 1.1rem; font-weight: 700; color: #f0f6fc; margin: 0;">Diagnostic Telemetry</h3>
                <span style="font-size: 0.75rem; color: #8b949e;">Deterministic rule-based checks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        findings = results.get("findings", [])
        if not findings:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 1rem 1.25rem;">
                    <div style="color: #34d399; font-weight: 600; font-size: 0.95rem;">✅ No readiness gaps detected</div>
                    <div style="color: #8b949e; font-size: 0.82rem; margin-top: 0.25rem;">All automated onboarding, documentation, license, and testing signals are present.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            severities = ["Critical", "High", "Medium", "Low", "Info"]
            for sev in severities:
                sev_findings = [f for f in findings if f.get("severity") == sev]
                if sev_findings:
                    cfg = SEVERITY_STYLES[sev]
                    st.markdown(
                        f"""
                        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {cfg['badge_color']}; margin: 1rem 0 0.5rem 0; display: flex; align-items: center; gap: 0.35rem;">
                            <span>{cfg['icon']}</span>
                            <span>{sev} Findings ({len(sev_findings)})</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for f in sev_findings:
                        with st.expander(f"[{f['id']}] {f['title']}"):
                            st.markdown(
                                f"""
                                <div style="display: flex; gap: 0.4rem; margin-bottom: 0.6rem;">
                                    <span class="finding-id">{f['id']}</span>
                                    <span class="finding-category-tag">Category: {f.get('category', 'General')}</span>
                                    <span class="finding-category-tag" style="color: {cfg['badge_color']}; border-color: {cfg['badge_border']};">Severity: {f.get('severity')}</span>
                                </div>
                                <div style="font-size: 0.88rem; color: #c9d1d9; margin-bottom: 0.6rem;">{f.get('explanation')}</div>
                                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #8b949e; font-family: ui-monospace, monospace; margin-bottom: 0.6rem;">
                                    <strong>Evidence:</strong> {f.get('evidence')}
                                </div>
                                <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 4px; padding: 0.6rem 0.8rem; font-size: 0.84rem; color: #34d399;">
                                    💡 <strong>Action:</strong> {f.get('recommendation')}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        st.markdown("<hr style='border-color: #21262d; margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # 5. AI Copilot / Advisor Section
        st.markdown(
            f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.6rem;">
                    <div>
                        <span style="font-size: 1.05rem; font-weight: 700; color: #f0f6fc;">🤖 Gemma AI Advisor</span>
                        <span class="cockpit-badge" style="margin-left: 0.4rem;">Model: {OLLAMA_MODEL}</span>
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #8b949e; line-height: 1.4; margin-bottom: 1rem;">
                    Synthesizes static telemetry into executive summaries and contributor onboarding roadmaps. Deterministic scores remain immutable.
                </div>
            """,
            unsafe_allow_html=True,
        )

        ai_analyzer = AIAnalyzer()
        current_cache_key = compute_repo_cache_key(
            repo_url=st.session_state.get("last_url", ""),
            metadata=metadata,
            score=results["score"],
            findings=findings,
            model=OLLAMA_MODEL,
            file_tree=bundle.get("file_tree"),
            files=bundle.get("files"),
        )

        col_ai_btn, _ = st.columns([1, 2])
        with col_ai_btn:
            generate_ai_clicked = st.button("Synthesize with Gemma", type="secondary", use_container_width=True)

        if generate_ai_clicked:
            # Check if matching cached insight already exists
            if (
                st.session_state.get("ai_insights")
                and st.session_state.get("ai_insights_cache_key") == current_cache_key
            ):
                st.info("Loaded recent analysis from repository session cache.")
            else:
                availability = ai_analyzer.check_availability()
                if not availability.get("available"):
                    st.warning(
                        f"⚠️ **Local Ollama daemon unreachable.**\n\n"
                        f"Start Ollama: `ollama serve` and pull `{OLLAMA_MODEL}`.\n\n"
                        f"*Deterministic static analysis above remains fully operational.*"
                    )
                elif not availability.get("model_installed"):
                    st.warning(
                        f"⚠️ **Model `{OLLAMA_MODEL}` not found in Ollama.** Run: `ollama pull {OLLAMA_MODEL}`."
                    )
                else:
                    with st.spinner(f"Generating repository intelligence with local {OLLAMA_MODEL}..."):
                        try:
                            insights = ai_analyzer.generate_readiness_insights(
                                metadata=metadata,
                                score=results["score"],
                                max_score=results["max_score"],
                                findings=findings,
                                ecosystems=results.get("ecosystems", ["General"]),
                                file_tree=bundle.get("file_tree"),
                                files=bundle.get("files"),
                            )
                            st.session_state["ai_insights"] = insights
                            st.session_state["ai_insights_cache_key"] = current_cache_key
                        except AIAnalyzerError as e:
                            st.error(f"**AI Engine Notice:** {e.user_message}")
                            if e.details:
                                st.caption(f"Details: `{e.details}`")
                            if e.troubleshooting_hint:
                                st.info(f"💡 {e.troubleshooting_hint}")
                        except Exception as e:
                            st.error("Unexpected error during AI synthesis.")
                            st.caption(f"Details: `{e}`")

        # Invalidate output if repo context changed
        if (
            st.session_state.get("ai_insights")
            and st.session_state.get("ai_insights_cache_key") == current_cache_key
        ):
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 1.15rem; font-size: 0.9rem; color: #e6edf3; line-height: 1.6;">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state["ai_insights"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 0.75rem; color: #484f58; margin-top: 2rem; border-top: 1px solid #21262d; padding-top: 1rem;">
            {APP_NAME} • 100% Free & Open-Source • Zero Tracking • Zero Remote Code Execution
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
