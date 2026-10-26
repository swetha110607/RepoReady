# 🚀 RepoReady

**RepoReady** is a free, open-source repository readiness analyzer built with Python and Streamlit, featuring local AI readiness insights powered by Google Gemma via Ollama.

RepoReady helps open-source maintainers and contributors inspect repositories to identify setup problems, missing documentation, testing gaps, and contribution-readiness issues.

---

## 📁 Project Structure

```text
repoready/
├── .gitignore              # Ignores virtual environments, caches, and temp files
├── pytest.ini              # Pytest configuration and Python path root
├── README.md               # Project documentation and quickstart instructions
├── requirements.txt        # Minimal Python dependencies (Streamlit & Pytest)
├── app.py                  # Streamlit application UI & entry point
├── src/                    # Core application logic and modular components
│   ├── __init__.py         # Package initialization
│   ├── ai_analyzer.py      # Google Gemma integration via Ollama REST API
│   ├── config.py           # Configuration, scoring weights, and constants
│   ├── github_fetcher.py   # Safe GitHub REST API client & file tree extractor
│   ├── static_analyzer.py  # Deterministic, ecosystem-aware readiness analyzer
│   └── utils.py            # URL validation & repo info extraction helpers
└── tests/                  # Automated test suite
    ├── test_basic.py       # Package metadata and URL parsing tests
    ├── test_fetcher.py     # GitHub API fetcher, secret filtering & mock error tests
    ├── test_analyzer.py    # Static analyzer checks, scoring boundaries & ecosystem tests
    └── test_ai.py          # Gemma / Ollama AI integration & resilience tests
```

---

## ⚖️ Scoring Logic & Deterministic Methodology

RepoReady evaluates open-source repository readiness across 6 key pillars using a transparent, 100-point rubric:

| Pillar | Weight | Description |
| :--- | :---: | :--- |
| **Documentation (README)** | 20 pts | Presence, non-empty content, and clear setup/usage guidance. |
| **Open-Source License** | 15 pts | Verified license file or recognized GitHub metadata. |
| **Automated Testing** | 20 pts | Recognizable test directories, test suites, or ecosystem test files. |
| **Continuous Integration (CI)** | 15 pts | GitHub Actions workflows or recognized CI configs. |
| **Contribution & Community** | 15 pts | `CONTRIBUTING.md`, Code of Conduct, and issue/PR templates. |
| **Package Manifest & Env** | 15 pts | Language manifests (`pyproject.toml`, `package.json`, etc.) & safe `.env.example` templates. |
| **Total Maximum Score** | **100 pts** | |

> **Disclaimer:** The readiness score measures contributor onboarding, repository hygiene, and documentation signals. It is not an assessment of runtime performance, code architecture, or security vulnerabilities.

---

## 🤖 Google Gemma AI Insights via Ollama

RepoReady integrates **Google Gemma** (defaulting to `gemma3:1b`) using local Ollama:
- **Preserved Source of Truth:** Gemma **does not** alter deterministic findings, severity ratings, or the readiness score.
- **Explain & Prioritize:** Gemma provides plain-English summaries, prioritizes fixes for maintainers, and suggests onboarding paths for first-time contributors.
- **Untrusted Input Protection:** Repository content is never fed directly to the model as executable prompts. Secrets (`.env`, keys) are never sent.
- **Graceful Degradation:** RepoReady is fully functional without Ollama or Gemma. If Ollama is not running, clear instructions are displayed in the UI while the static analyzer continues working.

### Setting Up Ollama & Gemma (Optional for AI Insights)

1. **Install Ollama:**
   Download from [ollama.com](https://ollama.com) and install it on your system.

2. **Start Ollama service:**
   ```powershell
   ollama serve
   ```

3. **Download the Gemma model:**
   ```powershell
   ollama pull gemma3:1b
   ```
   *(To use a different model, set `$env:OLLAMA_MODEL = "your-model"` before launching RepoReady).*

4. **Troubleshooting missing model errors:**
   - If you see `Model 'gemma3:1b' not found`, ensure you ran `ollama pull gemma3:1b`.
   - Verify installed models with: `ollama list`.

---

## 🔒 Security & Privacy Guarantees

- **No Secret Access:** Secret-bearing files (`.env`, private keys, certificates, credentials) are strictly excluded from fetching and are never sent to the model or displayed.
- **Safe Tree Inspection:** Only public repository file metadata and small, whitelisted text files (`README`, `CONTRIBUTING`, `LICENSE`, `.env.example`) are fetched.
- **Zero Remote Code Execution:** Repository code is never executed, downloaded into a shell, or run in any environment.
- **Zero Paid Dependencies:** No paid APIs, API keys, external databases, or third-party cloud services are required.

---

## 🛠️ Quickstart (Windows PowerShell)

Follow these steps in PowerShell to set up and run RepoReady locally:

### 1. Open PowerShell and navigate to the project directory
```powershell
cd c:\Users\DELL\Downloads\repoready
```

### 2. Create a virtual environment
```powershell
python -m venv .venv
```

### 3. Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```
> *Note: If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.*

### 4. Upgrade pip and install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run automated tests
```powershell
pytest
```

### 6. Launch the Streamlit application
```powershell
streamlit run app.py
```
Open your browser at the URL shown in your terminal (typically `http://localhost:8501`).

---

## 📌 Status
- [x] Step 1: Project foundation & landing page (Completed)
- [x] Step 2: Deterministic static repository analysis engine (Completed)
- [x] Step 3: Google Gemma AI integration via Ollama (Completed)
