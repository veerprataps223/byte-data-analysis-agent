"""
Configuration module for the Data Analysis Agent.
Handles Qwen open model selection, local endpoints, file paths, and runtime settings.
"""
import os
import urllib.request
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "data_artifacts"
MEMORY_FILE_PATH = PROJECT_ROOT / "persistent_memory.json"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Providers Configuration: "qwen" (Local Open Model) or "fallback" (Deterministic Planner)
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

# Local & Hosted Qwen Endpoints (vLLM, Ollama, LM Studio, SGLang, or local OpenAI-compatible server)
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "local-qwen")

# Agent Limits
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "8"))
EXECUTION_TIMEOUT_SECONDS = int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "30"))

def detect_best_provider() -> str:
    """Detect available LLM provider: defaults to Qwen if active server reachable, else deterministic fallback."""
    prov = os.getenv("LLM_PROVIDER", "").lower()
    if prov == "qwen":
        return "qwen"
    if prov in ["fallback", "deterministic"]:
        return "fallback"

    # Fast probe (0.2s) to check if a local Qwen server is actively running
    try:
        probe_url = QWEN_BASE_URL.rstrip("/") + "/models"
        req = urllib.request.Request(probe_url, headers={"Authorization": f"Bearer {QWEN_API_KEY}"})
        with urllib.request.urlopen(req, timeout=0.2):
            return "qwen"
    except Exception:
        pass

    return "fallback"
