import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen2.5-coder:7b")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITER", "15"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))  # seconds; increase for large models
SNIPPET_DIR = os.path.join(os.path.dirname(__file__), "snippets")
BASH_TIMEOUT = int(os.getenv("BASH_TIMEOUT", "60"))
RAG_FETCH_TIMEOUT = int(os.getenv("RAG_FETCH_TIMEOUT", "30"))
RAG_DIR = os.path.join(os.path.dirname(__file__), "rag_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
STREAM_OUTPUT = os.getenv("STREAM_OUTPUT", "true").lower() not in ("0", "false", "no")

# Per-agent model overrides — Preset A (balanced, ~7.3 GB, targets 16 GB RAM / 8 cores).
# Override individually: CODER_MODEL=qwen2.5-coder:7b python agent.py
AGENT_MODELS: dict[str, str] = {
    "orchestrator": os.getenv("ORCHESTRATOR_MODEL", "qwen2.5:0.5b"),
    "coder":        os.getenv("CODER_MODEL",        "qwen2.5-coder:7b"),
    "lint":         os.getenv("LINT_MODEL",         "qwen2.5-coder:7b"),
    "planner":      os.getenv("PLANNER_MODEL",      "phi4-mini:3.8b"),
    "general":      os.getenv("GENERAL_MODEL",      "phi4-mini:3.8b"),
}
