import os

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Cloud provider (OpenAI-compatible) ────────────────────────────────────────
# Set LLM_PROVIDER=cloud plus LLM_BASE_URL and LLM_API_KEY to route all
# inference to any OpenAI-compatible endpoint (Groq, Mistral, Together,
# OpenRouter, vLLM, etc.).  AGENT_MODELS entries should then be cloud model
# names (e.g. "mistral-large-latest").  Embeddings always use Ollama.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "cloud"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen2.5-coder:7b")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITER", "15"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # seconds; 7b+ models need 5-10 min on CPU
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))  # max tokens per generation; caps runaway output
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
    "orchestrator": os.getenv("ORCHESTRATOR_MODEL", "phi4-mini:3.8b"),
    "coder":        os.getenv("CODER_MODEL",        "qwen2.5-coder:7b"),
    "lint":         os.getenv("LINT_MODEL",         "qwen2.5-coder:7b"),
    "planner":      os.getenv("PLANNER_MODEL",      "phi4-mini:3.8b"),
    "tester":       os.getenv("TESTER_MODEL",       "qwen2.5-coder:7b"),
    "breakdown":    os.getenv("BREAKDOWN_MODEL",    "phi4-mini:3.8b"),
    "general":      os.getenv("GENERAL_MODEL",      "phi4-mini:3.8b"),
}
