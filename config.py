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
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "bonsai-2:27b")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITER", "15"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # seconds; 7b+ models need 5-10 min on CPU
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))  # max tokens per generation; caps runaway output
SNIPPET_DIR = os.path.join(os.path.dirname(__file__), "snippets")
BASH_TIMEOUT = int(os.getenv("BASH_TIMEOUT", "60"))
RAG_FETCH_TIMEOUT = int(os.getenv("RAG_FETCH_TIMEOUT", "30"))
RAG_DIR = os.path.join(os.path.dirname(__file__), "rag_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bonsai-2:27b")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
STREAM_OUTPUT = os.getenv("STREAM_OUTPUT", "true").lower() not in ("0", "false", "no")

# Ordered stages the orchestrator runs for an implementation ("pipeline") task.
# Default is read-only planning + coding. Append "lint" and/or "tester" to add a
# deterministic verification stage after the coder, e.g.
#   PIPELINE_STAGES=planner,breakdown,coder,lint,tester
# Each extra stage is another full ReAct run (minutes per stage on CPU), so it is
# opt-in. Unknown names are ignored at runtime.
PIPELINE_STAGES = tuple(
    s.strip() for s in os.getenv("PIPELINE_STAGES", "planner,breakdown,coder").split(",") if s.strip()
)

# Per-agent model overrides — Preset A (balanced, ~7.3 GB, targets 16 GB RAM / 8 cores).
# Override individually: CODER_MODEL=qwen2.5-coder:7b python agent.py
AGENT_MODELS: dict[str, str] = {
    "orchestrator": os.getenv("ORCHESTRATOR_MODEL", "bonsai-2:27b"),
    "coder":        os.getenv("CODER_MODEL",        "bonsai-2:27b"),
    "lint":         os.getenv("LINT_MODEL",         "bonsai-2:27b"),
    "planner":      os.getenv("PLANNER_MODEL",      "bonsai-2:27b"),
    "tester":       os.getenv("TESTER_MODEL",       "bonsai-2:27b"),
    "breakdown":    os.getenv("BREAKDOWN_MODEL",    "bonsai-2:27b"),
    "general":      os.getenv("GENERAL_MODEL",      "bonsai-2:27b")
}
