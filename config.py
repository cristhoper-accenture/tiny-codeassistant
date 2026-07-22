import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5:2b")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITER", "15"))
SNIPPET_DIR = os.path.join(os.path.dirname(__file__), "snippets")
BASH_TIMEOUT = int(os.getenv("BASH_TIMEOUT", "30"))
RAG_DIR = os.path.join(os.path.dirname(__file__), "rag_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
