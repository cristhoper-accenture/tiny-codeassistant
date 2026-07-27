"""Ollama client wrapper using the /api/chat endpoint."""

import os
import requests
from config import OLLAMA_BASE_URL, DEFAULT_MODEL

# Larger models (9b+) can take several minutes to generate long code blocks.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))


def chat(messages: list[dict], model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def embed(text: str, model: str = "qwen3-embedding:4b") -> list[float]:
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


def preload(model: str = DEFAULT_MODEL) -> None:
    """Load model weights into Ollama memory before the first real call.

    Uses /api/generate with no prompt — the canonical Ollama warm-up call.
    keep_alive=-1 holds the model in memory for the session.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "keep_alive": -1},
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()


def list_models() -> list[str]:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
