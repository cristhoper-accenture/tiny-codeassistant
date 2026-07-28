"""Ollama client wrapper using the /api/chat endpoint."""

import json
import os
import requests
from config import OLLAMA_BASE_URL, DEFAULT_MODEL

# Larger models (9b+) can take several minutes to generate long code blocks.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    on_chunk=None,
) -> str:
    """Send a chat request to Ollama.

    When *on_chunk* is provided the request streams and each text token is
    passed to ``on_chunk(chunk: str)`` as it arrives.  Either way the full
    accumulated response string is returned.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    stream = on_chunk is not None
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.2},
    }

    if stream:
        parts: list[str] = []
        with requests.post(url, json=payload, stream=True, timeout=LLM_TIMEOUT) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    parts.append(chunk)
                    on_chunk(chunk)
                if data.get("done"):
                    break
        return "".join(parts)

    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


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
