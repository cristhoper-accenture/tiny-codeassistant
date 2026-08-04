"""LLM client — Ollama (default) or any OpenAI-compatible cloud endpoint."""

import json
import os
import requests
from config import (
    OLLAMA_BASE_URL, DEFAULT_MODEL, LLM_TIMEOUT, LLM_MAX_TOKENS,
    LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY,
)

_CONNECT_TIMEOUT = 10  # seconds to establish TCP connection


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    on_chunk=None,
) -> str:
    """Send a chat request to the configured provider.

    When *on_chunk* is provided the request streams and each text token is
    passed to ``on_chunk(chunk: str)`` as it arrives.  Either way the full
    accumulated response string is returned.
    """
    if LLM_PROVIDER == "cloud":
        return _chat_cloud(messages, model, on_chunk)
    return _chat_ollama(messages, model, on_chunk)


# ── Ollama ────────────────────────────────────────────────────────────────────

def _chat_ollama(messages: list[dict], model: str, on_chunk) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    stream = on_chunk is not None
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.2, "num_predict": LLM_MAX_TOKENS},
    }

    if stream:
        parts: list[str] = []
        with requests.post(url, json=payload, stream=True, timeout=(_CONNECT_TIMEOUT, LLM_TIMEOUT)) as resp:
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

    resp = requests.post(url, json=payload, timeout=(_CONNECT_TIMEOUT, LLM_TIMEOUT))
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── Cloud (OpenAI-compatible) ─────────────────────────────────────────────────

def _chat_cloud(messages: list[dict], model: str, on_chunk) -> str:
    if not LLM_API_KEY:
        raise EnvironmentError("LLM_API_KEY is not set — required for LLM_PROVIDER=cloud")

    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    stream = on_chunk is not None
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": stream,
    }

    if stream:
        parts: list[str] = []
        with requests.post(url, json=payload, headers=headers, stream=True,
                           timeout=(_CONNECT_TIMEOUT, LLM_TIMEOUT)) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                chunk = data["choices"][0].get("delta", {}).get("content") or ""
                if chunk:
                    parts.append(chunk)
                    on_chunk(chunk)
        return "".join(parts)

    resp = requests.post(url, json=payload, headers=headers,
                         timeout=(_CONNECT_TIMEOUT, LLM_TIMEOUT))
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(text: str, model: str = "qwen3-embedding:4b") -> list[float]:
    """Embed text. Always uses Ollama regardless of LLM_PROVIDER."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text},
                         timeout=(_CONNECT_TIMEOUT, LLM_TIMEOUT))
    resp.raise_for_status()
    return resp.json()["embedding"]


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def preload(model: str = DEFAULT_MODEL) -> None:
    """Warm up a model. No-op for cloud providers (nothing to load locally)."""
    if LLM_PROVIDER != "ollama":
        return
    url = f"{OLLAMA_BASE_URL}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "keep_alive": -1},
        timeout=(_CONNECT_TIMEOUT, LLM_TIMEOUT),
    )
    resp.raise_for_status()


def list_models() -> list[str]:
    """List available models. Returns Ollama tags; empty for cloud providers."""
    if LLM_PROVIDER != "ollama":
        return []
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
