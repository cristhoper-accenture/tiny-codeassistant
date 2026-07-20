"""Ollama client wrapper using the /api/chat endpoint."""

import json
import requests
from config import OLLAMA_BASE_URL, DEFAULT_MODEL


def chat(messages: list[dict], model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def embed(text: str, model: str = "qwen3-embedding:4b") -> list[float]:
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


def list_models() -> list[str]:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
