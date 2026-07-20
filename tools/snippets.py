"""Save, list, and retrieve named code snippets."""

import os
import json
from datetime import datetime
from config import SNIPPET_DIR

os.makedirs(SNIPPET_DIR, exist_ok=True)
_INDEX = os.path.join(SNIPPET_DIR, "index.json")


def _load_index() -> dict:
    if not os.path.exists(_INDEX):
        return {}
    with open(_INDEX) as f:
        return json.load(f)


def _save_index(index: dict) -> None:
    with open(_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def save(name: str, code: str, language: str = "", description: str = "") -> str:
    """Save a snippet. Returns confirmation."""
    index = _load_index()
    safe_name = name.replace(" ", "_").replace("/", "_")
    ext = {"python": ".py", "javascript": ".js", "typescript": ".ts",
           "bash": ".sh", "sql": ".sql"}.get(language.lower(), ".txt")
    filename = f"{safe_name}{ext}"
    filepath = os.path.join(SNIPPET_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    index[safe_name] = {
        "file": filename,
        "language": language,
        "description": description,
        "created": datetime.now().isoformat(),
    }
    _save_index(index)
    return f"Snippet '{safe_name}' saved to {filepath}"


def get(name: str) -> str:
    """Retrieve a snippet by name."""
    index = _load_index()
    safe_name = name.replace(" ", "_")
    if safe_name not in index:
        return f"Snippet '{name}' not found. Available: {', '.join(index.keys()) or 'none'}"
    filepath = os.path.join(SNIPPET_DIR, index[safe_name]["file"])
    with open(filepath, encoding="utf-8") as f:
        code = f.read()
    meta = index[safe_name]
    header = f"# {safe_name} [{meta.get('language', '')}] — {meta.get('description', '')}\n\n"
    return header + code


def list_snippets() -> str:
    """List all saved snippets."""
    index = _load_index()
    if not index:
        return "No snippets saved yet."
    lines = []
    for name, meta in index.items():
        lines.append(
            f"• {name} [{meta.get('language', '?')}] — {meta.get('description', '')} ({meta.get('created', '')[:10]})"
        )
    return "\n".join(lines)


def delete(name: str) -> str:
    index = _load_index()
    safe_name = name.replace(" ", "_")
    if safe_name not in index:
        return f"Snippet '{name}' not found."
    filepath = os.path.join(SNIPPET_DIR, index[safe_name]["file"])
    if os.path.exists(filepath):
        os.remove(filepath)
    del index[safe_name]
    _save_index(index)
    return f"Snippet '{safe_name}' deleted."
