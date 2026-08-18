"""Shared agent memory — a JSON file persisted in the project cwd.

All agents read and write to .agent_memory.json so context (files created,
decisions made, completed steps, open questions) survives across agent turns.
"""

import json
import os

MEMORY_FILENAME = ".agent_memory.json"

_DISPLAY_KEYS_ORDER = [
    "project", "files_created", "files_modified", "decisions",
    "completed", "pending", "notes",
]


def _path(cwd: str) -> str:
    return os.path.join(cwd, MEMORY_FILENAME)


def load(cwd: str) -> dict:
    path = _path(cwd)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(cwd: str, data: dict) -> None:
    with open(_path(cwd), "w") as f:
        json.dump(data, f, indent=2)


def read(cwd: str, key: str = None) -> str:
    data = load(cwd)
    if not data:
        return "(memory is empty)"
    if key:
        if key not in data:
            return f"(no entry for '{key}')"
        return json.dumps({key: data[key]}, indent=2)
    return json.dumps(data, indent=2)


def write(cwd: str, key: str, value) -> str:
    data = load(cwd)
    # For list-valued keys, append instead of overwrite when value is a string.
    if isinstance(data.get(key), list) and isinstance(value, str):
        if value not in data[key]:
            data[key].append(value)
    else:
        data[key] = value
    _save(cwd, data)
    return f"Memory['{key}'] updated."


def format_for_prompt(cwd: str) -> str:
    """Return a compact summary suitable for injection into a system prompt."""
    data = load(cwd)
    if not data:
        return ""
    # Render known keys first, then any extras.
    ordered = {k: data[k] for k in _DISPLAY_KEYS_ORDER if k in data}
    extras  = {k: v for k, v in data.items() if k not in _DISPLAY_KEYS_ORDER}
    merged  = {**ordered, **extras}
    return json.dumps(merged, indent=2)
