"""File read, write, edit, and list operations."""

import os
import re


def read_file(path: str) -> str:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace first occurrence of old_text with new_text in the file."""
    path = os.path.expanduser(path)
    content = read_file(path)
    if old_text not in content:
        return f"ERROR: text not found in {path}"
    updated = content.replace(old_text, new_text, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return f"Replaced text in {path}"


def list_dir(path: str = ".") -> str:
    path = os.path.expanduser(path)
    entries = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        tag = "/" if os.path.isdir(full) else ""
        entries.append(f"{entry}{tag}")
    return "\n".join(entries) if entries else "(empty)"
