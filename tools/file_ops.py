"""File read, write, edit, and list operations — all relative to a cwd."""

import os


def _resolve(path: str, cwd: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return path


def read_file(path: str, cwd: str = ".") -> str:
    path = _resolve(path, cwd)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str, cwd: str = ".") -> str:
    path = _resolve(path, cwd)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"


def edit_file(path: str, old_text: str, new_text: str, cwd: str = ".") -> str:
    path = _resolve(path, cwd)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_text not in content:
        return f"ERROR: text not found in {path}"
    updated = content.replace(old_text, new_text, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return f"Replaced text in {path}"


def list_dir(path: str = ".", cwd: str = ".") -> str:
    path = _resolve(path, cwd)
    entries = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        tag = "/" if os.path.isdir(full) else ""
        entries.append(f"{entry}{tag}")
    return f"[{path}]\n" + ("\n".join(entries) if entries else "(empty)")
