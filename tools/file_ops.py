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
    if not old_text:
        return "ERROR: old_text must not be empty; use write_file to overwrite the entire file"
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


def make_dir(path: str, cwd: str = ".") -> str:
    path = _resolve(path, cwd)
    os.makedirs(path, exist_ok=True)
    return f"Directory created: {path}"


def remove_dir(path: str, recursive: bool = False, cwd: str = ".") -> str:
    import shutil
    path = _resolve(path, cwd)
    if not os.path.isdir(path):
        return f"ERROR: not a directory: {path}"
    if recursive:
        shutil.rmtree(path)
        return f"Removed directory (recursive): {path}"
    try:
        os.rmdir(path)
        return f"Removed directory: {path}"
    except OSError as e:
        return f"ERROR: {e} (use recursive=true to remove non-empty directories)"
