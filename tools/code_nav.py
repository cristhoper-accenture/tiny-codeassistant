"""
Code navigation tools: grep, find files, AST outline, line-range reads.
"""

import os
import re
import ast
import glob
import fnmatch


# ── grep_code ──────────────────────────────────────────────────────────────────

def grep_code(
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    case_sensitive: bool = True,
    max_matches: int = 50,
    cwd: str = ".",
) -> str:
    """Regex search across files. Returns matched lines with file:line context."""
    root = path if os.path.isabs(path) else os.path.join(cwd, path)
    root = os.path.normpath(root)
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fnmatch.fnmatch(fname, file_glob):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, cwd)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if rx.search(line):
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(matches) >= max_matches:
                                matches.append(f"… (stopped at {max_matches} matches)")
                                return "\n".join(matches)
            except (OSError, PermissionError):
                continue

    return "\n".join(matches) if matches else f"No matches for '{pattern}' in {root}"


# ── find_files ─────────────────────────────────────────────────────────────────

def find_files(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
    cwd: str = ".",
) -> str:
    """Find files matching a glob pattern (e.g. '**/*.py', 'test_*.py')."""
    root = path if os.path.isabs(path) else os.path.join(cwd, path)
    root = os.path.normpath(root)
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules"}

    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if fnmatch.fnmatch(fname, pattern):
                fpath = os.path.join(dirpath, fname)
                results.append(os.path.relpath(fpath, cwd))
                if len(results) >= max_results:
                    break

    if not results:
        return f"No files matching '{pattern}' under {root}"
    return "\n".join(sorted(results))


# ── read_lines ─────────────────────────────────────────────────────────────────

def read_lines(path: str, start: int = 1, end: int = None, cwd: str = ".") -> str:
    """Read a specific line range from a file (1-indexed, inclusive)."""
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    path = os.path.expanduser(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    s = max(1, start) - 1
    e = min(total, end) if end is not None else total
    selected = lines[s:e]
    numbered = [f"{s + i + 1:>6}  {ln.rstrip()}" for i, ln in enumerate(selected)]
    header = f"[{os.path.relpath(path, cwd)}  lines {s+1}–{s+len(selected)} of {total}]\n"
    return header + "\n".join(numbered)


# ── code_outline ───────────────────────────────────────────────────────────────

def code_outline(path: str, cwd: str = ".") -> str:
    """
    Return the structural outline of a source file.
    For Python: uses AST — classes, functions, imports, decorators.
    For JS/TS: regex-based extraction of exports, classes, functions.
    For others: returns the first 60 lines.
    """
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    path = os.path.expanduser(path)

    with open(path, encoding="utf-8", errors="replace") as f:
        source = f.read()

    ext = os.path.splitext(path)[1].lower()
    rel = os.path.relpath(path, cwd)

    if ext == ".py":
        return _python_outline(source, rel)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        return _js_outline(source, rel)
    else:
        lines = source.splitlines()[:60]
        return f"[{rel} — first {len(lines)} lines]\n" + "\n".join(lines)


def _python_outline(source: str, label: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError in {label}: {e}"

    lines = []
    lines.append(f"[{label}]")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = ", ".join(a.name for a in node.names)
                lines.append(f"  import  from {mod} import {names}")
            else:
                names = ", ".join(a.name for a in node.names)
                lines.append(f"  import  {names}")
        break  # only top-level pass for imports — full walk below

    # Top-level definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            lines.append(f"\n  class {node.name}({bases})  [line {node.lineno}]")
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decs = "".join(f"@{ast.unparse(d)} " for d in item.decorator_list)
                    args = _fmt_args(item.args)
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    lines.append(f"    {decs}{prefix}def {item.name}({args})  [line {item.lineno}]")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = "".join(f"@{ast.unparse(d)} " for d in node.decorator_list)
            args = _fmt_args(node.args)
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            lines.append(f"\n  {decs}{prefix}def {node.name}({args})  [line {node.lineno}]")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = ", ".join(a.name for a in node.names)
                lines.append(f"  from {mod} import {names}")
            else:
                names = ", ".join(a.name for a in node.names)
                lines.append(f"  import {names}")

    return "\n".join(lines)


def _fmt_args(args: ast.arguments) -> str:
    parts = [a.arg for a in args.args]
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _js_outline(source: str, label: str) -> str:
    patterns = [
        (r"^export\s+default\s+(?:class|function)\s+(\w+)", "export default"),
        (r"^export\s+(?:async\s+)?function\s+(\w+)", "export function"),
        (r"^export\s+class\s+(\w+)", "export class"),
        (r"^(?:async\s+)?function\s+(\w+)", "function"),
        (r"^class\s+(\w+)", "class"),
        (r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", "const arrow"),
        (r"^export\s+const\s+(\w+)\s*=", "export const"),
    ]
    lines = ["[" + label + "]"]
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        for rx, kind in patterns:
            m = re.match(rx, stripped)
            if m:
                lines.append(f"  {kind} {m.group(1)}  [line {lineno}]")
                break
    return "\n".join(lines) if len(lines) > 1 else f"[{label}] — no top-level definitions found"
