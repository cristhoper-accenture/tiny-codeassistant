"""
Code-focused LLM tools: explain, fix, generate tests, and lint.
"""

import os
import subprocess
import shutil

import llm
from config import DEFAULT_MODEL


# ── LLM-powered tools ──────────────────────────────────────────────────────────

def explain_code(code: str, language: str = "", model: str = DEFAULT_MODEL) -> str:
    """Explain what a piece of code does in plain English."""
    lang_hint = f" ({language})" if language else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert code reviewer. Explain the given code clearly and concisely. "
                "Describe: what it does, key decisions, potential pitfalls, and complexity."
            ),
        },
        {"role": "user", "content": f"Explain this{lang_hint} code:\n\n```\n{code}\n```"},
    ]
    return llm.chat(messages, model=model)


def fix_code(code: str, error: str, language: str = "", model: str = DEFAULT_MODEL) -> str:
    """Given code and an error message, return fixed code with an explanation."""
    lang_hint = f"{language} " if language else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert debugger. Given code and an error, produce: "
                "1) the root cause, 2) the corrected code (full, runnable), 3) what changed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fix this {lang_hint}code.\n\n"
                f"**Code:**\n```\n{code}\n```\n\n"
                f"**Error:**\n```\n{error}\n```"
            ),
        },
    ]
    return llm.chat(messages, model=model)


def generate_tests(code: str, language: str = "python", framework: str = "", model: str = DEFAULT_MODEL) -> str:
    """Generate unit tests for the given code."""
    fw_hint = f" using {framework}" if framework else ""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert in {language} testing. "
                f"Generate comprehensive unit tests{fw_hint} for the given code. "
                "Cover: happy path, edge cases, error cases. Output only the test code."
            ),
        },
        {"role": "user", "content": f"Write tests for:\n\n```{language}\n{code}\n```"},
    ]
    return llm.chat(messages, model=model)


def review_code(code: str, language: str = "", model: str = DEFAULT_MODEL) -> str:
    """Review code for bugs, style issues, and improvements."""
    lang_hint = f" ({language})" if language else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior code reviewer. Review the code and provide structured feedback: "
                "bugs, security issues, performance, readability, and suggested improvements. "
                "Be specific and actionable."
            ),
        },
        {"role": "user", "content": f"Review this{lang_hint} code:\n\n```\n{code}\n```"},
    ]
    return llm.chat(messages, model=model)


# ── Lint / static analysis ─────────────────────────────────────────────────────

def lint(path: str, cwd: str = ".") -> str:
    """
    Run a linter on a file or directory.
    Auto-detects: ruff > flake8 > pylint for Python; eslint for JS/TS.
    Falls back to Python AST check if no linter is found.
    """
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    path = os.path.normpath(os.path.expanduser(path))

    ext = _ext(path)

    if ext == ".py":
        return _lint_python(path, cwd)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        return _lint_js(path, cwd)
    else:
        return f"No linter configured for {ext} files."


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _run(cmd: list[str], cwd: str) -> str:
    from config import BASH_TIMEOUT
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=BASH_TIMEOUT)
        out = (r.stdout + r.stderr).strip()
        return out or f"Exit code: {r.returncode} (no output)"
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Linter timed out."


def _lint_python(path: str, cwd: str) -> str:
    for tool, args in [
        ("ruff", ["ruff", "check", path]),
        ("flake8", ["flake8", path]),
        ("pylint", ["pylint", "--output-format=text", path]),
    ]:
        if shutil.which(tool):
            result = _run(args, cwd)
            if result is not None:
                return f"[{tool}]\n{result}"

    # Fallback: AST syntax check
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        import ast
        ast.parse(source)
        return f"No linter found. AST syntax check passed for {path}."
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def _lint_js(path: str, cwd: str) -> str:
    if shutil.which("eslint"):
        result = _run(["eslint", path], cwd)
        if result is not None:
            return f"[eslint]\n{result}"
    return f"No JS linter found (install eslint). Cannot lint {path}."


# ── Test runner ────────────────────────────────────────────────────────────────

def run_tests(
    path: str = ".",
    pattern: str = "",
    verbose: bool = True,
    cwd: str = ".",
) -> str:
    """
    Run tests using pytest (preferred) or unittest.
    path: file or directory to test.
    pattern: test name filter (pytest -k).
    """
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    path = os.path.normpath(path)

    if shutil.which("pytest"):
        cmd = ["pytest", path, "--tb=short", "--no-header"]
        if verbose:
            cmd.append("-v")
        if pattern:
            cmd += ["-k", pattern]
        result = _run(cmd, cwd)
        return f"[pytest]\n{result}"

    # Fallback: python -m unittest discover
    cmd = ["python", "-m", "unittest", "discover", "-s", path]
    if pattern:
        cmd += ["-p", pattern]
    result = _run(cmd, cwd)
    return f"[unittest]\n{result}"
