"""
Git operations: status, diff, log, commit, branch.
All commands run in the given cwd (the git repo root or any subdirectory).
"""

import subprocess
import os
from config import BASH_TIMEOUT


def _git(args: list[str], cwd: str) -> tuple[str, str, int]:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=BASH_TIMEOUT,
    )
    return result.stdout, result.stderr, result.returncode


def _fmt(stdout: str, stderr: str, rc: int) -> str:
    out = stdout.strip()
    err = stderr.strip()
    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"STDERR: {err}")
    if rc != 0:
        parts.append(f"Exit code: {rc}")
    return "\n".join(parts) if parts else "(no output)"


def status(cwd: str) -> str:
    """Show working tree status (short format + branch)."""
    out1, err1, rc1 = _git(["rev-parse", "--show-toplevel"], cwd)
    root = out1.strip() or cwd
    out, err, rc = _git(["status", "-sb"], cwd)
    return f"[repo: {root}]\n" + _fmt(out, err, rc)


def diff(path: str = None, staged: bool = False, cwd: str = ".") -> str:
    """Show unstaged (or --staged) diff, optionally scoped to a path."""
    args = ["diff"]
    if staged:
        args.append("--staged")
    args += ["--stat", "--patch", "--", path] if path else ["--stat", "--patch"]
    out, err, rc = _git(args, cwd)
    result = _fmt(out, err, rc)
    return result[:8000] + ("…" if len(result) > 8000 else "")


def log(n: int = 10, oneline: bool = True, cwd: str = ".") -> str:
    """Show recent git commits."""
    fmt = "--oneline" if oneline else "--format=%H %ad %an%n%s%n"
    out, err, rc = _git(["log", fmt, f"-{n}"], cwd)
    return _fmt(out, err, rc)


def commit(message: str, files: list[str] = None, cwd: str = ".") -> str:
    """Stage files (or all tracked changes) and create a commit."""
    if files:
        paths = [f if os.path.isabs(f) else os.path.join(cwd, f) for f in files]
        out, err, rc = _git(["add", "--"] + paths, cwd)
        if rc != 0:
            return f"git add failed:\n{_fmt(out, err, rc)}"
    else:
        out, err, rc = _git(["add", "-u"], cwd)
        if rc != 0:
            return f"git add -u failed:\n{_fmt(out, err, rc)}"

    out, err, rc = _git(["commit", "-m", message], cwd)
    return _fmt(out, err, rc)


def branch(cwd: str) -> str:
    """List branches and show the current one."""
    out, err, rc = _git(["branch", "-a", "--color=never"], cwd)
    return _fmt(out, err, rc)


def checkout(ref: str, create: bool = False, cwd: str = ".") -> str:
    """Checkout a branch or commit. Set create=True to create a new branch."""
    args = ["checkout", "-b", ref] if create else ["checkout", ref]
    out, err, rc = _git(args, cwd)
    return _fmt(out, err, rc)


def blame(path: str, cwd: str = ".") -> str:
    """Show who last modified each line of a file."""
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    out, err, rc = _git(["blame", "--date=short", path], cwd)
    result = _fmt(out, err, rc)
    return result[:6000] + ("…" if len(result) > 6000 else "")
