"""Safe bash command execution with timeout and output capture."""

import subprocess
import shlex
from config import BASH_TIMEOUT

# Commands blocked for safety
_BLOCKED = {"rm -rf /", "mkfs", ":(){:|:&};:", "dd if=/dev/zero"}


def run(command: str, timeout: int = BASH_TIMEOUT) -> dict:
    """Execute a shell command. Returns {stdout, stderr, returncode}."""
    for blocked in _BLOCKED:
        if blocked in command:
            return {"stdout": "", "stderr": f"Blocked dangerous command: {blocked}", "returncode": -1}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout[:8000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1}


def format_result(result: dict) -> str:
    parts = []
    if result["stdout"]:
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"STDERR:\n{result['stderr']}")
    parts.append(f"Exit code: {result['returncode']}")
    return "\n".join(parts)
