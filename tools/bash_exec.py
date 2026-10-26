"""Bash command execution with timeout and output capture.

NOTE: This is a best-effort guardrail against obviously destructive commands —
NOT a security sandbox. Commands run with the full privileges of the user that
launched the assistant and can read/write/delete anything that user can. The
block patterns below stop common accidents, not a determined bypass. Run the
assistant against directories you are willing to let it modify.
"""

import re
import subprocess
from config import BASH_TIMEOUT

# Exact/substring triggers for dangerous commands.
_BLOCKED = {
    "mkfs", ":(){:|:&};:", "dd if=/dev/zero", "sudo", "shutdown", "reboot",
    "init 0", "init 6", "poweroff", "ssh", "scp", "ftp", "telnet", "nc",
    "netcat", "curl -O", "wget -O", "chmod 777 /", "chown root:root /",
    "mount /dev/sda1 /mnt", "umount /mnt",
}

# Regex patterns catch whitespace/argument-order variants the literal list misses,
# e.g. "rm   -rf  /", "rm -rf ~", "rm -fr /*", "rm -rf $HOME".
_BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+(-[a-z]*\s+)*-?[a-z]*r[a-z]*f[a-z]*\s+(/|~|\$HOME|/\*)(\s|$)"),
    re.compile(r"\brm\s+(-[a-z]*\s+)*-?[a-z]*f[a-z]*r[a-z]*\s+(/|~|\$HOME|/\*)(\s|$)"),
    re.compile(r">\s*/dev/sd[a-z]"),
]


def run(command: str, cwd: str = None, timeout: int = BASH_TIMEOUT) -> dict:
    """Execute a shell command in cwd. Returns {stdout, stderr, returncode, cwd}."""
    normalized = " ".join(command.split())  # collapse runs of whitespace
    for blocked in _BLOCKED:
        if blocked in normalized:
            return {"stdout": "", "stderr": f"Blocked dangerous command: {blocked}", "returncode": -1, "cwd": cwd}
    for pat in _BLOCKED_PATTERNS:
        if pat.search(normalized):
            return {"stdout": "", "stderr": f"Blocked dangerous command (matched {pat.pattern})", "returncode": -1, "cwd": cwd}

    # Capture the final cwd after the command runs
    tracked = f"{command}\npwd"
    try:
        result = subprocess.run(
            tracked,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        stdout_lines = result.stdout.splitlines()
        new_cwd = stdout_lines[-1].strip() if stdout_lines else cwd
        stdout = "\n".join(stdout_lines[:-1])
        return {
            "stdout": stdout[:8000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
            "cwd": new_cwd if result.returncode == 0 else cwd,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1, "cwd": cwd}


def format_result(result: dict) -> str:
    parts = []
    if result["stdout"]:
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"STDERR:\n{result['stderr']}")
    parts.append(f"Exit code: {result['returncode']}")
    parts.append(f"Working dir: {result['cwd']}")
    return "\n".join(parts)
