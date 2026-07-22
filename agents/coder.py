"""
CoderAgent — specialized for writing, editing, and verifying code.

Workflow enforced by the system prompt:
  1. Explore  — read-only: understand structure, find relevant files
  2. Plan     — emit a `plan` action with explicit steps
  3. Implement — write/edit files, lint after each
  4. Verify   — run lint + tests on all changed files
  5. Report   — final_answer with summary of changes
"""

import json

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

from agents.base import BaseAgent

# Tools the coder is allowed to use, grouped by phase
_EXPLORE_TOOLS = [
    "find_files", "list_dir", "grep_code", "code_outline",
    "read_file", "read_lines", "git_status", "git_log", "git_diff",
    "rag_search", "get_snippet", "list_snippets",
]
_IMPLEMENT_TOOLS = [
    "write_file", "edit_file", "bash", "change_dir",
    "save_snippet", "rag_add_file", "rag_add_text",
]
_VERIFY_TOOLS = [
    "lint", "run_tests", "bash",
    "git_status", "git_diff", "git_commit", "git_branch",
]
_LLM_TOOLS = [
    "explain_code", "fix_code", "generate_tests", "review_code",
    "websearch", "summarize",
]


class CoderAgent(BaseAgent):
    name = "coder"
    label = "Coder"
    border_color = "magenta"

    def build_system_prompt(self) -> str:
        all_docs = self._tool_docs()
        return f"""You are an expert software engineer. Your role is to write, edit, and verify code.

## Current working directory
`{self.cwd}`

## Mandatory workflow

You MUST follow these phases in order. Do not skip any phase.

---

### Phase 1 — EXPLORE (read-only, no file writes)
Understand the codebase before touching anything:
- `find_files` / `list_dir` — map the project structure
- `code_outline` — see classes and functions without reading full files
- `grep_code` — find where relevant symbols are defined or used
- `read_lines` — read specific sections of large files
- `git_status` — see what is already changed
- `rag_search` — query ingested documentation if available

### Phase 2 — PLAN
After exploration, emit a `plan` action:
```json
{{
  "thought": "<what you found and decided>",
  "action": "plan",
  "action_input": {{
    "summary": "<one sentence describing the task>",
    "steps": ["<step 1>", "<step 2>", "..."],
    "files_create": ["<path>", "..."],
    "files_modify": ["<path>", "..."]
  }}
}}
```
Wait for acknowledgement before proceeding.

### Phase 3 — IMPLEMENT
Execute the plan:
- Prefer `edit_file` over `write_file` for existing files
- Use `read_lines` to inspect exact content before editing
- After writing each file, run `lint` to catch issues immediately
- Use `bash` for package installs, script generation, or build steps
- Save reusable patterns with `save_snippet`

### Phase 4 — VERIFY
After all changes:
- `lint` every file you created or modified
- `run_tests` if a test suite exists
- Fix any errors before reporting
- `git_status` to confirm the change set

### Phase 5 — REPORT
Use `final_answer` with a structured summary:
```
## Changes
- Created/modified: <file list>

## What was done
<brief description>

## Verification
- Lint: passed / warnings: <list>
- Tests: passed N / failed M / skipped K
```

---

## Rules
- Never write to a file you have not first read (use `read_file` or `read_lines`)
- One tool call per response; wait for each result
- Prefer relative paths — they resolve from the cwd above
- If lint or tests fail, fix them before calling `final_answer`
- Do not invent file names or APIs — verify they exist with `find_files` or `grep_code` first

## Available tools

{all_docs}

## Response format

```json
{{
  "thought": "<reasoning>",
  "action": "<tool_name or plan or final_answer>",
  "action_input": {{<parameters>}}
}}
```
"""

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        if action != "plan":
            return None

        summary = action_input.get("summary", "")
        steps = action_input.get("steps", [])
        files_create = action_input.get("files_create", [])
        files_modify = action_input.get("files_modify", [])

        self._print_plan(summary, steps, files_create, files_modify)

        observation_parts = [f"Plan acknowledged: {summary}"]
        if steps:
            observation_parts.append("Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)))
        if files_create:
            observation_parts.append("Files to create: " + ", ".join(files_create))
        if files_modify:
            observation_parts.append("Files to modify: " + ", ".join(files_modify))
        observation_parts.append("Proceed with Phase 3 — Implement.")
        return "\n".join(observation_parts)

    def _print_plan(self, summary: str, steps: list, files_create: list, files_modify: list) -> None:
        if not _RICH:
            print(f"\n=== PLAN: {summary} ===")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s}")
            if files_create:
                print(f"  Create: {', '.join(files_create)}")
            if files_modify:
                print(f"  Modify: {', '.join(files_modify)}")
            return

        content_lines = []
        if summary:
            content_lines.append(f"[bold]{summary}[/bold]\n")
        for i, step in enumerate(steps, 1):
            content_lines.append(f"  [cyan]{i}.[/cyan] {step}")
        if files_create:
            content_lines.append(f"\n  [green]Create:[/green] {', '.join(files_create)}")
        if files_modify:
            content_lines.append(f"  [yellow]Modify:[/yellow] {', '.join(files_modify)}")

        _console.print(Panel(
            "\n".join(content_lines),
            title="[bold magenta]Implementation Plan[/bold magenta]",
            border_style="magenta",
        ))
