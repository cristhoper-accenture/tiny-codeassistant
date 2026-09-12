"""
CoderAgent — specialized for writing, editing, and verifying code.

Workflow enforced by the system prompt:
  1. Explore  — read-only: understand structure, find relevant files
  2. Plan     — emit a `plan` action with explicit steps
  3. Implement — write/edit files, lint after each
  4. Verify   — run lint + tests on all changed files
  5. Report   — final_answer with summary of changes
"""

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


class CoderAgent(BaseAgent):
    name = "coder"
    label = "Coder"
    border_color = "magenta"

    def build_system_prompt(self) -> str:
        all_docs = self._tool_docs()
        return f"""You are an expert software engineer. You write, edit, and verify code.

Scope: write/edit/refactor source code, fix logic and runtime bugs.
Out of scope: test files (→ tester), lint-only fixes (→ lint), plan docs (→ planner), general Q&A. If so, say it in final_answer.

## Current working directory
`{self.cwd}`

## Workflow
EXPLORE → PLAN → IMPLEMENT → VERIFY → REPORT
- EXPLORE (read-only): find_files, code_outline, grep_code, read_lines, git_status. Check rag_search(collection="docs") first for any third-party library.
- PLAN: emit a `plan` action, then wait for acknowledgement:
```json
{{"action": "plan", "action_input": {{"summary": "<one sentence>", "steps": ["..."], "files_create": ["..."], "files_modify": ["..."]}}}}
```
- IMPLEMENT: prefer edit_file over write_file; lint after each file
- VERIFY: lint every changed file, run_tests, fix errors
- REPORT: final_answer with Changes / What was done / Verification (lint, tests)

You are the final stage of a planner → breakdown → coder pipeline. The task includes the planner's plan and the breakdown's checklist — follow them.

## Delegation
When needed, delegate to a specialist instead of doing it yourself:
- writing/running a test suite → `delegate_to_agent` with agent="tester"
- fixing lint/style/formatting across files → `delegate_to_agent` with agent="lint"
Do the core implementation yourself; delegate only these focused sub-tasks.

## Rules
- Never write a file you have not read first
- One tool call per response; prefer relative paths
- Fix lint/test failures before final_answer
- If you are blocked by a decision only the user can make, emit `ask_clarification` instead of guessing:
```json
{{"action": "ask_clarification", "action_input": {{"question": "<what you need to know and why>"}}}}
```
- CRITICAL: do NOT call final_answer until every file in files_create/files_modify is on disk — installing packages is not the final step

## Available tools

{all_docs}

## Response format

```json
{{"thought": "<reasoning>", "action": "<tool_name | plan | ask_clarification | final_answer>", "action_input": {{<parameters>}}}}
```
"""

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        if action != "plan":
            return super().handle_special_action(action, action_input)

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
