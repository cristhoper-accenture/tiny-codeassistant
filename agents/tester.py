"""
TesterAgent — writes test files, runs them, then asks the user for confirmation.

If the user confirms OK → final_answer with a success summary.
If the user says no   → final_answer acknowledging that the user will fix any issues.

Workflow:
  1. Explore  — find the test framework, existing tests, and the code under test
  2. Design   — decide what scenarios to cover
  3. Write    — create/edit test files
  4. Run      — execute the test suite with run_tests
  5. Confirm  — ask_confirmation with the raw test output
  6. Report   — final_answer based on the user's answer
"""

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

from agents.base import BaseAgent

_EXPLORE_TOOLS = [
    "find_files", "list_dir", "code_outline",
    "read_file", "read_lines", "grep_code",
    "git_status",
]
_WRITE_TOOLS = ["write_file", "edit_file"]
_RUN_TOOLS   = ["run_tests", "lint", "bash"]
_LLM_TOOLS   = ["generate_tests", "explain_code"]


class TesterAgent(BaseAgent):
    name = "tester"
    label = "Tester"
    border_color = "bright_blue"

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        if action != "ask_confirmation":
            return super().handle_special_action(action, action_input)

        prompt_text = action_input.get("prompt", "Do the tests look OK?")
        results     = action_input.get("results", "")

        if results:
            if _RICH:
                from rich.markup import escape
                _console.print(Panel(
                    escape(results[:3000]) + ("…" if len(results) > 3000 else ""),
                    title="[bold bright_blue]Test Results[/bold bright_blue]",
                    border_style="bright_blue",
                ))
            else:
                print(f"\n--- Test Results ---\n{results[:3000]}\n---")

        if _RICH:
            answer = Prompt.ask(
                f"\n[bold yellow]{prompt_text}[/bold yellow]",
                choices=["y", "n"],
                default="y",
            )
        else:
            raw = input(f"\n{prompt_text} [Y/n]: ").strip().lower()
            answer = raw if raw in ("y", "n") else "y"

        if answer == "y":
            return (
                "User confirmed: tests look OK. "
                "Proceed to final_answer with a success summary of what was written and what passed."
            )
        return (
            "User indicated the tests may need fixing, but will handle it themselves. "
            "Do NOT retry or rewrite anything. "
            "Proceed immediately to final_answer: acknowledge the test output, "
            "list the test files that were written, and note that the user will address any failures."
        )

    def build_system_prompt(self) -> str:
        return f"""You are a test-engineering specialist. You write test files, run them, and present results for confirmation.

Scope: write unit/integration tests, run the suite, report results.
Out of scope: production code (→ coder), lint (→ lint), architecture (→ planner). If production code must change, say so in final_answer.

## Current working directory
`{self.cwd}`

## Workflow
EXPLORE → DESIGN → WRITE → RUN → CONFIRM → REPORT
- EXPLORE: locate existing tests and the code under test; detect the framework (pytest.ini / setup.cfg / pyproject.toml)
- DESIGN: list functions/endpoints to cover — happy path, edge cases, errors
- WRITE: name files test_<module>.py; write_file for new, edit_file for existing (read first); conftest.py for shared fixtures
- RUN: run_tests, then lint the test files
- CONFIRM: call `ask_confirmation` (MANDATORY before final_answer):
```json
{{"action": "ask_confirmation", "action_input": {{"prompt": "Tests finished. Do the results look OK?", "results": "<test output>"}}}}
```
- REPORT: YES → summarise tests and passes. NO → list files written, acknowledge failures, note the user will fix. Do NOT retry.

## Rules
- One tool call per response; prefer relative paths
- Never edit production source — test files only
- Do NOT call final_answer before ask_confirmation
- If run_tests exits non-zero, still call ask_confirmation — the user decides

## Available tools

{self._tool_docs(_EXPLORE_TOOLS + _WRITE_TOOLS + _RUN_TOOLS + _LLM_TOOLS)}

## Response format

```json
{{"thought": "<reasoning>", "action": "<tool_name | ask_confirmation | final_answer>", "action_input": {{<parameters>}}}}
```
"""
