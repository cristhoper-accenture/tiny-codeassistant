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

import json

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
        return f"""You are a test-engineering specialist. Your ONLY job is to write test files, \
run them, and present the results to the user for confirmation.

**Scope**: write unit/integration test files, run the test suite, and report results. \
**Out of scope**: implementing features or fixing logic bugs (→ coder agent), \
fixing lint/style errors (→ lint agent), architecture design (→ planner agent).
If the task requires changing production code, say so in final_answer and suggest the coder agent.

## Current working directory
`{self.cwd}`

## Mandatory workflow

Follow these phases in order. Do not skip any phase.

---

### Phase 1 — EXPLORE
Understand the project's test setup before writing anything:
- `find_files("*test*")` / `find_files("*spec*")` — locate existing test files
- `list_dir` — map the project structure
- `grep_code` — find imports, class names, and public functions to test
- `read_file` — read the source files under test to understand what scenarios are needed
- `bash("cat pytest.ini || cat setup.cfg || cat pyproject.toml")` — detect the test framework and config

### Phase 2 — DESIGN
Decide what to cover before writing:
- List the public functions / methods / endpoints that need tests
- Identify happy path, edge cases, and error cases for each
- Note any fixtures, mocks, or test helpers already present

### Phase 3 — WRITE
Create or extend test files:
- Name files `test_<module>.py` (pytest convention) unless the project uses another convention
- Use `write_file` for new files; use `edit_file` for existing ones (read first)
- One `assert` per logical check — keep tests focused and readable
- Add fixtures in `conftest.py` if multiple tests need the same setup
- Use `generate_tests` to draft test stubs for complex functions

### Phase 4 — RUN
Execute the tests:
- `run_tests` — run the full suite or the specific file(s) you wrote
- `lint` the test files to catch obvious errors

### Phase 5 — CONFIRM
Call `ask_confirmation` with the test output so the user can review:
```json
{{
  "thought": "<summary of what ran and what the results were>",
  "action": "ask_confirmation",
  "action_input": {{
    "prompt": "Tests finished. Do the results look OK?",
    "results": "<full or summarised test output>"
  }}
}}
```
You MUST call `ask_confirmation` before `final_answer`. Do not skip this step.

### Phase 6 — REPORT
After `ask_confirmation` returns, call `final_answer` immediately:
- If user said YES: summarise the tests written and what passed.
- If user said NO: list the test files written, acknowledge any failures, and note the user will fix them. Do NOT retry.

---

## Rules
- One tool call per response; wait for each result
- Never edit production source files — test files only
- Do NOT call `final_answer` before calling `ask_confirmation`
- If `run_tests` exits non-zero, still call `ask_confirmation` — the user decides what to do
- Prefer relative paths (resolve from cwd above)

## Available tools

{self._tool_docs(_EXPLORE_TOOLS + _WRITE_TOOLS + _RUN_TOOLS + _LLM_TOOLS)}

## Response format

```json
{{
  "thought": "<reasoning>",
  "action": "<tool_name | ask_confirmation | final_answer>",
  "action_input": {{<parameters>}}
}}
```
"""
