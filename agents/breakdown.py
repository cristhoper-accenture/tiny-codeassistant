"""
BreakdownAgent — decomposes a task into ordered, atomic, actionable steps.

Unlike the planner (which produces architecture docs and design specs), the
breakdown agent focuses exclusively on the execution sequence: what to do,
in what order, with explicit per-step acceptance criteria.

Workflow:
  1. Understand — clarify scope and final goal
  2. Explore    — read enough context to sequence correctly
  3. Sequence   — order steps with dependencies marked
  4. Report     — final_answer as a numbered checklist; optionally save breakdown_*.md
"""

from agents.base import BaseAgent

_READ_TOOLS = [
    "find_files", "list_dir", "code_outline", "read_file", "read_lines",
    "grep_code", "git_status", "git_log",
]
_ANALYSIS_TOOLS = [
    "rag_search", "rag_list",
    "get_snippet", "list_snippets",
    "websearch",
]
_OUTPUT_TOOLS = ["write_file"]


class BreakdownAgent(BaseAgent):
    name = "breakdown"
    label = "Breakdown"
    border_color = "green"

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        if action == "write_file":
            import os
            fname = os.path.basename(action_input.get("path", ""))
            if not (fname.startswith("breakdown_") and fname.endswith(".md")):
                return (
                    f"BLOCKED: breakdown agent may only write breakdown_*.md files, not '{fname}'. "
                    "Save your checklist as breakdown_<slug>.md, or return it inline via final_answer. "
                    "Source code must be written by the coder agent."
                )
        return super().handle_special_action(action, action_input)

    def build_system_prompt(self) -> str:
        return f"""You are a task decomposition specialist. You break a task into an ordered, numbered checklist of atomic steps.

Scope: step-by-step breakdowns, execution sequences, dependency-aware task lists. Save as breakdown_<slug>.md when non-trivial.
Out of scope: architecture/risk/interface design (→ planner), writing code (→ coder), general Q&A (→ general).

## Current working directory
`{self.cwd}`

## Workflow
UNDERSTAND → EXPLORE → SEQUENCE → REPORT
- UNDERSTAND: restate the goal in one line; identify the target artifact and out-of-scope items
- EXPLORE: list_dir/find_files, code_outline/grep_code, rag_search for library-specific steps, websearch for gaps
- SEQUENCE: each step atomic, with exact file/command, a "done when" criterion, and "⚠ requires step N" for dependencies
- REPORT: output Goal / Prerequisites / Steps / Estimated effort; write_file to breakdown_<slug>.md if non-trivial, then final_answer

You are the middle stage of a planner → breakdown → coder pipeline. The task includes the planner's plan; turn it into the execution checklist the coder will follow.

## Rules
- Never write, edit, or delete source code
- One tool call per response; steps must name exact files, functions, commands
- If you are blocked by a decision only the user can make, emit `ask_clarification` instead of guessing:
```json
{{"action": "ask_clarification", "action_input": {{"question": "<what you need to know and why>"}}}}
```

## Available tools

{self._tool_docs(_READ_TOOLS + _ANALYSIS_TOOLS + _OUTPUT_TOOLS)}

## Response format

```json
{{"thought": "<reasoning>", "action": "<tool_name | ask_clarification | final_answer>", "action_input": {{<parameters>}}}}
```
"""
