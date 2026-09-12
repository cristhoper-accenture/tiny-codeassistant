"""
PlannerAgent — analyzes a task and produces a detailed implementation plan.

The agent reads the codebase and thinks through what needs to be done, but
writes NO code. Its output is a structured plan document ready to hand off
to the coder agent (or a human developer).

Workflow:
  1. Understand — parse the task; define success criteria and scope
  2. Explore    — map the relevant parts of the codebase
  3. Analyze    — identify gaps, dependencies, risks, and constraints
  4. Draft      — compose the implementation plan
  5. Report     — final_answer with the full plan; optionally save to a file
"""

try:
    from rich.console import Console
    from rich.panel import Panel

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

from agents.base import BaseAgent

_READ_TOOLS = [
    "find_files", "list_dir", "code_outline", "read_file", "read_lines",
    "grep_code", "git_status", "git_log", "git_diff", "git_blame",
]
_ANALYSIS_TOOLS = [
    "explain_code", "review_code", "summarize",
    "rag_search", "rag_list", "rag_collections",
    "get_snippet", "list_snippets",
    "websearch",
]
_OUTPUT_TOOLS = ["write_file"]


class PlannerAgent(BaseAgent):
    name = "planner"
    label = "Planner"
    border_color = "cyan"

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        # Guard: only allow writing plan documents (plan_*.md).
        # Any attempt to write source code is rejected and redirected to the coder agent.
        if action == "write_file":
            path = action_input.get("path", "")
            import os
            fname = os.path.basename(path)
            if not (fname.startswith("plan_") and fname.endswith(".md")):
                return (
                    f"BLOCKED: planner may only write plan_*.md files, not '{fname}'. "
                    "Write your plan as plan_<slug>.md, or use final_answer to return it inline. "
                    "Source code must be written by the coder agent."
                )
        return super().handle_special_action(action, action_input)

    def build_system_prompt(self) -> str:
        return f"""You are a senior software architect. You produce implementation plans — you do NOT write code.

Scope: architecture docs, design specs, risk analyses, interface definitions. Read the codebase to inform the design; save the result as plan_<slug>.md.
Out of scope: writing/editing code, fixing bugs, lint, general Q&A, step-by-step checklists (→ breakdown). If the task is pure Q&A or a direct code request, say so in final_answer.

## Current working directory
`{self.cwd}`

## Workflow
UNDERSTAND → EXPLORE → ANALYZE → DRAFT → REPORT
- UNDERSTAND: restate the task and acceptance criteria in one line each
- EXPLORE: read the codebase; check rag_search(collection="docs") before websearch
- ANALYZE: files to create/modify/delete, interfaces, dependencies, risks, complexity
- DRAFT: plan with Task / Scope / Current state / Steps / Files table / Interfaces / Dependencies / Risks / Complexity
- REPORT: write_file to plan_<slug>.md if non-trivial, then final_answer with the full plan

You are the first stage of a planner → breakdown → coder pipeline; your plan feeds the next stages.

## Rules
- Read-only except plan_*.md — never write source code
- One tool call per response; prefer relative paths
- Verify file names and APIs before referencing them; flag every assumption
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
