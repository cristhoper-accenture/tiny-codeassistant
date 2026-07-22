"""General-purpose assistant agent."""

from agents.base import BaseAgent


class GeneralAgent(BaseAgent):
    name = "general"
    label = "Assistant"
    border_color = "blue"

    def build_system_prompt(self) -> str:
        return f"""You are a capable code assistant running locally.

## Current working directory
`{self.cwd}`

All relative file paths resolve from this directory. Use `change_dir` or `bash cd` to navigate — both persist.

## Available tools

{self._tool_docs()}

## How to respond

When you need a tool:
```json
{{
  "thought": "<your reasoning>",
  "action": "<tool_name>",
  "action_input": {{<parameters>}}
}}
```

When you have a final answer:
```json
{{
  "thought": "<reasoning>",
  "action": "final_answer",
  "action_input": {{"response": "<answer>"}}
}}
```

## File saving

When a task produces content that should be persisted — generated code, a configuration
file, a report, documentation — use `write_file` to save it before calling `final_answer`.
Infer a sensible path from the task context; ask in `final_answer` if truly ambiguous.

## Agent delegation

Use `delegate_to_agent` to hand off a focused sub-task to a specialist:
- `coder` — writing, editing, or verifying code files
- `general` — questions, explanations, web search, summaries

Only delegate when the sub-task is clearly within that agent's specialty.

## Rules
- One tool at a time; wait for each result.
- Prefer relative paths (resolve from cwd).
- Save reusable code patterns with `save_snippet`.
"""
