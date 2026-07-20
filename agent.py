#!/usr/bin/env python3
"""
Code Assistant Agent — ReAct loop over a local Ollama LLM.

Usage:
  python agent.py                    # interactive REPL
  python agent.py "your question"    # single-shot
  python agent.py --model qwen3.5:9b "your question"
"""

import json
import re
import sys
import argparse

import llm
from config import DEFAULT_MODEL, MAX_ITERATIONS
from tools import TOOLS, execute_tool

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import print as rprint

    console = Console()
    _RICH = True
except ImportError:
    console = None
    _RICH = False


# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    tool_docs = "\n".join(
        f"- **{t['name']}**: {t['description']}\n"
        + "  Parameters: "
        + json.dumps(t["parameters"])
        for t in TOOLS
    )
    return f"""You are a capable code assistant running locally. You have access to these tools:

{tool_docs}

## How to use tools

When you need to use a tool, respond with a JSON block (and nothing else outside it):

```json
{{
  "thought": "<your reasoning>",
  "action": "<tool_name>",
  "action_input": {{<parameters as JSON object>}}
}}
```

When you have a final answer and don't need more tools, respond with:

```json
{{
  "thought": "<final reasoning>",
  "action": "final_answer",
  "action_input": {{"response": "<your complete answer>"}}
}}
```

Rules:
- Always include "thought", "action", and "action_input".
- Use tools one at a time; wait for the result before calling the next.
- If a tool fails, try an alternative approach.
- Be concise and accurate in your final answer.
- When writing or editing code, save it as a snippet if it may be reused.
"""


# ── JSON extraction ────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_action(text: str) -> dict | None:
    """Extract the first JSON action block from the LLM response."""
    # Try fenced code block first
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: find first { ... } spanning the whole response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_tool_call(action: str, params: dict) -> None:
    if _RICH:
        console.print(
            Panel(
                f"[bold cyan]{action}[/bold cyan]\n{json.dumps(params, indent=2)}",
                title="[yellow]Tool call[/yellow]",
                border_style="yellow",
            )
        )
    else:
        print(f"\n[Tool] {action}({json.dumps(params)})")


def _print_tool_result(result: str) -> None:
    preview = result[:500] + ("…" if len(result) > 500 else "")
    if _RICH:
        console.print(
            Panel(preview, title="[green]Tool result[/green]", border_style="green")
        )
    else:
        print(f"[Result] {preview}\n")


def _print_thought(thought: str) -> None:
    if _RICH:
        console.print(f"[dim italic]Thought: {thought}[/dim italic]")
    else:
        print(f"  Thought: {thought}")


def _print_answer(text: str) -> None:
    if _RICH:
        console.print(Panel(Markdown(text), title="[bold green]Answer[/bold green]", border_style="green"))
    else:
        print(f"\n=== Answer ===\n{text}\n")


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(user_input: str, model: str = DEFAULT_MODEL) -> str:
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_input},
    ]

    for iteration in range(MAX_ITERATIONS):
        raw = llm.chat(messages, model=model)
        messages.append({"role": "assistant", "content": raw})

        action_obj = _extract_action(raw)

        if action_obj is None:
            # LLM responded in plain text — treat as final answer
            _print_answer(raw)
            return raw

        thought = action_obj.get("thought", "")
        action = action_obj.get("action", "")
        action_input = action_obj.get("action_input", {})

        if thought:
            _print_thought(thought)

        if action == "final_answer":
            answer = action_input.get("response", str(action_input))
            _print_answer(answer)
            return answer

        # Execute tool
        _print_tool_call(action, action_input)
        result = execute_tool(action, action_input)
        _print_tool_result(result)

        # Feed result back as a user message (observation)
        messages.append({"role": "user", "content": f"Observation: {result}"})

    return "Max iterations reached without a final answer."


# ── REPL ───────────────────────────────────────────────────────────────────────

def repl(model: str = DEFAULT_MODEL) -> None:
    if _RICH:
        console.print(
            Panel(
                f"[bold]Code Assistant[/bold] — model: [cyan]{model}[/cyan]\n"
                "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
                border_style="blue",
            )
        )
    else:
        print(f"Code Assistant (model: {model}) — type 'exit' to quit\n")

    while True:
        if _RICH:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
        else:
            user_input = input("\nYou: ").strip()

        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        run_agent(user_input, model=model)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM code assistant agent")
    parser.add_argument("query", nargs="?", help="Single-shot query (omit for REPL)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    args = parser.parse_args()

    if args.list_models:
        for m in llm.list_models():
            print(m)
        return

    if args.query:
        run_agent(args.query, model=args.model)
    else:
        repl(model=args.model)


if __name__ == "__main__":
    main()
