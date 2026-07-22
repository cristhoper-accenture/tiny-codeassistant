#!/usr/bin/env python3
"""
Code Assistant Agent — ReAct loop over a local Ollama LLM.

Usage:
  python agent.py                    # interactive REPL
  python agent.py "your question"    # single-shot
  python agent.py --model qwen3.5:9b --cwd /some/path "your question"
"""

import json
import re
import os
import sys
import argparse

import llm
from config import DEFAULT_MODEL, MAX_ITERATIONS
from tools.registry import TOOLS, execute_tool

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    _RICH = True
except ImportError:
    console = None
    _RICH = False


# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(cwd: str) -> str:
    tool_docs = "\n".join(
        f"- **{t['name']}**: {t['description']}\n"
        + "  Parameters: "
        + json.dumps(t["parameters"])
        for t in TOOLS
    )
    return f"""You are a capable code assistant running locally.

## Current working directory
`{cwd}`

All relative file paths resolve from this directory. New files are created here by default.
Use `change_dir` to navigate, or `bash` with `cd` — both persist the new directory for subsequent calls.

## Available tools

{tool_docs}

## How to use tools

When you need a tool, respond with ONLY a JSON block:

```json
{{
  "thought": "<your reasoning>",
  "action": "<tool_name>",
  "action_input": {{<parameters as JSON object>}}
}}
```

When you have a final answer:

```json
{{
  "thought": "<final reasoning>",
  "action": "final_answer",
  "action_input": {{"response": "<your complete answer>"}}
}}
```

Rules:
- Always include "thought", "action", and "action_input".
- Use tools one at a time; wait for each result before calling the next.
- Prefer relative paths — they resolve from the current working directory.
- When creating code files, use write_file with a relative path.
- Save reusable code as snippets with save_snippet.
"""


# ── JSON extraction ────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_action(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_tool_call(action: str, params: dict) -> None:
    if _RICH:
        console.print(Panel(
            f"[bold cyan]{action}[/bold cyan]\n{json.dumps(params, indent=2)}",
            title="[yellow]Tool call[/yellow]", border_style="yellow",
        ))
    else:
        print(f"\n[Tool] {action}({json.dumps(params)})")


def _print_tool_result(result: str, new_cwd: str) -> None:
    preview = result[:500] + ("…" if len(result) > 500 else "")
    if _RICH:
        console.print(Panel(
            f"{preview}\n\n[dim]cwd: {new_cwd}[/dim]",
            title="[green]Tool result[/green]", border_style="green",
        ))
    else:
        print(f"[Result] {preview}\n[cwd: {new_cwd}]")


def _print_thought(thought: str) -> None:
    if _RICH:
        console.print(f"[dim italic]  {thought}[/dim italic]")
    else:
        print(f"  Thought: {thought}")


def _print_answer(text: str) -> None:
    if _RICH:
        console.print(Panel(Markdown(text), title="[bold green]Answer[/bold green]", border_style="green"))
    else:
        print(f"\n=== Answer ===\n{text}\n")


def _cwd_prompt(cwd: str) -> str:
    home = os.path.expanduser("~")
    display = cwd.replace(home, "~") if cwd.startswith(home) else cwd
    return display


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(user_input: str, model: str = DEFAULT_MODEL, cwd: str = None) -> tuple[str, str]:
    """Run one query through the agent. Returns (answer, final_cwd)."""
    if cwd is None:
        cwd = os.getcwd()

    messages = [
        {"role": "system", "content": _build_system_prompt(cwd)},
        {"role": "user", "content": user_input},
    ]

    for _ in range(MAX_ITERATIONS):
        raw = llm.chat(messages, model=model)
        messages.append({"role": "assistant", "content": raw})

        action_obj = _extract_action(raw)

        if action_obj is None:
            _print_answer(raw)
            return raw, cwd

        thought = action_obj.get("thought", "")
        action = action_obj.get("action", "")
        action_input = action_obj.get("action_input", {})

        if thought:
            _print_thought(thought)

        if action == "final_answer":
            answer = action_input.get("response", str(action_input))
            _print_answer(answer)
            return answer, cwd

        _print_tool_call(action, action_input)
        result, cwd = execute_tool(action, action_input, cwd)
        _print_tool_result(result, cwd)

        # Update system message so the LLM always sees the current cwd
        messages[0] = {"role": "system", "content": _build_system_prompt(cwd)}
        messages.append({"role": "user", "content": f"Observation: {result}"})

    return "Max iterations reached without a final answer.", cwd


# ── REPL ───────────────────────────────────────────────────────────────────────

def repl(model: str = DEFAULT_MODEL, cwd: str = None) -> None:
    if cwd is None:
        cwd = os.getcwd()

    if _RICH:
        console.print(Panel(
            f"[bold]Code Assistant[/bold] — model: [cyan]{model}[/cyan]\n"
            f"Working dir: [yellow]{cwd}[/yellow]\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
            border_style="blue",
        ))
    else:
        print(f"Code Assistant (model: {model})\nWorking dir: {cwd}\n")

    while True:
        prompt_label = f"\n[bold blue]You[/bold blue] [dim]({_cwd_prompt(cwd)})[/dim]"
        if _RICH:
            user_input = Prompt.ask(prompt_label)
        else:
            user_input = input(f"\nYou ({_cwd_prompt(cwd)}): ").strip()

        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        _, cwd = run_agent(user_input, model=model, cwd=cwd)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM code assistant agent")
    parser.add_argument("query", nargs="?", help="Single-shot query (omit for REPL)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--cwd", default=None, help="Starting working directory (default: current dir)")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    args = parser.parse_args()

    start_cwd = os.path.abspath(args.cwd) if args.cwd else os.getcwd()

    if args.list_models:
        for m in llm.list_models():
            print(m)
        return

    if args.query:
        run_agent(args.query, model=args.model, cwd=start_cwd)
    else:
        repl(model=args.model, cwd=start_cwd)


if __name__ == "__main__":
    main()
