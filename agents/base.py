"""
BaseAgent — shared ReAct loop, display helpers, and JSON extraction.
Subclasses override build_system_prompt() and optionally handle_special_action().
"""

import json
import re
import os

import llm
from config import DEFAULT_MODEL, MAX_ITERATIONS
from tools.registry import TOOLS, execute_tool

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text

    _console = Console()
    _RICH = True
except ImportError:
    _console = None
    _RICH = False

_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class BaseAgent:
    name: str = "base"
    label: str = "Assistant"
    border_color: str = "blue"

    def __init__(self, model: str = DEFAULT_MODEL, cwd: str = None):
        self.model = model
        self.cwd = cwd or os.getcwd()

    # ── Subclasses override these ──────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        """
        Handle actions that are not tool calls (e.g. 'plan').
        Return an observation string to feed back, or None to treat as unknown tool.
        """
        return None

    # ── ReAct loop ─────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> tuple[str, str]:
        """Execute one query. Returns (answer, final_cwd)."""
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for _ in range(MAX_ITERATIONS):
            raw = llm.chat(messages, model=self.model)
            messages.append({"role": "assistant", "content": raw})

            action_obj = self._extract_action(raw)

            if action_obj is None:
                self._print_answer(raw)
                return raw, self.cwd

            thought = action_obj.get("thought", "")
            action = action_obj.get("action", "")
            action_input = action_obj.get("action_input", {})

            if thought:
                self._print_thought(thought)

            if action == "final_answer":
                answer = action_input.get("response", str(action_input))
                self._print_answer(answer)
                return answer, self.cwd

            # Let subclass handle special actions first
            special_result = self.handle_special_action(action, action_input)
            if special_result is not None:
                messages[0] = {"role": "system", "content": self.build_system_prompt()}
                messages.append({"role": "user", "content": f"Observation: {special_result}"})
                continue

            # Standard tool dispatch
            self._print_tool_call(action, action_input)
            result, self.cwd = execute_tool(action, action_input, self.cwd)
            self._print_tool_result(result)

            messages[0] = {"role": "system", "content": self.build_system_prompt()}
            messages.append({"role": "user", "content": f"Observation: {result}"})

        return "Max iterations reached without a final answer.", self.cwd

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _tool_docs(self, names: list[str] = None) -> str:
        tools = TOOLS if names is None else [t for t in TOOLS if t["name"] in names]
        return "\n".join(
            f"- **{t['name']}**: {t['description']}\n  Parameters: {json.dumps(t['parameters'])}"
            for t in tools
        )

    @staticmethod
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

    def _cwd_display(self) -> str:
        home = os.path.expanduser("~")
        return self.cwd.replace(home, "~") if self.cwd.startswith(home) else self.cwd

    # ── Display ────────────────────────────────────────────────────────────────

    def _print_thought(self, thought: str) -> None:
        if _RICH:
            _console.print(f"[dim italic]  {thought}[/dim italic]")
        else:
            print(f"  Thought: {thought}")

    def _print_tool_call(self, action: str, params: dict) -> None:
        if _RICH:
            _console.print(Panel(
                f"[bold cyan]{action}[/bold cyan]\n{json.dumps(params, indent=2)}",
                title="[yellow]Tool[/yellow]", border_style="yellow",
            ))
        else:
            print(f"\n[Tool] {action}({json.dumps(params)})")

    def _print_tool_result(self, result: str) -> None:
        preview = result[:500] + ("…" if len(result) > 500 else "")
        if _RICH:
            _console.print(Panel(
                f"{preview}\n[dim]cwd: {self._cwd_display()}[/dim]",
                title="[green]Result[/green]", border_style="green",
            ))
        else:
            print(f"[Result] {preview}\n[cwd: {self.cwd}]")

    def _print_answer(self, text: str) -> None:
        if _RICH:
            _console.print(Panel(
                Markdown(text),
                title=f"[bold green]{self.label}[/bold green]",
                border_style="green",
            ))
        else:
            print(f"\n=== {self.label} ===\n{text}\n")

    def _print_banner(self) -> None:
        if _RICH:
            _console.print(Panel(
                f"[bold]{self.label}[/bold] — model: [cyan]{self.model}[/cyan]\n"
                f"cwd: [yellow]{self._cwd_display()}[/yellow]\n"
                "Type [bold]exit[/bold] to quit.",
                border_style=self.border_color,
            ))
        else:
            print(f"{self.label} (model: {self.model})\ncwd: {self.cwd}\n")

    # ── REPL ───────────────────────────────────────────────────────────────────

    def repl(self) -> None:
        self._print_banner()
        while True:
            if _RICH:
                user_input = Prompt.ask(
                    f"\n[bold {self.border_color}]You[/bold {self.border_color}]"
                    f" [dim]({self._cwd_display()})[/dim]"
                )
            else:
                user_input = input(f"\nYou ({self._cwd_display()}): ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                break
            if not user_input:
                continue

            self.run(user_input)
