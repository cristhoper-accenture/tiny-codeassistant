"""
BaseAgent — shared ReAct loop, display helpers, and JSON extraction.
Subclasses override build_system_prompt() and optionally handle_special_action().
"""

import json
import re
import os

import llm
from config import DEFAULT_MODEL, AGENT_MODELS, MAX_ITERATIONS
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

    def __init__(self, model: str = None, cwd: str = None, _depth: int = 0):
        # None → use the per-agent model from AGENT_MODELS; explicit value → override for all agents.
        self._model_override = model
        self.model = model if model is not None else AGENT_MODELS.get(self.name, DEFAULT_MODEL)
        self.cwd = cwd or os.getcwd()
        self._depth = _depth

    # ── Subclasses override these ──────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def handle_special_action(self, action: str, action_input: dict) -> str | None:
        """
        Handle actions that are not standard tool calls (e.g. 'plan', 'delegate_to_agent').
        Subclasses should handle their own actions and fall through to super() for the rest.
        Returns an observation string, or None to fall through to the tool registry.
        """
        if action != "delegate_to_agent":
            return None

        from agents import REGISTRY
        agent_name = action_input.get("agent", "")
        task = action_input.get("task", "")

        if not agent_name or not task:
            return "ERROR: delegate_to_agent requires 'agent' and 'task' parameters."

        valid = [n for n in REGISTRY if n != "orchestrator"]
        if agent_name not in valid:
            return f"ERROR: Unknown agent '{agent_name}'. Available: {', '.join(valid)}"

        if self._depth >= 1:
            return "ERROR: Delegation depth limit reached. Handle this task directly."

        self._print_delegation(agent_name, task)
        sub = REGISTRY[agent_name](model=self._model_override, cwd=self.cwd, _depth=self._depth + 1)
        answer, new_cwd = sub.run(task)
        self.cwd = new_cwd
        return f"[Delegated to {agent_name} agent]\n{answer}"

    # ── ReAct loop ─────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> tuple[str, str]:
        """Execute one query. Returns (answer, final_cwd)."""
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        _format_misses = 0
        for _ in range(MAX_ITERATIONS):
            raw = llm.chat(messages, model=self.model)
            messages.append({"role": "assistant", "content": raw})

            action_obj = self._extract_action(raw)

            if action_obj is None:
                # If the response looks like prose or code (not a final answer), nudge the
                # model back to the required JSON format instead of silently bailing out.
                _format_misses += 1
                if _format_misses <= 2 and any(tok in raw for tok in ("```", "def ", "import ", "class ")):
                    self._print_format_warning(raw)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response was not in the required format. "
                            "You MUST reply with a single JSON block like this:\n"
                            "```json\n"
                            "{\"thought\": \"<reasoning>\", "
                            "\"action\": \"<tool_name or final_answer>\", "
                            "\"action_input\": {<parameters>}}\n"
                            "```\n"
                            "Do not write prose or raw code outside that block."
                        ),
                    })
                    continue
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
        # Sub-agents (depth >= 1) cannot delegate further — hide the tool to avoid confusion.
        if self._depth >= 1:
            tools = [t for t in tools if t["name"] != "delegate_to_agent"]
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
            from rich.markup import escape
            _console.print(Panel(
                f"{escape(preview)}\n[dim]cwd: {self._cwd_display()}[/dim]",
                title="[green]Result[/green]", border_style="green",
            ))
        else:
            print(f"[Result] {preview}\n[cwd: {self.cwd}]")

    def _print_format_warning(self, raw: str) -> None:
        preview = raw[:120] + ("…" if len(raw) > 120 else "")
        if _RICH:
            _console.print(f"[bold yellow]  ⚠ JSON format missing — nudging model.[/bold yellow] [dim]{preview}[/dim]")
        else:
            print(f"  [format warning] Expected JSON, got prose. Retrying.\n  {preview}")

    def _print_delegation(self, agent_name: str, task: str) -> None:
        preview = task[:120] + ("…" if len(task) > 120 else "")
        if _RICH:
            _console.print(
                f"\n[dim]  Delegating to[/dim] [bold cyan]{agent_name}[/bold cyan]"
                f"[dim]: {preview}[/dim]"
            )
        else:
            print(f"\n  [Delegate → {agent_name}] {preview}")

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
