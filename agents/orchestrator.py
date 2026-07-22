"""
OrchestratorAgent — central router that classifies each task and delegates
to the most appropriate sub-agent.

Routing is done via a fast, single LLM call using a tiny classification
prompt (no tools, minimal tokens). The selected sub-agent then runs the
full ReAct loop. CWD is shared and persisted across all turns.
"""

import os

import llm
from config import AGENT_MODELS, DEFAULT_MODEL
from agents.base import BaseAgent

try:
    from rich.console import Console
    from rich.text import Text
    from rich.rule import Rule

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

# Descriptions shown to the routing LLM
_AGENT_DESCRIPTIONS = {
    "coder": (
        "write, create, edit, refactor, or delete code; "
        "implement a feature or function; fix a bug; "
        "generate tests; run or lint code; "
        "make changes to files in a project"
    ),
    "general": (
        "answer a question; explain a concept or piece of code; "
        "search the web; summarize text; "
        "manage snippets or RAG documents; "
        "any task that does NOT require writing or modifying code files"
    ),
}

_ROUTE_PROMPT = """You are a task router. Given a user request, choose the most appropriate agent.

Agents:
{agent_lines}

Rules:
- If the task involves writing, creating, editing, or modifying code/files → coder
- If the task is a question, explanation, search, or general assistance → general
- When in doubt, prefer coder for anything that sounds like software development work

Respond with ONLY the agent name, one word, no punctuation.

User request: {request}
Agent:"""


def _build_route_prompt(request: str) -> str:
    lines = "\n".join(f'- "{name}": {desc}' for name, desc in _AGENT_DESCRIPTIONS.items())
    return _ROUTE_PROMPT.format(agent_lines=lines, request=request)


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    label = "Orchestrator"
    border_color = "white"

    # Fast model for routing — uses AGENT_MODELS["orchestrator"] from config.
    @property
    def _ROUTER_MODEL(self) -> str:
        return AGENT_MODELS.get("orchestrator", DEFAULT_MODEL)

    def build_system_prompt(self) -> str:
        # The orchestrator never runs its own ReAct loop, so this is unused,
        # but required by BaseAgent's abstract interface.
        return ""

    # ── Routing ────────────────────────────────────────────────────────────────

    def route(self, user_input: str) -> str:
        """Return the name of the sub-agent best suited for user_input."""
        prompt = _build_route_prompt(user_input)
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            model=self._ROUTER_MODEL,
        ).strip().lower()

        # Normalise — take the first word and match against known agents
        first_word = raw.split()[0].strip(".:,\"'") if raw else ""
        from agents import REGISTRY
        if first_word in REGISTRY and first_word != "orchestrator":
            return first_word

        # Fallback heuristic if LLM returned something unexpected
        coding_keywords = {
            "write", "create", "implement", "fix", "edit", "refactor",
            "generate", "build", "add", "delete", "modify", "code",
            "function", "class", "test", "lint", "run", "debug",
        }
        words = set(user_input.lower().split())
        if words & coding_keywords:
            return "coder"
        return "general"

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> tuple[str, str]:
        agent_name = self.route(user_input)
        self._print_routing(agent_name, user_input)

        from agents import REGISTRY
        # Propagate the raw override (None = each sub-agent uses its own configured model).
        sub = REGISTRY[agent_name](model=self._model_override, cwd=self.cwd)
        answer, new_cwd = sub.run(user_input)
        self.cwd = new_cwd  # persist cwd for the next turn
        return answer, new_cwd

    # ── REPL (overrides BaseAgent to show orchestrator banner) ─────────────────

    def repl(self) -> None:
        self._print_banner()
        while True:
            if _RICH:
                from rich.prompt import Prompt
                user_input = Prompt.ask(
                    f"\n[bold white]You[/bold white] [dim]({self._cwd_display()})[/dim]"
                )
            else:
                user_input = input(f"\nYou ({self._cwd_display()}): ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                break
            if not user_input:
                continue

            self.run(user_input)

    # ── Display ────────────────────────────────────────────────────────────────

    def _print_routing(self, agent_name: str, user_input: str) -> None:
        colors = {"coder": "magenta", "general": "blue"}
        color = colors.get(agent_name, "white")

        if _RICH:
            _console.print(
                f"\n[dim]Routing to[/dim] [bold {color}]{agent_name}[/bold {color}]"
                f"[dim] agent...[/dim]"
            )
        else:
            print(f"\n[Orchestrator] → {agent_name} agent")

    def _print_banner(self) -> None:
        from agents import REGISTRY
        agent_list = "  |  ".join(
            f"[bold]{n}[/bold]" if _RICH else n
            for n in REGISTRY
            if n != "orchestrator"
        )
        if _RICH:
            from rich.panel import Panel
            _console.print(Panel(
                f"[bold]Code Assistant[/bold] — model: [cyan]{self.model}[/cyan]\n"
                f"cwd: [yellow]{self._cwd_display()}[/yellow]\n"
                f"Agents: {agent_list}\n"
                "Type [bold]exit[/bold] to quit.",
                border_style="white",
                title="[bold white]Orchestrator[/bold white]",
            ))
        else:
            print(f"Code Assistant (orchestrator, model: {self.model})\ncwd: {self.cwd}\n")
