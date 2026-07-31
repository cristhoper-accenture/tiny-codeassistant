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

# Each description is exclusive — no overlap with the others.
# Phrased as "use ONLY when" so the routing LLM can apply the decision tree.
_AGENT_DESCRIPTIONS = {
    "lint": (
        "ONLY for fixing lint/style/formatting errors produced by a linter (ruff, flake8, "
        "pylint, mypy, eslint). Must NOT change business logic. "
        "Examples: 'fix ruff errors', 'clean up pylint warnings', 'enforce PEP8 in src/'."
    ),
    "coder": (
        "ONLY for writing, editing, refactoring, or deleting source code files right now. "
        "Use when actual file changes are needed: add a feature, fix a logic/runtime bug, "
        "generate tests, update an existing function. "
        "Examples: 'add rate limiting', 'fix the login crash', 'refactor auth.py', 'write tests for utils'."
    ),
    "planner": (
        "ONLY for producing a written plan, spec, or design document — no code is written. "
        "Use when the user wants to think through an implementation before starting. "
        "Examples: 'plan the OAuth2 integration', 'create a design spec for the new API', "
        "'break down the migration into steps'."
    ),
    "general": (
        "For everything that does NOT require modifying source code: answer questions, "
        "explain concepts or code, search the web, summarize text, index or refresh "
        "library documentation in the RAG knowledge base, manage snippets. "
        "Examples: 'what is a decorator?', 'explain this function', 'search httpx docs', "
        "'update the FastAPI docs in the knowledge base'."
    ),
}

_ROUTE_PROMPT = """You are a task router. Pick one agent using this priority order — stop at the first match:

1. lint    — task is ONLY about running a linter and fixing style/formatting (no logic changes)
2. coder   — task requires writing or editing source code files right now
3. planner — task requires a written plan or design document (no code written yet)
4. general — everything else: questions, explanations, web search, doc indexing, snippets

Agent definitions:
{agent_lines}

Respond with ONLY the agent name (lint / coder / planner / general), one word, no punctuation.

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
        words = set(user_input.lower().split())

        # Fast pre-check: strong keyword signals bypass the LLM entirely.
        # Ordered from most-specific to least-specific to avoid false matches.
        _pre = self._keyword_route(words)
        if _pre:
            return _pre

        # LLM routing for genuinely ambiguous requests.
        prompt = _build_route_prompt(user_input)
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            model=self._ROUTER_MODEL,
        ).strip().lower()

        first_word = raw.split()[0].strip(".:,\"'") if raw else ""
        from agents import REGISTRY
        if first_word in REGISTRY and first_word != "orchestrator":
            return first_word

        # Final heuristic fallback if the LLM returned something unexpected.
        return self._keyword_route(words) or "general"

    @staticmethod
    def _keyword_route(words: set[str]) -> str | None:
        """Return an agent name if strong, unambiguous keywords match, else None.

        Priority: lint > coder > planner > general.
        Only signals that are unambiguously tied to ONE agent are listed here.
        """
        # Lint — linter tool names are always unambiguous
        if words & {"lint", "linter", "linting", "flake8", "ruff", "pylint",
                    "eslint", "mypy", "bandit", "pycodestyle", "pyflakes"}:
            return "lint"

        # Coder — action verbs that always mean "write code now"
        if words & {"implement", "refactor", "rewrite", "scaffold"}:
            return "coder"

        # Planner — explicit planning intent (checked before weak "create" signal)
        if words & {"plan", "planify", "spec", "specification", "blueprint"}:
            return "planner"

        # "create/build/write/add" are weak coder signals — only use them when there
        # are no planner or doc-management keywords that would override the intent.
        _plan_words = {"plan", "spec", "specification", "blueprint", "document", "doc", "docs"}
        if words & {"create", "build", "write", "add", "generate"} and not words & _plan_words:
            return "coder"

        # General — doc-indexing signals and pure question words
        if words & {"docs", "documentation", "ingest", "rag", "readthedocs", "changelog"}:
            return "general"
        if words & {"what", "why", "when", "where", "explain", "describe", "summarize", "summarise"}:
            return "general"

        return None

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> tuple[str, str]:
        agent_name = self.route(user_input)
        self._print_routing(agent_name, user_input)

        from agents import REGISTRY
        # Propagate model override and streaming preference to sub-agents.
        sub = REGISTRY[agent_name](model=self._model_override, cwd=self.cwd, streaming=self._streaming)
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
        colors = {"coder": "magenta", "general": "blue", "lint": "yellow", "planner": "cyan"}
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
