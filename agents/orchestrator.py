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
        "ONLY for producing high-level architecture documents, design specs, risk analyses, and "
        "interface definitions — no code is written. Use when the user wants to design the "
        "architecture or think through trade-offs before implementation. "
        "Examples: 'design the OAuth2 architecture', 'create a technical spec for the new API', "
        "'what is the best data model for this feature'."
    ),
    "tester": (
        "ONLY for writing test files and running the test suite. Asks the user to confirm results "
        "before finishing — if user says no, assumes the user will fix it. "
        "Examples: 'write tests for auth.py', 'add unit tests for the parser', "
        "'generate pytest tests for utils', 'create a test suite for the API endpoints'."
    ),
    "breakdown": (
        "ONLY for decomposing a task into an ordered, numbered, atomic step-by-step checklist. "
        "Use when the user knows WHAT to build and wants HOW to execute it step by step. "
        "No architecture design, no code written. "
        "Examples: 'break down the migration into steps', 'give me a checklist to add JWT auth', "
        "'list the steps to deploy this to AWS', 'create a task sequence for the refactor'."
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

1. lint      — task is ONLY about running a linter and fixing style/formatting (no logic changes)
2. tester    — task is about writing test files or running the test suite
3. coder     — task requires writing or editing source code files right now
4. planner   — task requires high-level architecture design, a spec, or trade-off analysis (no code)
5. breakdown — task requires an ordered step-by-step checklist for how to execute something (no code)
6. general   — everything else: questions, explanations, web search, doc indexing, snippets

Agent definitions:
{agent_lines}

Respond with ONLY the agent name (lint / tester / coder / planner / breakdown / general), one word, no punctuation.

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
        text  = user_input.lower()
        words = set(text.split())

        # Fast pre-check: strong keyword signals bypass the LLM entirely.
        # Ordered from most-specific to least-specific to avoid false matches.
        _pre = self._keyword_route(words, text)
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
        return self._keyword_route(words, text) or "general"

    @staticmethod
    def _keyword_route(words: set[str], text: str = "") -> str | None:
        """Return an agent name if strong, unambiguous keywords match, else None.

        Priority: lint > tester > breakdown > coder > planner > general.
        `text` is the lowercased original input — used for reliable multi-word phrase matching.
        """
        # Lint — linter tool names are always unambiguous
        if words & {"lint", "linter", "linting", "flake8", "ruff", "pylint",
                    "eslint", "mypy", "bandit", "pycodestyle", "pyflakes"}:
            return "lint"

        # Tester — checked before coder so "write tests" beats the weak "write" coder signal
        _test_tool_words = {"pytest", "unittest", "jest", "mocha", "vitest"}
        _test_phrases = (
            "write tests", "add tests", "create tests", "generate tests",
            "unit tests", "integration tests", "test suite", "test file",
            "write test", "add test", "create test",
        )
        if words & _test_tool_words or any(p in text for p in _test_phrases):
            return "tester"

        # Breakdown — checked before coder so "sequence for the auth refactor" beats "refactor"
        _breakdown_words = {
            "breakdown", "checklist", "steps", "sequence",
            "roadmap", "milestones", "decompose", "decomposition",
            "tasklist", "sprint", "iterations",
        }
        _breakdown_phrases = (
            "break down", "step by step", "step-by-step", "task sequence", "task list",
        )
        if words & _breakdown_words or any(p in text for p in _breakdown_phrases):
            return "breakdown"

        # Coder — action verbs that always mean "write code now"
        if words & {"implement", "refactor", "rewrite", "scaffold"}:
            return "coder"

        # Planner — high-level design/architecture intent (checked before weak "create" signal)
        if words & {"plan", "planify", "spec", "specification", "blueprint",
                    "architecture", "design", "tradeoff", "trade-off"}:
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
        colors = {"coder": "magenta", "general": "blue", "lint": "yellow", "planner": "cyan", "breakdown": "green", "tester": "bright_blue"}
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
