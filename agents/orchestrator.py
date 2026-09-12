"""
OrchestratorAgent — classifies each task, then either answers it directly,
hands it to the general agent, or runs the implementation pipeline.

Flow:
  1. Classify the request as `direct` (simple greeting or factual question the
     LLM can answer from its own knowledge — no tools needed), `general` (a
     question, doc lookup, or summary that may require tools), or `pipeline`
     (build/change/fix code).
  2. direct   → answer immediately with a single LLM call; no sub-agent.
  3. general  → hand off to the general agent, which responds.
  4. pipeline → run planner → breakdown → coder in sequence, threading each
     stage's output into the next. If a stage emits `ask_clarification`, the
     pipeline pauses and the general agent surfaces the question to the user;
     the answer is fed back and the stage re-runs.

CWD is shared and persisted across all turns.
"""

import os

import llm
from config import AGENT_MODELS, DEFAULT_MODEL
from agents.base import BaseAgent

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.markdown import Markdown

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

# Ordered stages of the implementation pipeline.
_PIPELINE = ("planner", "breakdown", "coder")

# Max clarification rounds per stage before proceeding with whatever we have.
_MAX_CLARIFY_ROUNDS = 2

_CLASSIFY_PROMPT = """You classify a developer's request into exactly one mode.

- direct: a simple greeting, casual remark, or short factual question the \
assistant can answer from its own knowledge without reading files or searching \
the web. No tools needed.
  Examples: "hi", "hello", "thanks", "what is a Python decorator?",
            "what does async/await do?", "what's the difference between list and tuple?".
- general: a question, explanation, concept, code walkthrough, web search, \
documentation lookup/indexing, or summary that may require reading files or \
searching. Nothing needs to be built or changed.
  Examples: "explain this function", "search the httpx docs",
            "summarize this file", "index the FastAPI documentation".
- pipeline: the user wants to build, implement, add, change, refactor, fix, or \
otherwise produce or modify code. This runs planner -> breakdown -> coder.
  Examples: "add rate limiting", "fix the login crash", "refactor auth.py",
            "build a CLI for the parser", "add OAuth2 login".

Respond with ONLY one word: direct OR general OR pipeline

User request: {request}
Mode:"""

_DIRECT_PROMPT = """You are a helpful coding assistant. Answer the following question \
concisely and accurately.

{request}"""

# Greetings / trivial phrases answered directly without any agent.
_DIRECT_WORDS = {
    "hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "ok", "okay",
    "cool", "great", "nice", "sure", "yes", "no", "yep", "nope",
}

# Question words / doc signals that reliably mean "general context" — skip the LLM.
_GENERAL_WORDS = {
    "what", "why", "how", "when", "where", "who", "which",
    "explain", "describe", "summarize", "summarise", "define",
    "docs", "documentation", "ingest", "rag", "readthedocs", "changelog",
}
# Verbs that reliably mean "build/change code" — skip the LLM and go straight to pipeline.
_PIPELINE_WORDS = {
    "implement", "refactor", "rewrite", "scaffold", "build", "add",
    "create", "fix", "change", "update", "migrate", "delete", "remove",
}


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    label = "Orchestrator"
    border_color = "white"

    # Fast model for classification — uses AGENT_MODELS["orchestrator"] from config.
    @property
    def _ROUTER_MODEL(self) -> str:
        return AGENT_MODELS.get("orchestrator", DEFAULT_MODEL)

    def build_system_prompt(self) -> str:
        # The orchestrator never runs its own ReAct loop, so this is unused,
        # but required by BaseAgent's abstract interface.
        return ""

    # ── Classification ───────────────────────────────────────────────────────────

    def classify(self, user_input: str) -> str:
        """Return 'direct', 'general', or 'pipeline' for user_input."""
        words = set(user_input.lower().split())

        # Single-word greetings / trivial acks answered directly.
        if len(words) <= 3 and words & _DIRECT_WORDS and not (words & _PIPELINE_WORDS):
            return "direct"

        # Fast keyword pre-check: build verbs win over question words
        # ("fix the bug and explain why" is still an implementation task).
        if words & _PIPELINE_WORDS:
            return "pipeline"
        if words & _GENERAL_WORDS:
            return "general"

        # LLM tie-breaker for genuinely ambiguous requests.
        raw = llm.chat(
            [{"role": "user", "content": _CLASSIFY_PROMPT.format(request=user_input)}],
            model=self._ROUTER_MODEL,
        ).strip().lower()
        first = raw.split()[0].strip(".:,\"'") if raw else ""
        if first == "direct":
            return "direct"
        if first == "general":
            return "general"
        return "pipeline"

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> tuple[str, str]:
        mode = self.classify(user_input)

        if mode == "direct":
            answer = llm.chat(
                [{"role": "user", "content": _DIRECT_PROMPT.format(request=user_input)}],
                model=self._ROUTER_MODEL,
            ).strip()
            self._print_answer(answer)
            return answer, self.cwd

        if mode == "general":
            self._print_routing("general", user_input)
            answer = self._run_agent("general", user_input)
            return answer, self.cwd

        return self._run_pipeline(user_input)

    def _run_pipeline(self, user_input: str) -> tuple[str, str]:
        self._print_pipeline_start()
        outputs: dict[str, str] = {}
        clarifications: list[str] = []

        for stage in _PIPELINE:
            self._print_routing(stage, user_input)
            answer = ""
            for attempt in range(_MAX_CLARIFY_ROUNDS + 1):
                task = self._compose_task(user_input, outputs, clarifications)
                sub = self._make_agent(stage)
                answer, self.cwd = sub.run(task)

                question = getattr(sub, "_clarification", None)
                if question and attempt < _MAX_CLARIFY_ROUNDS:
                    user_answer = self._clarify(stage, question)
                    clarifications.append(f"Q ({stage}): {question}\nA: {user_answer}")
                    continue
                break
            outputs[stage] = answer

        # The coder's report is the pipeline's final answer.
        return outputs.get("coder", ""), self.cwd

    def _compose_task(
        self, user_input: str, outputs: dict[str, str], clarifications: list[str]
    ) -> str:
        """Thread the original request plus upstream stage outputs into the next stage."""
        parts = [f"User request:\n{user_input}"]
        if outputs.get("planner"):
            parts.append(f"Implementation plan (from the planner agent):\n{outputs['planner']}")
        if outputs.get("breakdown"):
            parts.append(f"Step-by-step breakdown (from the breakdown agent):\n{outputs['breakdown']}")
        if clarifications:
            parts.append("User clarifications (answered during this pipeline):\n" + "\n\n".join(clarifications))
        return "\n\n---\n\n".join(parts)

    def _make_agent(self, name: str) -> BaseAgent:
        from agents import REGISTRY
        return REGISTRY[name](model=self._model_override, cwd=self.cwd, streaming=self._streaming)

    def _run_agent(self, name: str, task: str) -> str:
        sub = self._make_agent(name)
        answer, self.cwd = sub.run(task)
        return answer

    def _clarify(self, stage: str, question: str) -> str:
        """A pipeline stage is blocked — the general agent surfaces the question to the user."""
        self._print_routing("general", f"clarifying for {stage}")
        prompt_text = (
            f"The {stage} agent needs input to continue:\n\n{question}\n\n"
            "Please provide the answer."
        )
        if _RICH:
            _console.print(Panel(
                Markdown(f"**The {stage} agent needs input to continue:**\n\n{question}"),
                title="[bold blue]Assistant[/bold blue]",
                border_style="blue",
            ))
            return Prompt.ask("[bold blue]Your answer[/bold blue]")
        print(f"\n{prompt_text}")
        return input("Your answer: ").strip()

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

    def _print_answer(self, text: str) -> None:
        if _RICH:
            _console.print(Panel(Markdown(text), title="[bold green]Assistant[/bold green]", border_style="green"))
        else:
            print(f"\nAssistant: {text}")

    def _print_pipeline_start(self) -> None:
        arrow = " → ".join(_PIPELINE)
        if _RICH:
            _console.print(f"\n[dim]Running implementation pipeline:[/dim] [bold]{arrow}[/bold]")
        else:
            print(f"\n[Orchestrator] pipeline: {arrow}")

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
