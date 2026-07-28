#!/usr/bin/env python3
"""
Code Assistant — local LLM agent launcher.

The orchestrator agent is the default entry point. It automatically routes
each task to the most appropriate sub-agent (coder, general, …).

Usage:
  python agent.py                              # orchestrator REPL (default)
  python agent.py "your task"                  # orchestrator, single-shot
  python agent.py --agent coder "task"         # force a specific agent
  python agent.py --model qwen3.5:9b "task"    # override model
  python agent.py --cwd /some/project "task"   # set working directory
  python agent.py --no-stream "task"           # disable streaming output
  python agent.py --list-agents                # show available agents
  python agent.py --list-models                # show available Ollama models
"""

import os
import argparse

import llm
from agents import REGISTRY, DEFAULT_AGENT
from config import AGENT_MODELS, DEFAULT_MODEL


def _preload_models(agent_name: str, model_override: str | None) -> None:
    """Warm up Ollama models so the first real inference call is fast."""
    if model_override:
        models = [model_override]
    elif agent_name == "orchestrator":
        # Preload all unique models the orchestrator may delegate to.
        seen: dict[str, None] = {}
        for m in AGENT_MODELS.values():
            seen[m] = None
        models = list(seen)
    else:
        models = [AGENT_MODELS.get(agent_name, DEFAULT_MODEL)]

    for model in models:
        print(f"  Preloading {model}...", end=" ", flush=True)
        try:
            llm.preload(model)
            print("ready.")
        except Exception as e:
            print(f"warning: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local LLM code assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?", help="Single-shot query (omit for REPL)")
    parser.add_argument(
        "--agent", "-a",
        default=DEFAULT_AGENT,
        choices=list(REGISTRY.keys()),
        help=f"Agent to use (default: {DEFAULT_AGENT}). Choices: {', '.join(REGISTRY)}",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Override model for all agents (default: each agent uses its AGENT_MODELS config entry)",
    )
    parser.add_argument("--cwd", default=None, help="Starting working directory (default: current dir)")
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming output (wait for full response before displaying)",
    )
    parser.add_argument("--list-agents", action="store_true", help="List available agents and exit")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    args = parser.parse_args()

    if args.list_agents:
        for name, cls in REGISTRY.items():
            marker = " (default)" if name == DEFAULT_AGENT else ""
            print(f"  {name:14} — {cls.label}{marker}")
        return

    if args.list_models:
        for m in llm.list_models():
            print(m)
        return

    start_cwd = os.path.abspath(args.cwd) if args.cwd else os.getcwd()
    streaming = False if args.no_stream else None  # None → use STREAM_OUTPUT config default
    agent = REGISTRY[args.agent](model=args.model, cwd=start_cwd, streaming=streaming)

    _preload_models(args.agent, args.model)

    if args.query:
        agent.run(args.query)
    else:
        agent.repl()


if __name__ == "__main__":
    main()
