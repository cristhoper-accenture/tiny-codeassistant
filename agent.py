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
  python agent.py --list-agents                # show available agents
  python agent.py --list-models                # show available Ollama models
"""

import os
import argparse

import llm
from config import DEFAULT_MODEL
from agents import REGISTRY, DEFAULT_AGENT


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
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--cwd", default=None, help="Starting working directory (default: current dir)")
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
    agent = REGISTRY[args.agent](model=args.model, cwd=start_cwd)

    if args.query:
        agent.run(args.query)
    else:
        agent.repl()


if __name__ == "__main__":
    main()
