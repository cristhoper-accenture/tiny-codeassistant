# Code Assistant

A local, fully offline AI code assistant powered by [Ollama](https://ollama.com). It runs a multi-agent ReAct loop — the orchestrator classifies each task and routes it to a specialist agent (coder or general), which then reasons step-by-step and calls tools until the task is complete.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Agent Architecture](#agent-architecture)
6. [BaseAgent — the parent class](#baseagent--the-parent-class)
7. [Tools reference](#tools-reference)
8. [Extending the codebase](#extending-the-codebase)

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally on `http://localhost:11434`
- The models you intend to use must be pulled beforehand:

```bash
ollama pull qwen2.5-coder:3b    # default coder / general model
ollama pull qwen3.5:2b          # default orchestrator / router model
ollama pull qwen3-embedding:4b  # required for RAG features
```

---

## Installation

```bash
git clone <repo>
cd codeassistant

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Interactive REPL — orchestrator picks the right agent automatically
python agent.py

# Single-shot query
python agent.py "explain what this repo does"

# Force a specific agent
python agent.py --agent coder "add type hints to utils.py"
python agent.py --agent general "what is a REST API?"

# Override the model for all agents (one-off)
python agent.py --model qwen2.5-coder:7b "refactor main.py"

# Set the working directory the agent starts in
python agent.py --cwd /path/to/myproject "fix the auth bug"

# Convenience wrapper (uses the venv automatically)
./run.sh "your task"

# List available agents
python agent.py --list-agents

# List Ollama models currently pulled
python agent.py --list-models
```

### Environment variable overrides

All tunables can be set without touching code:

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `qwen2.5-coder:3b` | Global fallback model |
| `ORCHESTRATOR_MODEL` | `qwen3.5:2b` | Model used for routing |
| `CODER_MODEL` | `qwen2.5-coder:3b` | Model used by the coder agent |
| `GENERAL_MODEL` | `qwen2.5-coder:3b` | Model used by the general agent |
| `AGENT_MAX_ITER` | `15` | Max ReAct iterations per query |
| `LLM_TIMEOUT` | `300` | Seconds before an Ollama call times out |
| `BASH_TIMEOUT` | `30` | Seconds before a shell command is killed |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `EMBED_MODEL` | `qwen3-embedding:4b` | Embedding model for RAG |
| `RAG_CHUNK_SIZE` | `600` | Token size for RAG document chunks |
| `RAG_CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |

Example — run the coder with a larger, slower model without editing config:

```bash
CODER_MODEL=qwen2.5-coder:7b LLM_TIMEOUT=600 python agent.py --agent coder "rewrite auth.py"
```

---

## Configuration

`config.py` is the single source of truth for all defaults. Every value reads from an environment variable with a fallback:

```python
# config.py (abbreviated)
AGENT_MODELS = {
    "orchestrator": os.getenv("ORCHESTRATOR_MODEL", "qwen3.5:2b"),
    "coder":        os.getenv("CODER_MODEL",        "qwen2.5-coder:3b"),
    "general":      os.getenv("GENERAL_MODEL",      DEFAULT_MODEL),
}
```

To permanently change a model assignment, edit the fallback string in `AGENT_MODELS`. To change it for a single run, set the env var.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (agent.py)                   │
│  --agent  --model  --cwd  query                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              OrchestratorAgent                          │
│                                                         │
│  1. Single LLM call (ORCHESTRATOR_MODEL, fast)          │
│     → classifies task as "coder" or "general"           │
│  2. Instantiates the chosen sub-agent                   │
│  3. Persists cwd across REPL turns                      │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│     CoderAgent       │  │        GeneralAgent            │
│  (CODER_MODEL)       │  │     (GENERAL_MODEL)            │
│                      │  │                               │
│  5-phase workflow:   │  │  Open-ended ReAct loop        │
│  1. EXPLORE          │  │  with all tools               │
│  2. PLAN             │  │                               │
│  3. IMPLEMENT        │  │  Saves output to files        │
│  4. VERIFY           │  │  when task requires it        │
│  5. REPORT           │  │                               │
└──────────┬───────────┘  └───────────────┬───────────────┘
           │                              │
           │   delegate_to_agent          │   delegate_to_agent
           │ ◄────────────────────────────┘
           │ ──────────────────────────── ►
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                    Tool Registry                        │
│                  (tools/registry.py)                    │
│                                                         │
│  File ops     │ read_file, write_file, edit_file,       │
│               │ list_dir, change_dir, read_lines        │
│  Shell        │ bash                                    │
│  Code nav     │ grep_code, find_files, code_outline     │
│  Git          │ status, diff, log, commit, branch,      │
│               │ checkout, blame                         │
│  Code tools   │ explain_code, fix_code, generate_tests, │
│  (LLM-based)  │ review_code, lint, run_tests            │
│  Web          │ websearch                               │
│  Knowledge    │ snippets (save/get/list/delete)         │
│               │ RAG (add_text/file/url, search, list)   │
│  Agents       │ delegate_to_agent                       │
└─────────────────────────────────────────────────────────┘
```

### Agent communication

Any top-level agent (depth 0) can call `delegate_to_agent` to hand off a focused sub-task:

```json
{
  "thought": "I need to search the web for the correct API endpoint before I code",
  "action": "delegate_to_agent",
  "action_input": {
    "agent": "general",
    "task": "Search the web for the Open-Meteo API URL for current weather in Santiago, Chile and return the exact endpoint URL and field names."
  }
}
```

The sub-agent runs a full ReAct loop and returns its answer as an observation. Delegation is limited to one level deep to prevent recursive chains.

### CoderAgent — 5-phase workflow

The coder enforces a strict sequence via its system prompt and a `plan` special action:

| Phase | What happens | Key tools |
|---|---|---|
| 1. Explore | Read-only survey of the project | `find_files`, `code_outline`, `grep_code`, `git_status` |
| 2. Plan | Emits a `plan` action — lists steps, files to create/modify | *(special action, not a tool)* |
| 3. Implement | Writes and edits files, installs deps | `write_file`, `edit_file`, `bash` |
| 4. Verify | Lints every changed file, runs tests if present | `lint`, `run_tests`, `git_status` |
| 5. Report | Structured `final_answer` with summary | *(final_answer)* |

`final_answer` is blocked (by a system prompt rule) until all planned files have been written.

---

## BaseAgent — the parent class

`agents/base.py` defines `BaseAgent`, which all agents inherit from. It provides the complete ReAct loop, tool dispatch, display, and inter-agent delegation. Understanding it is the key to extending the system.

### Constructor

```python
class BaseAgent:
    name: str = "base"      # used to look up AGENT_MODELS
    label: str = "Assistant"
    border_color: str = "blue"

    def __init__(self, model: str = None, cwd: str = None, _depth: int = 0):
        self._model_override = model          # raw CLI arg — None means "use per-agent config"
        self.model = model if model is not None else AGENT_MODELS.get(self.name, DEFAULT_MODEL)
        self.cwd = cwd or os.getcwd()
        self._depth = _depth                  # delegation depth — max 1
```

- **`model=None`** means "use the model configured for this agent in `AGENT_MODELS`". Passing an explicit model string overrides that for this agent and all agents it delegates to.
- **`_depth`** prevents recursive delegation: depth-0 agents can call `delegate_to_agent`; depth-1 agents cannot.

### Abstract interface

Subclasses must implement one method:

```python
def build_system_prompt(self) -> str:
    """Return the full system prompt for this agent's ReAct loop."""
    raise NotImplementedError
```

And may optionally override:

```python
def handle_special_action(self, action: str, action_input: dict) -> str | None:
    """
    Handle non-tool actions before they reach the tool registry.
    Return an observation string to continue the loop, or None to fall through.
    Always call super().handle_special_action(action, action_input) for unknown actions.
    """
```

`BaseAgent.handle_special_action` handles `delegate_to_agent`. `CoderAgent` overrides it to handle `plan`, then calls `super()` for everything else.

### ReAct loop — step by step

```
run(user_input)
│
├── Build messages: [system_prompt, user_message]
│
└── for each iteration (up to MAX_ITERATIONS):
    │
    ├── llm.chat(messages) → raw string
    │
    ├── _extract_action(raw) → dict or None
    │   ├── searches for ```json ... ``` block
    │   └── falls back to first { ... } in the text
    │
    ├── if None (no JSON found):
    │   ├── if response looks like prose/code (≤2 misses):
    │   │   └── append correction message → continue  ← JSON-enforcement retry
    │   └── else: _print_answer(raw) → return
    │
    ├── if action == "final_answer":
    │   └── _print_answer(response) → return
    │
    ├── handle_special_action(action, action_input)
    │   ├── "delegate_to_agent" → spawn sub-agent, run task, return result
    │   ├── "plan" (CoderAgent) → display plan, return acknowledgement
    │   └── None → fall through to tool registry
    │
    ├── execute_tool(action, action_input, cwd) → (result, new_cwd)
    │
    ├── update cwd
    └── append "Observation: <result>" → continue
```

### JSON-enforcement retry

When the model returns prose instead of a JSON action block, the loop detects it (by looking for ` ``` `, `def `, `import `, or `class ` in the response) and injects a correction message before retrying. This happens at most twice per run; after that the response is treated as a final answer. This prevents small models from silently short-circuiting the loop.

### Tool docs injection

Each agent's system prompt includes `self._tool_docs()`, which renders every tool's name, description, and parameter schema into a markdown list. Sub-agents (`_depth >= 1`) automatically have `delegate_to_agent` removed from their docs to prevent recursive delegation.

### Display helpers

All terminal output goes through helpers that gracefully degrade when `rich` is not installed:

| Method | Output |
|---|---|
| `_print_thought(thought)` | Dimmed italic reasoning line |
| `_print_tool_call(action, params)` | Yellow bordered panel with tool name + JSON params |
| `_print_tool_result(result)` | Green bordered panel with result preview + cwd |
| `_print_answer(text)` | Green bordered panel with Markdown-rendered final answer |
| `_print_format_warning(raw)` | Yellow warning when JSON format is missing |
| `_print_delegation(agent, task)` | Dim line showing delegation target |
| `_print_banner()` | Startup header with model name and cwd |

---

## Tools reference

### File operations

| Tool | Parameters | Description |
|---|---|---|
| `read_file` | `path` | Read full file contents |
| `read_lines` | `path`, `start`, `end` | Read a line range (1-indexed) |
| `write_file` | `path`, `content` | Write or overwrite a file |
| `edit_file` | `path`, `old_text`, `new_text` | Replace the first occurrence of `old_text` |
| `list_dir` | `path?` | List directory entries |
| `change_dir` | `path` | Change cwd (persists for the rest of the run) |

### Shell

| Tool | Parameters | Description |
|---|---|---|
| `bash` | `command`, `timeout?` | Run a shell command; cwd persists via `pwd` tracking; dangerous commands are blocked |

### Code navigation

| Tool | Parameters | Description |
|---|---|---|
| `grep_code` | `pattern`, `path?`, `file_glob?`, `case_sensitive?`, `max_matches?` | Regex search across source files |
| `find_files` | `pattern`, `path?`, `max_results?` | Glob search for filenames |
| `code_outline` | `path` | Extract classes, functions, imports (AST for Python, regex for JS/TS) |

### Git

| Tool | Parameters | Description |
|---|---|---|
| `git_status` | — | Working tree status |
| `git_diff` | `path?`, `staged?` | Show unstaged or staged changes |
| `git_log` | `n?` | Recent commit history |
| `git_commit` | `message`, `files?` | Stage and commit |
| `git_branch` | — | List branches |
| `git_checkout` | `ref`, `create?` | Switch or create branch |
| `git_blame` | `path` | Per-line authorship |

### LLM-powered code tools

These call the local Ollama model directly — useful for tasks that are too complex for a single prompt line.

| Tool | Parameters | Description |
|---|---|---|
| `explain_code` | `code`, `language?` | Plain-English explanation |
| `fix_code` | `code`, `error`, `language?` | Root cause + corrected code |
| `generate_tests` | `code`, `language?`, `framework?` | Unit test generation |
| `review_code` | `code`, `language?` | Bugs, security, performance, style |
| `lint` | `path` | ruff → flake8 → pylint (Python); eslint (JS/TS); AST fallback |
| `run_tests` | `path?`, `pattern?`, `verbose?` | pytest or unittest |

### Knowledge base

| Tool | Parameters | Description |
|---|---|---|
| `save_snippet` | `name`, `code`, `language?`, `description?` | Save a reusable code snippet |
| `get_snippet` | `name` | Retrieve a snippet by name |
| `list_snippets` | — | List all snippets |
| `delete_snippet` | `name` | Remove a snippet |
| `rag_add_text` | `name`, `text`, `collection?` | Ingest raw text into the RAG store |
| `rag_add_file` | `path`, `collection?` | Ingest a file |
| `rag_add_url` | `url`, `collection?` | Fetch and ingest a web page |
| `rag_search` | `query`, `collection?`, `top_k?` | Semantic search |
| `rag_list` | `collection?` | List ingested documents |
| `rag_collections` | — | List all collections |
| `rag_delete` | `source`, `collection?` | Remove a document |

### Agent delegation

| Tool | Parameters | Description |
|---|---|---|
| `delegate_to_agent` | `agent`, `task` | Hand off a sub-task to `coder` or `general`; returns the result as an observation |

---

## Extending the codebase

### Add a new agent

1. Create `agents/myagent.py` inheriting from `BaseAgent`:

```python
from agents.base import BaseAgent

class MyAgent(BaseAgent):
    name = "myagent"          # used for AGENT_MODELS lookup and CLI --agent flag
    label = "My Agent"
    border_color = "cyan"

    def build_system_prompt(self) -> str:
        return f"""You are a specialist agent for ...

## Current working directory
`{self.cwd}`

## Available tools
{self._tool_docs()}

## Response format
```json
{{"thought": "...", "action": "...", "action_input": {{...}}}}
```
"""
```

2. Register it in `agents/__init__.py`:

```python
from agents.myagent import MyAgent

REGISTRY = {
    "orchestrator": OrchestratorAgent,
    "coder":        CoderAgent,
    "general":      GeneralAgent,
    "myagent":      MyAgent,       # add this line
}
```

3. Add its model to `config.py`:

```python
AGENT_MODELS = {
    ...
    "myagent": os.getenv("MYAGENT_MODEL", DEFAULT_MODEL),
}
```

4. Add it to the orchestrator's routing descriptions in `agents/orchestrator.py`:

```python
_AGENT_DESCRIPTIONS = {
    ...
    "myagent": "describe when to route here",
}
```

### Add a new tool

1. Implement the function in an existing or new file under `tools/`:

```python
# tools/mytools.py
def my_tool(param: str, cwd: str = ".") -> str:
    ...
    return "result string"
```

2. Add the definition to `TOOLS` in `tools/registry.py`:

```python
{
    "name": "my_tool",
    "description": "What it does and when to use it.",
    "parameters": {
        "param": "str — description",
    },
},
```

3. Add the dispatch case to `execute_tool` in `tools/registry.py`:

```python
elif name == "my_tool":
    from tools.mytools import my_tool
    return my_tool(params["param"], cwd), cwd
```

The tool now appears automatically in every agent's `_tool_docs()` output.

### Add a new special action

Special actions are handled before the tool registry — useful for actions that need access to agent state (like `plan` or `delegate_to_agent`).

Override `handle_special_action` in your agent:

```python
def handle_special_action(self, action: str, action_input: dict) -> str | None:
    if action == "my_action":
        # do something with self.cwd, self.model, etc.
        return "Observation: action completed"
    return super().handle_special_action(action, action_input)  # always fall through
```

The returned string is appended as an `Observation:` message and the loop continues.

### Project layout

```
codeassistant/
├── agent.py              # CLI entry point (argparse)
├── llm.py                # Ollama HTTP client (chat + embed + list_models)
├── config.py             # All tunables — env vars with defaults
├── run.sh                # Convenience wrapper (uses .venv)
│
├── agents/
│   ├── __init__.py       # REGISTRY dict + DEFAULT_AGENT
│   ├── base.py           # BaseAgent: ReAct loop, display, delegation
│   ├── orchestrator.py   # OrchestratorAgent: routes to sub-agents
│   ├── coder.py          # CoderAgent: 5-phase coding workflow
│   └── general.py        # GeneralAgent: open-ended Q&A and tasks
│
├── tools/
│   ├── registry.py       # TOOLS list + execute_tool dispatcher
│   ├── file_ops.py       # read/write/edit/list files
│   ├── bash_exec.py      # safe shell execution with cwd tracking
│   ├── code_nav.py       # grep, find, outline
│   ├── git_ops.py        # git status/diff/log/commit/branch/…
│   ├── code_tools.py     # lint, run_tests, explain/fix/generate/review
│   ├── websearch.py      # DuckDuckGo search via ddgs
│   ├── summarize.py      # LLM-powered text summarization
│   ├── snippets.py       # persistent code snippet store
│   └── rag.py            # RAG: chunk, embed, search via numpy
│
├── smoke-test/
│   └── test_agents.py    # Integration smoke tests (run from project root)
│
├── snippets/             # Saved snippet files (created at runtime)
└── rag_store/            # RAG vector store (created at runtime)
```
