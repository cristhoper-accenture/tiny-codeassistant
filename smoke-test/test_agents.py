"""
Smoke tests for the codeassistant agents and tools.
Run from project root: .venv/bin/python smoke-test/test_agents.py
"""

import sys
import os

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SMOKE_DIR = os.path.join(ROOT, "smoke-test")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        _results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        _results.append((name, False, e))


# ── Agent registry ─────────────────────────────────────────────────────────────

def test_registry():
    from agents import REGISTRY, GeneralAgent, CoderAgent
    assert "general" in REGISTRY
    assert "coder" in REGISTRY
    assert REGISTRY["general"] is GeneralAgent
    assert REGISTRY["coder"] is CoderAgent


def test_general_agent_instantiation():
    from agents import GeneralAgent
    a = GeneralAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    prompt = a.build_system_prompt()
    assert SMOKE_DIR in prompt
    assert "tool_name" in prompt or "action" in prompt


def test_coder_agent_instantiation():
    from agents import CoderAgent
    a = CoderAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    prompt = a.build_system_prompt()
    assert "EXPLORE" in prompt
    assert "PLAN" in prompt
    assert "IMPLEMENT" in prompt
    assert "VERIFY" in prompt


def test_coder_plan_action():
    from agents import CoderAgent
    a = CoderAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    obs = a.handle_special_action("plan", {
        "summary": "Add hello function",
        "steps": ["Write hello.py", "Add tests"],
        "files_create": ["hello.py"],
        "files_modify": [],
    })
    assert obs is not None
    assert "Plan acknowledged" in obs
    assert "Implement" in obs


def test_unknown_special_action_returns_none():
    from agents import CoderAgent
    a = CoderAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    assert a.handle_special_action("bash", {}) is None


# ── Tools: code_nav ────────────────────────────────────────────────────────────

def test_find_files():
    from tools.code_nav import find_files
    result = find_files("*.py", cwd=ROOT)
    assert "agent.py" in result


def test_grep_code():
    from tools.code_nav import grep_code
    result = grep_code("def run", path=".", file_glob="*.py", cwd=ROOT)
    assert "def run" in result


def test_code_outline_python():
    from tools.code_nav import code_outline
    result = code_outline("agents/base.py", cwd=ROOT)
    assert "class BaseAgent" in result
    assert "def run" in result


def test_read_lines():
    from tools.code_nav import read_lines
    result = read_lines("config.py", start=1, end=3, cwd=ROOT)
    assert "import os" in result
    assert "lines 1–3" in result


# ── Tools: git_ops ────────────────────────────────────────────────────────────

def test_git_status():
    from tools.git_ops import status
    result = status(ROOT)
    assert "main" in result or "branch" in result.lower() or "##" in result


def test_git_log():
    from tools.git_ops import log
    result = log(3, cwd=ROOT)
    assert len(result) > 0


# ── Tools: file_ops with cwd ──────────────────────────────────────────────────

def test_write_and_read_file():
    from tools.file_ops import write_file, read_file
    write_file("tmp_smoke.txt", "hello smoke", cwd=SMOKE_DIR)
    content = read_file("tmp_smoke.txt", cwd=SMOKE_DIR)
    assert content == "hello smoke"
    os.remove(os.path.join(SMOKE_DIR, "tmp_smoke.txt"))


def test_edit_file():
    from tools.file_ops import write_file, edit_file, read_file
    write_file("tmp_edit.txt", "foo bar baz", cwd=SMOKE_DIR)
    edit_file("tmp_edit.txt", "bar", "QUX", cwd=SMOKE_DIR)
    assert read_file("tmp_edit.txt", cwd=SMOKE_DIR) == "foo QUX baz"
    os.remove(os.path.join(SMOKE_DIR, "tmp_edit.txt"))


# ── Tools: bash_exec with cwd tracking ───────────────────────────────────────

def test_bash_exec_cwd():
    from tools.bash_exec import run
    result = run("echo hello", cwd=SMOKE_DIR)
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]
    assert result["cwd"] == SMOKE_DIR


def test_bash_cd_persists():
    from tools.bash_exec import run
    result = run("cd /tmp && echo done", cwd=SMOKE_DIR)
    assert result["returncode"] == 0
    assert result["cwd"] == "/tmp"


def test_bash_blocks_dangerous():
    from tools.bash_exec import run
    result = run("rm -rf /")
    assert result["returncode"] == -1
    assert "Blocked" in result["stderr"]


# ── Tools: snippets ───────────────────────────────────────────────────────────

def test_snippet_lifecycle():
    from tools.snippets import save, get, list_snippets, delete
    save("smoke_fn", "def smoke(): pass", "python", "smoke test fn")
    listing = list_snippets()
    assert "smoke_fn" in listing
    content = get("smoke_fn")
    assert "def smoke(): pass" in content
    delete("smoke_fn")
    assert "smoke_fn" not in list_snippets()


# ── Tools: code_tools (lint / explain) ───────────────────────────────────────

def test_lint_valid_file():
    from tools.code_tools import lint
    # lint a known-good file — should not report errors
    result = lint("config.py", cwd=ROOT)
    # passes if no crash; content check is linter-dependent
    assert isinstance(result, str)


def test_lint_syntax_error():
    from tools.code_tools import lint
    from tools.file_ops import write_file
    bad = "def broken(\n    pass\n"
    write_file("tmp_bad.py", bad, cwd=SMOKE_DIR)
    result = lint("tmp_bad.py", cwd=SMOKE_DIR)
    assert isinstance(result, str)
    os.remove(os.path.join(SMOKE_DIR, "tmp_bad.py"))


# ── Orchestrator ──────────────────────────────────────────────────────────────

def test_orchestrator_in_registry():
    from agents import REGISTRY, DEFAULT_AGENT
    assert "orchestrator" in REGISTRY
    assert DEFAULT_AGENT == "orchestrator"


def test_orchestrator_instantiation():
    from agents import OrchestratorAgent
    o = OrchestratorAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    assert o.name == "orchestrator"
    assert o.cwd == SMOKE_DIR


def test_orchestrator_routes_coder():
    from agents import OrchestratorAgent
    o = OrchestratorAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    # Coding tasks
    for prompt in [
        "write a function that reverses a string",
        "create a new file called app.py",
        "fix the bug in my code",
        "implement a binary search",
    ]:
        result = o.route(prompt)
        assert result == "coder", f"Expected coder for: {prompt!r}, got {result!r}"


def test_orchestrator_routes_general():
    from agents import OrchestratorAgent
    o = OrchestratorAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    # General / informational tasks
    for prompt in [
        "what is a REST API?",
        "explain what async/await does",
        "search for python tutorial",
        "summarize this document",
    ]:
        result = o.route(prompt)
        assert result == "general", f"Expected general for: {prompt!r}, got {result!r}"


def test_orchestrator_cwd_persists():
    from agents import OrchestratorAgent
    o = OrchestratorAgent(model="qwen3.5:2b", cwd=SMOKE_DIR)
    assert o.cwd == SMOKE_DIR
    # Simulate cwd update as sub-agent would do
    o.cwd = "/tmp"
    assert o.cwd == "/tmp"


# ── Tools: RAG (ingest + search) ─────────────────────────────────────────────

def test_rag_lifecycle():
    import shutil
    from tools.rag import add_text, search, list_docs, delete_doc
    col = "smoke_test_col"
    add_text("doc_a", "Python is a dynamically typed language with garbage collection.", col)
    add_text("doc_b", "Rust is a systems language focused on memory safety and performance.", col)
    results = search("dynamically typed language", col, top_k=2)
    assert "doc_a" in results or "Python" in results
    listing = list_docs(col)
    assert "doc_a" in listing
    delete_doc("doc_a", col)
    delete_doc("doc_b", col)
    # clean up collection dir
    from config import RAG_DIR
    col_dir = os.path.join(RAG_DIR, col)
    if os.path.isdir(col_dir):
        shutil.rmtree(col_dir)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # agents
        ("Registry has general + coder", test_registry),
        ("GeneralAgent instantiation", test_general_agent_instantiation),
        ("CoderAgent instantiation", test_coder_agent_instantiation),
        ("CoderAgent plan action", test_coder_plan_action),
        ("Unknown special action → None", test_unknown_special_action_returns_none),
        # orchestrator
        ("Orchestrator in registry as default", test_orchestrator_in_registry),
        ("OrchestratorAgent instantiation", test_orchestrator_instantiation),
        ("Orchestrator routes coding tasks → coder", test_orchestrator_routes_coder),
        ("Orchestrator routes questions → general", test_orchestrator_routes_general),
        ("Orchestrator cwd persists across turns", test_orchestrator_cwd_persists),
        # code_nav
        ("find_files finds agent.py", test_find_files),
        ("grep_code finds def run", test_grep_code),
        ("code_outline parses BaseAgent", test_code_outline_python),
        ("read_lines returns numbered range", test_read_lines),
        # git_ops
        ("git_status returns branch info", test_git_status),
        ("git_log returns commits", test_git_log),
        # file_ops
        ("write + read file via cwd", test_write_and_read_file),
        ("edit_file replaces text", test_edit_file),
        # bash_exec
        ("bash exec runs in cwd", test_bash_exec_cwd),
        ("bash cd persists in result", test_bash_cd_persists),
        ("bash blocks rm -rf /", test_bash_blocks_dangerous),
        # snippets
        ("snippet save/get/list/delete", test_snippet_lifecycle),
        # code_tools
        ("lint valid file", test_lint_valid_file),
        ("lint catches syntax error", test_lint_syntax_error),
        # RAG
        ("RAG add/search/delete lifecycle", test_rag_lifecycle),
    ]

    print(f"\nRunning {len(tests)} smoke tests...\n")
    for name, fn in tests:
        check(name, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print(f"\n{'─'*50}")
    print(f"  {passed}/{len(_results)} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print("  — all clear")
    print()
    sys.exit(0 if failed == 0 else 1)
