"""Tool registry: definitions and dispatcher."""

import os
from tools import websearch, file_ops, bash_exec, summarize, snippets, rag
from tools import code_nav, git_ops, code_tools

TOOLS: list[dict] = [
    # ── Agent delegation ─────────────────────────────────────────────────────
    {
        "name": "delegate_to_agent",
        "description": (
            "Hand off a focused sub-task to a specialist agent and get its result back. "
            "Use 'coder' for writing/editing/verifying code files; "
            "use 'general' for questions, explanations, web search, or summaries."
        ),
        "parameters": {
            "agent": "str — target agent name: 'coder' or 'general'",
            "task": "str — complete, self-contained description of the sub-task",
        },
    },
    # ── Web ──────────────────────────────────────────────────────────────────
    {
        "name": "websearch",
        "description": "Search the web for up-to-date information using DuckDuckGo.",
        "parameters": {
            "query": "str — the search query",
            "max_results": "int (optional, default 5) — number of results to return",
        },
    },
    # ── File ops ─────────────────────────────────────────────────────────────
    {
        "name": "read_file",
        "description": "Read the full contents of a file. Relative paths resolve from cwd.",
        "parameters": {"path": "str — file path (relative or absolute)"},
    },
    {
        "name": "read_lines",
        "description": "Read a specific line range of a file (1-indexed). Use for large files.",
        "parameters": {
            "path": "str — file path",
            "start": "int (optional, default 1) — first line to read",
            "end": "int (optional) — last line to read (inclusive); omit for end of file",
        },
    },
    {
        "name": "write_file",
        "description": "Write (or overwrite) a file. Relative paths resolve from cwd.",
        "parameters": {
            "path": "str — file path (relative or absolute)",
            "content": "str — full file content",
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the first occurrence of old_text with new_text in a file.",
        "parameters": {
            "path": "str — file path",
            "old_text": "str — exact text to find",
            "new_text": "str — replacement text",
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories. Defaults to cwd.",
        "parameters": {"path": "str (optional, default '.') — directory path"},
    },
    {
        "name": "change_dir",
        "description": "Change the current working directory. Persists for all subsequent calls.",
        "parameters": {"path": "str — absolute or relative path"},
    },
    # ── Shell ─────────────────────────────────────────────────────────────────
    {
        "name": "bash",
        "description": "Execute a shell command in cwd. cd commands persist across calls.",
        "parameters": {
            "command": "str — the shell command to run",
            "timeout": f"int (optional, default {bash_exec.BASH_TIMEOUT}s)",
        },
    },
    # ── Code navigation ───────────────────────────────────────────────────────
    {
        "name": "grep_code",
        "description": (
            "Regex search across source files. Returns matching lines with file:line context. "
            "Skips .git, .venv, __pycache__, node_modules."
        ),
        "parameters": {
            "pattern": "str — regex pattern to search for",
            "path": "str (optional, default '.') — directory or file to search",
            "file_glob": "str (optional, default '*') — filename glob filter, e.g. '*.py'",
            "case_sensitive": "bool (optional, default true)",
            "max_matches": "int (optional, default 50)",
        },
    },
    {
        "name": "find_files",
        "description": "Find files by name glob pattern (e.g. '*.py', 'test_*.js'). Skips .git, .venv, node_modules.",
        "parameters": {
            "pattern": "str — glob pattern for filenames",
            "path": "str (optional, default '.') — root directory to search",
            "max_results": "int (optional, default 100)",
        },
    },
    {
        "name": "code_outline",
        "description": (
            "Show the structural outline of a source file: classes, functions, imports. "
            "Python: AST-based. JS/TS: regex-based. Other: first 60 lines."
        ),
        "parameters": {"path": "str — file path"},
    },
    # ── Git ───────────────────────────────────────────────────────────────────
    {
        "name": "git_status",
        "description": "Show the git working tree status (branch + changed files).",
        "parameters": {},
    },
    {
        "name": "git_diff",
        "description": "Show git diff of unstaged (or staged) changes, optionally scoped to a file.",
        "parameters": {
            "path": "str (optional) — scope diff to this file",
            "staged": "bool (optional, default false) — show staged changes",
        },
    },
    {
        "name": "git_log",
        "description": "Show recent git commit history.",
        "parameters": {
            "n": "int (optional, default 10) — number of commits to show",
        },
    },
    {
        "name": "git_commit",
        "description": "Stage files and create a git commit. Stages all tracked changes if no files given.",
        "parameters": {
            "message": "str — commit message",
            "files": "list[str] (optional) — specific files to stage; omit to stage all tracked changes",
        },
    },
    {
        "name": "git_branch",
        "description": "List all branches and show the current one.",
        "parameters": {},
    },
    {
        "name": "git_checkout",
        "description": "Checkout a branch or commit. Optionally create a new branch.",
        "parameters": {
            "ref": "str — branch name, tag, or commit hash",
            "create": "bool (optional, default false) — create the branch if it doesn't exist",
        },
    },
    {
        "name": "git_blame",
        "description": "Show who last modified each line of a file.",
        "parameters": {"path": "str — file path"},
    },
    # ── Code tools (LLM-powered) ──────────────────────────────────────────────
    {
        "name": "explain_code",
        "description": "Explain what a piece of code does — purpose, logic, pitfalls, complexity.",
        "parameters": {
            "code": "str — the code to explain",
            "language": "str (optional) — e.g. python, javascript",
        },
    },
    {
        "name": "fix_code",
        "description": "Given code and an error message, diagnose the root cause and return corrected code.",
        "parameters": {
            "code": "str — the broken code",
            "error": "str — the error message or traceback",
            "language": "str (optional)",
        },
    },
    {
        "name": "generate_tests",
        "description": "Generate unit tests for a given piece of code.",
        "parameters": {
            "code": "str — the code to test",
            "language": "str (optional, default 'python')",
            "framework": "str (optional) — e.g. pytest, unittest, jest",
        },
    },
    {
        "name": "review_code",
        "description": "Review code for bugs, security issues, performance, and style. Returns structured feedback.",
        "parameters": {
            "code": "str — the code to review",
            "language": "str (optional)",
        },
    },
    {
        "name": "lint",
        "description": (
            "Run a linter on a file or directory. "
            "Auto-selects: ruff/flake8/pylint for Python, eslint for JS/TS."
        ),
        "parameters": {"path": "str — file or directory to lint"},
    },
    {
        "name": "run_tests",
        "description": "Run tests using pytest (preferred) or unittest. Returns test results.",
        "parameters": {
            "path": "str (optional, default '.') — test file or directory",
            "pattern": "str (optional) — test name filter (pytest -k / unittest -p)",
            "verbose": "bool (optional, default true)",
        },
    },
    # ── Summarize ─────────────────────────────────────────────────────────────
    {
        "name": "summarize",
        "description": "Summarize a long piece of text using the local LLM.",
        "parameters": {
            "text": "str — the text to summarize",
            "focus": "str (optional) — aspect to focus on",
        },
    },
    # ── Snippets ──────────────────────────────────────────────────────────────
    {
        "name": "save_snippet",
        "description": "Save a reusable code snippet with a name and optional description.",
        "parameters": {
            "name": "str — unique snippet name (snake_case)",
            "code": "str — the code content",
            "language": "str (optional) — e.g. python, javascript, bash",
            "description": "str (optional) — short description",
        },
    },
    {
        "name": "get_snippet",
        "description": "Retrieve a saved code snippet by name.",
        "parameters": {"name": "str — snippet name"},
    },
    {
        "name": "list_snippets",
        "description": "List all saved code snippets.",
        "parameters": {},
    },
    {
        "name": "delete_snippet",
        "description": "Delete a saved snippet by name.",
        "parameters": {"name": "str — snippet name"},
    },
    # ── RAG ───────────────────────────────────────────────────────────────────
    {
        "name": "rag_add_text",
        "description": "Add raw text to the RAG knowledge base.",
        "parameters": {
            "name": "str — label for this document",
            "text": "str — document content",
            "collection": "str (optional, default 'default')",
        },
    },
    {
        "name": "rag_add_file",
        "description": "Read a file and ingest it into the RAG knowledge base. Relative paths resolve from cwd.",
        "parameters": {
            "path": "str — file path",
            "collection": "str (optional, default 'default')",
        },
    },
    {
        "name": "rag_add_url",
        "description": "Fetch a URL and ingest the page text into the RAG knowledge base.",
        "parameters": {
            "url": "str — web URL to fetch",
            "collection": "str (optional, default 'default')",
        },
    },
    {
        "name": "rag_search",
        "description": "Semantic search over the RAG knowledge base. Returns the most relevant chunks.",
        "parameters": {
            "query": "str — natural language query",
            "collection": "str (optional, default 'default')",
            "top_k": "int (optional, default 5)",
        },
    },
    {
        "name": "rag_list",
        "description": "List documents in a RAG collection.",
        "parameters": {"collection": "str (optional, default 'default')"},
    },
    {
        "name": "rag_collections",
        "description": "List all RAG collections.",
        "parameters": {},
    },
    {
        "name": "rag_delete",
        "description": "Remove a document from a RAG collection.",
        "parameters": {
            "source": "str — source label used when the document was added",
            "collection": "str (optional, default 'default')",
        },
    },
]


def execute_tool(name: str, params: dict, cwd: str) -> tuple[str, str]:
    """Dispatch a tool call. Returns (result_str, updated_cwd)."""
    try:
        # ── Web ──────────────────────────────────────────────────────────────
        if name == "websearch":
            results = websearch.search(params["query"], params.get("max_results", 5))
            return websearch.format_results(results), cwd

        # ── File ops ─────────────────────────────────────────────────────────
        elif name == "read_file":
            return file_ops.read_file(params["path"], cwd), cwd

        elif name == "read_lines":
            return code_nav.read_lines(params["path"], params.get("start", 1), params.get("end"), cwd), cwd

        elif name == "write_file":
            return file_ops.write_file(params["path"], params["content"], cwd), cwd

        elif name == "edit_file":
            return file_ops.edit_file(params["path"], params["old_text"], params["new_text"], cwd), cwd

        elif name == "list_dir":
            return file_ops.list_dir(params.get("path", "."), cwd), cwd

        elif name == "change_dir":
            target = os.path.expanduser(params["path"])
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(cwd, target))
            if not os.path.isdir(target):
                return f"ERROR: not a directory: {target}", cwd
            return f"Working directory changed to {target}", target

        # ── Shell ─────────────────────────────────────────────────────────────
        elif name == "bash":
            result = bash_exec.run(params["command"], cwd=cwd, timeout=params.get("timeout", bash_exec.BASH_TIMEOUT))
            return bash_exec.format_result(result), result.get("cwd") or cwd

        # ── Code navigation ───────────────────────────────────────────────────
        elif name == "grep_code":
            return code_nav.grep_code(
                params["pattern"],
                params.get("path", "."),
                params.get("file_glob", "*"),
                params.get("case_sensitive", True),
                params.get("max_matches", 50),
                cwd,
            ), cwd

        elif name == "find_files":
            return code_nav.find_files(
                params["pattern"],
                params.get("path", "."),
                params.get("max_results", 100),
                cwd,
            ), cwd

        elif name == "code_outline":
            return code_nav.code_outline(params["path"], cwd), cwd

        # ── Git ───────────────────────────────────────────────────────────────
        elif name == "git_status":
            return git_ops.status(cwd), cwd

        elif name == "git_diff":
            return git_ops.diff(params.get("path"), params.get("staged", False), cwd), cwd

        elif name == "git_log":
            return git_ops.log(params.get("n", 10), cwd=cwd), cwd

        elif name == "git_commit":
            return git_ops.commit(params["message"], params.get("files"), cwd), cwd

        elif name == "git_branch":
            return git_ops.branch(cwd), cwd

        elif name == "git_checkout":
            return git_ops.checkout(params["ref"], params.get("create", False), cwd), cwd

        elif name == "git_blame":
            return git_ops.blame(params["path"], cwd), cwd

        # ── Code tools ────────────────────────────────────────────────────────
        elif name == "explain_code":
            return code_tools.explain_code(params["code"], params.get("language", "")), cwd

        elif name == "fix_code":
            return code_tools.fix_code(params["code"], params["error"], params.get("language", "")), cwd

        elif name == "generate_tests":
            return code_tools.generate_tests(
                params["code"], params.get("language", "python"), params.get("framework", "")
            ), cwd

        elif name == "review_code":
            return code_tools.review_code(params["code"], params.get("language", "")), cwd

        elif name == "lint":
            return code_tools.lint(params["path"], cwd), cwd

        elif name == "run_tests":
            return code_tools.run_tests(
                params.get("path", "."), params.get("pattern", ""), params.get("verbose", True), cwd
            ), cwd

        # ── Summarize ─────────────────────────────────────────────────────────
        elif name == "summarize":
            return summarize.summarize(params["text"], params.get("focus", "")), cwd

        # ── Snippets ──────────────────────────────────────────────────────────
        elif name == "save_snippet":
            return snippets.save(params["name"], params["code"], params.get("language", ""), params.get("description", "")), cwd

        elif name == "get_snippet":
            return snippets.get(params["name"]), cwd

        elif name == "list_snippets":
            return snippets.list_snippets(), cwd

        elif name == "delete_snippet":
            return snippets.delete(params["name"]), cwd

        # ── RAG ───────────────────────────────────────────────────────────────
        elif name == "rag_add_text":
            return rag.add_text(params["name"], params["text"], params.get("collection", "default")), cwd

        elif name == "rag_add_file":
            return rag.add_file(params["path"], params.get("collection", "default"), cwd), cwd

        elif name == "rag_add_url":
            return rag.add_url(params["url"], params.get("collection", "default")), cwd

        elif name == "rag_search":
            return rag.search(params["query"], params.get("collection", "default"), params.get("top_k", 5)), cwd

        elif name == "rag_list":
            return rag.list_docs(params.get("collection", "default")), cwd

        elif name == "rag_collections":
            return rag.list_collections(), cwd

        elif name == "rag_delete":
            return rag.delete_doc(params["source"], params.get("collection", "default")), cwd

        else:
            return f"Unknown tool: {name}", cwd

    except KeyError as e:
        return f"Missing required parameter: {e}", cwd
    except Exception as e:
        return f"Tool error ({name}): {type(e).__name__}: {e}", cwd
