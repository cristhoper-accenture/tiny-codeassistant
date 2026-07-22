"""Tool registry: definitions and dispatcher."""

import os
from tools import websearch, file_ops, bash_exec, summarize, snippets

TOOLS: list[dict] = [
    {
        "name": "websearch",
        "description": "Search the web for up-to-date information using DuckDuckGo.",
        "parameters": {
            "query": "str — the search query",
            "max_results": "int (optional, default 5) — number of results to return",
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Relative paths resolve from the current working directory.",
        "parameters": {"path": "str — file path (relative or absolute)"},
    },
    {
        "name": "write_file",
        "description": "Write (or overwrite) a file. Relative paths are created inside the current working directory.",
        "parameters": {
            "path": "str — file path (relative or absolute)",
            "content": "str — full file content",
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the first occurrence of old_text with new_text in a file.",
        "parameters": {
            "path": "str — file path (relative or absolute)",
            "old_text": "str — exact text to find",
            "new_text": "str — replacement text",
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories. Defaults to the current working directory.",
        "parameters": {"path": "str (optional, default '.') — directory path"},
    },
    {
        "name": "change_dir",
        "description": "Change the current working directory. Affects all subsequent file and bash operations.",
        "parameters": {"path": "str — absolute or relative path to cd into"},
    },
    {
        "name": "bash",
        "description": (
            "Execute a shell command in the current working directory. "
            "cd commands persist across calls."
        ),
        "parameters": {
            "command": "str — the shell command to run",
            "timeout": f"int (optional, default {bash_exec.BASH_TIMEOUT}s)",
        },
    },
    {
        "name": "summarize",
        "description": "Summarize a long piece of text using the local LLM.",
        "parameters": {
            "text": "str — the text to summarize",
            "focus": "str (optional) — specific aspect to focus on",
        },
    },
    {
        "name": "save_snippet",
        "description": "Save a code snippet for later reuse.",
        "parameters": {
            "name": "str — unique snippet name (snake_case)",
            "code": "str — the code content",
            "language": "str (optional) — e.g. python, javascript, bash",
            "description": "str (optional) — short description",
        },
    },
    {
        "name": "get_snippet",
        "description": "Retrieve a previously saved code snippet by name.",
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
]


def execute_tool(name: str, params: dict, cwd: str) -> tuple[str, str]:
    """Dispatch a tool call. Returns (result_str, updated_cwd)."""
    try:
        if name == "websearch":
            results = websearch.search(params["query"], params.get("max_results", 5))
            return websearch.format_results(results), cwd

        elif name == "read_file":
            return file_ops.read_file(params["path"], cwd), cwd

        elif name == "write_file":
            return file_ops.write_file(params["path"], params["content"], cwd), cwd

        elif name == "edit_file":
            return file_ops.edit_file(params["path"], params["old_text"], params["new_text"], cwd), cwd

        elif name == "list_dir":
            return file_ops.list_dir(params.get("path", "."), cwd), cwd

        elif name == "change_dir":
            target = params["path"]
            target = os.path.expanduser(target)
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(cwd, target))
            if not os.path.isdir(target):
                return f"ERROR: not a directory: {target}", cwd
            return f"Working directory changed to {target}", target

        elif name == "bash":
            result = bash_exec.run(params["command"], cwd=cwd, timeout=params.get("timeout", bash_exec.BASH_TIMEOUT))
            new_cwd = result.get("cwd") or cwd
            return bash_exec.format_result(result), new_cwd

        elif name == "summarize":
            return summarize.summarize(params["text"], params.get("focus", "")), cwd

        elif name == "save_snippet":
            return snippets.save(
                params["name"],
                params["code"],
                params.get("language", ""),
                params.get("description", ""),
            ), cwd

        elif name == "get_snippet":
            return snippets.get(params["name"]), cwd

        elif name == "list_snippets":
            return snippets.list_snippets(), cwd

        elif name == "delete_snippet":
            return snippets.delete(params["name"]), cwd

        else:
            return f"Unknown tool: {name}", cwd

    except KeyError as e:
        return f"Missing required parameter: {e}", cwd
    except Exception as e:
        return f"Tool error ({name}): {type(e).__name__}: {e}", cwd
