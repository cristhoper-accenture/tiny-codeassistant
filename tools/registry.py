"""Tool registry: definitions and dispatcher."""

import json
from tools import websearch, file_ops, bash_exec, summarize, snippets

# Each tool definition describes inputs so the LLM knows how to call it.
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
        "description": "Read the contents of a file from disk.",
        "parameters": {"path": "str — absolute or relative file path"},
    },
    {
        "name": "write_file",
        "description": "Write (or overwrite) a file with given content.",
        "parameters": {
            "path": "str — file path",
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
        "description": "List files and directories at a given path.",
        "parameters": {"path": "str (optional, default '.') — directory path"},
    },
    {
        "name": "bash",
        "description": "Execute a shell command and return stdout/stderr. Use carefully.",
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


def execute_tool(name: str, params: dict) -> str:
    """Dispatch a tool call and return the string result."""
    try:
        if name == "websearch":
            results = websearch.search(params["query"], params.get("max_results", 5))
            return websearch.format_results(results)

        elif name == "read_file":
            return file_ops.read_file(params["path"])

        elif name == "write_file":
            return file_ops.write_file(params["path"], params["content"])

        elif name == "edit_file":
            return file_ops.edit_file(params["path"], params["old_text"], params["new_text"])

        elif name == "list_dir":
            return file_ops.list_dir(params.get("path", "."))

        elif name == "bash":
            result = bash_exec.run(params["command"], params.get("timeout", bash_exec.BASH_TIMEOUT))
            return bash_exec.format_result(result)

        elif name == "summarize":
            return summarize.summarize(params["text"], params.get("focus", ""))

        elif name == "save_snippet":
            return snippets.save(
                params["name"],
                params["code"],
                params.get("language", ""),
                params.get("description", ""),
            )

        elif name == "get_snippet":
            return snippets.get(params["name"])

        elif name == "list_snippets":
            return snippets.list_snippets()

        elif name == "delete_snippet":
            return snippets.delete(params["name"])

        else:
            return f"Unknown tool: {name}"

    except KeyError as e:
        return f"Missing required parameter: {e}"
    except Exception as e:
        return f"Tool error ({name}): {type(e).__name__}: {e}"
