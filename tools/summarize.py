"""Summarization via local LLM."""

import llm
from config import DEFAULT_MODEL

_CHUNK = 6000  # chars per chunk to stay within context


def summarize(text: str, focus: str = "", model: str = DEFAULT_MODEL) -> str:
    """Summarize text, optionally focusing on a specific aspect."""
    chunks = [text[i:i + _CHUNK] for i in range(0, len(text), _CHUNK)]

    if len(chunks) == 1:
        return _summarize_chunk(chunks[0], focus, model)

    # Recursive: summarize each chunk, then summarize summaries
    chunk_summaries = [_summarize_chunk(c, focus, model) for c in chunks]
    combined = "\n\n---\n\n".join(chunk_summaries)
    return _summarize_chunk(combined, focus, model)


def _summarize_chunk(text: str, focus: str, model: str) -> str:
    focus_clause = f" Focus on: {focus}." if focus else ""
    messages = [
        {
            "role": "system",
            "content": f"You are a concise summarizer.{focus_clause} Produce a clear, dense summary.",
        },
        {"role": "user", "content": f"Summarize this:\n\n{text}"},
    ]
    return llm.chat(messages, model=model)
