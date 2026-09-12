"""Web search via DuckDuckGo (no API key required) and URL content reader."""

import html.parser
import urllib.request

from config import RAG_FETCH_TIMEOUT
from ddgs import DDGS


# ── HTML text extractor (mirrors rag._TextExtractor) ──────────────────────────

class _TextExtractor(html.parser.HTMLParser):
    _SKIP = {"script", "style", "head", "nav", "footer", "aside"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)


def read_url(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return its readable text content."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=RAG_FETCH_TIMEOUT) as resp:
        raw = resp.read().decode(resp.headers.get_content_charset("utf-8"), errors="replace")
    parser = _TextExtractor()
    parser.feed(raw)
    text = " ".join(parser.parts)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated — {len(text) - max_chars} chars remaining]"
    return text


def search(query: str, max_results: int = 5) -> list[dict]:
    """Return list of {title, url, snippet} dicts."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results


def format_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**\n   {r['url']}\n   {r['snippet']}")
    return "\n\n".join(lines)
