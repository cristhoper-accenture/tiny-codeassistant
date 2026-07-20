"""Web search via DuckDuckGo (no API key required)."""

from ddgs import DDGS


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
