"""Internet adapter (implements WebPort): DuckDuckGo to search, requests + BeautifulSoup to read.

Dependencies are imported lazily: the agent starts even when they are missing and
only fails, with a readable message, if the model actually tries to use the web.
"""
from __future__ import annotations


class WebClient:
    def __init__(self, timeout: int = 10, max_chars: int = 8000):
        self._timeout, self._max = timeout, max_chars

    def search(self, query: str, max_results: int = 5) -> str:
        """Search DuckDuckGo and return the top results as text."""
        try:
            try:
                from ddgs import DDGS                      # current package name
            except ImportError:
                from duckduckgo_search import DDGS         # legacy package name
        except ImportError:
            return "Missing dependency: pip install ddgs"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            return "\n\n".join(
                f"Title: {r.get('title', '')}\nLink: {r.get('href', '')}\nSummary: {r.get('body', '')}"
                for r in results)
        except Exception as e:
            return f"Web search failed: {e}"

    def fetch(self, url: str) -> str:
        """Extract the main text of a URL (truncated so it does not flood the context)."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return "Missing dependencies: pip install requests beautifulsoup4"
        if not url.startswith(("http://", "https://")):
            return "Invalid URL: it must start with http:// or https://"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                             timeout=self._timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            for el in soup(["script", "style", "nav", "footer"]):
                el.extract()
            return soup.get_text(separator="\n", strip=True)[:self._max] or "(the page has no readable text)"
        except Exception as e:
            return f"Could not fetch the URL: {e}"
