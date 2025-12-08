# src/tools/search_mcp_server.py

from __future__ import annotations

import logging
from typing import Dict, Any, List

import requests
from fastmcp import FastMCP, Context

from src.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

# Production-style MCP server
mcp = FastMCP(
    "math_agent_tools",
    mask_error_details=True,
    log_level="INFO",
)


# -----------------------------------------------------------
# 1. Tavily Web Search Tool (Primary Fallback Search)
# -----------------------------------------------------------
@mcp.tool()  #  FIXED: Added parentheses
def tavily_search(query: str, n_results: int = 5, ctx: Context = None) -> Dict[str, Any]:
    """
    Web search using Tavily API.

    Used when:
      - Local RAG confidence is low
      - Query not covered by local KB
      - Hybrid reasoning is required

    Args:
        query: Search query text.
        n_results: Number of results (1–10 recommended).
        ctx: FastMCP context (auto-injected).

    Returns:
        dict: {
          "query": str,
          "results": [
            {
              "title": str,
              "url": str,
              "content": str,
            }, ...
          ]
        }
        or {"error": str} on failure.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Query cannot be empty."}

    if not TAVILY_API_KEY:
        logger.error("[MCP:TAVILY] TAVILY_API_KEY missing in environment.")
        return {"error": "Tavily API key not configured."}

    # Clamp results to sane range
    if n_results <= 0:
        n_results = 3
    if n_results > 10:
        n_results = 10

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": n_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        raw_results: List[Dict[str, Any]] = data.get("results", [])
        results: List[Dict[str, Any]] = []
        for item in raw_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:1000],
                }
            )

        logger.info("[MCP:TAVILY] Query='%s' → %d results", query, len(results))

        return {
            "query": query,
            "results": results,
        }

    except Exception as exc:
        logger.exception("[MCP:TAVILY] Tavily search failed: %s", exc)
        return {"error": "Tavily search failed."}


# -----------------------------------------------------------
# 2. Wikipedia Search Tool (Structured Knowledge)
# -----------------------------------------------------------
@mcp.tool()  # ✅ FIXED: Added parentheses
def wiki_search(query: str, limit: int = 3, ctx: Context = None) -> Dict[str, Any]:
    """
    Wikipedia search using the MediaWiki API (no key required).

    Args:
        query: Topic to search.
        limit: Number of wiki results (1–10 recommended).
        ctx: FastMCP context (auto-injected).

    Returns:
        dict: {
          "query": str,
          "results": [
            {
              "title": str,
              "url": str,
              "snippet": str,
            }, ...
          ]
        }
        or {"error": str} on failure.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Query cannot be empty."}

    if limit <= 0:
        limit = 3
    if limit > 10:
        limit = 10

    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }

    try:
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])

        formatted: List[Dict[str, Any]] = []
        for r in results:
            title = r.get("title", "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
            snippet = r.get("snippet", "")

            formatted.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        logger.info("[MCP:WIKI] Query='%s' → %d results", query, len(formatted))

        return {
            "query": query,
            "results": formatted,
        }

    except Exception as exc:
        logger.exception("[MCP:WIKI] Wikipedia search failed: %s", exc)
        return {"error": "Wikipedia search failed."}


# -----------------------------------------------------------
# 3. Webpage Fetcher (For direct URL scraping)
# -----------------------------------------------------------
@mcp.tool()  # ✅ FIXED: Added parentheses
def web_fetch(url: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Fetch raw HTML/text from a URL.

    Typical usage:
      - Tavily returns URLs → agent selects one → calls web_fetch
      - Agent wants deeper inspection of a single page

    Args:
        url: The URL to fetch.
        ctx: FastMCP context (auto-injected).

    Returns:
        dict: {
          "url": str,
          "content": str (truncated at 20k chars)
        }
        or {"error": str} on failure.
    """
    url = (url or "").strip()
    if not url:
        return {"error": "URL cannot be empty."}

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        content = resp.text[:20000]

        logger.info("[MCP:FETCH] Fetched %d chars from %s", len(content), url)

        return {
            "url": url,
            "content": content,
        }

    except Exception as exc:
        logger.exception("[MCP:FETCH] Failed to fetch URL %s: %s", url, exc)
        return {"error": "Failed to fetch URL."}


# -----------------------------------------------------------
# Run MCP Server
# -----------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[MCP] Math Agent Search MCP Server is running...")
    print("[MCP] Available tools:")
    print("  • tavily_search - Web search via Tavily API")
    print("  • wiki_search - Wikipedia knowledge base search")
    print("  • web_fetch - Fetch full webpage content")
    mcp.run()