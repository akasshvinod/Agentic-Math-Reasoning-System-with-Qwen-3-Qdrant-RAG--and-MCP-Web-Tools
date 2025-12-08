from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

from src.graph.state import MathState

logger = logging.getLogger(__name__)

BRIDGE_MODULE = "src.tools.mcp_bridge"
SUBPROCESS_TIMEOUT = 60  # seconds
PROJECT_ROOT = r"C:\Users\91730\Math_Agent_Project"


def _run_bridge(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any] | str:
    """Call MCP via a separate Python process and return JSON (dict or raw str)."""
    cmd = [
        sys.executable,
        "-m",
        BRIDGE_MODULE,
        tool_name,
        json.dumps(payload),
    ]

    env = os.environ.copy    ()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("[MCP BRIDGE] %s timed out after %ss.", tool_name, SUBPROCESS_TIMEOUT)
        return {"error": "bridge_timeout"}

    elapsed = time.time() - start
    logger.info(
        "[MCP BRIDGE] %s finished in %.2fs (returncode=%s)",
        tool_name,
        elapsed,
        proc.returncode,
    )

    if proc.returncode != 0:
        logger.error("[MCP BRIDGE] %s failed: %s", tool_name, proc.stderr.strip())
        # proc.stdout may still contain JSON error from bridge
        out = proc.stdout or ""
        try:
            return json.loads(out) if out else {"error": "bridge_error"}
        except Exception:
            return out or {"error": "bridge_error"}

    # Successful exit: stdout should be JSON
    out = proc.stdout or ""
    try:
        return json.loads(out)
    except Exception as exc:
        logger.error("[MCP BRIDGE] Invalid JSON from %s: %s", tool_name, exc)
        return out or {"error": "invalid_json"}


async def mcp_search_node(state: MathState) -> MathState:
    """
    Node 5: External web search via MCP bridge (Tavily).

    Populates:
      state["web_results"] = [
        {"source": "tavily", "title": str | None, "url": str | None, "content": str},
        ...
      ]
    """
    query = (state.get("query") or "").strip()
    logger.info("[MCP SEARCH] Running external search for: %r", query)

    if not query:
        logger.warning("[MCP SEARCH] Empty query → skipping external search.")
        state["web_results"] = []
        return state

    aggregated_results: List[Dict[str, Any]] = []

    # Tavily via MCP bridge
    tavily_out = _run_bridge(
        "tavily_search",
        {"query": query, "n_results": 3},
    )

    # If bridge returned a raw JSON string, decode it
    if isinstance(tavily_out, str):
        try:
            tavily_out = json.loads(tavily_out)
        except json.JSONDecodeError:
            logger.error("[MCP SEARCH] Tavily bridge returned non-JSON string.")
            tavily_out = {}

    if isinstance(tavily_out, dict) and "results" in tavily_out:
        tavily_results = tavily_out.get("results") or []
        logger.info(
            "[MCP SEARCH] Tavily via MCP returned %d results.",
            len(tavily_results),
        )
        for item in tavily_results:
            aggregated_results.append(
                {
                    "source": "tavily",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content", ""),
                }
            )
    else:
        logger.warning(
            "[MCP SEARCH] Tavily via MCP returned no results: %s",
            tavily_out.get("error") if isinstance(tavily_out, dict) else tavily_out,
        )

    if not aggregated_results:
        logger.warning("[MCP SEARCH] No external results found.")
        state["web_results"] = []
    else:
        state["web_results"] = aggregated_results
        logger.info(
            "[MCP SEARCH] Total aggregated web results: %d",
            len(aggregated_results),
        )

    return state
