# src/tools/mcp_client.py

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient  # [web:18]
from langchain_core.tools import BaseTool  # [web:25]

logger = logging.getLogger(__name__)

MCP_CALL_TIMEOUT = 15  # seconds


class MCPClientManager:
    """
    Manages a MultiServerMCPClient for calling MCP tools (tavily_search, web_fetch, etc.)
    from LangGraph nodes.

    - Starts one or more MCP servers (stdio transport).
    - Loads all MCP tools as LangChain BaseTool objects.
    - Provides async `.call(tool_name, **kwargs)` helper.
    """

    def __init__(self, server_config: Dict[str, Dict[str, Any]]) -> None:
        """
        server_config example (for your math_agent_tools server):

        {
            "search": {
                "command": "C:\\Users\\91730\\miniconda3\\envs\\faiss-env\\python.exe",
                "args": ["-m", "src.mcp.search_mcp"],
                "transport": "stdio",
            }
        }
        """
        self.server_config = server_config
        self.client: Optional[MultiServerMCPClient] = None
        self.tools_by_name: Dict[str, BaseTool] = {}

    async def initialize(self) -> None:
        """Initialize MCP client and load all tools from all servers."""
        logger.info("[MCP CLIENT] Initializing MultiServerMCPClient...")

        # Create client with your server definitions.
        self.client = MultiServerMCPClient(self.server_config)

        # Load tools from all servers (flattened list).
        tools: List[BaseTool] = await self.client.get_tools()
        self.tools_by_name = {tool.name: tool for tool in tools}

        logger.info(
            "[MCP CLIENT] Loaded %d tools: %s",
            len(self.tools_by_name),
            list(self.tools_by_name.keys()),
        )

    async def call(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Call a tool by name with keyword arguments.

        Example:
            await mcp_client.call("tavily_search", query="linear algebra", n_results=3)
        """
        if self.client is None or not self.tools_by_name:
            raise RuntimeError("MCPClientManager not initialized. Call await initialize().")

        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            raise ValueError(f"MCP tool not found: {tool_name}")

        logger.info("[MCP CLIENT] Calling tool '%s' with args=%s", tool_name, kwargs)

        try:
            # LangChain tools are async-callable via .ainvoke; .invoke works sync.
            result = await asyncio.wait_for(
                tool.ainvoke(kwargs),
                timeout=MCP_CALL_TIMEOUT,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "[MCP CLIENT] Tool '%s' timed out after %ss.",
                tool_name,
                MCP_CALL_TIMEOUT,
            )
            raise


# Singleton instance ---------------------------------------------------------

_mcp_instance: Optional[MCPClientManager] = None


def get_mcp_client() -> MCPClientManager:
    global _mcp_instance
    if _mcp_instance is None:
        _mcp_instance = MCPClientManager(
            server_config={
                "search": {
                    "command": r"C:\Users\91730\Math_Agent_Project\run_math_mcp.bat",
                    "args": [],  # batch handles python + module
                    "transport": "stdio",
                }
            }
        )
    return _mcp_instance


# Manual test ----------------------------------------------------------------

if __name__ == "__main__":

    async def _test() -> None:
        logging.basicConfig(level=logging.INFO)

        mcp = get_mcp_client()
        await mcp.initialize()

        try:
            out = await mcp.call(
                "tavily_search",
                query="solve d/dx(x^3)",
                n_results=2,
            )
            print("\n[TEST] tavily_search via MCP:\n", out)
        except asyncio.TimeoutError:
            print("[ERROR] tavily_search via MCP timed out.")

    asyncio.run(_test())
