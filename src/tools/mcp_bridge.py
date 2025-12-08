# src/tools/mcp_bridge.py
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from src.tools.mcp_clients import get_mcp_client

# Force UTF-8 stdout on Windows to avoid 'charmap' codec errors
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: mcp_bridge TOOL_NAME PAYLOAD_JSON"}))
        sys.exit(1)

    tool_name = sys.argv[1]
    try:
        payload = json.loads(sys.argv[2])
    except Exception:
        print(json.dumps({"error": "Invalid JSON payload"}))
        sys.exit(1)

    mcp = get_mcp_client()
    await mcp.initialize()

    try:
        out: Any = await mcp.call(tool_name, **payload)
        # Only JSON to stdout, logs go to stderr via logging
        print(json.dumps(out, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": f"Tool call failed: {exc}"}))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
