"""MCP protocol smoke test for vctx-mcp."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "server.py"


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            expected = {
                "vctx_buffer",
                "vctx_archive",
                "vctx_read",
                "vctx_search",
                "vctx_list",
                "vctx_index",
                "vctx_decay",
                "vctx_delete",
                "vctx_status",
            }
            missing = expected - names
            if missing:
                raise RuntimeError(f"Missing tools: {sorted(missing)}")

            archived = await session.call_tool(
                "vctx_archive",
                {
                    "title": "smoke-test",
                    "content": "VCTX smoke test memory: sqlite storage, MCP read, keyword search.",
                    "conclusion": "smoke test validates archive search read",
                    "keywords": ["VCTX", "sqlite", "MCP", "smoke"],
                    "session_id": "smoke-test",
                },
            )
            archived_text = archived.content[0].text
            block_id = json.loads(archived_text)["block_id"]

            search = await session.call_tool("vctx_search", {"query": "sqlite"})
            search_data = json.loads(search.content[0].text)
            if search_data["count"] < 1:
                raise RuntimeError("Search did not return the archived block")

            read = await session.call_tool("vctx_read", {"block_id": block_id})
            read_data = json.loads(read.content[0].text)
            if "sqlite storage" not in read_data["content"]:
                raise RuntimeError("Read returned unexpected content")

            status = await session.call_tool("vctx_status", {"session_id": "smoke-test"})
            json.loads(status.content[0].text)

            await session.call_tool("vctx_delete", {"block_id": block_id})

    print("vctx-mcp smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
