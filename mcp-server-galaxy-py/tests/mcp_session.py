"""Drive the MCP tools over a real client session, the way every real caller does.

The live integration tests used to invoke the tool functions directly, with no MCP
request context. That matters more than it looks: without a context there is no session
id, so `connect()` has nowhere to put the validated client and every later call trips
`ensure_connected()`. The whole suite failed that way, and the shape of the failure
invited the wrong conclusion -- that `connect()` was broken -- when in fact no real
client is ever in that state. FastMCP hands stdio, SSE and in-memory transports a
generated session id, and HTTP carries one in a header.

So the tests drive an in-memory FastMCP client instead. One connection, one stable
session id, the same code path a packaged client takes, and the session-scoped
connection store doing the job it was designed for.

`call()` keeps the old ergonomics on purpose: it binds positional arguments against the
tool's real signature and rebuilds the `GalaxyResult`, so assertions written against
`result.data` / `.success` / `.count` did not have to change.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastmcp import Client

from galaxy_mcp import server as server_module
from galaxy_mcp.server import GalaxyResult, mcp

from .test_helpers import get_function


class ToolCallError(RuntimeError):
    """A tool raised on the server side.

    The server raises ValueError; FastMCP turns that into a protocol-level error, so the
    original type is gone by the time it reaches a client. Tests that want to assert on
    a failure match on the message, which does survive.
    """


class LiveMCPSession:
    """A connected in-memory MCP client, usable from synchronous tests.

    One event loop and one client for the whole session so the session id -- and with it
    the session-scoped Galaxy connection -- stays stable across calls.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Client | None = None

    def __enter__(self) -> LiveMCPSession:
        self._loop = asyncio.new_event_loop()
        self._client = Client(mcp)
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *exc_info: Any) -> None:
        assert self._loop is not None
        try:
            if self._client is not None:
                self._loop.run_until_complete(self._client.__aexit__(*exc_info))
        finally:
            self._loop.close()
            self._loop = None
            self._client = None

    def _require_open(self) -> tuple[asyncio.AbstractEventLoop, Client]:
        if self._loop is None or self._client is None:
            raise RuntimeError("LiveMCPSession is not open -- use it as a context manager")
        return self._loop, self._client

    def call(self, tool_name: str, *args: Any, **kwargs: Any) -> GalaxyResult:
        """Invoke a tool by name, binding positional args against its real signature."""
        loop, client = self._require_open()

        fn = get_function(getattr(server_module, tool_name))
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        arguments = dict(bound.arguments)

        try:
            result = loop.run_until_complete(client.call_tool(tool_name, arguments))
        except Exception as exc:  # the server-side failure, flattened by the protocol
            raise ToolCallError(str(exc)) from exc

        payload = result.structured_content
        if not isinstance(payload, dict):
            raise ToolCallError(f"{tool_name} returned no structured content: {result.content!r}")
        return GalaxyResult(**payload)
