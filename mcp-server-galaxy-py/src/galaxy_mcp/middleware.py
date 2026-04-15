"""Session-aware tool visibility middleware for the Galaxy MCP server.

Filters which tools are exposed to each MCP client based on:

* Per-session Galaxy capabilities (admin privileges, unprivileged_tools
  support) -- hides tags the caller can't meaningfully use.
* Static include / exclude tag lists supplied at server startup.

Tools rely on `@mcp.tool(tags={...})` registrations added in Phase 1 of the
progressive tool discovery plan.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

SessionStateFn = Callable[[], dict[str, Any]]


class ToolVisibilityMiddleware(Middleware):
    """Filter tool listings and guard direct calls based on session capabilities."""

    def __init__(
        self,
        *,
        get_session_state: SessionStateFn,
        include_tags: set[str] | None = None,
        exclude_tags: set[str] | None = None,
    ) -> None:
        self._get_session_state = get_session_state
        self._include_tags = set(include_tags) if include_tags else None
        self._exclude_tags = set(exclude_tags) if exclude_tags else set()
        self._admin_cache: dict[tuple[str, str], bool] = {}
        self._user_tools_cache: dict[str, bool] = {}

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        tools = await call_next(context)
        hidden_tags = self._hidden_tags_for_session()
        return [tool for tool in tools if self._is_visible(tool, hidden_tags)]

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool = await self._resolve_tool(context)
        if tool is not None:
            hidden_tags = self._hidden_tags_for_session()
            if not self._is_visible(tool, hidden_tags):
                raise ToolError(f"Tool '{context.message.name}' is not available in this session")
        return await call_next(context)

    async def _resolve_tool(
        self, context: MiddlewareContext[mt.CallToolRequestParams]
    ) -> Tool | None:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None:
            return None
        server = getattr(fastmcp_context, "fastmcp", None)
        if server is None:
            return None
        try:
            return await server.get_tool(context.message.name)
        except Exception as exc:
            logger.debug("get_tool lookup failed for %s: %s", context.message.name, exc)
            return None

    def _is_visible(self, tool: Tool, hidden_tags: set[str]) -> bool:
        tags = set(tool.tags or ())
        if self._exclude_tags & tags:
            return False
        if hidden_tags & tags:
            return False
        if self._include_tags is None:
            return True
        return bool(self._include_tags & tags)

    def _hidden_tags_for_session(self) -> set[str]:
        hidden: set[str] = set()
        state: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            state = self._get_session_state()

        if not self._session_is_admin(state):
            hidden.add("admin")
        if not self._session_supports_user_tools(state):
            hidden.add("user_tools")
        return hidden

    def _session_is_admin(self, state: dict[str, Any]) -> bool:
        gi = state.get("gi")
        if gi is None:
            return False
        key = (str(state.get("url") or ""), str(state.get("api_key") or ""))
        if not all(key):
            return False
        cached = self._admin_cache.get(key)
        if cached is not None:
            return cached
        try:
            user = gi.users.get_current_user()
        except Exception as exc:
            logger.debug("Admin probe failed for %s: %s", key[0], exc)
            return False
        is_admin = bool(user.get("is_admin"))
        self._admin_cache[key] = is_admin
        return is_admin

    def _session_supports_user_tools(self, state: dict[str, Any]) -> bool:
        gi = state.get("gi")
        if gi is None:
            return False
        url = str(state.get("url") or "")
        if not url:
            return False
        cached = self._user_tools_cache.get(url)
        if cached is not None:
            return cached
        try:
            config = gi.config.get_config()
        except Exception as exc:
            logger.debug("Capability probe failed for %s: %s", url, exc)
            return False
        supported = bool(config.get("enable_unprivileged_tools"))
        self._user_tools_cache[url] = supported
        return supported
