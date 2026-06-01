"""Tests for ToolVisibilityMiddleware."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import MiddlewareContext

from galaxy_mcp.middleware import ToolVisibilityMiddleware


def _tool(name: str, tags: set[str]) -> SimpleNamespace:
    return SimpleNamespace(name=name, tags=tags)


def _session(*, is_admin: bool = False, user_tools: bool = False) -> dict:
    gi = Mock()
    gi.users.get_current_user.return_value = {"id": "u1", "is_admin": is_admin}
    gi.config.get_config.return_value = {"enable_unprivileged_tools": user_tools}
    return {"gi": gi, "url": "https://galaxy.test/", "api_key": "k"}


def _run_list_tools(middleware: ToolVisibilityMiddleware, tools: list) -> list:
    async def call_next(_ctx):
        return tools

    ctx = MiddlewareContext(message=None, method="tools/list")
    return asyncio.run(middleware.on_list_tools(ctx, call_next))


def _run_call_tool(middleware: ToolVisibilityMiddleware, tool, *, name: str | None = None):
    server = Mock()

    async def get_tool(_name):
        return tool

    server.get_tool.side_effect = get_tool
    fastmcp_context = SimpleNamespace(fastmcp=server)
    ctx = MiddlewareContext(
        message=SimpleNamespace(name=name or getattr(tool, "name", "t")),
        method="tools/call",
        fastmcp_context=fastmcp_context,
    )

    async def call_next(_ctx):
        return "ok"

    return asyncio.run(middleware.on_call_tool(ctx, call_next))


class TestAdminGating:
    def test_admin_tools_hidden_for_non_admin(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(is_admin=False))
        tools = [
            _tool("get_users", {"admin", "read", "niche"}),
            _tool("get_histories", {"histories", "read", "core"}),
        ]
        visible = _run_list_tools(mw, tools)
        assert [t.name for t in visible] == ["get_histories"]

    def test_admin_tools_visible_for_admin(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(is_admin=True))
        tools = [
            _tool("get_users", {"admin", "read", "niche"}),
            _tool("get_histories", {"histories", "read", "core"}),
        ]
        visible = _run_list_tools(mw, tools)
        assert {t.name for t in visible} == {"get_users", "get_histories"}


class TestUserToolsGating:
    def test_user_tools_hidden_when_galaxy_lacks_capability(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(user_tools=False))
        tools = [
            _tool("create_user_tool", {"user_tools", "write", "niche"}),
            _tool("get_histories", {"histories", "read", "core"}),
        ]
        visible = _run_list_tools(mw, tools)
        assert [t.name for t in visible] == ["get_histories"]

    def test_user_tools_visible_when_capability_advertised(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(user_tools=True))
        tools = [_tool("create_user_tool", {"user_tools", "write", "niche"})]
        visible = _run_list_tools(mw, tools)
        assert [t.name for t in visible] == ["create_user_tool"]


class TestStaticTagFilters:
    def test_exclude_tags_drops_matching_tools(self):
        mw = ToolVisibilityMiddleware(
            get_session_state=lambda: _session(is_admin=True, user_tools=True),
            exclude_tags={"niche"},
        )
        tools = [
            _tool("a", {"histories", "read", "core"}),
            _tool("b", {"iwc", "read", "niche"}),
        ]
        visible = _run_list_tools(mw, tools)
        assert [t.name for t in visible] == ["a"]

    def test_include_tags_keeps_only_matching_tools(self):
        mw = ToolVisibilityMiddleware(
            get_session_state=lambda: _session(is_admin=True, user_tools=True),
            include_tags={"core"},
        )
        tools = [
            _tool("a", {"histories", "read", "core"}),
            _tool("b", {"histories", "read", "extended"}),
        ]
        visible = _run_list_tools(mw, tools)
        assert [t.name for t in visible] == ["a"]


class TestOnCallToolGuard:
    def test_hidden_tool_raises(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(is_admin=False))
        hidden = _tool("get_users", {"admin", "read", "niche"})
        with pytest.raises(ToolError, match="not available"):
            _run_call_tool(mw, hidden, name="get_users")

    def test_visible_tool_passes_through(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(is_admin=False))
        visible = _tool("get_histories", {"histories", "read", "core"})
        result = _run_call_tool(mw, visible, name="get_histories")
        assert result == "ok"

    def test_unknown_tool_is_not_blocked(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(is_admin=False))
        # When the tool can't be resolved, the middleware should defer to downstream handling.
        result = _run_call_tool(mw, None, name="missing")
        assert result == "ok"

    def test_tool_lookup_exception_defers(self):
        mw = ToolVisibilityMiddleware(get_session_state=lambda: _session(is_admin=False))
        server = Mock()

        async def boom(_name):
            raise RuntimeError("lookup failed")

        server.get_tool.side_effect = boom
        fastmcp_context = SimpleNamespace(fastmcp=server)
        ctx = MiddlewareContext(
            message=SimpleNamespace(name="anything"),
            method="tools/call",
            fastmcp_context=fastmcp_context,
        )

        async def call_next(_ctx):
            return "ok"

        result = asyncio.run(mw.on_call_tool(ctx, call_next))
        assert result == "ok"


class TestCaching:
    def test_admin_probe_cached_per_session(self):
        session = _session(is_admin=True)
        mw = ToolVisibilityMiddleware(get_session_state=lambda: session)
        tools = [_tool("t", {"admin", "read"})]
        _run_list_tools(mw, tools)
        _run_list_tools(mw, tools)
        assert session["gi"].users.get_current_user.call_count == 1

    def test_user_tools_probe_cached_per_url(self):
        session = _session(user_tools=True)
        mw = ToolVisibilityMiddleware(get_session_state=lambda: session)
        tools = [_tool("t", {"user_tools", "write"})]
        _run_list_tools(mw, tools)
        _run_list_tools(mw, tools)
        assert session["gi"].config.get_config.call_count == 1
