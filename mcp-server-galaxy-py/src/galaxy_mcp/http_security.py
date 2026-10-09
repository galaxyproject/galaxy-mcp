"""Guards for the HTTP transports.

Over stdio the caller is whoever launched the process, so there is nothing to defend.
Over HTTP anything that can reach the socket is a caller -- other hosts on the network,
and any web page open in a browser on the same machine. These helpers cover the pieces
of that gap the server can close on its own: which Host/Origin headers it answers,
whether it will start unauthenticated on a routable address, and whether tools may touch
the server's filesystem.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

from fastmcp.server.dependencies import get_http_request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse, Response

ALLOW_LOCAL_FILES_ENV = "GALAXY_MCP_ALLOW_LOCAL_FILES"
ALLOW_UNAUTHENTICATED_ENV = "GALAXY_MCP_ALLOW_UNAUTHENTICATED"
ALLOWED_HOSTS_ENV = "GALAXY_MCP_ALLOWED_HOSTS"
ALLOWED_ORIGINS_ENV = "GALAXY_MCP_ALLOWED_ORIGINS"

DEFAULT_HTTP_HOST = "127.0.0.1"

_TRUTHY = {"1", "true", "yes", "on"}


class HTTPStartupError(ValueError):
    """The requested HTTP configuration is not safe to serve."""


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def env_list(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def is_loopback_host(host: str) -> bool:
    candidate = host.strip().strip("[]").rstrip(".").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def in_http_request() -> bool:
    """True when the current tool call arrived over an HTTP transport."""
    try:
        get_http_request()
    except RuntimeError:
        return False
    return True


_serving_http = False


def mark_serving_http() -> None:
    """Record that this process serves an HTTP transport.

    in_http_request() can only see a request whose context reached the current call, and a
    call that runs somewhere the context did not follow would otherwise pass for the
    operator's. Once a process serves HTTP, no call is treated as the operator's.
    """
    global _serving_http
    _serving_http = True


def remote_caller_possible() -> bool:
    """False only when this call can come from nobody but the operator, over stdio."""
    return _serving_http or in_http_request()


def local_files_allowed() -> bool:
    """Whether tools may read or write paths on the server's filesystem."""
    return env_flag(ALLOW_LOCAL_FILES_ENV) or not remote_caller_possible()


def require_local_files(action: str, alternative: str) -> None:
    if local_files_allowed():
        return
    raise ValueError(
        f"{action} is disabled over HTTP transports, because the path would refer to the "
        f"MCP server's filesystem rather than yours. {alternative} Operators of a trusted "
        f"single-user deployment can re-enable it with {ALLOW_LOCAL_FILES_ENV}=1."
    )


def check_http_startup(host: str, *, auth_enabled: bool, allow_unauthenticated: bool) -> None:
    """Refuse to expose an unauthenticated server beyond loopback by accident."""
    if auth_enabled or is_loopback_host(host) or allow_unauthenticated:
        return
    raise HTTPStartupError(
        f"Refusing to serve HTTP on '{host}' without authentication: anyone who can reach "
        "that address could use every tool, with any Galaxy credentials configured in this "
        "environment. Bind to 127.0.0.1, configure OAuth (GALAXY_MCP_PUBLIC_URL and "
        "GALAXY_MCP_SESSION_SECRET), or pass --allow-unauthenticated "
        f"({ALLOW_UNAUTHENTICATED_ENV}=1) if something else controls access to the listener. "
        f"Clients that reach it by a non-loopback name also need {ALLOWED_HOSTS_ENV}."
    )


def _hostname(value: str) -> str:
    # urlsplit copes with ports and bracketed IPv6 literals, which a split(":") would not
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


class HTTPSecurityMiddleware(BaseHTTPMiddleware):
    """Host and Origin checks for the HTTP transports.

    With OAuth every request carries a bearer token that a hostile web page does not have,
    so browser clients from any origin are answered as before. Without it the token is
    gone and these headers are the only thing between a web page and the tools, so the
    Host must be one we expect (defeats DNS rebinding) and cross-origin access is limited
    to pages served from this machine plus an explicit allowlist.
    """

    def __init__(
        self,
        app,
        *,
        auth_enabled: bool,
        allowed_hosts: set[str] | None = None,
        allowed_origins: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._auth_enabled = auth_enabled
        self._allowed_hosts = {h.lower() for h in allowed_hosts or set()}
        self._allowed_origins = {o.lower().rstrip("/") for o in allowed_origins or set()}

    def _host_allowed(self, host_header: str) -> bool:
        if self._auth_enabled or "*" in self._allowed_hosts:
            return True
        hostname = _hostname(f"//{host_header}")
        return is_loopback_host(hostname) or hostname in self._allowed_hosts

    def _origin_allowed(self, origin: str) -> bool:
        if "*" in self._allowed_origins:
            return True
        if self._auth_enabled and not self._allowed_origins:
            return True
        normalized = origin.lower().rstrip("/")
        if normalized in self._allowed_origins:
            return True
        # Pages served from this machine (MCP Inspector and friends) are the user's own
        return is_loopback_host(_hostname(normalized))

    async def dispatch(self, request, call_next):
        if not self._host_allowed(request.headers.get("host", "")):
            return PlainTextResponse("Invalid Host header", status_code=421)

        origin = request.headers.get("origin")
        if origin is None:
            return await call_next(request)

        if not self._origin_allowed(origin):
            return PlainTextResponse("Origin not allowed", status_code=403)

        cors_headers = {
            "access-control-allow-origin": origin,
            "access-control-allow-methods": request.headers.get(
                "access-control-request-method", "POST,GET,OPTIONS"
            ),
            "access-control-allow-headers": request.headers.get(
                "access-control-request-headers", "authorization,content-type"
            ),
            "access-control-max-age": "600",
            "vary": "Origin",
        }

        # Preflights carry no credentials, so they have to be answered ahead of FastMCP auth
        if request.method.upper() == "OPTIONS":
            return Response(status_code=204, headers=cors_headers)

        response = await call_next(request)
        for header, value in cors_headers.items():
            response.headers.setdefault(header, value)
        return response
