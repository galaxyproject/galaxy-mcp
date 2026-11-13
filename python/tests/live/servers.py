"""Each MCP server under test, as a real stdio subprocess behind a synchronous client.

The environment is built from nothing on purpose: PATH, HOME, GALAXY_URL and
GALAXY_API_KEY, so a developer's discovery-mode or tag-filter settings cannot change the
surface under test. The cwd is a temp dir outside the repo because the Python server
loads the first .env it finds walking up from its cwd.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.types import CallToolResult

from .loader import REPO

TS_PACKAGES = REPO / "typescript" / "packages"
TS_ENTRY = TS_PACKAGES / "galaxy-mcp" / "dist" / "index.js"
TS_BUILT_PACKAGES = ("galaxy-ops", "galaxy-mcp")
CMD_OVERRIDE_ENV = {"python": "GALAXY_LIVE_PYTHON_CMD", "typescript": "GALAXY_LIVE_TS_CMD"}
CALL_TIMEOUT = 300.0


@dataclass(frozen=True)
class LaunchSpec:
    server: str
    command: list[str]
    overridden: bool


def launch_spec(server: str, env: dict[str, str | None]) -> LaunchSpec:
    override = env.get(CMD_OVERRIDE_ENV[server])
    if override:
        return LaunchSpec(server, shlex.split(override), True)
    if server == "python":
        return LaunchSpec(server, [sys.executable, "-m", "galaxy_mcp"], False)
    return LaunchSpec(server, ["node", str(TS_ENTRY)], False)


def _newest(paths: list[Path]) -> float:
    return max((p.stat().st_mtime for p in paths if p.is_file()), default=0.0)


def ts_build_problem() -> str | None:
    """Why the TS dist cannot be trusted to be the source under test, or None."""
    if not TS_ENTRY.is_file():
        return f"{TS_ENTRY.relative_to(REPO)} does not exist"
    for name in TS_BUILT_PACKAGES:
        pkg = TS_PACKAGES / name
        src = _newest(list((pkg / "src").rglob("*")))
        dist = _newest(list((pkg / "dist").rglob("*")))
        if dist < src:
            return f"typescript/packages/{name}/dist is older than its src"
    return None


class ServerDiedError(Exception):
    pass


class StdioSession:
    """One server subprocess, one client session, one event loop.

    The loop and client live as long as the session so the server keeps its connection
    state between calls, the way a real client's would.
    """

    def __init__(
        self,
        spec: LaunchSpec,
        galaxy_url: str,
        api_key: str,
        path_env: str,
        home: str,
        log_dir: Path,
    ) -> None:
        self.spec = spec
        self.env = {
            "PATH": path_env,
            "HOME": home,
            "GALAXY_URL": galaxy_url,
            "GALAXY_API_KEY": api_key,
        }
        self.log_path = log_dir / f"{spec.server}-server.log"
        self.cwd = tempfile.mkdtemp(prefix=f"mcp-live-{spec.server}-cwd-")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Client | None = None
        self._transport: StdioTransport | None = None
        self._log: IO[str] | None = None
        self.starts = 0

    @property
    def server(self) -> str:
        return self.spec.server

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._log = self.log_path.open("a")
        self._log.write(f"\n=== start {self.starts + 1}: {shlex.join(self.spec.command)}\n")
        self._log.flush()
        # keep_alive=False: leaving the client must stop the subprocess, or a restart
        # would leave the old server running until garbage collection.
        self._transport = StdioTransport(
            self.spec.command[0],
            self.spec.command[1:],
            env=self.env,
            cwd=self.cwd,
            log_file=self._log,
            keep_alive=False,
        )
        self._client = Client(self._transport, timeout=CALL_TIMEOUT, init_timeout=60)
        try:
            self._loop.run_until_complete(self._client.__aenter__())
        except Exception as exc:
            self.close()
            raise ServerDiedError(
                f"{self.server} server did not start ({exc}); its stderr is in {self.log_path}"
            ) from exc
        self.starts += 1

    def close(self) -> None:
        try:
            # A dead process cannot be shut down cleanly; what matters is that it is gone.
            if self._loop is not None and self._client is not None:
                with contextlib.suppress(Exception):
                    self._loop.run_until_complete(self._client.__aexit__(None, None, None))
            if self._loop is not None and self._transport is not None:
                with contextlib.suppress(Exception):
                    self._loop.run_until_complete(self._transport.close())
        finally:
            if self._loop is not None:
                self._loop.close()
            if self._log is not None:
                self._log.close()
            self._loop = None
            self._client = None
            self._transport = None
            self._log = None

    def ensure_alive(self) -> None:
        if self._client is None or self._loop is None or not self._client.is_connected():
            self.close()
            self.start()

    def _run(self, coro: Any) -> Any:
        if self._loop is None or self._client is None:
            raise ServerDiedError(f"{self.server} server is not running")
        return self._loop.run_until_complete(coro)

    def list_tools(self) -> list[str]:
        assert self._client is not None
        tools = self._run(self._client.list_tools())
        return sorted(t.name for t in tools)

    def call(
        self, tool: str, args: dict[str, Any], timeout: float = CALL_TIMEOUT
    ) -> CallToolResult:
        assert self._client is not None
        try:
            result = self._run(self._client.call_tool_mcp(tool, args, timeout=timeout))
        except Exception as exc:
            if self._client is None or not self._client.is_connected():
                self.close()
                raise ServerDiedError(
                    f"{self.server} server died during {tool} ({exc}); stderr in {self.log_path}"
                ) from exc
            raise
        assert isinstance(result, CallToolResult)
        return result


class ServerPool:
    """One StdioSession per server for the whole pytest session, started on first use."""

    def __init__(
        self,
        env: dict[str, str | None],
        galaxy_url: str,
        api_key: str,
        path_env: str,
        home: str,
        log_dir: Path,
    ) -> None:
        self._env = env
        self._args = (galaxy_url, api_key, path_env, home, log_dir)
        self._sessions: dict[str, StdioSession] = {}

    def get(self, server: str) -> StdioSession:
        if server not in self._sessions:
            session = StdioSession(launch_spec(server, self._env), *self._args)
            session.start()
            self._sessions[server] = session
        return self._sessions[server]

    def dedicated(self, server: str) -> StdioSession:
        """A session of its own for one scenario, not started and not kept.

        For a scenario with `server_env`: the runner adds the resolved variables to its
        environment once the fixtures exist, starts it, and closes it when the scenario ends.
        """
        return StdioSession(launch_spec(server, self._env), *self._args)

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
