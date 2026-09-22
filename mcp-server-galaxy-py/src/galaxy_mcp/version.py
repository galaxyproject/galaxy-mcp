"""What Galaxy's ``/api/version`` says, and the minimum a tool is allowed to declare.

``version_major`` is the only field worth comparing. Galaxy sets it to ``YY.M`` --
``"26.0"``, ``"26.1"``, ``"26.2"`` on a dev checkout -- and keeps the patch level in
``version_minor``, so two integers are the whole ordering: no semver dependency here and
nothing to keep in step with one.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Lenient on the tail, because a caller who states a version reaches for the release tag:
# "26.1.1" and "26.2.dev0" mean the same servers as "26.1" and "26.2".
_OBSERVED = re.compile(r"^(\d+)\.(\d+)(?:[.\-+].*)?$")
# Strict, because a tool's declaration is ours to write: ">=MAJOR.MINOR" and nothing else.
_REQUIREMENT = re.compile(r"^>=\s*(\d+)\.(\d+)$")


@dataclass(frozen=True)
class GalaxyVersion:
    """A version a server reported, kept with the text it reported so a refusal can quote it."""

    raw: str
    major: int
    minor: int


@dataclass(frozen=True)
class VersionRequirement:
    """The lower bound a tool declares."""

    major: int
    minor: int


@dataclass(frozen=True)
class GalaxyVersionReport:
    """Everything one ``/api/version`` lookup produced.

    The guard and ``get_server_info`` both read this, through the same lookup and the same
    entry, so neither goes off and asks on its own account. That is not quite the same as
    never differing: a reader that arrives while a refresh is in flight gets the previous
    answer, which is the snapshot the refresh is in the middle of replacing.
    """

    version: GalaxyVersion | None
    payload: dict[str, Any]


class GalaxyVersionError(ValueError):
    """Raised instead of running a tool the connected Galaxy is too old for.

    A ``ValueError`` like every other tool failure, so nothing that already handles one has
    to learn a new type, but distinct enough to test for and to recognize in a log.
    """


def parse_version(raw: Any) -> GalaxyVersion | None:
    """Read a version a server reported.

    Anything that does not start with two numbers is unknown rather than an error: an
    unreadable version has to refuse nothing.
    """
    if not isinstance(raw, str):
        return None
    match = _OBSERVED.match(raw.strip())
    if not match:
        return None
    return GalaxyVersion(raw=raw.strip(), major=int(match.group(1)), minor=int(match.group(2)))


def parse_requirement(spec: str) -> VersionRequirement:
    """Read a tool's declared minimum.

    The grammar is ``>=MAJOR.MINOR`` and nothing else: that is the shape ``version_major``
    comes in, so a third component could never be answered, and no tool has yet needed an
    upper bound. A malformed one raises where the tool is declared -- at import -- rather
    than at the one call it would have been asked to judge.
    """
    match = _REQUIREMENT.match(spec.strip()) if isinstance(spec, str) else None
    if not match:
        raise ValueError(f"Galaxy requirement {spec!r} is not of the form '>=MAJOR.MINOR'")
    return VersionRequirement(major=int(match.group(1)), minor=int(match.group(2)))


def satisfies(version: GalaxyVersion, spec: str) -> bool:
    """Whether a known server version meets a requirement."""
    want = parse_requirement(spec)
    if version.major != want.major:
        return version.major > want.major
    return version.minor >= want.minor


def requirement_sentence(spec: str) -> str:
    """The sentence a tool description carries, built from the spec the guard compares.

    One source for both, so the prose and the bound cannot drift apart.
    """
    want = parse_requirement(spec)
    return f"Requires Galaxy {want.major}.{want.minor} or newer."


# Tool name -> the minimum it declares. Filled at import by the decorator in server.py, and
# read by the guard, by get_server_info and by the surface manifest.
TOOL_REQUIREMENTS: dict[str, str] = {}


def record_requirement(tool_name: str, spec: str) -> None:
    """Remember what a tool needs, rejecting a spec nothing could ever satisfy."""
    parse_requirement(spec)
    TOOL_REQUIREMENTS[tool_name] = spec


def unsupported_tools(version: GalaxyVersion | None) -> list[dict[str, str]]:
    """The declared tools a server of this version cannot run.

    Empty for an unknown version because nothing is known, not because everything is
    supported -- which is also exactly what the guard will do with it.
    """
    if version is None:
        return []
    return [
        {"name": name, "requires": spec}
        for name, spec in sorted(TOOL_REQUIREMENTS.items())
        if not satisfies(version, spec)
    ]


@dataclass
class _Server:
    """One server's identity: the lock callers queue on, and what it last answered.

    Lock and answer live on the same object, and everything a lookup does it does to the
    object it is holding rather than to the dictionary by key. That is what makes eviction
    harmless: a fetch still in flight when its server is dropped writes into an entry
    nobody can reach any more, instead of landing on top of whatever a later caller has
    since stored under the same name.
    """

    lock: threading.Lock
    report: GalaxyVersionReport | None = None


_servers: dict[str, _Server] = {}
_servers_lock = threading.Lock()
# A remembered server costs a few dozen bytes, but in a deployment where callers bring
# their own Galaxy URL the set of keys is theirs to grow, so it is capped like the session
# connection store. Entries are counted, not answers, so a run of servers whose version
# endpoint fails is bounded too. Oldest first: a server nobody has asked about in a while
# is the one worth re-asking anyway.
MAX_REMEMBERED_SERVERS = 128


def _read_version(fetch: Callable[[], Any]) -> GalaxyVersionReport:
    payload = fetch()
    if not isinstance(payload, dict):
        payload = {}
    return GalaxyVersionReport(version=parse_version(payload.get("version_major")), payload=payload)


def _server_for(key: str) -> _Server:
    """The one entry that stands for this server, made on first sight."""
    with _servers_lock:
        entry = _servers.get(key)
        if entry is None:
            entry = _servers[key] = _Server(lock=threading.Lock())
            while len(_servers) > MAX_REMEMBERED_SERVERS:
                _servers.pop(next(iter(_servers)))
        return entry


def lookup_version(
    key: str, fetch: Callable[[], Any], *, refresh: bool = False
) -> GalaxyVersionReport:
    """Ask a server its version once and hand every later caller the same answer.

    Only an answer is remembered. A lookup that raised is not remembered, because one 502
    from a restarting Galaxy would otherwise leave every requirement unchecked for the life
    of the process.

    Asking and remembering are one step, under the server's own lock held for the whole of
    it, or a slow answer can land on top of a newer one: a lookup that fetched 26.0 and then
    waited would otherwise overwrite the 26.1 a refresh had since stored, and every later
    call would be judged against a version the server had already left behind. The lock
    belongs to the server rather than to all of them together, because it is held across a
    network call and a Galaxy that is slow to answer has no business delaying anyone else's
    first request.

    ``refresh`` asks the server again and replaces what was remembered, which is how a
    Galaxy upgraded underneath a long-running process stops being judged on its old
    version. A refresh that fails leaves the previous answer alone: a server that has gone
    quiet is not evidence that it changed. A caller with no key is answered without
    anything being remembered.

    The key is bioblend's ``base_url``, byte for byte, and nothing here tidies it up. Two
    ways of writing one address therefore cost two entries and one extra request, which is
    the price of never letting two addresses cost one answer: a path, a query, a trailing
    slash or an empty ``?`` can all be what a gateway routes on, so there is nothing in a
    URL we can safely decide is decoration.
    """
    if not key:
        return _read_version(fetch)

    entry = _server_for(key)
    if not refresh and entry.report is not None:
        return entry.report

    with entry.lock:
        if not refresh and entry.report is not None:
            # Somebody else asked while this call was waiting for the lock.
            return entry.report
        report = _read_version(fetch)
        entry.report = report
        return report


def clear_version_cache() -> None:
    """Forget every remembered version. For tests, and for a server that was reconfigured."""
    with _servers_lock:
        _servers.clear()
