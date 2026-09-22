"""A tool that declares a minimum Galaxy refuses an older one before it sends anything.

These tests drive a real bioblend client against `responses` rather than a mock, because
the thing worth proving is that no request leaves for the endpoint the tool would have
called -- which a mocked transport cannot show.
"""

import ast
import asyncio
import re
import threading
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import responses
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp import server
from galaxy_mcp.server import galaxy_state
from galaxy_mcp.version import (
    TOOL_REQUIREMENTS,
    GalaxyVersionError,
    _servers,
    parse_version,
    unsupported_tools,
)
from tests.mcp_session import LiveMCPSession, ToolCallError
from tests.test_helpers import create_page_fn, get_page_fn, get_server_info_fn, list_pages_fn

GALAXY_BASE_URL = "http://version-test.invalid"
OTHER_BASE_URL = "http://other-galaxy.invalid"

# Enough arguments to reach each gated tool's body, so a missing guard shows up as a
# request rather than as a TypeError.
GATED_CALLS = {
    "list_pages": (),
    "create_page": (),
    "update_page": ("page1",),
    "list_page_revisions": ("page1",),
    "get_page_revision": ("page1", "rev1"),
    "revert_page_revision": ("page1", "rev1"),
}


def _connect(base_url=GALAXY_BASE_URL):
    """Point the server at a Galaxy that only exists inside `responses`."""
    gi = GalaxyInstance(url=base_url, key="notakey")
    galaxy_state.update({"connected": True, "gi": gi, "url": f"{base_url}/", "api_key": "notakey"})
    return gi


def _serves_version(version_major, base_url=GALAXY_BASE_URL, **kwargs):
    responses.add(
        responses.GET,
        f"{base_url}/api/version",
        json={"version_major": version_major, "version_minor": "1"},
        **kwargs,
    )


def _version_calls(base_url=GALAXY_BASE_URL):
    return [call for call in responses.calls if call.request.url.startswith(f"{base_url}/api/vers")]


class TestRefusal:
    @responses.activate
    def test_an_older_server_is_refused_without_a_request_going_out(self):
        _connect()
        _serves_version("26.0")

        with pytest.raises(GalaxyVersionError) as excinfo:
            list_pages_fn()

        # One call, and it is the version lookup: nothing was sent to /api/pages.
        assert len(responses.calls) == 1
        assert responses.calls[0].request.url.startswith(f"{GALAXY_BASE_URL}/api/version")
        message = str(excinfo.value)
        assert "list_pages" in message
        assert "26.1 or newer" in message
        assert "26.0" in message
        assert "Nothing was sent to Galaxy" in message

    @responses.activate
    @pytest.mark.parametrize("tool_name", sorted(GATED_CALLS))
    def test_every_gated_tool_refuses_rather_than_calling_galaxy(self, tool_name):
        """No tool declares a minimum and then goes around the check."""
        assert tool_name in TOOL_REQUIREMENTS, f"{tool_name} declares no requirement"
        _connect()
        _serves_version("26.0")

        with pytest.raises(GalaxyVersionError, match=tool_name):
            getattr(server, tool_name)(*GATED_CALLS[tool_name])

        assert [call.request.url for call in responses.calls] == [f"{GALAXY_BASE_URL}/api/version"]

    def test_the_gated_tools_are_exactly_the_ones_that_need_the_newer_pages_api(self):
        """26.0 answers get_page in full, so gating it would refuse something that works."""
        assert set(TOOL_REQUIREMENTS) == set(GATED_CALLS)
        assert "get_page" not in TOOL_REQUIREMENTS

    @responses.activate
    def test_the_refusal_survives_the_trip_over_mcp(self):
        """The guard is on the tool, so the protocol path gets it too."""
        _connect()
        _serves_version("26.0")

        with LiveMCPSession() as session:
            with pytest.raises(ToolCallError, match="26.1 or newer"):
                session.call("list_pages")

        assert not [call for call in responses.calls if "/api/pages" in call.request.url]


class TestAcceptance:
    @responses.activate
    @pytest.mark.parametrize("version_major", ["26.1", "26.1.1", "26.2", "27.0"])
    def test_an_equal_or_newer_server_runs_the_tool(self, version_major):
        _connect()
        _serves_version(version_major)
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/pages", json=[], status=200)

        assert list_pages_fn().success is True

    @responses.activate
    def test_a_server_that_will_not_say_refuses_nothing(self):
        """An unreadable version is not a reason to gate; the tool stands on its own."""
        _connect()
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/version", status=500)
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/pages", json=[], status=200)

        assert list_pages_fn().success is True

    @responses.activate
    def test_a_version_the_server_words_oddly_refuses_nothing(self):
        _connect()
        responses.add(
            responses.GET, f"{GALAXY_BASE_URL}/api/version", json={"version_major": "twenty-six"}
        )
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/pages", json=[], status=200)

        assert list_pages_fn().success is True

    @responses.activate
    def test_a_tool_with_no_requirement_never_asks_for_the_version(self):
        """The lookup only happens where it could change the answer."""
        _connect()
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/pages/page1", json={"id": "page1"})

        assert get_page_fn("page1").success is True
        assert _version_calls() == []


class TestTheLookup:
    @responses.activate
    def test_the_version_is_asked_for_once_and_remembered(self):
        _connect()
        _serves_version("26.0")

        for _ in range(3):
            with pytest.raises(GalaxyVersionError):
                list_pages_fn()

        assert len(_version_calls()) == 1

    @responses.activate
    def test_a_lookup_that_failed_is_asked_again(self):
        """A restarting Galaxy must not leave every requirement unchecked for good."""
        _connect()
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/version", status=502)
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/pages", json=[], status=200)
        _serves_version("26.0")

        assert list_pages_fn().success is True
        with pytest.raises(GalaxyVersionError):
            list_pages_fn()

        assert len(_version_calls()) == 2

    @responses.activate
    def test_two_mcp_sessions_on_two_galaxies_do_not_share_a_version(self):
        """The key comes from the connection this request resolved to, not from a global."""
        galaxy_state.update({"connected": False, "gi": None, "url": None})
        _serves_version("26.1", base_url=OTHER_BASE_URL)
        responses.add(responses.GET, f"{OTHER_BASE_URL}/api/pages", json=[], status=200)
        _serves_version("26.0")

        for session_id, base_url in (("newer", OTHER_BASE_URL), ("older", GALAXY_BASE_URL)):
            server._set_session_connection(
                session_id,
                url=f"{base_url}/",
                api_key="notakey",
                gi=GalaxyInstance(url=base_url, key="notakey"),
            )

        with patch.object(server, "_get_current_session_id", return_value="newer"):
            assert list_pages_fn().success is True
        with patch.object(server, "_get_current_session_id", return_value="older"):
            with pytest.raises(GalaxyVersionError, match="26.0"):
                list_pages_fn()

    @responses.activate
    def test_one_galaxy_reached_two_ways_shares_its_version(self):
        """A trailing slash is not a different server, and should not cost a second lookup."""
        _connect(GALAXY_BASE_URL)
        _serves_version("26.0")

        with pytest.raises(GalaxyVersionError):
            list_pages_fn()
        galaxy_state["url"] = GALAXY_BASE_URL
        with pytest.raises(GalaxyVersionError):
            list_pages_fn()

        assert len(_version_calls()) == 1

    @responses.activate
    def test_two_galaxies_do_not_share_a_version(self):
        _connect(OTHER_BASE_URL)
        _serves_version("26.1", base_url=OTHER_BASE_URL)
        responses.add(responses.GET, f"{OTHER_BASE_URL}/api/pages", json=[], status=200)
        assert list_pages_fn().success is True

        _connect(GALAXY_BASE_URL)
        _serves_version("26.0")
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()


class TestDeclaration:
    def test_a_requirement_nothing_could_satisfy_is_rejected_where_it_is_written(self):
        """A typo fails at import rather than at the one call it would have judged."""
        with pytest.raises(ValueError, match=r">=MAJOR\.MINOR"):

            @server.requires_galaxy(">=26")
            def bogus() -> None: ...

    def test_the_description_says_what_the_guard_will_check(self):
        tools = {t.name: t for t in asyncio.run(server.mcp.list_tools(run_middleware=False))}
        for name in TOOL_REQUIREMENTS:
            assert "Requires Galaxy 26.1 or newer." in tools[name].description
        assert "Requires Galaxy" not in tools["get_page"].description

    def test_the_declared_tools_are_what_an_old_server_is_told_it_cannot_run(self):
        listed = unsupported_tools(parse_version("26.0"))
        assert [entry["name"] for entry in listed] == sorted(TOOL_REQUIREMENTS)
        assert all(entry["requires"] == ">=26.1" for entry in listed)
        assert unsupported_tools(parse_version("26.1")) == []


class TestServerInfo:
    def _serves_config(self, base_url=GALAXY_BASE_URL):
        responses.add(responses.GET, f"{base_url}/api/configuration", json={"brand": "Galaxy"})

    @responses.activate
    def test_it_names_what_this_galaxy_cannot_run(self):
        _connect()
        _serves_version("26.0")
        self._serves_config()

        data = get_server_info_fn().data

        assert data["version_known"] is True
        assert data["version"]["version_major"] == "26.0"
        assert [entry["name"] for entry in data["unsupported_tools"]] == sorted(TOOL_REQUIREMENTS)

    @responses.activate
    def test_a_newer_galaxy_is_told_it_can_run_everything(self):
        _connect()
        _serves_version("26.1")
        self._serves_config()

        data = get_server_info_fn().data

        assert data["version_known"] is True
        assert data["unsupported_tools"] == []

    @responses.activate
    def test_an_empty_list_from_an_unreadable_version_says_so(self):
        """Nothing is ruled out because nothing is known, which is not the same reassurance."""
        _connect()
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/version", json={"version_major": "?"})
        self._serves_config()

        data = get_server_info_fn().data

        assert data["version_known"] is False
        assert data["unsupported_tools"] == []

    @responses.activate
    def test_it_cannot_disagree_with_the_guard(self):
        """One lookup, one answer, whichever of the two asks first."""
        _connect()
        _serves_version("26.0")
        self._serves_config()

        reported = get_server_info_fn().data
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()

        assert [entry["name"] for entry in reported["unsupported_tools"]] == sorted(
            TOOL_REQUIREMENTS
        )
        # One lookup between them: the guard reuses what this tool just found.
        assert len(_version_calls()) == 1

    @responses.activate
    def test_it_asks_again_so_an_upgraded_galaxy_stops_being_refused(self):
        """Reporting the server is this tool's job, so a remembered answer will not do."""
        _connect()
        _serves_version("26.0")
        self._serves_config()
        self._serves_config()

        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()
        assert get_server_info_fn().data["version"]["version_major"] == "26.0"

        # The server is upgraded underneath a process that already remembered 26.0.
        _serves_version("26.1")
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/pages", json=[], status=200)

        assert get_server_info_fn().data["unsupported_tools"] == []
        # ...and what it learned is what the guard compares against from then on.
        assert list_pages_fn().success is True

    @responses.activate
    def test_a_refresh_that_fails_leaves_the_answer_it_had(self):
        """A server gone quiet is not evidence that it changed underneath us."""
        _connect()
        _serves_version("26.0")
        self._serves_config()
        responses.add(responses.GET, f"{GALAXY_BASE_URL}/api/version", status=502)

        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()
        with pytest.raises(ValueError, match="Failed to get server information"):
            get_server_info_fn()

        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()


@responses.activate
def test_what_fastmcp_registered_refuses_too():
    """Every path FastMCP dispatches -- the protocol, code mode, any transform -- ends here.

    The guard is applied under @mcp.tool, so the registered tool holds the wrapper rather
    than the bare function. If that ever inverts, the protocol keeps working and the check
    quietly stops happening, which is the failure this pins -- by calling what FastMCP
    holds and requiring it to refuse, rather than by looking for a wrapper.
    """
    _connect()
    _serves_version("26.0")
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools(run_middleware=False))}

    for name in TOOL_REQUIREMENTS:
        assert tools[name].fn is getattr(server, name)
        with pytest.raises(GalaxyVersionError, match=name):
            tools[name].fn(*GATED_CALLS[name])

    assert [call.request.url for call in responses.calls] == [f"{GALAXY_BASE_URL}/api/version"]


def _state(base_url, typed_url=None):
    """A connection state shaped like the one _get_request_connection_state returns."""
    return {
        "url": typed_url if typed_url is not None else f"{base_url}/",
        "api_key": "notakey",
        "gi": GalaxyInstance(url=base_url, key="notakey"),
        "connected": True,
        "source": "session",
        "session": None,
    }


class TestTheConnectionThatIsChecked:
    """A version checked against the server we had is no promise about the one we write to."""

    @responses.activate
    def test_the_server_that_gets_the_write_is_the_server_that_was_checked(self, monkeypatch):
        """A reconnect mid-call must not slip an unchecked Galaxy under the body."""
        _serves_version("26.1")
        _serves_version("26.0", base_url=OTHER_BASE_URL)
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/pages", json={"id": "on-checked"})
        responses.add(responses.POST, f"{OTHER_BASE_URL}/api/pages", json={"id": "on-switched"})

        states = [_state(GALAXY_BASE_URL), _state(OTHER_BASE_URL)]
        seen: list[int] = []

        def next_state():
            state = states[min(len(seen), 1)]
            seen.append(1)
            return state

        monkeypatch.setattr(server, "_get_request_connection_state", next_state)

        result = create_page_fn(title="t", slug="s")

        assert result.data["id"] == "on-checked"
        assert not [c for c in responses.calls if c.request.url.startswith(OTHER_BASE_URL)]

    @responses.activate
    def test_every_connection_a_body_obtains_is_checked(self, monkeypatch):
        """Not just the first: a body that asks twice is checked twice."""
        _serves_version("26.1")
        _serves_version("26.0", base_url=OTHER_BASE_URL)
        states = [_state(GALAXY_BASE_URL), _state(OTHER_BASE_URL)]
        seen: list[int] = []

        def next_state():
            state = states[min(len(seen), 1)]
            seen.append(1)
            return state

        monkeypatch.setattr(server, "_get_request_connection_state", next_state)

        @server.requires_galaxy(">=26.1")
        def probe() -> str:
            server.ensure_connected()
            server.ensure_connected()
            return "never reached"

        with pytest.raises(GalaxyVersionError, match="probe needs Galaxy 26.1"):
            probe()
        assert len(seen) == 2

    @responses.activate
    def test_a_requirement_does_not_reach_another_caller(self, monkeypatch):
        """FastMCP runs sync tools in worker threads; one call's gate is not another's."""
        _serves_version("26.0", base_url=OTHER_BASE_URL)
        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(OTHER_BASE_URL))
        inside = threading.Event()
        release = threading.Event()
        elsewhere: list[object] = []

        @server.requires_galaxy(">=26.1")
        def slow_gated_tool() -> None:
            inside.set()
            assert release.wait(5), "the other thread never finished"

        def unrelated_caller():
            try:
                elsewhere.append(server.ensure_connected()["url"])
            except BaseException as exc:  # noqa: BLE001 -- recorded and asserted on below
                elsewhere.append(exc)

        gated = threading.Thread(target=slow_gated_tool, daemon=True)
        gated.start()
        assert inside.wait(5), "the gated tool never started"

        other = threading.Thread(target=unrelated_caller, daemon=True)
        other.start()
        other.join(5)
        release.set()
        gated.join(5)

        assert elsewhere == [f"{OTHER_BASE_URL}/"]


class TestWhichServerAnAnswerBelongsTo:
    @responses.activate
    def test_one_typed_url_resolving_to_two_servers_does_not_share(self, monkeypatch):
        """A scheme-less url is two candidate servers; bioblend picks, and they differ."""
        _serves_version("26.1", base_url="https://ambiguous.invalid")
        _serves_version("26.0", base_url="http://ambiguous.invalid")
        responses.add(responses.GET, "https://ambiguous.invalid/api/pages", json=[])

        monkeypatch.setattr(
            server,
            "_get_request_connection_state",
            lambda: _state("https://ambiguous.invalid", typed_url="ambiguous.invalid"),
        )
        assert list_pages_fn().success is True

        monkeypatch.setattr(
            server,
            "_get_request_connection_state",
            lambda: _state("http://ambiguous.invalid", typed_url="ambiguous.invalid"),
        )
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()

    @responses.activate
    def test_two_galaxies_behind_one_query_routed_gateway_do_not_share(self, monkeypatch):
        """A gateway can route by query string, so the query is part of the address."""
        first = "https://gateway.invalid/proxy?target=a&path="
        second = "https://gateway.invalid/proxy?target=b&path="
        _serves_version("26.1", base_url=first)
        _serves_version("26.0", base_url=second)
        # A regex, because list_pages appends its own params and this gateway's address
        # already ends in a query that responses would otherwise insist on matching whole.
        responses.add(
            responses.GET, re.compile(re.escape(f"{first}/api/pages") + r"[&?].*"), json=[]
        )

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(first))
        assert list_pages_fn().success is True

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(second))
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()

    @responses.activate
    def test_two_identities_at_one_host_do_not_share(self, monkeypatch):
        """A proxy may route by the Basic-auth identity, whose case is its own."""
        upper = "https://Alice:pw@identity.invalid"
        lower = "https://alice:pw@identity.invalid"
        _serves_version("26.1", base_url=upper)
        _serves_version("26.0", base_url=lower)
        responses.add(responses.GET, f"{upper}/api/pages", json=[])

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(upper))
        assert list_pages_fn().success is True

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(lower))
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()

    @responses.activate
    def test_a_galaxy_mounted_under_another_ones_api_path_is_its_own_server(self, monkeypatch):
        """Two Galaxies can sit at /galaxy and /galaxy/api; neither answers for the other."""
        mounted = "https://host.invalid:8443/galaxy"
        nested = "https://host.invalid:8443/galaxy/api"
        _serves_version("26.1", base_url=mounted)
        _serves_version("26.0", base_url=nested)
        responses.add(responses.GET, f"{mounted}/api/pages", json=[])

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(mounted))
        assert list_pages_fn().success is True

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(nested))
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()

    @responses.activate
    def test_what_the_caller_typed_plays_no_part(self, monkeypatch):
        """One resolved server is one entry however the caller spelled it to connect."""
        _serves_version("26.0")

        for typed in (f"{GALAXY_BASE_URL}/", "whatever-the-user-typed"):
            monkeypatch.setattr(
                server,
                "_get_request_connection_state",
                lambda typed=typed: _state(f"{GALAXY_BASE_URL}/", typed_url=typed),
            )
            with pytest.raises(GalaxyVersionError, match="26.0"):
                list_pages_fn()

        assert len(_version_calls()) == 1

    @responses.activate
    @pytest.mark.parametrize(
        ("one", "other"),
        [
            # An empty query is a different request: bioblend asks /g/api/version for the
            # first and /g?/api/version for the second.
            ("https://gateway.invalid/g", "https://gateway.invalid/g?"),
            # bioblend's rstrip applies to the whole URL, so with a query the path's
            # trailing slash survives and the two are different routes.
            ("https://gateway.invalid/g?target=x", "https://gateway.invalid/g/?target=x"),
        ],
    )
    def test_addresses_that_differ_at_all_are_different_servers(self, monkeypatch, one, other):
        _serves_version("26.1", base_url=one)
        _serves_version("26.0", base_url=other)
        responses.add(responses.GET, re.compile(re.escape(f"{one}/api/pages") + r"[&?].*"), json=[])

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(one))
        assert list_pages_fn().success is True

        monkeypatch.setattr(server, "_get_request_connection_state", lambda: _state(other))
        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()

    @responses.activate
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://[2001:db8::1]:8443/g",
            "https://gateway.invalid/g?",
            "https://gateway.invalid/g/?target=x",
            "https://HOST.Invalid/g/",
        ],
    )
    def test_an_address_is_used_as_bioblend_holds_it(self, monkeypatch, base_url):
        """Whatever bioblend can send a request to, this can key on -- without parsing it."""
        state = _state(base_url)
        # From the resolved base_url, because bioblend rstrips the URL it was handed.
        _serves_version("26.0", base_url=state["gi"].base_url)
        monkeypatch.setattr(server, "_get_request_connection_state", lambda: state)

        with pytest.raises(GalaxyVersionError, match="26.0"):
            list_pages_fn()


@pytest.mark.parametrize("tool_name", sorted(GATED_CALLS))
def test_a_gated_body_asks_for_its_connection_before_it_asks_galaxy_for_anything(tool_name):
    """The check lives in ensure_connected, so a body that skips it would go unchecked.

    Also pins where the call sits: inside one of these bodies' try blocks the refusal would
    be caught and re-raised through format_error as though Galaxy had rejected something,
    when in fact nothing was sent.
    """
    source = Path(server.__file__).read_text()
    body = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == tool_name
    )

    def calls_ensure_connected(node):
        return any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "ensure_connected"
            for inner in ast.walk(node)
        )

    statements = [
        node
        for node in body.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    assert calls_ensure_connected(statements[0]), f"{tool_name} does not start by connecting"
    assert not any(
        calls_ensure_connected(node) for node in ast.walk(body) if isinstance(node, ast.Try)
    ), f"{tool_name} connects inside a try, which would dress a refusal up as a Galaxy failure"


class TestTheKeyIsTakenAsGiven:
    """Whatever bioblend can send a request to is a usable key, unexamined."""

    def _state_answering(self, base_url, version_major, asked):
        state = _state(base_url)

        def get_version():
            asked.append(base_url)
            return {"version_major": version_major}

        state["gi"].config.get_version = get_version
        return state

    def test_an_address_python_refuses_to_parse_is_still_a_key(self):
        """requests encodes this password's bracket and the call works; urlsplit raises.

        Nothing here may be fussier about a URL than the library that sends the request, or
        a connection that works everywhere else fails the moment it needs its version.
        """
        awkward = "https://alice:p[wo@host.invalid/g"
        with pytest.raises(ValueError, match="Invalid IPv6 URL"):
            urlsplit(awkward)

        asked: list[str] = []
        state = self._state_answering(awkward, "26.0", asked)

        assert server._galaxy_version_report(state).version == parse_version("26.0")
        assert server._galaxy_version_report(state).version == parse_version("26.0")
        assert len(asked) == 1
        assert awkward in _servers

    @pytest.mark.parametrize(
        ("one", "other"),
        [
            ("https://gateway.invalid/g", "https://gateway.invalid/g?"),
            ("https://gateway.invalid/g?target=x", "https://gateway.invalid/g/?target=x"),
            ("https://HOST.Invalid/g", "https://host.invalid/g"),
            ("https://Alice:pw@host.invalid/g", "https://alice:pw@host.invalid/g"),
            ("https://host.invalid/g", "https://host.invalid/g/api"),
        ],
    )
    def test_two_spellings_cost_two_entries_rather_than_one_answer(self, one, other):
        """The accepted price: a duplicate entry and one extra request, never a shared answer."""
        asked: list[str] = []

        assert server._galaxy_version_report(
            self._state_answering(one, "26.0", asked)
        ).version == parse_version("26.0")
        assert server._galaxy_version_report(
            self._state_answering(other, "26.1", asked)
        ).version == parse_version("26.1")

        assert asked == [one, other]
        assert {one, other} <= set(_servers)
