"""Reading a Galaxy version, and the minimum a tool may declare against it."""

import threading

import pytest

from galaxy_mcp.version import (
    MAX_REMEMBERED_SERVERS,
    GalaxyVersion,
    _servers,
    lookup_version,
    parse_requirement,
    parse_version,
    requirement_sentence,
    satisfies,
    unsupported_tools,
)


def _at(raw: str) -> GalaxyVersion:
    version = parse_version(raw)
    assert version is not None, f"{raw} did not parse"
    return version


class TestParseVersion:
    def test_reads_the_two_components_version_major_carries(self):
        assert parse_version("26.1") == GalaxyVersion(raw="26.1", major=26, minor=1)
        assert parse_version("26.0") == GalaxyVersion(raw="26.0", major=26, minor=0)

    @pytest.mark.parametrize(
        ("raw", "major", "minor"),
        [("26.1.1", 26, 1), ("26.2.dev0", 26, 2), ("26.1-rc1", 26, 1)],
    )
    def test_takes_the_first_two_components_of_a_fuller_string(self, raw, major, minor):
        """A caller reading the release tag says 26.1.1; a dev checkout answers 26.2.dev0."""
        version = _at(raw)
        assert (version.major, version.minor) == (major, minor)

    def test_keeps_what_the_server_said_so_a_refusal_can_quote_it(self):
        assert _at(" 26.1.1 ").raw == "26.1.1"

    @pytest.mark.parametrize(
        "bad", ["", "26", "twenty-six.one", "v26.1", ".1", None, 26.1, {"version_major": "26.1"}]
    )
    def test_is_unknown_rather_than_an_error_for_anything_it_cannot_read(self, bad):
        assert parse_version(bad) is None


class TestParseRequirement:
    @pytest.mark.parametrize(
        ("spec", "major", "minor"), [(">=26.1", 26, 1), (">= 26.1", 26, 1), ("  >=24.0  ", 24, 0)]
    )
    def test_accepts_a_major_minor_lower_bound(self, spec, major, minor):
        want = parse_requirement(spec)
        assert (want.major, want.minor) == (major, minor)

    @pytest.mark.parametrize(
        "bad", [">=26.1.1", ">26.1", "26.1", "<=26.1", "^26.1", ">=26", ">=26.x", "", None]
    )
    def test_refuses_a_bound_version_major_could_never_answer(self, bad):
        """The patch level lives in version_minor, so a three-component bound is unjudgeable."""
        with pytest.raises(ValueError, match=r">=MAJOR\.MINOR"):
            parse_requirement(bad)


class TestSatisfies:
    @pytest.mark.parametrize("raw", ["26.1", "26.2", "27.0", "26.1.1"])
    def test_passes_an_equal_or_newer_server(self, raw):
        assert satisfies(_at(raw), ">=26.1") is True

    @pytest.mark.parametrize("raw", ["26.0", "25.9", "9.9"])
    def test_fails_an_older_one_including_a_bigger_minor_under_a_smaller_major(self, raw):
        assert satisfies(_at(raw), ">=26.1") is False


class TestRequirementSentence:
    def test_says_the_bound_once_the_same_way_for_every_surface(self):
        assert requirement_sentence(">=26.1") == "Requires Galaxy 26.1 or newer."
        assert requirement_sentence(">= 24.0") == "Requires Galaxy 24.0 or newer."


class TestUnsupportedTools:
    def test_an_unknown_version_rules_nothing_out(self):
        assert unsupported_tools(None) == []


class TestTheCache:
    def _counting_fetch(self, version_major="26.1"):
        calls = []

        def fetch():
            calls.append(version_major)
            return {"version_major": version_major}

        return fetch, calls

    def test_one_server_is_asked_once(self):
        fetch, calls = self._counting_fetch()
        assert lookup_version("http://a", fetch).version == parse_version("26.1")
        assert lookup_version("http://a", fetch).version == parse_version("26.1")
        assert len(calls) == 1

    def test_a_refresh_asks_again_and_replaces_what_was_remembered(self):
        first, _ = self._counting_fetch("26.0")
        second, _ = self._counting_fetch("26.1")
        lookup_version("http://a", first)

        assert lookup_version("http://a", second, refresh=True).version == parse_version("26.1")
        assert lookup_version("http://a", first).version == parse_version("26.1")

    def test_a_caller_with_no_key_shares_nothing(self):
        """Two servers must not land in one bucket just because neither named itself."""
        first, first_calls = self._counting_fetch("26.0")
        second, _ = self._counting_fetch("26.1")

        assert lookup_version("", first).version == parse_version("26.0")
        assert lookup_version("", second).version == parse_version("26.1")
        assert lookup_version("", first).version == parse_version("26.0")
        assert len(first_calls) == 2

    def test_it_does_not_remember_more_servers_than_it_said_it_would(self):
        fetch, _ = self._counting_fetch()
        for index in range(MAX_REMEMBERED_SERVERS + 10):
            lookup_version(f"http://galaxy-{index}", fetch)

        assert len(_servers) == MAX_REMEMBERED_SERVERS
        assert "http://galaxy-0" not in _servers
        assert f"http://galaxy-{MAX_REMEMBERED_SERVERS + 9}" in _servers

    def test_servers_that_never_answer_are_bounded_too(self):
        """A run of unreachable Galaxies must not leave a lock behind for each one."""

        def unreachable():
            raise ConnectionError("no route to this Galaxy")

        for index in range(MAX_REMEMBERED_SERVERS + 10):
            with pytest.raises(ConnectionError):
                lookup_version(f"http://broken-{index}", unreachable)

        assert len(_servers) == MAX_REMEMBERED_SERVERS
        assert all(entry.report is None for entry in _servers.values())


class TestConcurrentLookups:
    def test_an_evicted_server_takes_its_straggler_with_it(self, monkeypatch):
        """Eviction must not hand a waiting fetch a fresh identity to overwrite."""
        monkeypatch.setattr("galaxy_mcp.version.MAX_REMEMBERED_SERVERS", 1)
        key = "http://evicted.invalid"
        fetching = threading.Event()
        release = threading.Event()

        def slow():
            fetching.set()
            assert release.wait(5), "the slow fetch was never released"
            return {"version_major": "26.0"}

        straggler = threading.Thread(target=lookup_version, args=(key, slow), daemon=True)
        straggler.start()
        assert fetching.wait(5), "the slow fetch never started"

        # Another server pushes this one out while its fetch is still in flight.
        lookup_version("http://someone-else.invalid", lambda: {"version_major": "26.1"})
        assert key not in _servers

        # The server comes back, upgraded, and is asked again from scratch.
        assert lookup_version(key, lambda: {"version_major": "26.1"}).version == parse_version(
            "26.1"
        )

        release.set()
        straggler.join(5)

        def unreachable():
            raise AssertionError("asked again instead of using what was remembered")

        assert lookup_version(key, unreachable).version == parse_version("26.1")

    def test_a_slow_answer_cannot_land_on_top_of_a_newer_one(self):
        """Asking and remembering are one step, or an upgrade gets undone by a straggler."""
        key = "http://slow.invalid"
        fetching = threading.Event()
        release = threading.Event()

        def slow():
            fetching.set()
            assert release.wait(5), "the slow fetch was never released"
            return {"version_major": "26.0"}

        straggler = threading.Thread(target=lookup_version, args=(key, slow), daemon=True)
        straggler.start()
        assert fetching.wait(5), "the slow fetch never started"

        refresh = threading.Thread(
            target=lookup_version,
            args=(key, lambda: {"version_major": "26.1"}),
            kwargs={"refresh": True},
            daemon=True,
        )
        refresh.start()
        # Long enough for the refresh to finish if nothing is holding it back.
        refresh.join(0.5)

        release.set()
        straggler.join(5)
        refresh.join(5)

        def unreachable():
            raise AssertionError("asked again instead of using what was remembered")

        assert lookup_version(key, unreachable).version == parse_version("26.1")
