"""A GALAXY_URL from .env is only used with the GALAXY_API_KEY from that same .env.

An exported GALAXY_API_KEY only goes to a GALAXY_URL that was exported (or a url passed to
connect() together with a key), so a .env that comes with a directory -- often a cloned
repository -- can't choose where that key is sent.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from galaxy_mcp import server

from .test_helpers import connect_fn

EXPORTED_URL = "https://exported.example/"
DOTENV_URL = "https://dotenv.example/"
EXPORTED_KEY = "EXPORTED-KEY"
DOTENV_KEY = "DOTENV-KEY"

# The real reader, before conftest's autouse fixture swaps it out for every test.
_REAL_READ_DOTENV = getattr(server, "_read_dotenv", None)


def _write_dotenv(directory: Path, url: str | None, key: str | None) -> None:
    lines = ["DOTENV_MARKER=applied"]
    if url:
        lines.append(f"GALAXY_URL={url}")
    if key:
        lines.append(f"GALAXY_API_KEY={key}")
    (directory / ".env").write_text("\n".join(lines) + "\n")


# --- startup: what the module builds when it is imported -------------------------------------

_STARTUP_CHILD = """
import json, os
from galaxy_mcp import server
state = server.galaxy_state
print("STARTUP-STATE " + json.dumps({
    "url": state["url"],
    "api_key": state["api_key"],
    "connected": bool(state["connected"] and state["gi"]),
    "environ_url": os.environ.get("GALAXY_URL"),
    "environ_key": os.environ.get("GALAXY_API_KEY"),
    "marker": os.environ.get("DOTENV_MARKER"),
}))
"""

# (exported url, exported key, .env url, .env key, .env two directories up) -> what startup does.
# None for url/key in the result means no connection is made at startup.
STARTUP_CASES = {
    "url exported, key exported": (EXPORTED_URL, EXPORTED_KEY, None, None, False),
    "url exported, key from .env": (EXPORTED_URL, None, None, DOTENV_KEY, False),
    "url exported, no key": (EXPORTED_URL, None, None, None, False),
    "url from .env, key exported": (None, EXPORTED_KEY, DOTENV_URL, None, False),
    "url from parent .env, key exported": (None, EXPORTED_KEY, DOTENV_URL, None, True),
    "url from .env, key from .env": (None, None, DOTENV_URL, DOTENV_KEY, False),
    "url from parent .env, key from parent .env": (None, None, DOTENV_URL, DOTENV_KEY, True),
    "url from .env, no key": (None, None, DOTENV_URL, None, False),
    "url and key from .env, key also exported": (None, EXPORTED_KEY, DOTENV_URL, DOTENV_KEY, False),
    "url exported and in .env, key exported": (EXPORTED_URL, EXPORTED_KEY, DOTENV_URL, None, False),
    "no url, key exported": (None, EXPORTED_KEY, None, None, False),
    "no url, key from .env": (None, None, None, DOTENV_KEY, False),
}

STARTUP_EXPECTED = {
    # name: (galaxy_state url, galaxy_state api_key, connected)
    "url exported, key exported": (EXPORTED_URL, EXPORTED_KEY, True),
    "url exported, key from .env": (EXPORTED_URL, DOTENV_KEY, True),
    "url exported, no key": (EXPORTED_URL, None, False),
    "url from .env, key exported": (DOTENV_URL, None, False),
    "url from parent .env, key exported": (DOTENV_URL, None, False),
    "url from .env, key from .env": (DOTENV_URL, DOTENV_KEY, True),
    "url from parent .env, key from parent .env": (DOTENV_URL, DOTENV_KEY, True),
    "url from .env, no key": (DOTENV_URL, None, False),
    "url and key from .env, key also exported": (DOTENV_URL, None, False),
    "url exported and in .env, key exported": (EXPORTED_URL, EXPORTED_KEY, True),
    "no url, key exported": (None, EXPORTED_KEY, False),
    "no url, key from .env": (None, DOTENV_KEY, False),
}


def _run_startup(name: str, top: Path) -> dict:
    exported_url, exported_key, dotenv_url, dotenv_key, in_parent = STARTUP_CASES[name]
    cwd = top / "repo" / "sub" if in_parent else top
    cwd.mkdir(parents=True, exist_ok=True)
    if dotenv_url or dotenv_key:
        _write_dotenv(top, dotenv_url, dotenv_key)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(top)}
    if exported_url:
        env["GALAXY_URL"] = exported_url
    if exported_key:
        env["GALAXY_API_KEY"] = exported_key
    out = subprocess.run(
        [sys.executable, "-c", _STARTUP_CHILD],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    for line in out.stdout.splitlines():
        if line.startswith("STARTUP-STATE "):
            return json.loads(line.removeprefix("STARTUP-STATE "))
    raise AssertionError(f"startup child failed:\n{out.stdout}\n{out.stderr}")


@pytest.fixture(scope="module")
def startup_results(tmp_path_factory):
    # Each case needs its own interpreter, since .env is read at import; run them together.
    tops = {name: tmp_path_factory.mktemp("startup") for name in STARTUP_CASES}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {name: pool.submit(_run_startup, name, top) for name, top in tops.items()}
        return {name: future.result() for name, future in futures.items()}


@pytest.mark.parametrize("name", list(STARTUP_CASES))
def test_startup_pairs_dotenv_url_only_with_dotenv_key(startup_results, name):
    result = startup_results[name]
    url, api_key, connected = STARTUP_EXPECTED[name]
    assert (result["url"], result["api_key"], result["connected"]) == (url, api_key, connected)


@pytest.mark.parametrize("name", [n for n, c in STARTUP_CASES.items() if c[2] or c[3]])
def test_startup_keeps_dotenv_credentials_out_of_the_environment(startup_results, name):
    # os.environ holds only what was exported, so nothing later mistakes a .env value for it;
    # other .env settings still apply.
    exported_url, exported_key, *_ = STARTUP_CASES[name]
    result = startup_results[name]
    assert result["environ_url"] == exported_url
    assert result["environ_key"] == exported_key
    assert result["marker"] == "applied"


# --- connect() over stdio --------------------------------------------------------------------


@pytest.fixture
def galaxy_instance(monkeypatch):
    """Record every client connect() builds, instead of contacting a Galaxy."""
    factory = Mock()
    factory.return_value.users.get_current_user.return_value = {"id": "u", "username": "user"}
    monkeypatch.setattr(server, "GalaxyInstance", factory)
    return factory


@pytest.fixture
def stdio(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "remote_caller_possible", lambda: False)
    monkeypatch.setattr(server, "auth_provider", None)
    monkeypatch.setattr(server, "normalized_galaxy_url", None)
    monkeypatch.setattr(server, "_get_current_session_id", lambda: None)
    if _REAL_READ_DOTENV is not None:
        monkeypatch.setattr(server, "_read_dotenv", _REAL_READ_DOTENV)
    monkeypatch.chdir(tmp_path)
    # A .env read into os.environ (as connect() used to) must not outlive the test.
    with patch.dict(os.environ):
        os.environ.pop("GALAXY_URL", None)
        os.environ.pop("GALAXY_API_KEY", None)
        yield tmp_path


def _configure(
    monkeypatch, directory, *, exported_url, exported_key, dotenv_url, dotenv_key, at_startup
):
    """Export what's given, and write a .env -- present at startup, or only written afterwards."""
    if exported_url:
        os.environ["GALAXY_URL"] = exported_url
    if exported_key:
        os.environ["GALAXY_API_KEY"] = exported_key
    if dotenv_url or dotenv_key:
        _write_dotenv(directory, dotenv_url, dotenv_key)
        if at_startup and _REAL_READ_DOTENV is not None:
            path, values = _REAL_READ_DOTENV()
            monkeypatch.setattr(server, "startup_dotenv_path", path, raising=False)
            monkeypatch.setattr(server, "startup_dotenv", values, raising=False)


def _sent(galaxy_instance):
    return [(c.kwargs["url"], c.kwargs["key"]) for c in galaxy_instance.call_args_list]


# (exported url, exported key, .env url, .env key) -> (url, key) connect() uses, or None: refused
CONNECT_CASES = {
    "url exported, key exported": (
        EXPORTED_URL,
        EXPORTED_KEY,
        None,
        None,
        (EXPORTED_URL, EXPORTED_KEY),
    ),
    "url exported, key from .env": (
        EXPORTED_URL,
        None,
        None,
        DOTENV_KEY,
        (EXPORTED_URL, DOTENV_KEY),
    ),
    "url from .env, key exported": (None, EXPORTED_KEY, DOTENV_URL, None, None),
    "url from .env, key from .env": (None, None, DOTENV_URL, DOTENV_KEY, (DOTENV_URL, DOTENV_KEY)),
    "url and key from .env, key also exported": (None, EXPORTED_KEY, DOTENV_URL, DOTENV_KEY, None),
    "url exported and in .env, key exported": (
        EXPORTED_URL,
        EXPORTED_KEY,
        DOTENV_URL,
        None,
        (EXPORTED_URL, EXPORTED_KEY),
    ),
    "url exported and in .env, both keys": (
        EXPORTED_URL,
        EXPORTED_KEY,
        DOTENV_URL,
        DOTENV_KEY,
        (EXPORTED_URL, EXPORTED_KEY),
    ),
}


@pytest.mark.parametrize(
    "at_startup", [True, False], ids=["dotenv at startup", "dotenv written after startup"]
)
@pytest.mark.parametrize("name", list(CONNECT_CASES))
def test_connect_without_arguments(stdio, galaxy_instance, monkeypatch, name, at_startup):
    exported_url, exported_key, dotenv_url, dotenv_key, expected = CONNECT_CASES[name]
    _configure(
        monkeypatch,
        stdio,
        exported_url=exported_url,
        exported_key=exported_key,
        dotenv_url=dotenv_url,
        dotenv_key=dotenv_key,
        at_startup=at_startup,
    )
    if expected is None:
        with pytest.raises(ValueError, match="only used with the GALAXY_API_KEY in the same file"):
            connect_fn()
        assert _sent(galaxy_instance) == []
    else:
        connect_fn()
        assert _sent(galaxy_instance) == [expected]


def test_connect_without_a_key_anywhere_reports_it_missing(stdio, galaxy_instance, monkeypatch):
    _configure(
        monkeypatch,
        stdio,
        exported_url=None,
        exported_key=None,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=True,
    )
    with pytest.raises(ValueError, match="Missing Galaxy API key"):
        connect_fn()
    assert _sent(galaxy_instance) == []


def test_refusal_names_the_dotenv_and_how_to_proceed(stdio, galaxy_instance, monkeypatch):
    _configure(
        monkeypatch,
        stdio,
        exported_url=None,
        exported_key=EXPORTED_KEY,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=False,
    )
    with pytest.raises(ValueError) as refused:
        connect_fn()
    message = str(refused.value)
    assert f"GALAXY_URL from {stdio / '.env'}" in message
    assert "not with GALAXY_API_KEY from the environment" in message
    assert "Set GALAXY_URL in the environment, or call connect(url=..., api_key=...)" in message


def test_explicit_url_does_not_pick_up_the_exported_key_for_a_dotenv_url(
    stdio, galaxy_instance, monkeypatch
):
    # Naming the .env's URL doesn't make it the exported URL; the exported key stays put.
    _configure(
        monkeypatch,
        stdio,
        exported_url=None,
        exported_key=EXPORTED_KEY,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=True,
    )
    with pytest.raises(ValueError, match="only used with the configured GALAXY_URL"):
        connect_fn(url=DOTENV_URL)
    assert _sent(galaxy_instance) == []


def test_explicit_url_matching_a_dotenv_pair_uses_its_key(stdio, galaxy_instance, monkeypatch):
    _configure(
        monkeypatch,
        stdio,
        exported_url=None,
        exported_key=None,
        dotenv_url=DOTENV_URL,
        dotenv_key=DOTENV_KEY,
        at_startup=True,
    )
    connect_fn(url=DOTENV_URL)
    assert _sent(galaxy_instance) == [(DOTENV_URL, DOTENV_KEY)]


def test_explicit_url_matching_the_exported_url_uses_the_exported_key(
    stdio, galaxy_instance, monkeypatch
):
    _configure(
        monkeypatch,
        stdio,
        exported_url=EXPORTED_URL,
        exported_key=EXPORTED_KEY,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=True,
    )
    connect_fn(url=EXPORTED_URL)
    assert _sent(galaxy_instance) == [(EXPORTED_URL, EXPORTED_KEY)]


@pytest.mark.parametrize(
    "at_startup", [True, False], ids=["dotenv at startup", "dotenv written after startup"]
)
def test_explicit_key_is_not_sent_to_a_dotenv_url(stdio, galaxy_instance, monkeypatch, at_startup):
    _configure(
        monkeypatch,
        stdio,
        exported_url=None,
        exported_key=None,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=at_startup,
    )
    with pytest.raises(ValueError, match="not with an api_key passed to connect"):
        connect_fn(api_key="typed-key")
    assert _sent(galaxy_instance) == []


def test_explicit_key_goes_to_the_exported_url(stdio, galaxy_instance, monkeypatch):
    _configure(
        monkeypatch,
        stdio,
        exported_url=EXPORTED_URL,
        exported_key=EXPORTED_KEY,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=True,
    )
    connect_fn(api_key="typed-key")
    assert _sent(galaxy_instance) == [(EXPORTED_URL, "typed-key")]


def test_explicit_url_and_key_are_used_as_given(stdio, galaxy_instance, monkeypatch):
    _configure(
        monkeypatch,
        stdio,
        exported_url=None,
        exported_key=EXPORTED_KEY,
        dotenv_url=DOTENV_URL,
        dotenv_key=None,
        at_startup=True,
    )
    connect_fn(url=DOTENV_URL, api_key="typed-key")
    assert _sent(galaxy_instance) == [(DOTENV_URL, "typed-key")]


def test_connect_leaves_the_environment_alone(stdio, galaxy_instance, monkeypatch):
    # A .env read on a connect() that is missing credentials is never merged into os.environ,
    # so it can't replace an exported GALAXY_URL, now or for later calls.
    _configure(
        monkeypatch,
        stdio,
        exported_url=EXPORTED_URL,
        exported_key=None,
        dotenv_url=DOTENV_URL,
        dotenv_key=DOTENV_KEY,
        at_startup=False,
    )
    connect_fn()
    assert _sent(galaxy_instance) == [(EXPORTED_URL, DOTENV_KEY)]
    assert os.environ.get("GALAXY_URL") == EXPORTED_URL
    assert "GALAXY_API_KEY" not in os.environ


# --- connect() over HTTP ---------------------------------------------------------------------


@pytest.fixture
def http(monkeypatch, tmp_path):
    """An HTTP server whose GALAXY_URL came from the .env it was started next to."""
    monkeypatch.setattr(server, "remote_caller_possible", lambda: True)
    monkeypatch.setattr(server, "auth_provider", None)
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ())
    monkeypatch.setattr(server, "normalized_galaxy_url", DOTENV_URL)
    monkeypatch.setattr(server, "_get_current_session_id", lambda: "dotenv-test")
    if _REAL_READ_DOTENV is not None:
        monkeypatch.setattr(server, "_read_dotenv", _REAL_READ_DOTENV)
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ):
        os.environ.pop("GALAXY_URL", None)
        os.environ.pop("GALAXY_API_KEY", None)
        yield tmp_path


def _http_startup(monkeypatch, directory, *, exported_key, dotenv_key):
    if exported_key:
        os.environ["GALAXY_API_KEY"] = exported_key
    _write_dotenv(directory, DOTENV_URL, dotenv_key)
    if _REAL_READ_DOTENV is not None:
        path, values = _REAL_READ_DOTENV()
        monkeypatch.setattr(server, "startup_dotenv_path", path, raising=False)
        monkeypatch.setattr(server, "startup_dotenv", values, raising=False)
        monkeypatch.setattr(
            server,
            "startup_credentials",
            server._configured_credentials(os.environ, values, path),
            raising=False,
        )


# Over HTTP an argument-less connect() never reaches these credentials (it reuses the startup
# connection or refuses), so the rule is checked where connect() fills in its arguments.


def test_http_exported_key_is_not_filled_in_for_a_dotenv_url(http, monkeypatch):
    _http_startup(monkeypatch, http, exported_key=EXPORTED_KEY, dotenv_key=None)
    assert server._resolve_connect_credentials(None, None) == (DOTENV_URL, None)
    assert server._resolve_connect_credentials(DOTENV_URL, None) == (DOTENV_URL, None)


def test_http_dotenv_pair_is_filled_in(http, monkeypatch):
    _http_startup(monkeypatch, http, exported_key=None, dotenv_key=DOTENV_KEY)
    assert server._resolve_connect_credentials(None, None) == (DOTENV_URL, DOTENV_KEY)


def test_http_caller_key_needs_the_url_named(http, galaxy_instance, monkeypatch):
    _http_startup(monkeypatch, http, exported_key=None, dotenv_key=None)
    with pytest.raises(ValueError, match="not with an api_key passed to connect"):
        connect_fn(api_key="caller-key")
    connect_fn(url=DOTENV_URL, api_key="caller-key")
    assert _sent(galaxy_instance) == [(DOTENV_URL, "caller-key")]


def test_http_reloaded_dotenv_key_does_not_follow_a_changed_url(http, monkeypatch):
    # The allowlist is fixed at startup; a .env rewritten to point elsewhere must not send its
    # key to the startup URL either.
    _http_startup(monkeypatch, http, exported_key=None, dotenv_key=None)
    _write_dotenv(http, "https://elsewhere.example/", "new-key")
    reloaded = _REAL_READ_DOTENV() if _REAL_READ_DOTENV is not None else None
    assert server._resolve_connect_credentials(None, None, reloaded) == (DOTENV_URL, None)
