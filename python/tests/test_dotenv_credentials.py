"""A GALAXY_URL from .env is only used with the GALAXY_API_KEY from that same .env.

An exported GALAXY_API_KEY only goes to a GALAXY_URL that was exported (or to a url passed to
connect() together with a key), whichever directory the server runs in.
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

URL_E = {"GALAXY_URL": EXPORTED_URL}
KEY_E = {"GALAXY_API_KEY": EXPORTED_KEY}
URL_D = {"GALAXY_URL": DOTENV_URL}
KEY_D = {"GALAXY_API_KEY": DOTENV_KEY}

# The real reader, before conftest's autouse fixture swaps it out for every test.
_REAL_READ_DOTENV = server._read_dotenv


def _write_dotenv(directory: Path, values: dict[str, str]) -> None:
    lines = ["DOTENV_MARKER=applied", *(f"{name}={value}" for name, value in values.items())]
    (directory / ".env").write_text("\n".join(lines) + "\n")


# --- startup: what the module builds when it is imported -------------------------------------

_STARTUP_CHILD = """
import json, os
from galaxy_mcp import server
state = server.galaxy_state
# What connect() would fill in over HTTP, where the startup GALAXY_URL is the only destination.
server.remote_caller_possible = lambda: True
try:
    http_fill = list(server._resolve_connect_credentials(None, None))
except ValueError:
    http_fill = "refused"
print("STARTUP-STATE " + json.dumps({
    "url": state["url"],
    "api_key": state["api_key"],
    "connected": bool(state["connected"] and state["gi"]),
    "http_fill": http_fill,
    "environ_url": os.environ.get("GALAXY_URL"),
    "environ_key": os.environ.get("GALAXY_API_KEY"),
    "marker": os.environ.get("DOTENV_MARKER"),
}))
"""

# name: (exported, .env or None, .env two directories up, (galaxy_state url, api_key)).
# A startup connection is made exactly when both are set.
STARTUP_CASES = {
    "url exported, key exported": ({**URL_E, **KEY_E}, None, False, (EXPORTED_URL, EXPORTED_KEY)),
    "url exported, key from .env": (URL_E, KEY_D, False, (EXPORTED_URL, DOTENV_KEY)),
    "url exported, no key": (URL_E, None, False, (EXPORTED_URL, None)),
    "url from .env, key exported": (KEY_E, URL_D, False, (DOTENV_URL, None)),
    "url from parent .env, key exported": (KEY_E, URL_D, True, (DOTENV_URL, None)),
    "url from .env, key from .env": ({}, {**URL_D, **KEY_D}, False, (DOTENV_URL, DOTENV_KEY)),
    "url from parent .env, key from parent .env": (
        {},
        {**URL_D, **KEY_D},
        True,
        (DOTENV_URL, DOTENV_KEY),
    ),
    "url from .env, no key": ({}, URL_D, False, (DOTENV_URL, None)),
    "url and key from .env, key also exported": (
        KEY_E,
        {**URL_D, **KEY_D},
        False,
        (DOTENV_URL, None),
    ),
    "url exported and in .env, key exported": (
        {**URL_E, **KEY_E},
        URL_D,
        False,
        (EXPORTED_URL, EXPORTED_KEY),
    ),
    "no url, key exported": (KEY_E, None, False, (None, EXPORTED_KEY)),
    "no url, key from .env": ({}, KEY_D, False, (None, DOTENV_KEY)),
    "no url, no key": ({}, None, False, (None, None)),
    # ${...} is not expanded in the credentials, so a .env can't copy exported values into them.
    ".env url expanding the exported key": (
        KEY_E,
        {"GALAXY_URL": "https://dotenv.example/${GALAXY_API_KEY}/"},
        False,
        ("https://dotenv.example/${GALAXY_API_KEY}/", None),
    ),
    ".env key expanding another exported value": (
        {"OTHER_SECRET": "OTHER"},
        {**URL_D, "GALAXY_API_KEY": "${OTHER_SECRET}"},
        False,
        (DOTENV_URL, "${OTHER_SECRET}"),
    ),
    # An exported variable shadows the .env's even when empty, as load_dotenv() does.
    "empty exported url, .env pair": (
        {"GALAXY_URL": ""},
        {**URL_D, **KEY_D},
        False,
        (None, DOTENV_KEY),
    ),
    "PYTHON_DOTENV_DISABLED, .env pair": (
        {"PYTHON_DOTENV_DISABLED": "1"},
        {**URL_D, **KEY_D},
        False,
        (None, None),
    ),
}


def _run_startup(name: str, top: Path) -> dict:
    exported, dotenv, in_parent, _ = STARTUP_CASES[name]
    cwd = top / "repo" / "sub" if in_parent else top
    cwd.mkdir(parents=True, exist_ok=True)
    if dotenv is not None:
        _write_dotenv(top, dotenv)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(top), **exported}
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
    url, api_key = STARTUP_CASES[name][3]
    assert (result["url"], result["api_key"]) == (url, api_key)
    assert result["connected"] == bool(url and api_key)
    # Over HTTP, connect() fills in the same pair, and refuses when there is no GALAXY_URL.
    assert result["http_fill"] == ([url, api_key] if url else "refused")


@pytest.mark.parametrize(
    "name", ["url from .env, key exported", "url exported, key from .env", "no url, no key"]
)
def test_startup_keeps_dotenv_credentials_out_of_the_environment(startup_results, name):
    # os.environ holds only what was exported, so nothing later mistakes a .env value for it.
    exported, dotenv, *_ = STARTUP_CASES[name]
    result = startup_results[name]
    assert result["environ_url"] == exported.get("GALAXY_URL")
    assert result["environ_key"] == exported.get("GALAXY_API_KEY")
    # Other .env settings still apply.
    assert result["marker"] == ("applied" if dotenv is not None else None)


def test_startup_honours_python_dotenv_disabled(startup_results):
    assert startup_results["PYTHON_DOTENV_DISABLED, .env pair"]["marker"] is None


# --- connect() over stdio --------------------------------------------------------------------


@pytest.fixture
def galaxy_instance(monkeypatch):
    """Record every client connect() builds, instead of contacting a Galaxy."""
    factory = Mock()
    factory.return_value.users.get_current_user.return_value = {"id": "u", "username": "user"}
    monkeypatch.setattr(server, "GalaxyInstance", factory)
    return factory


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    """A working directory whose .env is read for real, and an environment restored afterwards."""
    monkeypatch.setattr(server, "auth_provider", None)
    monkeypatch.setattr(server, "_read_dotenv", _REAL_READ_DOTENV)
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ):
        os.environ.pop("GALAXY_URL", None)
        os.environ.pop("GALAXY_API_KEY", None)
        yield tmp_path


@pytest.fixture
def stdio(monkeypatch, workdir):
    monkeypatch.setattr(server, "remote_caller_possible", lambda: False)
    monkeypatch.setattr(server, "normalized_galaxy_url", None)
    monkeypatch.setattr(server, "_get_current_session_id", lambda: None)
    return workdir


def _start(monkeypatch, directory, exported, dotenv, *, at_startup=True):
    """Export what's given and write the .env: there at startup, or only written afterwards."""
    os.environ.update(exported)
    if dotenv is not None:
        _write_dotenv(directory, dotenv)
    startup_dotenv = _REAL_READ_DOTENV() if at_startup else server._Dotenv()
    monkeypatch.setattr(server, "startup_dotenv", startup_dotenv)
    monkeypatch.setattr(
        server, "startup_credentials", server._configured_credentials(os.environ, startup_dotenv)
    )


def _sent(galaxy_instance):
    return [(c.kwargs["url"], c.kwargs["key"]) for c in galaxy_instance.call_args_list]


PAIRING = "only used with the GALAXY_API_KEY in the same file"
MISSING = "Missing Galaxy"

# name: (exported, .env, connect() arguments, (url, key) sent -- or the refusal's text)
CONNECT_CASES = {
    "url exported, key exported": ({**URL_E, **KEY_E}, None, {}, (EXPORTED_URL, EXPORTED_KEY)),
    "url exported, key from .env": (URL_E, KEY_D, {}, (EXPORTED_URL, DOTENV_KEY)),
    "url exported, no key": (URL_E, None, {}, MISSING),
    "url from .env, key exported": (KEY_E, URL_D, {}, PAIRING),
    "url from .env, key from .env": ({}, {**URL_D, **KEY_D}, {}, (DOTENV_URL, DOTENV_KEY)),
    "url from .env, no key": ({}, URL_D, {}, MISSING),
    "no url, key exported": (KEY_E, None, {}, MISSING),
    "no url, key from .env": ({}, KEY_D, {}, MISSING),
    "no url, no key": ({}, None, {}, MISSING),
    "url and key from .env, key also exported": (KEY_E, {**URL_D, **KEY_D}, {}, PAIRING),
    "url exported and in .env, key exported": (
        {**URL_E, **KEY_E},
        URL_D,
        {},
        (EXPORTED_URL, EXPORTED_KEY),
    ),
    "url exported and in .env, both keys": (
        {**URL_E, **KEY_E},
        {**URL_D, **KEY_D},
        {},
        (EXPORTED_URL, EXPORTED_KEY),
    ),
    # Naming the .env's URL doesn't make it the exported URL; the exported key stays put.
    "url= the .env url, key exported": (
        KEY_E,
        URL_D,
        {"url": DOTENV_URL},
        "only used with the configured GALAXY_URL",
    ),
    "url= the .env url, .env pair": (
        {},
        {**URL_D, **KEY_D},
        {"url": DOTENV_URL},
        (DOTENV_URL, DOTENV_KEY),
    ),
    "url= another url, .env pair": (
        {},
        {**URL_D, **KEY_D},
        {"url": "https://other.example/"},
        "only used with the configured GALAXY_URL",
    ),
    "url= the exported url, key exported": (
        {**URL_E, **KEY_E},
        URL_D,
        {"url": EXPORTED_URL},
        (EXPORTED_URL, EXPORTED_KEY),
    ),
    "api_key= with a .env url": ({}, URL_D, {"api_key": "typed"}, "not with an api_key passed"),
    "api_key= and an empty url, with a .env url": (
        {},
        URL_D,
        {"url": "", "api_key": "typed"},
        "not with an api_key passed",
    ),
    "api_key= with an exported url": (
        {**URL_E, **KEY_E},
        URL_D,
        {"api_key": "typed"},
        (EXPORTED_URL, "typed"),
    ),
    "url= and api_key=, with a .env url and key exported": (
        KEY_E,
        URL_D,
        {"url": DOTENV_URL, "api_key": "typed"},
        (DOTENV_URL, "typed"),
    ),
}


@pytest.mark.parametrize(
    "at_startup", [True, False], ids=["dotenv at startup", "dotenv written after startup"]
)
@pytest.mark.parametrize("name", list(CONNECT_CASES))
def test_connect(stdio, galaxy_instance, monkeypatch, name, at_startup):
    exported, dotenv, kwargs, expected = CONNECT_CASES[name]
    _start(monkeypatch, stdio, exported, dotenv, at_startup=at_startup)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            connect_fn(**kwargs)
        assert _sent(galaxy_instance) == []
    else:
        connect_fn(**kwargs)
        assert _sent(galaxy_instance) == [expected]


def test_stdio_ignores_an_oauth_provider(stdio, galaxy_instance, monkeypatch):
    # GALAXY_MCP_PUBLIC_URL builds the provider whatever the transport; it only governs HTTP.
    monkeypatch.setattr(server, "auth_provider", Mock())
    _start(monkeypatch, stdio, {}, {**URL_D, **KEY_D}, at_startup=False)
    connect_fn()
    assert _sent(galaxy_instance) == [(DOTENV_URL, DOTENV_KEY)]


@pytest.mark.parametrize(
    "at_startup", [True, False], ids=["dotenv at startup", "dotenv written after startup"]
)
def test_stdio_session_sees_the_pairing_refusal(stdio, galaxy_instance, monkeypatch, at_startup):
    # A stdio client has an MCP session, and an argument-less connect() from a session that
    # isn't connected stops early; it should still say why rather than "no connection".
    monkeypatch.setattr(server, "_get_current_session_id", lambda: "stdio-session")
    _start(monkeypatch, stdio, KEY_E, URL_D, at_startup=at_startup)
    with pytest.raises(ValueError, match=PAIRING):
        connect_fn()
    assert _sent(galaxy_instance) == []


def test_http_session_is_not_told_about_the_dotenv(http, galaxy_instance, monkeypatch):
    _start(monkeypatch, http, KEY_E, URL_D)
    with pytest.raises(ValueError, match="No Galaxy connection is available"):
        connect_fn()
    assert _sent(galaxy_instance) == []


def test_refusal_names_the_dotenv_and_how_to_proceed(stdio, galaxy_instance, monkeypatch):
    _start(monkeypatch, stdio, KEY_E, URL_D, at_startup=False)
    with pytest.raises(ValueError) as refused:
        connect_fn()
    message = str(refused.value)
    assert f"GALAXY_URL from {stdio / '.env'}" in message
    assert "not with GALAXY_API_KEY from the environment" in message
    assert "Set GALAXY_URL in the environment, or call connect(url=..., api_key=...)" in message


def test_connect_leaves_the_environment_alone(stdio, galaxy_instance, monkeypatch):
    # The .env read again on a connect() missing credentials is never merged into os.environ,
    # so it can't replace an exported GALAXY_URL, now or for later calls.
    _start(monkeypatch, stdio, URL_E, {**URL_D, **KEY_D}, at_startup=False)
    connect_fn()
    assert _sent(galaxy_instance) == [(EXPORTED_URL, DOTENV_KEY)]
    assert os.environ.get("GALAXY_URL") == EXPORTED_URL
    assert "GALAXY_API_KEY" not in os.environ


# --- connect() over HTTP ---------------------------------------------------------------------


@pytest.fixture
def http(monkeypatch, workdir):
    """An HTTP server whose GALAXY_URL came from the .env it was started next to."""
    monkeypatch.setattr(server, "remote_caller_possible", lambda: True)
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ())
    monkeypatch.setattr(server, "normalized_galaxy_url", DOTENV_URL)
    monkeypatch.setattr(server, "_get_current_session_id", lambda: "dotenv-test")
    return workdir


@pytest.mark.parametrize("oauth", [False, True], ids=["no oauth", "oauth"])
def test_http_caller_key_needs_the_url_named(http, galaxy_instance, monkeypatch, oauth):
    _start(monkeypatch, http, {}, URL_D)
    if oauth:
        monkeypatch.setattr(server, "auth_provider", Mock())
        monkeypatch.setattr(server, "_get_request_connection_state", lambda: {"connected": False})
        monkeypatch.setattr(server, "_untracked_remote_request", lambda: False)
    with pytest.raises(ValueError, match="not with an api_key passed"):
        connect_fn(api_key="caller-key")
    assert _sent(galaxy_instance) == []
    connect_fn(url=DOTENV_URL, api_key="caller-key")
    assert _sent(galaxy_instance) == [(DOTENV_URL, "caller-key")]


def test_http_reloaded_dotenv_key_does_not_follow_a_changed_url(http, monkeypatch):
    # The allowlist is fixed at startup; a .env rewritten to point elsewhere must not send its
    # key to the startup URL either.
    _start(monkeypatch, http, {}, URL_D)
    _write_dotenv(http, {"GALAXY_URL": "https://elsewhere.example/", "GALAXY_API_KEY": "new-key"})
    assert server._resolve_connect_credentials(None, None, _REAL_READ_DOTENV()) == (
        DOTENV_URL,
        None,
    )
