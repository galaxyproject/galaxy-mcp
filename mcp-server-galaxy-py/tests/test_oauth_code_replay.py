"""Regression tests: authorization codes must be single-use (RFC 6749 4.1.2 / 10.5).

A code is a stateless encrypted blob, so without server-side bookkeeping it could
be exchanged for tokens repeatedly until expiry. These tests pin the single-use
behavior of `exchange_authorization_code`.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from galaxy_mcp.auth import GalaxyAuthenticationError, GalaxyOAuthProvider

REDIRECT_URI = "https://example.test/callback"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def provider():
    return GalaxyOAuthProvider(
        base_url="https://mcp.test",
        galaxy_url="https://galaxy.test/",
        session_secret="unit-test-secret",
        client_registry_path=None,
    )


def _code(provider, *, client_id="client-1", exp_offset=300):
    return provider._encrypt_payload(
        {
            "typ": "authorization_code",
            "client_id": client_id,
            "scopes": ["galaxy:full"],
            "exp": int(time.time()) + exp_offset,
            "code_challenge": "test-challenge",
            "code_challenge_method": "S256",
            "redirect_uri": REDIRECT_URI,
            "redirect_uri_provided_explicitly": True,
            "galaxy": {
                "url": "https://galaxy.test/",
                "api_key": "k",
                "username": "u",
                "user_email": None,
            },
            "nonce": "test-nonce",
        }
    )


def test_authorization_code_cannot_be_replayed(provider):
    code = _code(provider)
    client = SimpleNamespace(client_id="client-1")

    # First exchange succeeds and issues a token.
    tokens = _run(provider.exchange_authorization_code(client, SimpleNamespace(code=code)))
    assert tokens.access_token

    # Re-presenting the same code must be rejected, not re-issue tokens.
    with pytest.raises(GalaxyAuthenticationError):
        _run(provider.exchange_authorization_code(client, SimpleNamespace(code=code)))


def test_distinct_codes_both_exchange(provider):
    # The spent-code set must not produce false positives for different codes.
    client = SimpleNamespace(client_id="client-1")
    first = _run(
        provider.exchange_authorization_code(client, SimpleNamespace(code=_code(provider)))
    )
    second = _run(
        provider.exchange_authorization_code(client, SimpleNamespace(code=_code(provider)))
    )
    assert first.access_token
    assert second.access_token


def test_expired_spent_entries_are_pruned(provider):
    # An already-expired code is rejected on the expiry check before it can be
    # recorded, and stale spent entries don't accumulate: exchanging a fresh code
    # leaves only its own hash behind.
    expired = _code(provider, exp_offset=-10)
    with pytest.raises(GalaxyAuthenticationError):
        _run(
            provider.exchange_authorization_code(
                SimpleNamespace(client_id="client-1"), SimpleNamespace(code=expired)
            )
        )

    _run(
        provider.exchange_authorization_code(
            SimpleNamespace(client_id="client-1"), SimpleNamespace(code=_code(provider))
        )
    )
    assert len(provider._spent_auth_codes) == 1
