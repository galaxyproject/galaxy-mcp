"""Token-lifecycle tests for :class:`GalaxyOAuthProvider`.

These exercise the self-contained crypto and token paths -- payload encryption,
token issuance, the authorization-code and refresh-token load/exchange logic,
the client-id guards, and the client registry -- without a live Galaxy server.

Authorization-code and refresh-token carriers are crafted with the provider's
own ``_encrypt_payload`` so the inputs are exactly what the provider would have
produced. The exchange methods re-decrypt the carrier themselves, so a lightweight
object exposing only ``.code`` / ``.token`` is a faithful stand-in for the
``AuthorizationCode`` / ``RefreshToken`` models there.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from cryptography.fernet import InvalidToken
from mcp.shared.auth import OAuthClientInformationFull

from galaxy_mcp.auth import (
    ACCESS_TOKEN_TTL_SECONDS,
    GalaxyAuthenticationError,
    GalaxyOAuthProvider,
)

REDIRECT_URI = "https://example.test/callback"
# Mirrors the galaxy payload _authenticate_and_complete actually stores in a token
# (keys: url / api_key / username / user_email). The provider treats it as opaque,
# but keeping the real shape here keeps the round-trip assertions honest.
GALAXY_SESSION = {
    "url": "https://galaxy.test/",
    "api_key": "galaxy-api-key",
    "username": "alice",
    "user_email": "alice@example.test",
}


def _run(coro):
    return asyncio.run(coro)


def _client(client_id="client-1"):
    return OAuthClientInformationFull(client_id=client_id, redirect_uris=[REDIRECT_URI])


def _anonymous_client():
    # redirect_uris is the only required field; client_id defaults to None.
    return OAuthClientInformationFull(redirect_uris=[REDIRECT_URI])


@pytest.fixture
def provider():
    return GalaxyOAuthProvider(
        base_url="https://mcp.test",
        galaxy_url="https://galaxy.test/",
        session_secret="unit-test-secret",
        client_registry_path=None,
    )


def _auth_code(provider, *, client_id="client-1", exp_offset=300, scopes=None, galaxy=None):
    return provider._encrypt_payload(
        {
            "typ": "authorization_code",
            "client_id": client_id,
            "scopes": scopes or ["galaxy:full"],
            "exp": int(time.time()) + exp_offset,
            "code_challenge": "test-challenge",
            "redirect_uri": REDIRECT_URI,
            "redirect_uri_provided_explicitly": True,
            "galaxy": galaxy if galaxy is not None else dict(GALAXY_SESSION),
        }
    )


def _refresh(provider, *, client_id="client-1", exp_offset=3600, scopes=None, galaxy=None):
    return provider._encrypt_payload(
        {
            "typ": "refresh",
            "client_id": client_id,
            "scopes": scopes or ["galaxy:full"],
            "exp": int(time.time()) + exp_offset,
            "galaxy": galaxy if galaxy is not None else dict(GALAXY_SESSION),
        }
    )


class TestTokenCrypto:
    def test_encrypt_decrypt_roundtrip(self, provider):
        payload = {"typ": "access", "client_id": "client-1", "value": 42}
        token = provider._encrypt_payload(payload)
        assert provider._decrypt_payload(token, expected_type="access") == payload

    def test_decrypt_rejects_type_mismatch(self, provider):
        token = provider._encrypt_payload({"typ": "access", "client_id": "client-1"})
        with pytest.raises(InvalidToken):
            provider._decrypt_payload(token, expected_type="refresh")

    def test_decrypt_rejects_tampered_token(self, provider):
        token = provider._encrypt_payload({"typ": "access", "client_id": "client-1"})
        tampered = token[:-4] + ("AAAA" if token[-4:] != "AAAA" else "BBBB")
        with pytest.raises(InvalidToken):
            provider._decrypt_payload(tampered, expected_type="access")

    def test_token_not_decryptable_with_other_secret(self, provider):
        other = GalaxyOAuthProvider(
            base_url="https://mcp.test",
            galaxy_url="https://galaxy.test/",
            session_secret="a-different-secret",
            client_registry_path=None,
        )
        token = provider._encrypt_payload({"typ": "access", "client_id": "client-1"})
        with pytest.raises(InvalidToken):
            other._decrypt_payload(token, expected_type="access")


class TestIssueAndDecode:
    def test_issue_tokens_shape_and_roundtrip(self, provider):
        tokens = provider._issue_tokens(
            client_id="client-1", scopes=["galaxy:full"], galaxy_payload=dict(GALAXY_SESSION)
        )
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in == ACCESS_TOKEN_TTL_SECONDS
        assert tokens.scope == "galaxy:full"

        access = provider.decode_access_token(tokens.access_token)
        assert access is not None
        assert access["typ"] == "access"
        assert access["client_id"] == "client-1"
        assert access["scopes"] == ["galaxy:full"]
        assert access["galaxy"] == GALAXY_SESSION

        refresh = provider._decrypt_payload(tokens.refresh_token, expected_type="refresh")
        assert refresh["client_id"] == "client-1"
        assert refresh["galaxy"] == GALAXY_SESSION

    def test_load_access_token_valid(self, provider):
        tokens = provider._issue_tokens(
            client_id="client-1", scopes=["galaxy:full"], galaxy_payload=dict(GALAXY_SESSION)
        )
        access = _run(provider.load_access_token(tokens.access_token))
        assert access is not None
        assert access.client_id == "client-1"
        assert access.scopes == ["galaxy:full"]
        assert access.token == tokens.access_token

    def test_load_access_token_expired_returns_none(self, provider):
        expired = provider._encrypt_payload(
            {
                "typ": "access",
                "client_id": "client-1",
                "scopes": ["galaxy:full"],
                "exp": int(time.time()) - 5,
            }
        )
        assert _run(provider.load_access_token(expired)) is None

    def test_load_access_token_tampered_returns_none(self, provider):
        assert _run(provider.load_access_token("not-a-real-token")) is None

    def test_decode_access_token_rejects_refresh_token(self, provider):
        tokens = provider._issue_tokens(
            client_id="client-1", scopes=["galaxy:full"], galaxy_payload=dict(GALAXY_SESSION)
        )
        # A refresh token must not validate as an access token.
        assert provider.decode_access_token(tokens.refresh_token) is None

    def test_decode_access_token_expired_returns_none(self, provider):
        expired = provider._encrypt_payload(
            {"typ": "access", "client_id": "client-1", "scopes": [], "exp": int(time.time()) - 1}
        )
        assert provider.decode_access_token(expired) is None


class TestAuthorizationCode:
    def test_load_valid(self, provider):
        code = _auth_code(provider)
        loaded = _run(provider.load_authorization_code(_client(), code))
        assert loaded is not None
        assert loaded.client_id == "client-1"
        assert loaded.scopes == ["galaxy:full"]
        assert loaded.code_challenge == "test-challenge"
        assert str(loaded.redirect_uri) == REDIRECT_URI

    def test_load_expired_returns_none(self, provider):
        code = _auth_code(provider, exp_offset=-10)
        assert _run(provider.load_authorization_code(_client(), code)) is None

    def test_load_client_mismatch_returns_none(self, provider):
        code = _auth_code(provider, client_id="client-1")
        assert _run(provider.load_authorization_code(_client("client-2"), code)) is None

    def test_load_null_client_id_returns_none(self, provider):
        code = _auth_code(provider, client_id="client-1")
        assert _run(provider.load_authorization_code(_anonymous_client(), code)) is None

    def test_load_invalid_token_returns_none(self, provider):
        assert _run(provider.load_authorization_code(_client(), "garbage")) is None

    def test_exchange_valid_issues_tokens(self, provider):
        # Drive the real load -> exchange chain (not a stand-in carrier) on the happy path.
        code = _auth_code(provider)
        loaded = _run(provider.load_authorization_code(_client(), code))
        assert loaded is not None
        tokens = _run(provider.exchange_authorization_code(_client(), loaded))
        access = provider.decode_access_token(tokens.access_token)
        assert access is not None
        assert access["client_id"] == "client-1"
        assert access["galaxy"] == GALAXY_SESSION

    def test_exchange_expired_raises(self, provider):
        code = _auth_code(provider, exp_offset=-10)
        with pytest.raises(GalaxyAuthenticationError):
            _run(provider.exchange_authorization_code(_client(), SimpleNamespace(code=code)))

    def test_exchange_client_mismatch_raises(self, provider):
        code = _auth_code(provider, client_id="client-1")
        with pytest.raises(GalaxyAuthenticationError):
            _run(
                provider.exchange_authorization_code(
                    _client("client-2"), SimpleNamespace(code=code)
                )
            )

    def test_exchange_null_client_id_raises(self, provider):
        code = _auth_code(provider, client_id="client-1")
        with pytest.raises(GalaxyAuthenticationError):
            _run(
                provider.exchange_authorization_code(
                    _anonymous_client(), SimpleNamespace(code=code)
                )
            )


class TestRefreshToken:
    def test_load_valid(self, provider):
        token = _refresh(provider)
        loaded = _run(provider.load_refresh_token(_client(), token))
        assert loaded is not None
        assert loaded.client_id == "client-1"
        assert loaded.scopes == ["galaxy:full"]
        assert loaded.token == token

    def test_load_client_mismatch_returns_none(self, provider):
        token = _refresh(provider, client_id="client-1")
        assert _run(provider.load_refresh_token(_client("client-2"), token)) is None

    def test_load_anonymous_client_returns_none(self, provider):
        # load_refresh_token lacks the explicit `client_id is None` guard the other
        # three methods got, but a client-bound token still can't be loaded by an
        # anonymous client because None != the token's client_id. Lock that in.
        token = _refresh(provider, client_id="client-1")
        assert _run(provider.load_refresh_token(_anonymous_client(), token)) is None

    def test_load_expired_returns_none(self, provider):
        token = _refresh(provider, exp_offset=-10)
        assert _run(provider.load_refresh_token(_client(), token)) is None

    def test_exchange_valid_reissues_tokens(self, provider):
        # Drive the real load -> exchange chain on the happy path.
        token = _refresh(provider)
        loaded = _run(provider.load_refresh_token(_client(), token))
        assert loaded is not None
        tokens = _run(provider.exchange_refresh_token(_client(), loaded, []))
        access = provider.decode_access_token(tokens.access_token)
        assert access is not None
        assert access["client_id"] == "client-1"
        assert access["galaxy"] == GALAXY_SESSION

    def test_exchange_honors_requested_scopes(self, provider):
        token = _refresh(provider, scopes=["galaxy:full"])
        tokens = _run(
            provider.exchange_refresh_token(
                _client(), SimpleNamespace(token=token), ["galaxy:read"]
            )
        )
        access = provider.decode_access_token(tokens.access_token)
        assert access is not None
        assert access["scopes"] == ["galaxy:read"]

    def test_exchange_expired_raises(self, provider):
        token = _refresh(provider, exp_offset=-10)
        with pytest.raises(GalaxyAuthenticationError):
            _run(provider.exchange_refresh_token(_client(), SimpleNamespace(token=token), []))

    def test_exchange_client_mismatch_raises(self, provider):
        token = _refresh(provider, client_id="client-1")
        with pytest.raises(GalaxyAuthenticationError):
            _run(
                provider.exchange_refresh_token(
                    _client("client-2"), SimpleNamespace(token=token), []
                )
            )

    def test_exchange_null_client_id_raises(self, provider):
        token = _refresh(provider, client_id="client-1")
        with pytest.raises(GalaxyAuthenticationError):
            _run(
                provider.exchange_refresh_token(
                    _anonymous_client(), SimpleNamespace(token=token), []
                )
            )


class TestClientRegistryAndRevocation:
    def test_register_and_get_client(self, provider):
        client = _client("registered-client")
        _run(provider.register_client(client))
        assert _run(provider.get_client("registered-client")) is client
        assert _run(provider.get_client("unknown")) is None

    def test_register_requires_client_id(self, provider):
        with pytest.raises(ValueError, match="client_id"):
            _run(provider.register_client(_anonymous_client()))

    def test_revoke_token_is_noop_token_stays_valid(self, provider):
        # Stateless tokens can't be individually revoked. revoke_token is a no-op,
        # so the token still validates afterward -- assert that explicitly rather
        # than only that the call doesn't raise. This documents current behavior,
        # it is NOT an endorsement (see the revocation note in the PR/review).
        tokens = provider._issue_tokens(
            client_id="client-1", scopes=["galaxy:full"], galaxy_payload=dict(GALAXY_SESSION)
        )
        access = _run(provider.load_access_token(tokens.access_token))
        assert access is not None
        assert _run(provider.revoke_token(access)) is None
        assert _run(provider.load_access_token(tokens.access_token)) is not None
