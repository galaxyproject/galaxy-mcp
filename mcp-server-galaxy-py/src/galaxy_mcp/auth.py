"""Authentication helpers and stateless OAuth provider for the Galaxy MCP server."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import secrets
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio
import requests
from bioblend.galaxy import GalaxyInstance
from cryptography.fernet import Fernet, InvalidToken
from fastmcp.server.auth.auth import (
    AccessToken as FastMCPAccessToken,
)
from fastmcp.server.auth.auth import (
    ClientRegistrationOptions,
    OAuthProvider,
    RevocationOptions,
)
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route

try:  # pragma: no cover - fallback import for Python < 3.12
    from typing import override
except ImportError:  # pragma: no cover - Python < 3.12 without typing.override
    try:
        from typing_extensions import override  # type: ignore
    except ImportError:  # pragma: no cover - hard fallback if typing_extensions missing

        def override(func):  # type: ignore
            return func


logger = logging.getLogger(__name__)

AUTH_CODE_TTL_SECONDS = 5 * 60
ACCESS_TOKEN_TTL_SECONDS = 60 * 60
REFRESH_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60

LOGIN_PATH = "/galaxy-auth/login"
RESOURCE_METADATA_PATH = "/.well-known/oauth-protected-resource"

CHATGPT_LOGO_DATA_URI = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0iNzIiIGhlaWdodD0iNzIiIHZpZXdCb3g9IjAgMCA3MiA3MiIgZmlsbD0ibm9u"
    "ZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjcyIiBo"
    "ZWlnaHQ9IjcyIiByeD0iMTgiIGZpbGw9IiMxMDEzMUYiLz4KPHBhdGggZD0iTTM2IDE3QzMwLjQ3"
    "NyAxNyAyNS42MjEgMjAuNTY3IDI0LjAwMSAyNS42NzZDMjAuOTI3IDI2Ljc4OCAxOC41IDI5Ljg1"
    "NSAxOC41IDMzLjVDMTguNSAzNy42NDIgMjEuODU4IDQxIDI2IDQxSDI3LjAzQzI3LjAxMSA0MS4z"
    "MzIgMjcgNDEuNjY1IDI3IDQyQzI3IDQ4LjA3NSAzMS45MjUgNTMgMzggNTNDNDMuNTIzIDUzIDQ4"
    "LjM3OSA0OS40MzMgNDkuOTk5IDQ0LjMyNEM1My4wNzMgNDMuMjEyIDU1LjUgNDAuMTQ1IDU1LjUg"
    "MzYuNUM1NS41IDMyLjM1OCA1Mi4xNDIgMjkgNDggMjlINDYuOTdDNDYuOTg5IDI4LjY2OCA0NyAy"
    "OC4zMzUgNDcgMjhDNDcgMjEuOTI1IDQyLjA3NSAxNyAzNiAxN1oiIHN0cm9rZT0iIzczRkZBOSIgc"
    "3Ryb2tlLXdpZHRoPSIzLjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPS"
    "Jyb3VuZCIvPgo8cGF0aCBkPSJNMjggNDFDMjguNTM3IDQ3LjUzOSAzNC4wMDMgNTMgNDAuNyA1MyI"
    "gc3Ryb2tlPSIjNENFMEIzIiBzdHJva2Utd2lkdGg9IjIuNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5k"
    "Ii8+CjxwYXRoIGQ9Ik00NCAyOUM0My40NjMgMjIuNDYxIDM3Ljk5NyAxNyAzMS4zIDE3IiBzdHJva"
    "2U9IiM0Q0UwQjMiIHN0cm9rZS13aWR0aD0iMi40IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPC"
    "9zdmc+"
)

GALAXY_LOGO_DATA_URI = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0iNzIiIGhlaWdodD0iNzIiIHZpZXdCb3g9IjAgMCA3MiA3MiIgZmlsbD0ibm9u"
    "ZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjcyIiBo"
    "ZWlnaHQ9IjcyIiByeD0iMTgiIGZpbGw9InVybCgjZykiLz4KPGRlZnM+CjxsaW5lYXJHcmFkaWVu"
    "dCBpZD0iZyIgeDE9IjE0IiB5MT0iMTIiIHgyPSI1OCIgeTI9IjYwIiBncmFkaWVudFVuaXRzPSJ1"
    "c2VyU3BhY2VPblVzZSI+CjxzdG9wIHN0b3AtY29sb3I9IiMxMjE4M0YiLz4KPHN0b3Agb2Zmc2V0"
    "PSIxIiBzdG9wLWNvbG9yPSIjMkY2REZGIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwvZGVmcz4KPHBh"
    "dGggZD0iTTM2IDU3QzQ0LjI4NCA1NyA1MSA0MSAzNiA0MUMyMSA0MSAyNy43MTYgNTcgMzYgNTda"
    "IiBmaWxsPSIjRkNEMzREIiBmaWxsLW9wYWNpdHk9IjAuODUiLz4KPHBhdGggZD0iTTE0IDMxQzI2"
    "LjUgMzEgMzIuNSA0MSAzMS41IDQ3QzMwLjUgNTMgMTkgNTcgMTQgNTciIHN0cm9rZT0iI0Y5RjdG"
    "RiIgc3Ryb2tlLXdpZHRoPSIyLjYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8cGF0aCBkPSJN"
    "NTggMzFDNDUuNSAzMSAzOS41IDQxIDQwLjUgNDdDNDEuNSA1MyA1MyA1NyA1OCA1NyIgc3Ryb2tl"
    "PSIjQzBEQUZGIiBzdHJva2Utd2lkdGg9IjIuNiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+Cjxj"
    "aXJjbGUgY3g9IjIxIiBjeT0iMjMiIHI9IjMiIGZpbGw9IiNGQ0QzNEQiLz4KPGNpcmNsZSBjeD0i"
    "NTEiIGN5PSIxOSIgcj0iMi41IiBmaWxsPSIjQzBEQUZGIi8+Cjwvc3ZnPg=="
)


class GalaxyAuthenticationError(Exception):
    """Raised when Galaxy authentication fails."""


@dataclass
class AuthorizationTransaction:
    """Stored data for an in-flight authorization request."""

    client_id: str
    redirect_uri: str
    redirect_uri_provided_explicitly: bool
    state: str | None
    code_challenge: str
    code_challenge_method: str
    scopes: list[str]
    created_at: float


@dataclass(frozen=True)
class GalaxyCredentials:
    """Decoded Galaxy credentials from an access token."""

    galaxy_url: str
    api_key: str
    username: str
    user_email: str | None
    expires_at: int
    scopes: list[str]
    client_id: str


class GalaxyOAuthProvider(OAuthProvider):
    """OAuth provider that authenticates users against a Galaxy instance."""

    def __init__(
        self,
        *,
        base_url: str,
        galaxy_url: str,
        required_scopes: list[str] | None = None,
        session_secret: str | None = None,
        client_registry_path: str | Path | None = None,
    ):
        client_registration = ClientRegistrationOptions(enabled=True)
        revocation_options = RevocationOptions(enabled=True)

        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must be a non-empty string")

        super_init = super().__init__
        super_params = inspect.signature(super_init).parameters
        super_kwargs: dict[str, Any] = {}
        if "base_url" in super_params:
            super_kwargs["base_url"] = normalized_base_url
        if "issuer_url" in super_params:
            super_kwargs["issuer_url"] = normalized_base_url
        if "service_documentation_url" in super_params:
            super_kwargs["service_documentation_url"] = None
        if "client_registration_options" in super_params:
            super_kwargs["client_registration_options"] = client_registration
        if "revocation_options" in super_params:
            super_kwargs["revocation_options"] = revocation_options
        if "required_scopes" in super_params:
            super_kwargs["required_scopes"] = required_scopes or ["galaxy:full"]

        super_init(**super_kwargs)

        self.base_url = normalized_base_url
        self.required_scopes = required_scopes or ["galaxy:full"]
        self._galaxy_url = galaxy_url if galaxy_url.endswith("/") else f"{galaxy_url}/"
        self._transactions: dict[str, AuthorizationTransaction] = {}
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._fernet = Fernet(self._derive_key(session_secret))
        self._client_registry_path = (
            Path(client_registry_path).expanduser() if client_registry_path else None
        )

        self._load_client_registry()

    # ------------------------------------------------------------------
    # OAuth provider interface
    # ------------------------------------------------------------------

    @override
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    @override
    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info
        await self._persist_client_registry()

    @override
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        txn_id = secrets.token_urlsafe(32)
        transaction = AuthorizationTransaction(
            client_id=client.client_id,
            redirect_uri=str(params.redirect_uri),
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            state=params.state,
            code_challenge=params.code_challenge,
            code_challenge_method=getattr(params, "code_challenge_method", "S256"),
            scopes=params.scopes or self.required_scopes,
            created_at=time.time(),
        )
        self._transactions[txn_id] = transaction

        base_url = str(self.base_url)
        login_url = construct_redirect_uri(
            f"{base_url.rstrip('/')}{LOGIN_PATH}",
            txn=txn_id,
            galaxy=self._galaxy_url.rstrip("/"),
        )
        logger.debug("Created authorization transaction %s for client %s", txn_id, client.client_id)
        return login_url

    @override
    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        try:
            payload = self._decrypt_payload(authorization_code, expected_type="authorization_code")
        except InvalidToken:
            return None

        if payload["client_id"] != client.client_id:
            return None

        if payload["exp"] < time.time():
            return None

        return AuthorizationCode(
            code=authorization_code,
            client_id=client.client_id,
            scopes=payload["scopes"],
            expires_at=payload["exp"],
            code_challenge=payload["code_challenge"],
            redirect_uri=payload["redirect_uri"],
            redirect_uri_provided_explicitly=payload["redirect_uri_provided_explicitly"],
            resource=None,
        )

    @override
    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        payload = self._decrypt_payload(authorization_code.code, expected_type="authorization_code")
        if payload["exp"] < time.time():
            raise GalaxyAuthenticationError("Authorization code expired.")

        if payload["client_id"] != client.client_id:
            raise GalaxyAuthenticationError("Authorization code issued for a different client.")

        galaxy_payload = payload["galaxy"]
        return self._issue_tokens(
            client_id=client.client_id, scopes=payload["scopes"], galaxy_payload=galaxy_payload
        )

    @override
    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        try:
            payload = self._decrypt_payload(refresh_token, expected_type="refresh")
        except InvalidToken:
            return None

        if payload["client_id"] != client.client_id:
            return None
        if payload["exp"] < time.time():
            return None

        return RefreshToken(
            token=refresh_token,
            client_id=payload["client_id"],
            scopes=payload["scopes"],
            expires_at=payload["exp"],
        )

    @override
    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        payload = self._decrypt_payload(refresh_token.token, expected_type="refresh")
        if payload["exp"] < time.time():
            raise GalaxyAuthenticationError("Refresh token expired.")

        if payload["client_id"] != client.client_id:
            raise GalaxyAuthenticationError("Refresh token issued for a different client.")

        resolved_scopes = scopes or payload["scopes"]
        return self._issue_tokens(
            client_id=client.client_id, scopes=resolved_scopes, galaxy_payload=payload["galaxy"]
        )

    @override
    async def load_access_token(self, token: str) -> AccessToken | None:
        try:
            payload = self._decrypt_payload(token, expected_type="access")
        except InvalidToken:
            return None

        if payload["exp"] < time.time():
            return None

        galaxy_info = payload["galaxy"]
        return FastMCPAccessToken(
            token=token,
            client_id=payload["client_id"],
            scopes=payload["scopes"],
            expires_at=payload["exp"],
            claims={
                "galaxy_url": galaxy_info["url"],
                "username": galaxy_info["username"],
                "user_email": galaxy_info.get("user_email"),
            },
        )

    @override
    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        # Stateless tokens cannot be selectively revoked without external storage.
        logger.debug(
            "Revocation requested for token, but stateless tokens cannot be revoked individually."
        )

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_base_path(base_path: str | None) -> str | None:
        if not base_path:
            return None

        normalized = base_path if base_path.startswith("/") else f"/{base_path}"
        normalized = normalized.rstrip("/")
        if not normalized or normalized == "/":
            return None
        return normalized

    def get_login_paths(self, base_path: str | None = None) -> set[str]:
        login_paths = {LOGIN_PATH}
        normalized = self._normalize_base_path(base_path)
        if normalized:
            login_paths.add(f"{normalized}{LOGIN_PATH}")
        return login_paths

    def get_resource_metadata_paths(self, base_path: str | None = None) -> set[str]:
        metadata_paths = {RESOURCE_METADATA_PATH}
        normalized = self._normalize_base_path(base_path)
        if normalized:
            metadata_paths.add(f"{normalized}{RESOURCE_METADATA_PATH}")
        return metadata_paths

    @override
    async def handle_login(self, request: Request) -> Response:
        """Public wrapper for the login handler so it can be registered on FastMCP routes."""
        return await self._login_handler(request)

    @override
    def get_resource_metadata(self) -> dict[str, Any]:
        """Return OAuth protected resource metadata."""
        return {
            "resource": self._galaxy_url,
            "authorization_servers": [self.base_url],
            "scopes_supported": self.required_scopes,
            "token_types_supported": ["Bearer"],
        }

    @override
    async def handle_resource_metadata(self, request: Request) -> Response:
        """Return OAuth protected resource metadata."""
        return JSONResponse(self.get_resource_metadata())

    @override
    def get_routes(
        self, mcp_path: str | None = None, mcp_endpoint: Any | None = None
    ) -> list[Route]:
        """
        Return the Starlette routes that expose the OAuth surface.

        FastMCP's base OAuth implementation eagerly registers generic login/resource-metadata
        endpoints. Galaxy-specific login handling needs to replace those with custom handlers
        (including base-path-prefixed variants), so we strip the parent definitions first and
        then install our own. This defensive dedupe also shields us from future FastMCP routing
        changes that might otherwise create duplicate routes and confusing behaviour.
        """
        routes = super().get_routes(mcp_path, mcp_endpoint)

        base_path = self._normalize_base_path(
            urlparse(str(self.base_url)).path if self.base_url else None
        )
        login_paths = self.get_login_paths(base_path)
        metadata_paths = self.get_resource_metadata_paths(base_path)

        routes = [
            route
            for route in routes
            if not (isinstance(route, Route) and route.path in login_paths | metadata_paths)
        ]

        existing_paths = {route.path for route in routes if isinstance(route, Route)}

        for path in login_paths:
            if path not in existing_paths:
                routes.append(Route(path, endpoint=self.handle_login, methods=["GET", "POST"]))
                existing_paths.add(path)

        for path in metadata_paths:
            if path not in existing_paths:
                routes.append(Route(path, endpoint=self.handle_resource_metadata, methods=["GET"]))
                existing_paths.add(path)

        return routes

    def _load_client_registry(self) -> None:
        if not self._client_registry_path:
            return

        path = self._client_registry_path
        try:
            if not path.exists():
                return

            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                return
            payload = json.loads(raw)
            if not isinstance(payload, list):
                logger.warning("Client registry at %s is not a list; ignoring contents.", path)
                return

            for entry in payload:
                try:
                    client = OAuthClientInformationFull.model_validate(entry)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to load client entry from registry: %s", exc)
                    continue
                self._clients[client.client_id] = client
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load client registry from %s: %s", path, exc)

    async def _persist_client_registry(self) -> None:
        if not self._client_registry_path:
            return

        path = self._client_registry_path
        clients_data = [
            client.model_dump(mode="json")
            for client in sorted(self._clients.values(), key=lambda c: c.client_id)
        ]

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(clients_data, fh, separators=(",", ":"), sort_keys=True)
            tmp_path.replace(path)

        try:
            await anyio.to_thread.run_sync(_write, abandon_on_cancel=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to persist client registry to %s: %s", path, exc)

    def decode_access_token(self, token: str) -> dict[str, Any] | None:
        try:
            payload = self._decrypt_payload(token, expected_type="access")
        except InvalidToken:
            return None

        if payload["exp"] < time.time():
            return None

        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_key(self, secret: str | None) -> bytes:
        if secret:
            digest = hashlib.sha256(secret.encode("utf-8")).digest()
            return base64.urlsafe_b64encode(digest)
        key = Fernet.generate_key()
        logger.warning(
            "GALAXY_MCP_SESSION_SECRET is not set; generated a volatile secret. "
            "All tokens will become invalid on restart."
        )
        return key

    def _encrypt_payload(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self._fernet.encrypt(serialized).decode("utf-8")

    def _decrypt_payload(self, token: str, *, expected_type: str) -> dict[str, Any]:
        data = self._fernet.decrypt(token.encode("utf-8"))
        payload: dict[str, Any] = json.loads(data.decode("utf-8"))
        if payload.get("typ") != expected_type:
            raise InvalidToken("Token type mismatch")
        return payload

    def _issue_tokens(
        self, *, client_id: str, scopes: list[str], galaxy_payload: dict[str, Any]
    ) -> OAuthToken:
        now = int(time.time())
        access_payload = {
            "typ": "access",
            "client_id": client_id,
            "scopes": scopes,
            "galaxy": galaxy_payload,
            "exp": now + ACCESS_TOKEN_TTL_SECONDS,
            "iat": now,
            "nonce": secrets.token_urlsafe(8),
        }

        refresh_payload = {
            "typ": "refresh",
            "client_id": client_id,
            "scopes": scopes,
            "galaxy": galaxy_payload,
            "exp": now + REFRESH_TOKEN_TTL_SECONDS,
            "iat": now,
            "nonce": secrets.token_urlsafe(8),
        }

        access_token = self._encrypt_payload(access_payload)
        refresh_token = self._encrypt_payload(refresh_payload)

        return OAuthToken(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=refresh_token,
            scope=" ".join(scopes),
        )

    async def _login_handler(self, request: Request) -> Response:
        txn_id = request.query_params.get("txn")
        if not txn_id:
            return PlainTextResponse("Missing transaction identifier.", status_code=400)

        transaction = self._transactions.get(txn_id)
        if not transaction:
            return PlainTextResponse("Authorization request is no longer valid.", status_code=400)

        if request.method == "GET":
            return self._render_login_form(transaction, error=request.query_params.get("error"))

        form = await request.form()
        username = (form.get("username") or "").strip()
        password = (form.get("password") or "").strip()

        if not username or not password:
            return self._render_login_form(transaction, error="Username and password are required.")

        try:
            redirect_url = await self._authenticate_and_complete(txn_id, username, password)
        except GalaxyAuthenticationError as exc:
            return self._render_login_form(transaction, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error during Galaxy login: %s", exc)
            return self._render_login_form(
                transaction, error="Unexpected error during authentication."
            )

        return RedirectResponse(redirect_url, status_code=303)

    async def _authenticate_and_complete(self, txn_id: str, username: str, password: str) -> str:
        transaction = self._transactions.pop(txn_id, None)
        if not transaction:
            raise GalaxyAuthenticationError(
                "Authorization request expired. Please restart the flow."
            )

        api_key = await self._get_api_key(username, password)
        user_info = await self._get_user_info(api_key)

        galaxy_payload = {
            "url": self._galaxy_url,
            "api_key": api_key,
            "username": user_info.get("username") or username,
            "user_email": user_info.get("email"),
        }

        code_payload = {
            "typ": "authorization_code",
            "client_id": transaction.client_id,
            "scopes": transaction.scopes,
            "code_challenge": transaction.code_challenge,
            "code_challenge_method": transaction.code_challenge_method,
            "redirect_uri": transaction.redirect_uri,
            "redirect_uri_provided_explicitly": transaction.redirect_uri_provided_explicitly,
            "galaxy": galaxy_payload,
            "exp": time.time() + AUTH_CODE_TTL_SECONDS,
            "nonce": secrets.token_urlsafe(8),
        }

        code_value = self._encrypt_payload(code_payload)

        logger.info("Galaxy authentication successful for user %s", galaxy_payload["username"])
        return construct_redirect_uri(
            transaction.redirect_uri, code=code_value, state=transaction.state
        )

    async def _get_api_key(self, username: str, password: str) -> str:
        url = f"{self._galaxy_url}api/authenticate/baseauth"

        def _request_api_key() -> str:
            response = requests.get(url, auth=(username, password), timeout=15)
            if response.status_code == 401:
                raise GalaxyAuthenticationError("Invalid Galaxy credentials.")
            response.raise_for_status()
            payload = response.json()
            key = payload.get("api_key")
            if not key:
                raise GalaxyAuthenticationError("Galaxy did not return an API key.")
            return key

        return await anyio.to_thread.run_sync(_request_api_key)

    async def _get_user_info(self, api_key: str) -> dict[str, Any]:
        def _fetch() -> dict[str, Any]:
            gi = GalaxyInstance(url=self._galaxy_url, key=api_key)
            return gi.users.get_current_user()

        try:
            return await anyio.to_thread.run_sync(_fetch)
        except Exception as exc:
            raise GalaxyAuthenticationError("Failed to validate API key with Galaxy.") from exc

    def _render_login_form(
        self, transaction: AuthorizationTransaction, error: str | None = None
    ) -> HTMLResponse:
        scopes_text = ", ".join(transaction.scopes) if transaction.scopes else "galaxy:full"
        error_html = f'<div class="alert" role="alert">{error}</div>' if error else ""
        html = textwrap.dedent(
            f"""
            <html lang="en">
                <head>
                    <meta charset="utf-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1" />
                    <title>Authorize ChatGPT for Galaxy</title>
                    <style>
                        :root {{ color-scheme: light dark; }}
                        body {{
                            margin: 0;
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background:
                                radial-gradient(
                                    circle at 20% 20%,
                                    #4c6ef5 0%,
                                    #2a2f4f 60%,
                                    #0b0d26 100%
                                );
                            font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
                            color: #101223;
                        }}
                        .card {{
                            width:
                                min(440px, calc(100% - 40px));
                            background: rgba(255, 255, 255, 0.96);
                            border-radius: 18px;
                            box-shadow: 0 18px 45px rgba(14, 20, 56, 0.25);
                            padding: 34px 38px;
                            box-sizing: border-box;
                        }}
                        .logo-strip {{
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 18px;
                            margin-bottom: 20px;
                        }}
                        .logo {{
                            position: relative;
                            width: 72px;
                            height: 72px;
                            border-radius: 18px;
                            flex: 0 0 72px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            overflow: hidden;
                        }}
                        .logo::after {{
                            content: "";
                            position: absolute;
                            inset: 0;
                            background-size: cover;
                            background-position: center;
                            background-repeat: no-repeat;
                        }}
                        .logo-chatgpt::after {{
                            background-image:
                                url("{CHATGPT_LOGO_DATA_URI}");
                        }}
                        .logo-galaxy::after {{
                            background-image:
                                url("{GALAXY_LOGO_DATA_URI}");
                        }}
                        .arrow {{
                            font-size: 2.5rem;
                            color: rgba(20, 33, 92, 0.85);
                        }}
                        .card h1 {{
                            margin: 0 0 12px;
                            font-size: 1.6rem;
                            color: #14215c;
                        }}
                        .card p {{
                            margin: 0 0 18px;
                            line-height: 1.5;
                        }}
                        form {{
                            display: grid;
                            gap: 16px;
                        }}
                        label {{
                            font-weight: 600;
                            font-size: 0.95rem;
                            color: #202552;
                        }}
                        input {{
                            width: 100%;
                            padding: 12px 14px;
                            border-radius: 10px;
                            border: 1px solid rgba(20, 33, 92, 0.25);
                            font-size: 1rem;
                            transition: border-color 0.2s ease, box-shadow 0.2s ease;
                            background: rgba(255, 255, 255, 0.95);
                        }}
                        input:focus {{
                            outline: none;
                            border-color: #4c6ef5;
                            box-shadow: 0 0 0 3px rgba(76, 110, 245, 0.25);
                        }}
                        button {{
                            margin-top: 8px;
                            padding: 13px 16px;
                            border-radius: 12px;
                            border: none;
                            background: linear-gradient(135deg, #4c6ef5, #5f9bff);
                            color: white;
                            font-size: 1rem;
                            font-weight: 600;
                            cursor: pointer;
                            transition: transform 0.2s ease, box-shadow 0.2s ease;
                        }}
                        button:hover {{
                            transform: translateY(-1px);
                            box-shadow: 0 12px 24px rgba(19, 64, 216, 0.3);
                        }}
                        .fine-print {{
                            margin-top: 12px;
                            font-size: 0.85rem;
                            color: #4b4f70;
                        }}
                        .alert {{
                            padding: 12px 16px;
                            border-radius: 10px;
                            background: rgba(220, 53, 69, 0.1);
                            color: #9e1c2f;
                            border: 1px solid rgba(220, 53, 69, 0.35);
                            margin-bottom: 18px;
                        }}
                        @media (prefers-color-scheme: dark) {{
                            .card {{
                                background: rgba(16, 18, 35, 0.88);
                                color: #eef0ff;
                            }}
                            label {{ color: #c9d1ff; }}
                            input {{
                                background: rgba(10, 12, 26, 0.9);
                                color: #eef0ff;
                                border-color: rgba(128, 140, 255, 0.35);
                            }}
                            .fine-print {{ color: rgba(216, 220, 255, 0.75); }}
                            .arrow {{ color: rgba(224, 231, 255, 0.9); }}
                        }}
                    </style>
                </head>
                <body>
                    <main class="card" role="dialog" aria-labelledby="title">
                        <div class="logo-strip" aria-hidden="true">
                            <div class="logo logo-chatgpt"></div>
                            <span class="arrow">→</span>
                            <div class="logo logo-galaxy"></div>
                        </div>
                        <h1 id="title">Allow ChatGPT to access Galaxy</h1>
                        <p>Sign in to <strong>{self._galaxy_url.rstrip('/')}</strong>.</p>
                        <p class="fine-print">Scopes: <strong>{scopes_text}</strong></p>
                        {error_html}
                        <form method="post">
                            <div>
                                <label for="username">Galaxy username or email</label>
                                <input
                                    id="username"
                                    name="username"
                                    type="text"
                                    autocomplete="username"
                                    autofocus
                                    required
                                />
                            </div>
                            <div>
                                <label for="password">Galaxy password</label>
                                <input
                                    id="password"
                                    name="password"
                                    type="password"
                                    autocomplete="current-password"
                                    required
                                />
                            </div>
                            <button type="submit">Allow access</button>
                        </form>
                        <p class="fine-print">
                            Need help?
                            <a href="https://usegalaxy.org.au" target="_blank" rel="noopener">
                                usegalaxy.org.au
                            </a>
                        </p>
                    </main>
                </body>
            </html>
            """
        )
        return HTMLResponse(html)


_AUTH_PROVIDER: GalaxyOAuthProvider | None = None


def configure_auth_provider(provider: GalaxyOAuthProvider) -> None:
    """Register the global auth provider instance."""
    global _AUTH_PROVIDER
    _AUTH_PROVIDER = provider


def get_auth_provider() -> GalaxyOAuthProvider | None:
    """Return the configured auth provider, if any."""
    return _AUTH_PROVIDER


def get_active_session(
    get_token: Callable[[], AccessToken | None],
) -> tuple[GalaxyCredentials | None, str | None]:
    """Decode the access token from the request and extract Galaxy credentials."""
    provider = get_auth_provider()
    if not provider:
        return None, None

    access_token = get_token()
    if access_token is None:
        return None, None

    token_payload = provider.decode_access_token(access_token.token)
    if not token_payload:
        return None, None

    galaxy_payload = token_payload["galaxy"]
    credentials = GalaxyCredentials(
        galaxy_url=galaxy_payload["url"],
        api_key=galaxy_payload["api_key"],
        username=galaxy_payload["username"],
        user_email=galaxy_payload.get("user_email"),
        expires_at=token_payload["exp"],
        scopes=token_payload["scopes"],
        client_id=token_payload["client_id"],
    )
    return credentials, credentials.api_key
