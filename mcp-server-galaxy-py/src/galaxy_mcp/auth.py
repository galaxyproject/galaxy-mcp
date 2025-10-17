"""Authentication helpers and stateless OAuth provider for the Galaxy MCP server."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import secrets
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

logger = logging.getLogger(__name__)

AUTH_CODE_TTL_SECONDS = 5 * 60
ACCESS_TOKEN_TTL_SECONDS = 60 * 60
REFRESH_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60

LOGIN_PATH = "/galaxy-auth/login"
RESOURCE_METADATA_PATH = "/.well-known/oauth-protected-resource"


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

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info
        await self._persist_client_registry()

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

    async def handle_login(self, request: Request) -> Response:
        """Public wrapper for the login handler so it can be registered on FastMCP routes."""
        return await self._login_handler(request)

    def get_resource_metadata(self) -> dict[str, Any]:
        """Return OAuth protected resource metadata."""
        return {
            "resource": self._galaxy_url,
            "authorization_servers": [self.base_url],
            "scopes_supported": self.required_scopes,
            "token_types_supported": ["Bearer"],
        }

    async def handle_resource_metadata(self, request: Request) -> Response:
        """Return OAuth protected resource metadata."""
        return JSONResponse(self.get_resource_metadata())

    def get_routes(
        self, mcp_path: str | None = None, mcp_endpoint: Any | None = None
    ) -> list[Route]:
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
        message = (
            "<p>Authenticate with your Galaxy credentials to allow this MCP server to access "
            "Galaxy on your behalf.</p>"
        )
        if error:
            message += f'<p style="color: red;">{error}</p>'

        html = f"""
        <html>
            <head>
                <title>Galaxy Login</title>
            </head>
            <body>
                <h1>Sign in to Galaxy</h1>
                {message}
                <form method="post">
                    <label for="username">Username or email</label><br />
                    <input id="username" name="username" type="text" autofocus required /><br />
                    <label for="password">Password</label><br />
                    <input id="password" name="password" type="password" required /><br />
                    <button type="submit">Sign in</button>
                </form>
            </body>
        </html>
        """
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
