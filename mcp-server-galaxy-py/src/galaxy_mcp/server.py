# Galaxy MCP Server
import concurrent.futures
import json
import logging
import os
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from bioblend.galaxy import GalaxyInstance
from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP
from mcp.server.auth.middleware.auth_context import get_access_token
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from galaxy_mcp.auth import (
    GalaxyOAuthProvider,
    configure_auth_provider,
    get_active_session,
)

# Set up logging
logging_level = os.environ.get("GALAXY_MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))
logger = logging.getLogger(__name__)


def format_error(action: str, error: Exception, context: dict | None = None) -> str:
    """Format error messages consistently"""
    if context is None:
        context = {}
    msg = f"{action} failed: {str(error)}"

    # Add HTTP status code interpretations
    error_str = str(error)
    if "401" in error_str:
        msg += " (Authentication failed - check your API key)"
    elif "403" in error_str:
        msg += " (Permission denied - check your account permissions)"
    elif "404" in error_str:
        msg += " (Resource not found - check IDs and URLs)"
    elif "500" in error_str:
        msg += " (Server error - try again later or contact admin)"

    # Add context if provided
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        msg += f". Context: {context_str}"

    return msg


# Try to load environment variables from .env file
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")

# Galaxy configuration (target Galaxy instance)
raw_galaxy_url = os.environ.get("GALAXY_URL")
normalized_galaxy_url = (
    raw_galaxy_url if not raw_galaxy_url or raw_galaxy_url.endswith("/") else f"{raw_galaxy_url}/"
)
galaxy_state: dict[str, Any] = {
    "url": normalized_galaxy_url,
    "api_key": None,
    "gi": None,
    "connected": False,
}

# Configure OAuth provider if Galaxy URL and public URL are specified
public_base_url = os.environ.get("GALAXY_MCP_PUBLIC_URL")
session_secret = os.environ.get("GALAXY_MCP_SESSION_SECRET")
client_registry_path_env = os.environ.get("GALAXY_MCP_CLIENT_REGISTRY")
default_registry_path = Path.home() / ".galaxy-mcp" / "clients.json"
client_registry_path = (
    Path(client_registry_path_env).expanduser()
    if client_registry_path_env
    else default_registry_path
)
auth_provider: GalaxyOAuthProvider | None = None
if public_base_url and normalized_galaxy_url:
    try:
        auth_provider = GalaxyOAuthProvider(
            base_url=public_base_url,
            galaxy_url=normalized_galaxy_url,
            session_secret=session_secret,
            client_registry_path=client_registry_path,
        )
        configure_auth_provider(auth_provider)
        logger.info("OAuth login enabled for Galaxy at %s", normalized_galaxy_url)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to initialize OAuth provider: %s", exc, exc_info=True)
elif public_base_url and not normalized_galaxy_url:
    logger.warning(
        "GALAXY_MCP_PUBLIC_URL is set but GALAXY_URL is missing. OAuth login is disabled until "
        "GALAXY_URL is configured."
    )
elif not public_base_url:
    logger.info(
        "GALAXY_MCP_PUBLIC_URL not set. OAuth login is disabled; falling back to API key "
        "authentication."
    )

# Create an MCP server (with auth if available)
mcp: FastMCP = FastMCP("Galaxy", auth=auth_provider)


class _PreflightMiddleware(BaseHTTPMiddleware):
    """Allow browser CORS preflight requests to bypass FastMCP auth handling."""

    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin", "*")
        allow_methods = request.headers.get("access-control-request-method", "POST,GET,OPTIONS")
        allow_headers = request.headers.get(
            "access-control-request-headers", "authorization,content-type"
        )

        cors_headers = {
            "access-control-allow-origin": origin,
            "access-control-allow-methods": allow_methods,
            "access-control-allow-headers": allow_headers,
            "access-control-max-age": "600",
        }

        if request.method.upper() == "OPTIONS":
            return Response(status_code=204, headers=cors_headers)

        response = await call_next(request)
        for header, value in cors_headers.items():
            response.headers.setdefault(header, value)
        return response


_original_http_app = FastMCP.http_app


def _http_app_with_preflight(self, *args, **kwargs):
    app = _original_http_app(self, *args, **kwargs)
    app.add_middleware(_PreflightMiddleware)
    return app


mcp.http_app = types.MethodType(_http_app_with_preflight, mcp)


def _build_search_input_schema(
    *,
    include_deleted: bool = False,
    include_published: bool = False,
) -> dict[str, Any]:
    """Create a strong JSON schema for search tool inputs."""
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "term": {
                "type": "string",
                "description": (
                    "Search term matched against resource names, identifiers, and descriptions."
                ),
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "minimum": 1,
                "default": 5,
            },
        },
        "required": ["term"],
    }
    if include_deleted:
        schema["properties"]["include_deleted"] = {
            "type": "boolean",
            "description": "Include deleted resources when supported by the Galaxy API.",
            "default": False,
        }
    if include_published:
        schema["properties"]["include_published"] = {
            "type": "boolean",
            "description": (
                "Include published or shared resources when supported by the Galaxy API."
            ),
            "default": False,
        }
    return schema


def _search_result_item_schema(source: str) -> dict[str, Any]:
    """Create a JSON schema describing a search result entry."""
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "resource_id": {
                "type": "string",
                "description": "Identifier that can be used to reference the resource.",
            },
            "source": {"type": "string", "enum": [source]},
            "name": {"type": "string", "description": "Display name of the resource."},
            "summary": {"type": "string", "description": "Short human-readable summary."},
            "score": {
                "type": "number",
                "description": "Relative match score in the range [0, 1].",
            },
            "metadata": {
                "type": "object",
                "description": "Key highlights extracted from the resource.",
            },
            "details": {
                "type": "object",
                "description": "Full metadata payload returned by Galaxy for this resource.",
            },
        },
        "required": ["resource_id", "source", "name", "summary", "score", "details"],
    }


def _build_search_response_schema(source: str) -> dict[str, Any]:
    """Construct a JSON schema for search tool responses."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": {"type": "string", "enum": [source]},
            "term": {"type": "string", "description": "Sanitized search term."},
            "limit": {
                "type": "integer",
                "description": "Configured maximum number of results.",
                "minimum": 1,
            },
            "total": {
                "type": "integer",
                "description": "Number of results returned.",
                "minimum": 0,
            },
            "matches": {
                "type": "array",
                "description": f"List of matching {source.replace('_', ' ')}.",
                "items": _search_result_item_schema(source),
            },
        },
        "required": ["source", "term", "limit", "total", "matches"],
    }


def _normalize_term(term: str) -> str:
    return term.strip().lower()


def _score_match(name: str | None, identifier: str | None, term_lower: str) -> float:
    if not term_lower:
        return 0.5
    if name:
        name_l = name.lower()
        if name_l == term_lower:
            return 1.0
        if term_lower in name_l:
            return min(0.95, 0.6 + len(term_lower) / max(len(name_l), 1))
    if identifier:
        id_l = identifier.lower()
        if id_l == term_lower:
            return 0.9
        if term_lower in id_l:
            return 0.75
    return 0.5


def _format_search_result(
    source: str,
    identifier: str | tuple[str, ...],
    name: str | None,
    summary: str,
    term_lower: str,
    extra: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    encoded_id = ":".join(identifier) if isinstance(identifier, tuple) else identifier
    display_name = name or encoded_id
    metadata = extra or {}
    result: dict[str, Any] = {
        "resource_id": f"{source}:{encoded_id}",
        "source": source,
        "name": display_name,
        "summary": summary,
        "score": round(_score_match(display_name, encoded_id, term_lower), 3),
    }
    if metadata:
        result["metadata"] = metadata
    result["details"] = details or {}
    return result


def _split_tool_identifier(tool_id: str) -> tuple[str, str | None]:
    if not tool_id or "/" not in tool_id:
        return tool_id, None
    base_id, version = tool_id.rsplit("/", 1)
    if not version:
        return base_id, None
    return base_id, version


def _resolve_tool_version_name(version_entry: Any) -> str | None:
    if version_entry is None:
        return None
    if isinstance(version_entry, str):
        return version_entry
    if isinstance(version_entry, dict):
        if version_entry.get("version"):
            return version_entry["version"]
        if version_entry.get("name"):
            return version_entry["name"]
        version_id = version_entry.get("id")
        if isinstance(version_id, str):
            _, candidate = _split_tool_identifier(version_id)
            return candidate
    return None


def _safe_call(action: str, func: Callable[[], Any]) -> Any | None:
    try:
        return func()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("%s failed during search aggregation: %s", action, exc)
        return None


def _get_request_connection_state() -> dict[str, Any]:
    """
    Resolve the effective Galaxy connection for the current request.

    Returns a dictionary containing the Galaxy URL, API key, GalaxyInstance client,
    connection source, and (optionally) the active OAuth session object.
    """
    credentials, api_key = get_active_session(get_access_token)
    url = credentials.galaxy_url if credentials else normalized_galaxy_url
    if credentials and api_key:
        gi = GalaxyInstance(url=credentials.galaxy_url, key=api_key)
        return {
            "url": credentials.galaxy_url,
            "api_key": api_key,
            "gi": gi,
            "connected": True,
            "source": "oauth",
            "session": credentials,
        }

    # Test override (used by pytest to simulate connections without OAuth)
    if (
        os.environ.get("PYTEST_CURRENT_TEST")
        and galaxy_state.get("connected")
        and galaxy_state.get("gi")
    ):
        return {
            "url": galaxy_state.get("url") or normalized_galaxy_url,
            "api_key": galaxy_state.get("api_key"),
            "gi": galaxy_state.get("gi"),
            "connected": True,
            "source": "test",
            "session": None,
        }

    return {
        "url": url,
        "api_key": None,
        "gi": None,
        "connected": False,
        "source": None,
        "session": None,
    }


def ensure_connected() -> dict[str, Any]:
    """Resolve the active Galaxy connection, raising if none is available."""
    state = _get_request_connection_state()
    if not state["connected"] or not state["gi"]:
        raise ValueError(
            "Not connected to Galaxy. Please authenticate through your MCP client "
            "to establish a Galaxy session."
        )
    return state


# Shared JSON schema fragments used by MCP tool annotations.
EMPTY_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
    "required": [],
}

GENERIC_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}

STRING_SCHEMA = {"type": "string"}
INTEGER_SCHEMA = {"type": "integer"}
BOOLEAN_SCHEMA = {"type": "boolean"}

GENERIC_OBJECT_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": GENERIC_OBJECT_SCHEMA,
}


RUN_TOOL_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "history_id": {
            "type": "string",
            "description": "Target Galaxy history identifier (encoded).",
        },
        "tool_id": {
            "type": "string",
            "description": "Galaxy tool identifier to execute.",
        },
        "inputs": {
            "type": "object",
            "description": "Tool input payload matching Galaxy's tool schema.",
        },
    },
    "required": ["history_id", "tool_id", "inputs"],
}

CREATE_HISTORY_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "history_name": {
            "type": "string",
            "description": "Name for the new history.",
        }
    },
    "required": ["history_name"],
}

FILTER_TOOLS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dataset_type": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Keywords describing the dataset (e.g. ['csv', 'tabular']).",
        }
    },
    "required": ["dataset_type"],
}

GET_HISTORIES_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "limit": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "Maximum number of histories to return (None for all).",
        },
        "offset": {
            "type": "integer",
            "minimum": 0,
            "description": "Offset for pagination.",
        },
        "name": {
            "type": ["string", "null"],
            "description": "Optional name filter (case-sensitive substring).",
        },
        "ids_only": {
            "type": "boolean",
            "description": "Return simplified id/name pairs when true.",
            "default": False,
        },
    },
    "required": [],
}

GET_HISTORY_DETAILS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "history_id": {
            "type": "string",
            "description": "Galaxy history identifier to inspect.",
        }
    },
    "required": ["history_id"],
}

GET_HISTORY_CONTENTS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "history_id": {"type": "string", "description": "History id to list contents for."},
        "limit": {"type": "integer", "minimum": 1, "default": 100},
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "deleted": {"type": "boolean", "default": False},
        "visible": {"type": "boolean", "default": True},
        "details": {"type": "boolean", "default": False},
        "order": {"type": "string", "default": "hid-asc"},
    },
    "required": ["history_id"],
}

GET_JOB_DETAILS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dataset_id": {
            "type": "string",
            "description": "Dataset identifier whose producing job should be inspected.",
        },
        "history_id": {
            "type": ["string", "null"],
            "description": "Optional history id for disambiguation.",
        },
    },
    "required": ["dataset_id"],
}

GET_DATASET_DETAILS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dataset_id": {"type": "string"},
        "include_preview": {"type": "boolean", "default": True},
        "preview_lines": {"type": "integer", "minimum": 1, "default": 10},
    },
    "required": ["dataset_id"],
}

DOWNLOAD_DATASET_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dataset_id": {"type": "string"},
        "file_path": {"type": ["string", "null"], "description": "Filesystem path to save to."},
        "use_default_filename": {"type": "boolean", "default": True},
        "require_ok_state": {"type": "boolean", "default": True},
    },
    "required": ["dataset_id"],
}

UPLOAD_FILE_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "path": {"type": "string", "description": "Local path to upload."},
        "history_id": {
            "type": ["string", "null"],
            "description": "Target history id (defaults to current).",
        },
    },
    "required": ["path"],
}

GET_INVOCATIONS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "invocation_id": {"type": ["string", "null"]},
        "workflow_id": {"type": ["string", "null"]},
        "history_id": {"type": ["string", "null"]},
        "limit": {"type": ["integer", "null"], "minimum": 1},
        "view": {"type": "string", "enum": ["collection", "element"], "default": "collection"},
        "step_details": {"type": "boolean", "default": False},
    },
    "required": [],
}

IWC_WORKFLOWS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "term": {
            "type": ["string", "null"],
            "description": "Optional search term applied to workflow names, annotations, and tags.",
        }
    },
    "required": [],
}

IMPORT_IWC_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "trs_id": {
            "type": "string",
            "description": "TRS identifier for the workflow to import.",
        }
    },
    "required": ["trs_id"],
}

HISTORIES_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "histories": GENERIC_OBJECT_ARRAY_SCHEMA,
        "pagination": GENERIC_OBJECT_SCHEMA,
        "ids_only": {"type": "boolean"},
    },
    "required": ["histories", "pagination"],
}

HISTORY_DETAILS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "history": GENERIC_OBJECT_SCHEMA,
        "contents_summary": GENERIC_OBJECT_SCHEMA,
    },
    "required": ["history", "contents_summary"],
}

HISTORY_CONTENTS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "history_id": {"type": "string"},
        "contents": GENERIC_OBJECT_ARRAY_SCHEMA,
        "pagination": GENERIC_OBJECT_SCHEMA,
    },
    "required": ["history_id", "contents", "pagination"],
}

IWC_WORKFLOWS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "term": {"type": ["string", "null"]},
        "total": {"type": "integer", "minimum": 0},
        "matched": {"type": "integer", "minimum": 0},
        "workflows": GENERIC_OBJECT_ARRAY_SCHEMA,
    },
    "required": ["total", "matched", "workflows"],
}


def _collect_tool_details(
    gi: GalaxyInstance,
    tool_identifier: str,
    version_override: str | None = None,
) -> dict[str, Any]:
    """Retrieve detailed tool metadata including all known versions and citations."""
    base_id, embedded_version = _split_tool_identifier(tool_identifier)
    version_spec = version_override or embedded_version

    def _show_tool(version: str | None) -> dict[str, Any]:
        kwargs = {"io_details": True}
        if version:
            kwargs["tool_version"] = version
        return gi.tools.show_tool(base_id, **kwargs)

    requested_version = version_spec

    latest_details = _show_tool(None)
    version_entries = latest_details.get("versions") or []
    aggregated: list[dict[str, Any]] = []
    collected_citations: list[Any] = []
    seen_versions: set[str] = set()
    seen_citation_hashes: set[str] = set()

    def _collect_citations(details: dict[str, Any]) -> None:
        citations = details.get("citations") or []
        for citation in citations:
            try:
                signature = json.dumps(citation, sort_keys=True)
            except TypeError:
                signature = repr(citation)
            if signature in seen_citation_hashes:
                continue
            seen_citation_hashes.add(signature)
            collected_citations.append(citation)

    def _add_version(details: dict[str, Any]) -> None:
        resolved_version = details.get("version")
        aggregated.append(
            {
                "version": resolved_version,
                "id": details.get("id"),
                "resource_id": f"tools:{details.get('id')}" if details.get("id") else None,
                "details": details,
            }
        )
        if resolved_version:
            seen_versions.add(resolved_version)
        _collect_citations(details)

    _add_version(latest_details)

    for version_entry in version_entries:
        version_name = _resolve_tool_version_name(version_entry)
        if not version_name or version_name in seen_versions:
            continue
        details = _show_tool(version_name)
        if version_name in seen_versions:
            continue
        _add_version(details)

    if requested_version and requested_version not in seen_versions:
        requested_details = _show_tool(requested_version)
        _add_version(requested_details)

    if requested_version:
        aggregated.sort(
            key=lambda entry: (
                entry.get("version") != requested_version,
                entry.get("version") or "",
            )
        )

    return {
        "tool_id": base_id,
        "versions": aggregated,
        "citations": collected_citations,
    }


@mcp.tool(
    name="search_tools",
    description="Search Galaxy tools and return detailed metadata for matching tools.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Tools",
        "aiInputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "term": {
                    "type": "string",
                    "description": (
                        "Substring matched against tool names, identifiers, "
                        "descriptions, and labels."
                    ),
                    "minLength": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tool groups to return.",
                    "minimum": 1,
                    "default": 5,
                },
            },
            "required": ["term"],
        },
        "aiResponseSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source": {"type": "string", "enum": ["tools"]},
                "term": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
                "total": {"type": "integer", "minimum": 0},
                "matches": {
                    "type": "array",
                    "items": _search_result_item_schema("tools"),
                },
            },
            "required": ["source", "term", "limit", "total", "matches"],
        },
    },
)
def search_tools(term: str, limit: int = 5) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    grouped: dict[str, dict[str, Any]] = {}
    seen_tool_versions: set[str] = set()
    candidates: list[dict[str, Any]] = []

    filtered = _safe_call("tools.get_tools (filtered)", lambda: gi.tools.get_tools(name=term_lower))
    if filtered:
        candidates.extend(filtered)
    if len(candidates) < limit:
        all_tools = _safe_call("tools.get_tools", gi.tools.get_tools) or []
        candidates.extend(all_tools)

    for tool in candidates:
        tool_id = tool.get("id")
        if not tool_id or tool_id in seen_tool_versions:
            continue
        seen_tool_versions.add(tool_id)

        base_id, version_name = _split_tool_identifier(tool_id)
        name = tool.get("name") or base_id
        description = tool.get("description") or ""
        labels = tool.get("labels", [])
        searchable_text = " ".join(
            filter(
                None,
                [
                    name,
                    base_id,
                    tool_id,
                    version_name,
                    description,
                    " ".join(labels) if isinstance(labels, list) else "",
                ],
            )
        ).lower()
        if term_lower not in searchable_text:
            continue

        group = grouped.setdefault(
            base_id,
            {
                "name": name,
                "description": description,
                "tool_type": tool.get("tool_type"),
                "labels": set(),
                "versions": [],
                "score": 0.0,
            },
        )

        if description and not group["description"]:
            group["description"] = description
        if tool.get("tool_type") and not group["tool_type"]:
            group["tool_type"] = tool.get("tool_type")
        if isinstance(labels, list):
            group["labels"].update(labels)

        version_label = tool.get("version") or version_name
        group["versions"].append(
            {
                "id": tool_id,
                "version": version_label,
                "description": description,
                "resource_id": f"tools:{tool_id}",
            }
        )

        version_score = _score_match(name, tool_id, term_lower)
        if version_label:
            version_score = max(version_score, _score_match(version_label, tool_id, term_lower))
        group["score"] = max(group["score"], version_score)

    matches: list[dict[str, Any]] = []
    for base_id, info in sorted(
        grouped.items(),
        key=lambda item: (-item[1]["score"], item[1]["name"]),
    ):
        versions = info["versions"]
        versions.sort(
            key=lambda entry: (
                entry["version"] or "",
                entry["id"],
            ),
            reverse=True,
        )
        version_names = [entry["version"] for entry in versions if entry.get("version")]
        description = info["description"] or "Galaxy tool"
        if version_names:
            versions_preview = ", ".join(version_names[:3])
            if len(version_names) > 3:
                versions_preview += ", …"
            description = f"{description} (versions: {versions_preview})"
        metadata = {
            "tool_id": base_id,
            "tool_type": info["tool_type"],
            "labels": sorted(info["labels"]),
            "versions": versions,
        }
        details_payload = _safe_call(
            "tools.collect_details", lambda base=base_id: _collect_tool_details(gi, base)
        ) or {"tool_id": base_id, "versions": versions}
        result = _format_search_result(
            "tools",
            base_id,
            info["name"],
            description,
            term_lower,
            metadata,
            details_payload,
        )
        result["score"] = round(info["score"], 3)
        matches.append(result)
        if len(matches) >= limit:
            break
    return {
        "source": "tools",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    annotations={
        "readOnlyHint": "false",
        "aiInputSchema": RUN_TOOL_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def run_tool(history_id: str, tool_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Run a tool in Galaxy

    Args:
        history_id: Galaxy history ID where to run the tool - a hexadecimal hash string
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)
        tool_id: Galaxy tool identifier - typically in format 'toolshed.g2.bx.psu.edu/repos/...'
                (e.g., 'Cut1' for simple tools or full toolshed URLs for complex tools)
        inputs: Dictionary of tool input parameters and dataset references matching tool schema

    Returns:
        Dictionary containing tool execution information including job IDs and output dataset IDs
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Run the tool with provided inputs
        result = gi.tools.run_tool(history_id, tool_id, inputs)
        return result
    except Exception as e:
        error_msg = f"Failed to run tool '{tool_id}' in history '{history_id}': {str(e)}"
        if "400" in str(e) or "bad request" in str(e).lower():
            error_msg += " Check that all required tool parameters are provided correctly."
        elif "404" in str(e):
            error_msg += " Verify the tool ID and history ID are valid."
        else:
            error_msg += " Check the tool inputs format matches the tool's requirements."

        raise ValueError(error_msg) from e


@mcp.tool(
    annotations={
        "readOnlyHint": "false",
        "aiInputSchema": CREATE_HISTORY_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def create_history(history_name: str) -> dict[str, Any]:
    """
    Create a new history in Galaxy

    Args:
        history_name: Human-readable name for the new history (e.g., 'RNA-seq Analysis')

    Returns:
        Dictionary containing the created history details including the new history ID hash
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    return gi.histories.create_history(history_name)


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": FILTER_TOOLS_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def filter_tools_by_dataset(dataset_type: list[str]) -> dict[str, Any]:
    """
    Filter Galaxy tools that are potentially suitable for a given dataset type.

    Args:
        dataset_type (list[str]): A list of keywords or phrases describing the dataset type,
                                e.g., ['csv', 'tsv']. if the dataset type is csv or tsv,
                                please provide ['csv', 'tabular'] or ['tsv', 'tabular'].

    Returns:
        dict: A dictionary containing the list of recommended tools and the total count.
    """

    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    lock = threading.Lock()

    dataset_keywords = [dt.lower() for dt in dataset_type]

    try:
        tool_panel = gi.tools.get_tool_panel()

        def flatten_tools(panel):
            tools = []
            if isinstance(panel, list):
                for item in panel:
                    tools.extend(flatten_tools(item))
            elif isinstance(panel, dict):
                if "elems" in panel:
                    for item in panel["elems"]:
                        tools.extend(flatten_tools(item))
                else:
                    # Assume this dict represents a tool if no sub-elements exist.
                    tools.append(panel)
            return tools

        all_tools = flatten_tools(tool_panel)
        recommended_tools = []

        # Separate tools that already match by name/description.
        tools_to_fetch = []
        for tool in all_tools:
            name = (tool.get("name") or "").lower()
            description = (tool.get("description") or "").lower()
            if any(kw in name for kw in dataset_keywords) or any(
                kw in description for kw in dataset_keywords
            ):
                recommended_tools.append(tool)
            else:
                tools_to_fetch.append(tool)

        # Define a helper to check each tool's details.
        def check_tool(tool):
            tool_id = tool.get("id")
            if not tool_id:
                return None
            if tool_id.endswith("_label"):
                return None
            try:
                tool_details = gi.tools.show_tool(tool_id, io_details=True)
                tool_inputs = tool_details.get("inputs", [{}])
                for input_spec in tool_inputs:
                    if not isinstance(input_spec, dict):
                        continue
                    fmt = input_spec.get("extensions", "")
                    # 'extensions' might be a list or a string.
                    if isinstance(fmt, list):
                        for ext in fmt:
                            if ext and any(kw in ext.lower() for kw in dataset_keywords):
                                return tool
                    elif (
                        isinstance(fmt, str)
                        and fmt
                        and any(kw in fmt.lower() for kw in dataset_keywords)
                    ):
                        return tool
                return None
            except Exception:
                return None

        # Use a thread pool to concurrently check tools that require detail retrieval.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_tool = {executor.submit(check_tool, tool): tool for tool in tools_to_fetch}
            for future in concurrent.futures.as_completed(future_to_tool):
                result = future.result()
                if result is not None:
                    # Use the lock to ensure thread-safe appending.
                    with lock:
                        recommended_tools.append(result)

        slim_tools = []
        for tool in recommended_tools:
            slim_tools.append(
                {
                    "id": tool.get("id", ""),
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "versions": tool.get("versions", []),
                }
            )
        return {"recommended_tools": slim_tools, "count": len(slim_tools)}
    except Exception as e:
        raise ValueError(f"Failed to filter tools based on dataset: {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": EMPTY_OBJECT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def get_server_info() -> dict[str, Any]:
    """
    Get Galaxy server information including version, URL, and configuration details

    Returns:
        Server information including version, URL, and other configuration details
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get server configuration info
        config_info = gi.config.get_config()

        # Get server version info
        version_info = gi.config.get_version()

        # Build comprehensive server info response
        server_info = {
            "url": state["url"],
            "version": version_info,
            "config": {
                "brand": config_info.get("brand", "Galaxy"),
                "logo_url": config_info.get("logo_url"),
                "welcome_url": config_info.get("welcome_url"),
                "support_url": config_info.get("support_url"),
                "citation_url": config_info.get("citation_url"),
                "terms_url": config_info.get("terms_url"),
                "allow_user_creation": config_info.get("allow_user_creation"),
                "allow_user_deletion": config_info.get("allow_user_deletion"),
                "enable_quotas": config_info.get("enable_quotas"),
                "ftp_upload_site": config_info.get("ftp_upload_site"),
                "wiki_url": config_info.get("wiki_url"),
                "screencasts_url": config_info.get("screencasts_url"),
                "library_import_dir": config_info.get("library_import_dir"),
                "user_library_import_dir": config_info.get("user_library_import_dir"),
                "allow_library_path_paste": config_info.get("allow_library_path_paste"),
                "enable_unique_workflow_defaults": config_info.get(
                    "enable_unique_workflow_defaults"
                ),
            },
        }

        return server_info
    except Exception as e:
        raise ValueError(f"Failed to get server information: {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": EMPTY_OBJECT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def get_user() -> dict[str, Any]:
    """
    Get current user information

    Returns:
        Current user details
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        user_info = gi.users.get_current_user()
        return user_info
    except Exception as e:
        raise ValueError(f"Failed to get user: {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": GET_HISTORIES_INPUT_SCHEMA,
        "aiResponseSchema": HISTORIES_RESPONSE_SCHEMA,
    },
)
def get_histories(
    limit: int | None = None,
    offset: int = 0,
    name: str | None = None,
    ids_only: bool = False,
) -> dict[str, Any]:
    """
    Get paginated list of user histories

    Args:
        limit: Maximum number of histories to return (default: None for all histories)
        offset: Number of histories to skip from the beginning (default: 0, for pagination)
        name: Filter histories by name pattern (optional, case-sensitive partial match)

    Returns:
        Dictionary containing list of histories and pagination metadata
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get histories with pagination and optional filtering
        histories = gi.histories.get_histories(limit=limit, offset=offset, name=name)

        # If pagination is used, get total count for metadata
        if limit is not None:
            # Get total count without pagination
            all_histories = gi.histories.get_histories(name=name)
            total_items = len(all_histories) if all_histories else 0

            # Calculate pagination metadata
            has_next = (offset + limit) < total_items
            has_previous = offset > 0
            current_page = (offset // limit) + 1 if limit > 0 else 1
            total_pages = ((total_items - 1) // limit) + 1 if limit > 0 and total_items > 0 else 1

            pagination = {
                "total_items": total_items,
                "returned_items": len(histories),
                "limit": limit,
                "offset": offset,
                "current_page": current_page,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous,
                "next_offset": offset + limit if has_next else None,
                "previous_offset": max(0, offset - limit) if has_previous else None,
                "helper_text": f"Page {current_page} of {total_pages}. "
                + (
                    f"Use offset={offset + limit} for next page."
                    if has_next
                    else "This is the last page."
                ),
            }
        else:
            # No pagination requested, return simple count
            pagination = {
                "total_items": len(histories),
                "returned_items": len(histories),
                "paginated": False,
            }

        if ids_only:
            simplified = [
                {"id": history.get("id"), "name": history.get("name", "Unnamed")}
                for history in histories
                if history.get("id")
            ]
            return {
                "histories": simplified,
                "pagination": pagination,
                "ids_only": True,
            }

        return {"histories": histories, "pagination": pagination, "ids_only": False}
    except Exception as e:
        raise ValueError(
            f"Failed to get histories: {str(e)}. "
            "Check your connection to Galaxy and that you have "
            "permission to view histories."
        )


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": GET_HISTORY_DETAILS_INPUT_SCHEMA,
        "aiResponseSchema": HISTORY_DETAILS_RESPONSE_SCHEMA,
    },
)
def get_history_details(history_id: str) -> dict[str, Any]:
    """
    Get history metadata and summary count ONLY - does not return actual datasets

    This function provides quick access to history information without loading all datasets.
    For the actual datasets/contents, use get_history_contents() which supports
    pagination and ordering.

    Args:
        history_id: Galaxy history ID - a hexadecimal hash string identifying the history
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)

    Returns:
        Dictionary containing:
        - history: Basic history metadata (name, id, state, etc.)
        - contents_summary: Just the count of datasets, not the datasets themselves

        To get actual datasets: Use get_history_contents(history_id, limit=N,
                                         order="create_time-dsc")
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        logger.info(f"Getting details for history ID: {history_id}")

        # Get history details
        history_info = gi.histories.show_history(history_id, contents=False)
        logger.info(f"Successfully retrieved history info: {history_info.get('name', 'Unknown')}")

        # Get total count by calling without limit
        all_contents = gi.histories.show_history(history_id, contents=True)
        total_items = len(all_contents) if all_contents else 0

        return {
            "history": history_info,
            "contents_summary": {
                "total_items": total_items,
                "note": "This is just a count. To get actual datasets, use get_history_contents("
                "history_id, limit=25, order='create_time-dsc') for newest datasets first.",
            },
        }
    except Exception as e:
        logger.error(f"Failed to get history details for ID '{history_id}': {str(e)}")
        if "404" in str(e) or "No route" in str(e):
            raise ValueError(
                f"History ID '{history_id}' not found. Make sure to pass a valid history ID string."
            ) from e
        raise ValueError(f"Failed to get history details for ID '{history_id}': {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": GET_HISTORY_CONTENTS_INPUT_SCHEMA,
        "aiResponseSchema": HISTORY_CONTENTS_RESPONSE_SCHEMA,
    },
)
def get_history_contents(
    history_id: str,
    limit: int = 100,
    offset: int = 0,
    deleted: bool = False,
    visible: bool = True,
    details: bool = False,
    order: str = "hid-asc",
) -> dict[str, Any]:
    """
    Get paginated contents (datasets) from a specific history with ordering support

    Args:
        history_id: Galaxy history ID - a hexadecimal hash string identifying the history
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)
        limit: Maximum number of items to return per page (default: 100, max recommended: 500)
        offset: Number of items to skip from the beginning (default: 0, for pagination)
        deleted: Include deleted datasets in results (default: False)
        visible: Include only visible datasets (default: True, set False to include hidden)
        details: Include detailed metadata for each dataset (default: False, impacts performance)
        order: Sort order for results. Options include:
              - 'hid-asc': History ID ascending (default, oldest first)
              - 'hid-dsc': History ID descending (newest first)
              - 'create_time-dsc': Creation time descending (most recent first)
              - 'create_time-asc': Creation time ascending (oldest first)
              - 'update_time-dsc': Last updated descending (most recently modified first)
              - 'name-asc': Dataset name ascending (alphabetical)

    Returns:
        Dictionary containing paginated dataset list, pagination metadata, and history reference
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        logger.info(
            f"Getting contents for history ID: {history_id} "
            f"(limit={limit}, offset={offset}, order={order})"
        )

        # Use datasets API for better ordering support
        contents = gi.datasets.get_datasets(
            limit=limit,
            offset=offset,
            history_id=history_id,
            order=order,
            # Note: datasets API uses different parameter names
            # deleted and visible filtering is done post-query if needed
        )

        # Filter by visibility and deleted status if needed
        if not deleted:
            contents = [item for item in contents if not item.get("deleted", False)]
        if visible:
            contents = [item for item in contents if item.get("visible", True)]

        # Get total count for pagination metadata
        all_contents = gi.datasets.get_datasets(
            history_id=history_id,
            order=order,
        )

        # Apply same filtering to total count
        if not deleted:
            all_contents = [item for item in all_contents if not item.get("deleted", False)]
        if visible:
            all_contents = [item for item in all_contents if item.get("visible", True)]

        total_items = len(all_contents) if all_contents else 0

        # Calculate pagination metadata
        has_next = (offset + limit) < total_items
        has_previous = offset > 0
        current_page = (offset // limit) + 1 if limit > 0 else 1
        total_pages = ((total_items - 1) // limit) + 1 if limit > 0 and total_items > 0 else 1

        logger.info(f"Retrieved {len(contents)} items (page {current_page} of {total_pages})")

        return {
            "history_id": history_id,
            "contents": contents,
            "pagination": {
                "total_items": total_items,
                "returned_items": len(contents),
                "limit": limit,
                "offset": offset,
                "current_page": current_page,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous,
                "next_offset": offset + limit if has_next else None,
                "previous_offset": max(0, offset - limit) if has_previous else None,
                "helper_text": f"Showing page {current_page} of {total_pages}. "
                + (
                    f"Use offset={offset + limit} for next page."
                    if has_next
                    else "This is the last page."
                ),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get history contents for ID '{history_id}': {str(e)}")
        if "404" in str(e) or "No route" in str(e):
            raise ValueError(
                f"History ID '{history_id}' not found. Make sure to pass a valid history ID string."
            ) from e
        raise ValueError(f"Failed to get history contents for ID '{history_id}': {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": GET_JOB_DETAILS_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def get_job_details(dataset_id: str, history_id: str | None = None) -> dict[str, Any]:
    """
    Get detailed information about the job that created a specific dataset

    Args:
        dataset_id: Galaxy dataset ID - a hexadecimal hash string identifying the dataset
                   (e.g., 'f2db41e1fa331b3e', typically 16 characters)
        history_id: Galaxy history ID containing the dataset - optional for performance optimization
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)

    Returns:
        Dictionary containing job metadata, tool information, dataset ID, and job ID
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get dataset provenance to find the creating job
        try:
            provenance = gi.histories.show_dataset_provenance(
                history_id=history_id, dataset_id=dataset_id
            )

            # Extract job ID from provenance
            job_id = provenance.get("job_id")
            if not job_id:
                raise ValueError(
                    f"No job information found for dataset '{dataset_id}'. "
                    "The dataset may not have been created by a job."
                )

        except Exception as provenance_error:
            # If provenance fails, try getting dataset details which might contain job info
            try:
                dataset_details = gi.datasets.show_dataset(dataset_id)
                job_id = dataset_details.get("creating_job")
                if not job_id:
                    raise ValueError(
                        f"No job information found for dataset '{dataset_id}'. "
                        "The dataset may not have been created by a job."
                    )
            except Exception:
                raise ValueError(
                    f"Failed to get job information for dataset '{dataset_id}': "
                    f"{str(provenance_error)}"
                ) from provenance_error

        # Get job details using the Galaxy API directly
        # (Bioblend doesn't have a direct method for this)
        url = f"{state['url']}api/jobs/{job_id}"
        headers = {"x-api-key": state["api_key"]}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        job_info = response.json()

        return {"job": job_info, "dataset_id": dataset_id, "job_id": job_id}
    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Dataset ID '{dataset_id}' not found or job not accessible. "
                "Make sure the dataset exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to get job details for dataset '{dataset_id}': {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": GET_DATASET_DETAILS_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def get_dataset_details(
    dataset_id: str, include_preview: bool = True, preview_lines: int = 10
) -> dict[str, Any]:
    """
    Get detailed information about a specific dataset, optionally including a content preview

    Args:
        dataset_id: Galaxy dataset ID - a hexadecimal hash string identifying the dataset
                   (e.g., 'f2db41e1fa331b3e', typically 16 characters)
        include_preview: Whether to include a preview of the dataset content showing first N lines
                        (default: True, only works for datasets in 'ok' state)
        preview_lines: Number of lines to include in the content preview (default: 10)

    Returns:
        Dictionary containing dataset metadata (name, size, state, extension) and optional
        content preview with line count and truncation information
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get dataset details using bioblend
        dataset_info = gi.datasets.show_dataset(dataset_id)

        result = {"dataset": dataset_info, "dataset_id": dataset_id}

        # Add content preview if requested and dataset is in 'ok' state
        if include_preview and dataset_info.get("state") == "ok":
            try:
                # Get dataset content for preview
                content = gi.datasets.download_dataset(
                    dataset_id, use_default_filename=False, require_ok_state=False
                )

                # Convert bytes to string if needed
                if isinstance(content, bytes):
                    try:
                        content_str = content.decode("utf-8")
                    except UnicodeDecodeError:
                        # For binary files, show first part as hex
                        content_str = (
                            f"[Binary content - first 100 bytes as hex: {content[:100].hex()}]"
                        )
                else:
                    content_str = content

                # Get preview lines
                lines = content_str.split("\n")
                preview = "\n".join(lines[:preview_lines])

                result["preview"] = {
                    "lines": preview,
                    "total_lines": len(lines),
                    "preview_lines": min(preview_lines, len(lines)),
                    "truncated": len(lines) > preview_lines,
                }

            except Exception as preview_error:
                logger.warning(f"Could not get preview for dataset {dataset_id}: {preview_error}")
                result["preview"] = {
                    "error": f"Preview unavailable: {str(preview_error)}",
                    "lines": None,
                }

        return result

    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Dataset ID '{dataset_id}' not found. "
                "Make sure the dataset exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to get dataset details for '{dataset_id}': {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": DOWNLOAD_DATASET_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def download_dataset(
    dataset_id: str,
    file_path: str | None = None,
    use_default_filename: bool = True,
    require_ok_state: bool = True,
) -> dict[str, Any]:
    """
    Download a dataset from Galaxy to the local filesystem or memory

    Args:
        dataset_id: Galaxy dataset ID - a hexadecimal hash string identifying the dataset
                   (e.g., 'f2db41e1fa331b3e', typically 16 characters)
        file_path: Local filesystem path where to save the downloaded file
                  (e.g., '/path/to/data.txt', requires write access to filesystem)
                  If not provided, downloads to memory instead
        use_default_filename: Deprecated - use file_path for specific locations
                             (default: True, ignored when file_path not provided)
        require_ok_state: Only allow download if dataset processing state is 'ok'
                         (default: True, set False to download datasets in other states)

    Returns:
        Dictionary containing download information:
        - file_path: Path where file was saved (None if downloaded to memory)
        - suggested_filename: Recommended filename based on dataset name
        - content_available: Whether content was successfully downloaded
        - file_size: Size of downloaded content in bytes
        - dataset_info: Dataset metadata (name, extension, state, genome build)

    IMPORTANT FOR LLMs: If you don't have filesystem write access (common in sandboxed
    environments), omit the file_path parameter to download content to memory. Only
    specify file_path if you can actually write files to the local filesystem.
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get dataset info first to check state and get metadata
        dataset_info = gi.datasets.show_dataset(dataset_id)

        filename = dataset_info.get("name", f"dataset_{dataset_id}")
        extension = dataset_info.get("extension", "")
        if extension and not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"

        # Check dataset state if required
        if require_ok_state and dataset_info.get("state") != "ok":
            raise ValueError(
                f"Dataset '{dataset_id}' is in state '{dataset_info.get('state')}', not 'ok'. "
                "Set require_ok_state=False to download anyway."
            )

        # Download the dataset
        if file_path:
            # Download to specific path
            result_path = gi.datasets.download_dataset(
                dataset_id,
                file_path=file_path,
                use_default_filename=False,
                require_ok_state=require_ok_state,
            )
            download_path = file_path

            # Get file size
            import os

            file_size = os.path.getsize(download_path) if os.path.exists(download_path) else None

        else:
            # Download content to memory (don't save to filesystem)
            result_path = gi.datasets.download_dataset(
                dataset_id,
                use_default_filename=False,  # Get content in memory
                require_ok_state=require_ok_state,
            )

            download_path = None  # No file saved
            file_size = len(result_path) if isinstance(result_path, bytes | str) else None

        return {
            "dataset_id": dataset_id,
            "file_path": download_path,
            "suggested_filename": filename if not file_path else None,
            "content_available": result_path is not None,
            "file_size": file_size,  # Keep consistent with existing API
            "note": (
                "Content downloaded to memory. Use file_path parameter to save to a location."
                if not file_path
                else "File saved to specified path."
            ),
            "dataset_info": {
                "name": dataset_info.get("name"),
                "extension": dataset_info.get("extension"),
                "state": dataset_info.get("state"),
                "genome_build": dataset_info.get("genome_build"),
                "file_size": dataset_info.get("file_size"),
            },
        }

    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Dataset ID '{dataset_id}' not found. "
                "Make sure the dataset exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to download dataset '{dataset_id}': {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "false",
        "aiInputSchema": UPLOAD_FILE_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def upload_file(path: str, history_id: str | None = None) -> dict[str, Any]:
    """
    Upload a local file to Galaxy

    Args:
        path: Local filesystem path to the file to upload (e.g., '/path/to/data.csv')
        history_id: Galaxy history ID where to upload the file - optional, uses current history
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)

    Returns:
        Dictionary containing upload status and information about the created dataset(s)
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        if not os.path.exists(path):
            abs_path = os.path.abspath(path)
            raise ValueError(
                f"File not found: '{path}' (absolute: '{abs_path}'). "
                "Check that the file exists and you have read permissions."
            )

        result = gi.tools.upload_file(path, history_id=history_id)
        return result
    except Exception as e:
        raise ValueError(f"Failed to upload file: {str(e)}") from e


@mcp.tool(
    annotations={
        "readOnlyHint": "true",
        "aiInputSchema": GET_INVOCATIONS_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def get_invocations(
    invocation_id: str | None = None,
    workflow_id: str | None = None,
    history_id: str | None = None,
    limit: int | None = None,
    view: str = "collection",
    step_details: bool = False,
) -> dict[str, Any]:
    """
    View workflow invocations in Galaxy.

    Args:
        invocation_id: Specific workflow invocation ID to view.
        workflow_id: Filter invocations by workflow ID.
        history_id: Filter invocations by history ID.
        limit: Maximum number of invocations to return.
        view: Detail level ('element' or 'collection').
        step_details: Include step details when view is 'element'.
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        if invocation_id:
            invocation = gi.invocations.show_invocation(invocation_id)
            return {"invocation": invocation}

        invocations = gi.invocations.get_invocations(
            workflow_id=workflow_id,
            history_id=history_id,
            limit=limit,
            view=view,
            step_details=step_details,
        )
        return {"invocations": invocations}
    except Exception as e:
        raise ValueError(f"Failed to get workflow invocations: {str(e)}") from e


def _fetch_iwc_workflows() -> list[dict[str, Any]]:
    """Retrieve the flattened list of workflows from the IWC manifest."""
    response = requests.get("https://iwc.galaxyproject.org/workflow_manifest.json")
    response.raise_for_status()
    manifest = response.json()

    workflows: list[dict[str, Any]] = []
    for entry in manifest:
        if isinstance(entry, dict) and "workflows" in entry:
            workflows.extend(entry["workflows"] or [])
    return workflows


@mcp.tool(
    name="iwc_workflows",
    description="Fetch workflows from the IWC catalogue, optionally filtered by a search term.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "IWC Workflows",
        "aiInputSchema": IWC_WORKFLOWS_INPUT_SCHEMA,
        "aiResponseSchema": IWC_WORKFLOWS_RESPONSE_SCHEMA,
    },
)
def iwc_workflows(term: str | None = None) -> dict[str, Any]:
    """Return IWC workflows, optionally filtered by a search term."""
    try:
        workflows = _fetch_iwc_workflows()
    except Exception as exc:  # pragma: no cover - network failures
        raise ValueError(f"Failed to fetch IWC workflows: {exc}") from exc

    if term:
        query = term.lower().strip()
        matched = []
        for workflow in workflows:
            definition = workflow.get("definition") or {}
            name = str(definition.get("name", "")).lower()
            annotation = str(definition.get("annotation", "")).lower()
            tags = [str(tag).lower() for tag in definition.get("tags", []) if tag]
            if query in name or query in annotation or any(query in tag for tag in tags):
                matched.append(workflow)
    else:
        matched = workflows

    return {
        "term": term,
        "total": len(workflows),
        "matched": len(matched),
        "workflows": matched,
    }


@mcp.tool(
    annotations={
        "readOnlyHint": "false",
        "aiInputSchema": IMPORT_IWC_INPUT_SCHEMA,
        "aiResponseSchema": GENERIC_OBJECT_SCHEMA,
    },
)
def import_workflow_from_iwc(trs_id: str) -> dict[str, Any]:
    """
    Import a workflow from IWC to the user's Galaxy instance.

    Args:
        trs_id: TRS ID of the workflow in the IWC manifest.
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        workflows = _fetch_iwc_workflows()

        workflow = None
        for wf in workflows:
            if wf.get("trsID") == trs_id:
                workflow = wf
                break

        if not workflow:
            raise ValueError(
                f"Workflow with trsID '{trs_id}' not found in IWC manifest. "
                "Use iwc_workflows(term=...) to discover available workflows."
            )

        workflow_definition = workflow.get("definition")
        if not workflow_definition:
            raise ValueError(
                f"No definition found for workflow with trsID '{trs_id}'. "
                "This may be a problem with the IWC manifest."
            )

        imported_workflow = gi.workflows.import_workflow_dict(workflow_definition)
        return {"imported_workflow": imported_workflow}
    except Exception as e:
        raise ValueError(f"Failed to import workflow from IWC: {str(e)}") from e


def run_http_server(
    *,
    host: str | None = None,
    port: int | None = None,
    transport: str | None = None,
    path: str | None = None,
) -> None:
    """Run the MCP server over HTTP-based transport."""
    resolved_host = host or os.environ.get("GALAXY_MCP_HOST", "0.0.0.0")
    resolved_port = int(port or os.environ.get("GALAXY_MCP_PORT", "8000"))
    resolved_transport = (
        transport or os.environ.get("GALAXY_MCP_TRANSPORT") or "streamable-http"
    ).lower()
    if resolved_transport not in {"streamable-http", "sse"}:
        raise ValueError(
            f"Unsupported transport '{resolved_transport}'. Choose 'streamable-http' or 'sse'."
        )
    resolved_path = path or os.environ.get("GALAXY_MCP_HTTP_PATH")
    if resolved_path is None and resolved_transport == "streamable-http":
        resolved_path = "/"
    if resolved_path is not None and not resolved_path.startswith("/"):
        resolved_path = f"/{resolved_path}"

    mcp.run(
        transport=resolved_transport, host=resolved_host, port=resolved_port, path=resolved_path
    )


if __name__ == "__main__":
    run_http_server()


@mcp.tool(
    name="search_histories",
    description="Search Galaxy histories by name or identifier.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Histories",
        "aiInputSchema": _build_search_input_schema(include_deleted=True),
        "aiResponseSchema": _build_search_response_schema("histories"),
    },
)
def search_histories(term: str, limit: int = 5, include_deleted: bool = False) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    list_limit = max(limit * 3, 20)

    def _collect(deleted_flag: bool | None) -> list[dict[str, Any]]:
        return gi.histories.get_histories(limit=list_limit, deleted=deleted_flag)

    candidates: list[dict[str, Any]] = []
    non_deleted = _safe_call("histories.get_histories", lambda: _collect(False))
    if non_deleted:
        candidates.extend(non_deleted)
    if include_deleted:
        deleted_items = _safe_call("histories.get_histories (deleted)", lambda: _collect(True))
        if deleted_items:
            candidates.extend(deleted_items)

    for history in candidates:
        hist_id = history.get("id")
        if not hist_id or hist_id in seen:
            continue
        seen.add(hist_id)
        name = history.get("name") or hist_id
        if term_lower not in name.lower() and term_lower not in hist_id.lower():
            continue
        details = (
            _safe_call(
                "histories.show_history",
                lambda hid=hist_id: gi.histories.show_history(hid),
            )
            or {}
        )
        summary = f"History state: {details.get('state', history.get('state', 'unknown'))}"
        extra = {
            "update_time": details.get("update_time"),
            "size": details.get("size"),
            "tags": details.get("tags", []),
            "deleted": details.get("deleted", history.get("deleted")),
        }
        combined_details: dict[str, Any] = {}
        if isinstance(history, dict):
            combined_details.update(history)
        if isinstance(details, dict):
            combined_details.update(details)
        matches.append(
            _format_search_result(
                "histories",
                hist_id,
                name,
                summary,
                term_lower,
                extra,
                combined_details,
            )
        )
        if len(matches) >= limit:
            break
    return {
        "source": "histories",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_workflows",
    description="Search Galaxy workflows including optionally published workflows.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Workflows",
        "aiInputSchema": _build_search_input_schema(include_published=True),
        "aiResponseSchema": _build_search_response_schema("workflows"),
    },
)
def search_workflows(
    term: str,
    limit: int = 5,
    include_published: bool = False,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _collect(published: bool) -> list[dict[str, Any]]:
        return gi.workflows.get_workflows(published=published)

    candidates = _safe_call("workflows.get_workflows", lambda: _collect(include_published)) or []
    if include_published:
        owned = _safe_call("workflows.get_workflows (private)", lambda: _collect(False)) or []
        candidates.extend(owned)

    for workflow in candidates:
        wf_id = workflow.get("id")
        if not wf_id or wf_id in seen:
            continue
        seen.add(wf_id)
        name = workflow.get("name") or wf_id
        searchable = " ".join(
            [
                name,
                wf_id,
                workflow.get("annotation", "") or "",
                workflow.get("description", "") or "",
            ]
        ).lower()
        if term_lower not in searchable:
            continue
        summary = workflow.get("annotation") or "Galaxy workflow"
        extra = {
            "published": workflow.get("published", False),
            "latest_workflow_id": workflow.get("latest_workflow_id"),
        }
        details = (
            _safe_call("workflows.show_workflow", lambda wid=wf_id: gi.workflows.show_workflow(wid))
            or {}
        )
        combined_details: dict[str, Any] = {}
        if isinstance(workflow, dict):
            combined_details.update(workflow)
        if isinstance(details, dict):
            combined_details.update(details)
        matches.append(
            _format_search_result(
                "workflows",
                wf_id,
                name,
                summary,
                term_lower,
                extra,
                combined_details,
            )
        )
        if len(matches) >= limit:
            break
    return {
        "source": "workflows",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_datasets",
    description="Search Galaxy datasets within histories.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Datasets",
        "aiInputSchema": _build_search_input_schema(include_deleted=True),
        "aiResponseSchema": _build_search_response_schema("datasets"),
    },
)
def search_datasets(term: str, limit: int = 5, include_deleted: bool = False) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    dataset_limit = max(limit * 5, 50)
    deleted_flag: bool | None = None if include_deleted else False
    candidates = (
        _safe_call(
            "datasets.get_datasets",
            lambda: gi.datasets.get_datasets(limit=dataset_limit, deleted=deleted_flag),
        )
        or []
    )

    for ds in candidates:
        ds_id = ds.get("id")
        if not ds_id or ds_id in seen:
            continue
        seen.add(ds_id)
        name = ds.get("name") or ds_id
        searchable = " ".join([name, ds_id, ds.get("extension", ""), ds.get("state", "")]).lower()
        if term_lower not in searchable:
            continue
        summary = (
            f"Dataset state: {ds.get('state', 'unknown')} ({ds.get('extension', 'unknown')} format)"
        )
        extra = {
            "history_id": ds.get("history_id"),
            "deleted": ds.get("deleted"),
            "visible": ds.get("visible"),
            "tags": ds.get("tags", []),
        }
        details = (
            _safe_call("datasets.show_dataset", lambda did=ds_id: gi.datasets.show_dataset(did))
            or {}
        )
        combined_details: dict[str, Any] = {}
        if isinstance(ds, dict):
            combined_details.update(ds)
        if isinstance(details, dict):
            combined_details.update(details)
        matches.append(
            _format_search_result(
                "datasets",
                ds_id,
                name,
                summary,
                term_lower,
                extra,
                combined_details,
            )
        )
        if len(matches) >= limit:
            break
    return {
        "source": "datasets",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_dataset_collections",
    description="Search Galaxy dataset collections across available histories.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Dataset Collections",
        "aiInputSchema": _build_search_input_schema(include_deleted=True),
        "aiResponseSchema": _build_search_response_schema("dataset_collections"),
    },
)
def search_dataset_collections(
    term: str,
    limit: int = 5,
    include_deleted: bool = False,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    histories = (
        _safe_call(
            "histories.get_histories for collections",
            lambda: gi.histories.get_histories(limit=max(limit * 4, 20), deleted=False),
        )
        or []
    )
    if include_deleted:
        histories.extend(
            _safe_call(
                "histories.get_histories deleted for collections",
                lambda: gi.histories.get_histories(limit=10, deleted=True),
            )
            or []
        )

    for history in histories:
        hist_id = history.get("id")
        if not hist_id:
            continue
        collections = _safe_call(
            f"histories.show_history (collections) {hist_id}",
            lambda hid=hist_id: gi.histories.show_history(
                hid,
                contents=True,
                types=["dataset_collection"],
                deleted=include_deleted if include_deleted else None,
            ),
        )
        if not collections:
            continue
        for collection in collections:
            coll_id = collection.get("id")
            if not coll_id or coll_id in seen:
                continue
            seen.add(coll_id)
            name = collection.get("name") or coll_id
            searchable = " ".join([name, coll_id, collection.get("collection_type", "")]).lower()
            if term_lower not in searchable:
                continue
            summary = (
                f"{collection.get('collection_type', 'collection').title()} in history "
                f"{history.get('name') or hist_id}"
            )
            extra = {
                "history_id": hist_id,
                "collection_type": collection.get("collection_type"),
                "element_count": collection.get("element_count"),
                "deleted": collection.get("deleted"),
            }
            details = _safe_call(
                "dataset_collections.show_dataset_collection",
                lambda cid=coll_id: gi.dataset_collections.show_dataset_collection(cid),
            )
            if details is None:
                details = collection
            matches.append(
                _format_search_result(
                    "dataset_collections",
                    coll_id,
                    name,
                    summary,
                    term_lower,
                    extra,
                    details,
                )
            )
            if len(matches) >= limit:
                matches = matches[:limit]
                return {
                    "source": "dataset_collections",
                    "term": normalized_term,
                    "limit": limit,
                    "total": len(matches),
                    "matches": matches,
                }
    return {
        "source": "dataset_collections",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_libraries",
    description="Search Galaxy data libraries.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Libraries",
        "aiInputSchema": _build_search_input_schema(include_deleted=True),
        "aiResponseSchema": _build_search_response_schema("libraries"),
    },
)
def search_libraries(term: str, limit: int = 5, include_deleted: bool = False) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    libraries = (
        _safe_call(
            "libraries.get_libraries",
            lambda: gi.libraries.get_libraries(
                deleted=include_deleted if include_deleted else False
            ),
        )
        or []
    )

    for library in libraries:
        library_id = library.get("id")
        if not library_id or library_id in seen:
            continue
        seen.add(library_id)
        name = library.get("name") or library_id
        searchable = " ".join(
            [name, library_id, library.get("description", ""), library.get("synopsis", "")]
        ).lower()
        if term_lower not in searchable:
            continue
        summary = library.get("description") or "Galaxy data library"
        extra = {
            "synopsis": library.get("synopsis"),
            "deleted": library.get("deleted"),
        }
        details = (
            _safe_call(
                "libraries.show_library_full",
                lambda lid=library_id: gi.libraries.show_library(lid, contents=True),
            )
            or {}
        )
        combined_details: dict[str, Any] = {}
        if isinstance(library, dict):
            combined_details.update(library)
        if isinstance(details, dict):
            combined_details.update(details)
        matches.append(
            _format_search_result(
                "libraries",
                library_id,
                name,
                summary,
                term_lower,
                extra,
                combined_details,
            )
        )
        if len(matches) >= limit:
            break
    return {
        "source": "libraries",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_library_datasets",
    description="Search datasets stored within Galaxy data libraries.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Library Datasets",
        "aiInputSchema": _build_search_input_schema(include_deleted=True),
        "aiResponseSchema": _build_search_response_schema("library_datasets"),
    },
)
def search_library_datasets(
    term: str,
    limit: int = 5,
    include_deleted: bool = False,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    libraries = (
        _safe_call(
            "libraries.get_libraries for datasets",
            lambda: gi.libraries.get_libraries(
                deleted=include_deleted if include_deleted else False
            ),
        )
        or []
    )

    for library in libraries:
        library_id = library.get("id")
        if not library_id:
            continue
        contents = _safe_call(
            f"libraries.show_library {library_id}",
            lambda lid=library_id: gi.libraries.show_library(lid, contents=True),
        )
        if not contents:
            continue
        for entry in contents:
            if entry.get("type") not in {"file", "dataset"}:
                continue
            dataset_id = entry.get("id")
            if not dataset_id:
                continue
            name = entry.get("name") or dataset_id
            searchable = " ".join([name, dataset_id, entry.get("type", "")]).lower()
            if term_lower not in searchable:
                continue
            summary = f"Library dataset in {library.get('name') or library_id}"
            extra = {
                "library_id": library_id,
                "deleted": entry.get("deleted"),
                "type": entry.get("type"),
            }
            details = _safe_call(
                "libraries.show_dataset",
                lambda lid=library_id, did=dataset_id: gi.libraries.show_dataset(lid, did),
            )
            if details is None:
                details = entry
            matches.append(
                _format_search_result(
                    "library_datasets",
                    (library_id, dataset_id),
                    name,
                    summary,
                    term_lower,
                    extra,
                    details,
                )
            )
            if len(matches) >= limit:
                return {
                    "source": "library_datasets",
                    "term": normalized_term,
                    "limit": limit,
                    "total": len(matches),
                    "matches": matches[:limit],
                }
    return {
        "source": "library_datasets",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_jobs",
    description="Search recent Galaxy jobs by identifier, tool, or state.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Jobs",
        "aiInputSchema": _build_search_input_schema(),
        "aiResponseSchema": _build_search_response_schema("jobs"),
    },
)
def search_jobs(term: str, limit: int = 5) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    jobs = (
        _safe_call(
            "jobs.get_jobs",
            lambda: gi.jobs.get_jobs(limit=max(limit * 5, 50)),
        )
        or []
    )

    for job in jobs:
        job_id = job.get("id")
        if not job_id:
            continue
        searchable = " ".join(
            [
                job_id,
                job.get("tool_id", ""),
                job.get("state", ""),
                job.get("history_id", "") or "",
            ]
        ).lower()
        if term_lower not in searchable:
            continue
        summary = f"Job state: {job.get('state', 'unknown')} (tool {job.get('tool_id', 'unknown')})"
        extra = {
            "tool_id": job.get("tool_id"),
            "state": job.get("state"),
            "history_id": job.get("history_id"),
            "exit_code": job.get("exit_code"),
        }
        details = _safe_call(
            "jobs.show_job", lambda jid=job_id: gi.jobs.show_job(jid, full_details=True)
        )
        if details is None:
            details = job
        matches.append(
            _format_search_result(
                "jobs",
                job_id,
                job_id,
                summary,
                term_lower,
                extra,
                details,
            )
        )
        if len(matches) >= limit:
            break
    return {
        "source": "jobs",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }


@mcp.tool(
    name="search_invocations",
    description="Search Galaxy workflow invocations.",
    enabled=True,
    annotations={
        "readOnlyHint": "true",
        "title": "Search Invocations",
        "aiInputSchema": _build_search_input_schema(),
        "aiResponseSchema": _build_search_response_schema("invocations"),
    },
)
def search_invocations(term: str, limit: int = 5) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Search term must not be empty.")
    term_lower = _normalize_term(normalized_term)
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    matches: list[dict[str, Any]] = []
    invocations = (
        _safe_call(
            "invocations.get_invocations",
            lambda: gi.invocations.get_invocations(limit=max(limit * 4, 40), view="element"),
        )
        or []
    )

    for invocation in invocations:
        inv_id = invocation.get("id")
        if not inv_id:
            continue
        name = invocation.get("workflow_name") or invocation.get("workflow_id") or inv_id
        searchable = " ".join(
            [
                inv_id,
                name,
                invocation.get("state", ""),
                invocation.get("history_id", "") or "",
                invocation.get("workflow_id", "") or "",
            ]
        ).lower()
        if term_lower not in searchable:
            continue
        summary = f"Invocation state: {invocation.get('state', 'unknown')}"
        extra = {
            "history_id": invocation.get("history_id"),
            "workflow_id": invocation.get("workflow_id"),
            "update_time": invocation.get("update_time"),
        }
        details = _safe_call(
            "invocations.show_invocation",
            lambda iid=inv_id: gi.invocations.show_invocation(iid),
        )
        if details is None:
            details = invocation
        matches.append(
            _format_search_result(
                "invocations",
                inv_id,
                name,
                summary,
                term_lower,
                extra,
                details,
            )
        )
        if len(matches) >= limit:
            break
    return {
        "source": "invocations",
        "term": normalized_term,
        "limit": limit,
        "total": len(matches),
        "matches": matches,
    }
