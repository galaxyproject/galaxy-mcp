# Galaxy MCP Server
import concurrent.futures
import logging
import os
import threading
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import requests
from bioblend.galaxy import GalaxyInstance
from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP
from mcp.server.auth.middleware.auth_context import get_access_token
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from galaxy_mcp.auth import (
    GalaxyOAuthProvider,
    configure_auth_provider,
    get_active_session,
)


class PaginationInfo(BaseModel):
    """Pagination metadata for list operations."""

    total_items: int = Field(description="Total number of items available")
    returned_items: int = Field(description="Number of items in this response")
    limit: int = Field(description="Maximum items requested")
    offset: int = Field(description="Number of items skipped")
    has_next: bool = Field(description="Whether more items are available")
    has_previous: bool = Field(description="Whether previous items exist")
    next_offset: int | None = Field(default=None, description="Offset for next page")
    previous_offset: int | None = Field(default=None, description="Offset for previous page")
    helper_text: str | None = Field(default=None, description="Human-readable pagination hint")


class GalaxyResult(BaseModel):
    """Standardized response from Galaxy MCP tools."""

    data: Any = Field(description="Response data from Galaxy API")
    success: bool = Field(default=True, description="Whether the operation succeeded")
    message: str = Field(description="Human-readable status message")
    count: int | None = Field(default=None, description="Number of items returned")
    pagination: PaginationInfo | None = Field(
        default=None, description="Pagination info for list operations"
    )


# Set up logging
logging.basicConfig(level=logging.INFO)
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

# Configure Galaxy target and client state
raw_galaxy_url = os.environ.get("GALAXY_URL")
normalized_galaxy_url = (
    raw_galaxy_url if not raw_galaxy_url or raw_galaxy_url.endswith("/") else f"{raw_galaxy_url}/"
)
galaxy_state: dict[str, Any] = {
    "url": normalized_galaxy_url,
    "api_key": os.environ.get("GALAXY_API_KEY"),
    "gi": None,
    "connected": False,
}

# Configure OAuth provider if requested
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
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to initialize OAuth provider: %s", exc, exc_info=True)
elif public_base_url and not normalized_galaxy_url:
    logger.warning(
        "GALAXY_MCP_PUBLIC_URL is set but GALAXY_URL is missing. "
        "OAuth login remains disabled until GALAXY_URL is configured."
    )
else:
    logger.info(
        "OAuth login disabled. Configure GALAXY_MCP_PUBLIC_URL to enable browser-based login."
    )

# Create an MCP server (inject auth provider when available)
if auth_provider:
    mcp: FastMCP = FastMCP("Galaxy", auth=auth_provider)
else:
    mcp = FastMCP("Galaxy")

# Allow browser preflight CORS requests to bypass FastMCP auth


class _PreflightMiddleware(BaseHTTPMiddleware):
    """Ensure CORS preflight requests succeed for browser-based clients."""

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


class _OAuthPublicRoutes:
    """Expose OAuth login and metadata routes without auth headers."""

    def __init__(self, app, provider: GalaxyOAuthProvider, base_path: str | None):
        self._app = app
        self._provider = provider
        self._login_paths = provider.get_login_paths(base_path)
        self._metadata_paths = provider.get_resource_metadata_paths(base_path)
        self.state = getattr(app, "state", None)
        self.router = getattr(app, "router", None)

    def __getattr__(self, item):
        return getattr(self._app, item)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "").upper()
        if path in self._metadata_paths:
            if method not in {"GET", "HEAD"}:
                await self._app(scope, receive, send)
                return
            request = Request(scope, receive=receive)
            response = await self._provider.handle_resource_metadata(request)
            await response(scope, receive, send)
            return

        if path in self._login_paths and method in {"GET", "POST"}:
            request = Request(scope, receive=receive)
            response = await self._provider.handle_login(request)
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


def _http_app_with_preflight(self, *args, **kwargs):
    app = _original_http_app(self, *args, **kwargs)
    app.add_middleware(_PreflightMiddleware)
    if auth_provider:
        base_path = kwargs.get("path")
        app = _OAuthPublicRoutes(app, auth_provider, base_path)
    return app


mcp.http_app = types.MethodType(_http_app_with_preflight, mcp)  # type: ignore[method-assign]


# Initialize Galaxy client if environment variables are set
if galaxy_state["url"] and galaxy_state["api_key"]:
    try:
        galaxy_state["gi"] = GalaxyInstance(url=galaxy_state["url"], key=galaxy_state["api_key"])
        galaxy_state["connected"] = True
        logger.info(
            "Galaxy client initialized from environment variables (URL: %s)",
            galaxy_state["url"],
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Galaxy client from environment variables: {e}")
        logger.warning("You'll need to use connect() to establish a connection.")


def _get_request_connection_state() -> dict[str, Any]:
    """
    Determine the effective Galaxy connection, preferring OAuth credentials when available.
    """
    if auth_provider:
        credentials, api_key = get_active_session(get_access_token)
        if credentials and api_key:
            try:
                gi = GalaxyInstance(url=credentials.galaxy_url, key=api_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to create Galaxy client for OAuth session: %s", exc)
            else:
                return {
                    "url": credentials.galaxy_url,
                    "api_key": api_key,
                    "gi": gi,
                    "connected": True,
                    "source": "oauth",
                    "session": credentials,
                }

    return {
        "url": galaxy_state.get("url") or normalized_galaxy_url,
        "api_key": galaxy_state.get("api_key"),
        "gi": galaxy_state.get("gi"),
        "connected": galaxy_state.get("connected", False) and bool(galaxy_state.get("gi")),
        "source": "api_key" if galaxy_state.get("connected") else None,
        "session": None,
    }


def ensure_connected() -> dict[str, Any]:
    """Helper function to ensure Galaxy connection is established."""
    state = _get_request_connection_state()
    if not state["connected"] or not state["gi"]:
        raise ValueError(
            "Not connected to Galaxy. Authenticate via OAuth or run connect() with your "
            "Galaxy URL and API key. Example: connect(url='https://your-galaxy.org', "
            "api_key='your-key')"
        )
    return state


@mcp.tool()
def connect(url: str | None = None, api_key: str | None = None) -> GalaxyResult:
    """
    Connect to Galaxy server

    Args:
        url: Galaxy server URL (optional, uses GALAXY_URL env var if not provided)
        api_key: Galaxy API key (optional, uses GALAXY_API_KEY env var if not provided)

    Returns:
        GalaxyResult with connection status and user information in data field
    """
    try:
        # Reuse current OAuth session when available
        state = _get_request_connection_state()
        if state["connected"] and state.get("source") == "oauth" and state["gi"]:
            gi: GalaxyInstance = state["gi"]
            user_info = gi.users.get_current_user()
            return GalaxyResult(
                data={
                    "connected": True,
                    "user": user_info,
                    "url": state["url"],
                    "auth": "oauth",
                },
                success=True,
                message=f"Connected to Galaxy at {state['url']} via OAuth",
            )

        # Use provided parameters or fall back to environment variables
        use_url = url or os.environ.get("GALAXY_URL")
        use_api_key = api_key or os.environ.get("GALAXY_API_KEY")

        # Check if we have the necessary credentials
        if not use_url or not use_api_key:
            # Try to reload from .env file in case it was added after startup
            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                load_dotenv(dotenv_path, override=True)
                # Check again after loading .env
                use_url = url or os.environ.get("GALAXY_URL")
                use_api_key = api_key or os.environ.get("GALAXY_API_KEY")

            # If still missing credentials, report error
            if not use_url or not use_api_key:
                missing = []
                if not use_url:
                    missing.append("URL")
                if not use_api_key:
                    missing.append("API key")
                missing_str = " and ".join(missing)
                raise ValueError(
                    f"Missing Galaxy {missing_str}. Please provide as arguments, "
                    f"set environment variables, or create a .env file with "
                    f"GALAXY_URL and GALAXY_API_KEY."
                )

        galaxy_url = use_url if use_url.endswith("/") else f"{use_url}/"

        # Create a new Galaxy instance to test connection
        gi = GalaxyInstance(url=galaxy_url, key=use_api_key)

        # Test the connection by fetching user info
        user_info = gi.users.get_current_user()

        # Update galaxy state
        galaxy_state["url"] = galaxy_url
        galaxy_state["api_key"] = use_api_key
        galaxy_state["gi"] = gi
        galaxy_state["connected"] = True

        return GalaxyResult(
            data={"connected": True, "user": user_info},
            success=True,
            message=f"Connected to Galaxy at {galaxy_url}",
        )
    except Exception as e:
        # Reset state on failure
        galaxy_state["url"] = None
        galaxy_state["api_key"] = None
        galaxy_state["gi"] = None
        galaxy_state["connected"] = False

        galaxy_url = locals().get("galaxy_url") or use_url or normalized_galaxy_url or "unknown"
        error_msg = f"Failed to connect to Galaxy at {galaxy_url}: {str(e)}"
        if "401" in str(e) or "authentication" in str(e).lower():
            error_msg += " Check that your API key is valid and has the necessary permissions."
        elif "404" in str(e) or "not found" in str(e).lower():
            error_msg += " Check that the Galaxy URL is correct and accessible."
        elif "connection" in str(e).lower() or "timeout" in str(e).lower():
            error_msg += " Check your network connection and that the Galaxy server is running."
        else:
            error_msg += " Verify the URL format (should end with /) and API key."

        raise ValueError(error_msg) from e


@mcp.tool()
def search_tools_by_name(query: str) -> GalaxyResult:
    """
    Search Galaxy tools whose name, ID, or description contains the given query (substring match).

    RECOMMENDED WORKFLOW:
    1. Use this function to find tools by name/keyword
    2. Review the returned tool IDs and names
    3. Call get_tool_details(tool_id) for full input parameters
    4. Call run_tool() with the correct inputs

    Args:
        query: Search query - matches against tool name, ID, or description.
               Examples: "fastq", "alignment", "filter", "bwa"

    Returns:
        GalaxyResult with:
        - data: List of matching tools with id, name, version, description
        - count: Number of tools found
        - message: Summary of results

    Example:
        >>> search_tools_by_name("fastq")
        GalaxyResult(
            data=[
                {"id": "fastqc", "name": "FastQC", "version": "0.73+galaxy0", ...},
                {"id": "fastq_filter", "name": "Filter FASTQ", ...}
            ],
            count=15,
            message="Found 15 tools matching 'fastq'"
        )

    NEXT STEPS:
    - To see full tool parameters: get_tool_details(tool_id)
    - To see example inputs: get_tool_run_examples(tool_id)
    - To run a tool: run_tool(history_id, tool_id, inputs)
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get all tools and filter client-side for substring matching
        # The get_tools(name=query) parameter doesn't support substring matching
        all_tools = gi.tools.get_tools()
        query_lower = query.lower()

        # Filter tools by substring match in name, ID, or description
        matching_tools = [
            tool
            for tool in all_tools
            if query_lower in tool.get("name", "").lower()
            or query_lower in tool.get("id", "").lower()
            or query_lower in tool.get("description", "").lower()
        ]

        return GalaxyResult(
            data=matching_tools,
            success=True,
            message=f"Found {len(matching_tools)} tools matching '{query}'",
            count=len(matching_tools),
        )
    except Exception as e:
        raise ValueError(format_error("Search tools", e, {"query": query})) from e


@mcp.tool()
def get_tool_details(tool_id: str, io_details: bool = False) -> GalaxyResult:
    """
    Get detailed information about a specific tool including its input parameters.

    RECOMMENDED WORKFLOW:
    1. First find tools using search_tools_by_name() or get_tool_panel()
    2. Call this function with io_details=True to see all input parameters
    3. Use the inputs schema to construct the inputs dict for run_tool()

    Args:
        tool_id: Galaxy tool identifier. Common formats:
                 - Simple: "fastqc", "bwa", "upload1"
                 - Toolshed: "toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.73"
        io_details: Set True to include detailed input/output parameter schemas.
                    Essential for understanding how to call run_tool().

    Returns:
        GalaxyResult with tool info including:
        - id, name, version, description
        - inputs: Parameter definitions (when io_details=True)
        - outputs: Output file definitions

    Example:
        >>> get_tool_details("fastqc", io_details=True)
        GalaxyResult(
            data={
                "id": "fastqc",
                "name": "FastQC",
                "version": "0.73+galaxy0",
                "inputs": [
                    {"name": "input_file", "type": "data", "format": ["fastq"]},
                    {"name": "contaminants", "type": "data", "optional": True}
                ],
                ...
            }
        )

    NEXT STEPS:
    - To see example tool calls: get_tool_run_examples(tool_id)
    - To run the tool: run_tool(history_id, tool_id, inputs)

    ERROR HANDLING:
    - Tool not found: Check tool_id spelling or use search_tools_by_name()
    - Permission denied: Tool may be restricted to certain users
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get detailed information about the tool
        tool_info = gi.tools.show_tool(tool_id, io_details=io_details)
        return GalaxyResult(
            data=tool_info,
            success=True,
            message=f"Retrieved details for tool '{tool_id}'",
        )
    except Exception as e:
        raise ValueError(
            format_error("Get tool details", e, {"tool_id": tool_id, "io_details": io_details})
        ) from e


@mcp.tool()
def get_tool_run_examples(tool_id: str, tool_version: str | None = None) -> GalaxyResult:
    """
    Return the exact XML test definitions (inputs, outputs, assertions, required files)
    for a Galaxy tool so an LLM can study real, working run configurations.

    Args:
        tool_id: ID of the tool to inspect
        tool_version: Optional version selector (use '*' for all versions)

    Returns:
        GalaxyResult with test cases in data field
    """
    ensure_connected()

    try:
        test_cases = galaxy_state["gi"].tools.get_tool_tests(tool_id, tool_version=tool_version)
        return GalaxyResult(
            data={
                "tool_id": tool_id,
                "requested_version": tool_version,
                "test_cases": test_cases,
            },
            success=True,
            message=f"Retrieved {len(test_cases)} test cases for tool '{tool_id}'",
            count=len(test_cases),
        )
    except Exception as e:
        context = {"tool_id": tool_id}
        if tool_version:
            context["tool_version"] = tool_version
        raise ValueError(format_error("Get tool run examples", e, context)) from e


@mcp.tool()
def get_tool_citations(tool_id: str) -> GalaxyResult:
    """
    Get citation information for a specific tool

    Args:
        tool_id: ID of the tool

    Returns:
        GalaxyResult with tool citation information in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get the tool information which includes citations
        tool_info = gi.tools.show_tool(tool_id)

        # Extract citation information
        citations = tool_info.get("citations", [])

        return GalaxyResult(
            data={
                "tool_name": tool_info.get("name", tool_id),
                "tool_version": tool_info.get("version", "unknown"),
                "citations": citations,
            },
            success=True,
            message=f"Retrieved {len(citations)} citations for tool '{tool_id}'",
            count=len(citations),
        )
    except Exception as e:
        raise ValueError(format_error("Get tool citations", e, {"tool_id": tool_id})) from e


@mcp.tool()
def run_tool(history_id: str, tool_id: str, inputs: dict[str, Any]) -> GalaxyResult:
    """
    Run a Galaxy tool on datasets in a history.

    RECOMMENDED WORKFLOW:
    1. Create or select a history: create_history() or get_histories()
    2. Upload data: upload_file() or upload_file_from_url()
    3. Get tool parameters: get_tool_details(tool_id, io_details=True)
    4. Call this function with properly formatted inputs
    5. Monitor job: get_job_details() or check history contents

    Args:
        history_id: Galaxy history ID (16-char hex string like '1cd8e2f6b131e5aa').
                    Get from create_history() or get_histories().
        tool_id: Tool identifier. Common formats:
                 - Simple built-in: "cat1", "Cut1", "upload1"
                 - Toolshed: "toolshed.g2.bx.psu.edu/repos/iuc/fastqc/fastqc/0.73"
        inputs: Tool input parameters. Dataset inputs use this format:
                {"input_name": {"src": "hda", "id": "dataset_id"}}

    Returns:
        GalaxyResult with:
        - data.jobs: List of job objects with state and IDs
        - data.outputs: List of output datasets created
        - data.output_collections: List of output collections (if any)

    Example - Running FastQC:
        >>> run_tool(
        ...     history_id="abc123def456",
        ...     tool_id="fastqc",
        ...     inputs={"input_file": {"src": "hda", "id": "dataset123"}}
        ... )
        GalaxyResult(
            data={
                "jobs": [{"id": "job789", "state": "queued"}],
                "outputs": [{"id": "output456", "name": "FastQC on data 1"}]
            },
            message="Started tool 'fastqc' in history 'abc123def456'"
        )

    Example - Tool with multiple inputs:
        >>> run_tool(
        ...     history_id="abc123",
        ...     tool_id="bwa_mem",
        ...     inputs={
        ...         "fastq_input|fastq_input1": {"src": "hda", "id": "reads1"},
        ...         "reference_source|ref_file": {"src": "hda", "id": "genome"},
        ...         "analysis_type|analysis_type_selector": "simple"
        ...     }
        ... )

    NEXT STEPS:
    - Check job status: get_job_details(output_dataset_id)
    - View outputs: get_history_contents(history_id)
    - Download results: download_dataset(output_id)

    ERROR HANDLING:
    - "Tool not found": Verify tool_id with search_tools_by_name()
    - "Invalid input": Check input format with get_tool_details(io_details=True)
    - "Dataset not found": Verify dataset_id exists in the history
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Run the tool with provided inputs
        result = gi.tools.run_tool(history_id, tool_id, inputs)
        return GalaxyResult(
            data=result,
            success=True,
            message=f"Started tool '{tool_id}' in history '{history_id}'",
        )
    except Exception as e:
        raise ValueError(
            format_error(
                "Run tool", e, {"history_id": history_id, "tool_id": tool_id, "inputs": inputs}
            )
        ) from e


@mcp.tool()
def get_tool_panel() -> GalaxyResult:
    """
    Get the tool panel structure (toolbox)

    Returns:
        GalaxyResult with tool panel hierarchy in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get the tool panel structure
        tool_panel = gi.tools.get_tool_panel()
        return GalaxyResult(
            data=tool_panel,
            success=True,
            message="Retrieved tool panel structure",
        )
    except Exception as e:
        raise ValueError(format_error("Get tool panel", e)) from e


@mcp.tool()
def create_history(history_name: str) -> GalaxyResult:
    """
    Create a new history to organize datasets and analyses.

    A history is the primary workspace in Galaxy. Create a new history for each
    distinct project or analysis to keep your work organized.

    RECOMMENDED WORKFLOW:
    1. Create a history with a descriptive name
    2. Upload your input data: upload_file() or upload_file_from_url()
    3. Run tools on the data: run_tool()
    4. View results: get_history_contents()

    Args:
        history_name: Descriptive name for the history.
                      Best practices:
                      - Include project/sample name: "RNA-seq Sample A"
                      - Include date if relevant: "ChIP-seq 2024-01"
                      - Be specific: "BWA alignment of patient_001"

    Returns:
        GalaxyResult with:
        - data.id: The history ID (use this for subsequent operations)
        - data.name: The history name
        - data.create_time: When the history was created

    Example:
        >>> create_history("RNA-seq Analysis - Sample A")
        GalaxyResult(
            data={"id": "abc123def456", "name": "RNA-seq Analysis - Sample A", ...},
            message="Created history 'RNA-seq Analysis - Sample A'"
        )

    NEXT STEPS:
    - Upload data: upload_file(file_path, history_id)
    - Upload from URL: upload_file_from_url(url, history_id)
    - Run a tool: run_tool(history_id, tool_id, inputs)
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    history = gi.histories.create_history(history_name)
    return GalaxyResult(
        data=history,
        success=True,
        message=f"Created history '{history_name}'",
    )


@mcp.tool()
def search_tools_by_keywords(keywords: list[str]) -> GalaxyResult:
    """
    Recommend Galaxy tools based on a list of keywords.

    Args:
        keywords (list[str]): A list of keywords or phrases describing what you're looking for,
            e.g., ["csv", "rna", "alignment", "visualization"]. The search will match tools
            whose name, description, or accepted input formats contain any of these keywords.

    Returns:
        GalaxyResult with recommended tools in data field
    """

    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    lock = threading.Lock()

    keywords_lower = [k.lower() for k in keywords]

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
            if any(kw in name for kw in keywords_lower) or any(
                kw in description for kw in keywords_lower
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
                            if ext and any(kw in ext.lower() for kw in keywords_lower):
                                return tool
                    elif (
                        isinstance(fmt, str)
                        and fmt
                        and any(kw in fmt.lower() for kw in keywords_lower)
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
        return GalaxyResult(
            data=slim_tools,
            success=True,
            message=f"Found {len(slim_tools)} tools matching keywords: {', '.join(keywords)}",
            count=len(slim_tools),
        )
    except Exception as e:
        raise ValueError(f"Failed to search tools by keywords: {str(e)}") from e


@mcp.tool()
def get_server_info() -> GalaxyResult:
    """
    Get Galaxy server information including version, URL, and configuration details

    Returns:
        GalaxyResult with server information in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    url = state["url"] or normalized_galaxy_url

    try:
        # Get server configuration info
        config_info = gi.config.get_config()

        # Get server version info
        version_info = gi.config.get_version()

        # Build comprehensive server info response
        server_info = {
            "url": url,
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

        return GalaxyResult(
            data=server_info,
            success=True,
            message=f"Retrieved server info for {url}",
        )
    except Exception as e:
        raise ValueError(f"Failed to get server information: {str(e)}") from e


@mcp.tool()
def get_user() -> GalaxyResult:
    """
    Get current user information

    Returns:
        GalaxyResult with current user details in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        user_info = gi.users.get_current_user()
        return GalaxyResult(
            data=user_info,
            success=True,
            message=f"Retrieved user info for '{user_info.get('username', 'unknown')}'",
        )
    except Exception as e:
        raise ValueError(f"Failed to get user: {str(e)}") from e


@mcp.tool()
def get_histories(
    limit: int | None = None, offset: int = 0, name: str | None = None
) -> GalaxyResult:
    """
    Get list of user's histories with optional pagination and filtering.

    Histories are Galaxy's primary organizational unit - each contains datasets,
    collections, and records of analyses. Most operations require a history_id.

    RECOMMENDED WORKFLOW:
    1. Call get_histories() to see existing histories
    2. Either use an existing history_id or create_history() for new work
    3. Upload data or run tools in the selected history

    Args:
        limit: Maximum histories to return. Default None returns all.
               Use with offset for pagination on large history lists.
        offset: Skip this many histories (for pagination). Default 0.
        name: Filter by name pattern (case-sensitive partial match).
              Example: name="RNA" matches "RNA-seq analysis", "my RNA data"

    Returns:
        GalaxyResult with:
        - data: List of history objects with id, name, update_time, etc.
        - count: Number of histories returned
        - pagination: PaginationInfo when limit is specified

    Example - Get all histories:
        >>> get_histories()
        GalaxyResult(
            data=[
                {"id": "abc123", "name": "RNA-seq Analysis", "update_time": "2024-01-15"},
                {"id": "def456", "name": "ChIP-seq Data", "update_time": "2024-01-10"}
            ],
            count=2
        )

    Example - Paginated with filter:
        >>> get_histories(limit=10, offset=0, name="RNA")
        GalaxyResult(
            data=[...],
            pagination=PaginationInfo(total_items=25, has_next=True, next_offset=10)
        )

    NEXT STEPS:
    - View history contents: get_history_contents(history_id)
    - Get history details: get_history_details(history_id)
    - Create new history: create_history("Analysis Name")
    - Upload data: upload_file(file_path, history_id)
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

            pagination = PaginationInfo(
                total_items=total_items,
                returned_items=len(histories),
                limit=limit,
                offset=offset,
                has_next=has_next,
                has_previous=has_previous,
                next_offset=offset + limit if has_next else None,
                previous_offset=max(0, offset - limit) if has_previous else None,
                helper_text=f"Page {current_page} of {total_pages}. "
                + (
                    f"Use offset={offset + limit} for next page."
                    if has_next
                    else "This is the last page."
                ),
            )

            return GalaxyResult(
                data=histories,
                success=True,
                message=f"Retrieved {len(histories)} of {total_items} histories",
                count=len(histories),
                pagination=pagination,
            )
        else:
            # No pagination requested
            return GalaxyResult(
                data=histories,
                success=True,
                message=f"Retrieved {len(histories)} histories",
                count=len(histories),
            )
    except Exception as e:
        raise ValueError(
            f"Failed to get histories: {str(e)}. "
            "Check your connection to Galaxy and that you have "
            "permission to view histories."
        )


@mcp.tool()
def list_history_ids() -> GalaxyResult:
    """
    Get a simplified list of history IDs and names for easy reference

    Returns:
        GalaxyResult with list of {id, name} dictionaries in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        histories = gi.histories.get_histories()
        if not histories:
            return GalaxyResult(
                data=[],
                success=True,
                message="No histories found",
                count=0,
            )
        # Extract just the id and name for convenience
        simplified = [{"id": h["id"], "name": h.get("name", "Unnamed")} for h in histories]
        return GalaxyResult(
            data=simplified,
            success=True,
            message=f"Found {len(simplified)} histories",
            count=len(simplified),
        )
    except Exception as e:
        raise ValueError(f"Failed to list history IDs: {str(e)}") from e


@mcp.tool()
def get_history_details(history_id: str) -> GalaxyResult:
    """
    Get history metadata and summary count ONLY - does not return actual datasets

    This function provides quick access to history information without loading all datasets.
    For the actual datasets/contents, use get_history_contents() which supports
    pagination and ordering.

    Args:
        history_id: Galaxy history ID - a hexadecimal hash string identifying the history
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)

    Returns:
        GalaxyResult with history metadata and contents summary in data field

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

        return GalaxyResult(
            data={
                "history": history_info,
                "contents_summary": {
                    "total_items": total_items,
                    "note": "This is just a count. To get actual datasets, use "
                    "get_history_contents(history_id, limit=25, order='create_time-dsc') "
                    "for newest datasets first.",
                },
            },
            success=True,
            message=f"Retrieved details for history '{history_info.get('name', history_id)}'",
            count=total_items,
        )
    except Exception as e:
        logger.error(f"Failed to get history details for ID '{history_id}': {str(e)}")
        if "404" in str(e) or "No route" in str(e):
            raise ValueError(
                f"History ID '{history_id}' not found. Make sure to pass a valid history ID string."
            ) from e
        raise ValueError(f"Failed to get history details for ID '{history_id}': {str(e)}") from e


@mcp.tool()
def get_history_contents(
    history_id: str,
    limit: int = 100,
    offset: int = 0,
    deleted: bool = False,
    visible: bool = True,
    order: str = "hid-asc",
) -> GalaxyResult:
    """
    Get paginated contents (datasets and collections) from a specific history with ordering support

    Args:
        history_id: Galaxy history ID - a hexadecimal hash string identifying the history
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)
        limit: Maximum number of items to return per page (default: 100, max recommended: 500)
        offset: Number of items to skip from the beginning (default: 0, for pagination)
        deleted: Include deleted datasets in results (default: False)
        visible: Include only visible datasets (default: True, set False to include hidden)
        order: Sort order for results. Options include:
              - 'hid-asc': History ID ascending (default, oldest first)
              - 'hid-dsc': History ID descending (newest first)
              - 'create_time-dsc': Creation time descending (most recent first)
              - 'create_time-asc': Creation time ascending (oldest first)
              - 'update_time-dsc': Last updated descending (most recently modified first)
              - 'name-asc': Dataset name ascending (alphabetical)

    Returns:
        GalaxyResult with paginated dataset/collection list in data field and pagination metadata.
        Each item includes a 'history_content_type' field: 'dataset' or 'dataset_collection'

    Note:
        Performance: This function uses gi.histories.show_history(contents=True) to
        fetch all items and then paginates client-side. For very large histories,
        this may be slower than server-side pagination, but it is required to
        include dataset collections alongside datasets.
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        logger.info(
            f"Getting contents for history ID: {history_id} "
            f"(limit={limit}, offset={offset}, order={order})"
        )

        # Use show_history with contents=True to get both datasets and collections
        all_contents_raw = gi.histories.show_history(history_id, contents=True)

        # Add history_content_type field to distinguish datasets from collections
        all_contents = []
        for item in all_contents_raw:
            # Determine content type based on 'history_content_type' field if present,
            # otherwise infer from 'collection_type' or 'type' field
            if "history_content_type" in item:
                content_type = item["history_content_type"]
            elif item.get("collection_type") or item.get("type") == "collection":
                content_type = "dataset_collection"
            else:
                content_type = "dataset"

            # Add the field to the item (backward compatible - adds new field)
            item_with_type = {**item, "history_content_type": content_type}
            all_contents.append(item_with_type)

        # Filter by visibility and deleted status
        filtered_contents = all_contents
        if not deleted:
            filtered_contents = [
                item for item in filtered_contents if not item.get("deleted", False)
            ]
        if visible:
            filtered_contents = [item for item in filtered_contents if item.get("visible", True)]

        # Sort the contents based on order parameter
        def get_sort_key(item):
            if order.startswith("hid"):
                return item.get("hid", 0)
            elif order.startswith("create_time"):
                return item.get("create_time", "")
            elif order.startswith("update_time"):
                return item.get("update_time", "")
            elif order.startswith("name"):
                return item.get("name", "")
            else:
                return item.get("hid", 0)

        reverse = order.endswith("-dsc")
        sorted_contents = sorted(filtered_contents, key=get_sort_key, reverse=reverse)

        # Apply pagination
        total_items = len(sorted_contents)
        paginated_contents = sorted_contents[offset : offset + limit]

        # Calculate pagination metadata
        has_next = (offset + limit) < total_items
        has_previous = offset > 0
        current_page = (offset // limit) + 1 if limit > 0 else 1
        total_pages = ((total_items - 1) // limit) + 1 if limit > 0 and total_items > 0 else 1

        logger.info(
            f"Retrieved {len(paginated_contents)} items (page {current_page} of {total_pages})"
        )

        pagination = PaginationInfo(
            total_items=total_items,
            returned_items=len(paginated_contents),
            limit=limit,
            offset=offset,
            has_next=has_next,
            has_previous=has_previous,
            next_offset=offset + limit if has_next else None,
            previous_offset=max(0, offset - limit) if has_previous else None,
            helper_text=f"Showing page {current_page} of {total_pages}. "
            + (
                f"Use offset={offset + limit} for next page."
                if has_next
                else "This is the last page."
            ),
        )

        return GalaxyResult(
            data={"history_id": history_id, "contents": paginated_contents},
            success=True,
            message=f"Retrieved {len(paginated_contents)} items from history",
            count=len(paginated_contents),
            pagination=pagination,
        )
    except Exception as e:
        logger.error(f"Failed to get history contents for ID '{history_id}': {str(e)}")
        if "404" in str(e) or "No route" in str(e):
            raise ValueError(
                f"History ID '{history_id}' not found. Make sure to pass a valid history ID string."
            ) from e
        raise ValueError(f"Failed to get history contents for ID '{history_id}': {str(e)}") from e


@mcp.tool()
def get_job_details(dataset_id: str, history_id: str | None = None) -> GalaxyResult:
    """
    Get detailed information about the job that created a specific dataset

    Args:
        dataset_id: Galaxy dataset ID - a hexadecimal hash string identifying the dataset
                   (e.g., 'f2db41e1fa331b3e', typically 16 characters)
        history_id: Galaxy history ID containing the dataset - optional for performance optimization
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)

    Returns:
        GalaxyResult with job metadata, tool information, dataset ID, and job ID in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]
    base_url = state["url"] or normalized_galaxy_url or ""
    api_key = state["api_key"]
    if not base_url or not api_key:
        raise ValueError("Galaxy connection is missing URL or API key information.")

    try:
        # Get dataset provenance to find the creating job
        job_id: str | None = None
        provenance_error: Exception | None = None
        if history_id:
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

            except Exception as exc:
                provenance_error = exc

        if not job_id:
            # If provenance fails, try getting dataset details which might contain job info
            try:
                dataset_details = gi.datasets.show_dataset(dataset_id)
                job_id = dataset_details.get("creating_job")
                if not job_id:
                    raise ValueError(
                        f"No job information found for dataset '{dataset_id}'. "
                        "The dataset may not have been created by a job."
                    )
            except Exception as dataset_error:
                error_detail = str(provenance_error) if provenance_error else str(dataset_error)
                raise ValueError(
                    f"Failed to get job information for dataset '{dataset_id}': {error_detail}"
                ) from (provenance_error or dataset_error)

        # Get job details using the Galaxy API directly
        # (Bioblend doesn't have a direct method for this)
        url = f"{base_url}api/jobs/{job_id}"
        headers = {"x-api-key": api_key}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        job_info = response.json()

        return GalaxyResult(
            data={"job": job_info, "dataset_id": dataset_id, "job_id": job_id},
            success=True,
            message=f"Retrieved job details for dataset '{dataset_id}'",
        )
    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Dataset ID '{dataset_id}' not found or job not accessible. "
                "Make sure the dataset exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to get job details for dataset '{dataset_id}': {str(e)}") from e


@mcp.tool()
def get_dataset_details(
    dataset_id: str, include_preview: bool = True, preview_lines: int = 10
) -> GalaxyResult:
    """
    Get detailed information about a specific dataset, optionally including a content preview

    Args:
        dataset_id: Galaxy dataset ID - a hexadecimal hash string identifying the dataset
                   (e.g., 'f2db41e1fa331b3e', typically 16 characters)
        include_preview: Whether to include a preview of the dataset content showing first N lines
                        (default: True, only works for datasets in 'ok' state)
        preview_lines: Number of lines to include in the content preview (default: 10)

    Returns:
        GalaxyResult with dataset metadata (name, size, state, extension) and optional
        content preview in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get dataset details using bioblend
        dataset_info = gi.datasets.show_dataset(dataset_id)

        result_data: dict[str, Any] = {"dataset": dataset_info, "dataset_id": dataset_id}

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

                result_data["preview"] = {
                    "lines": preview,
                    "total_lines": len(lines),
                    "preview_lines": min(preview_lines, len(lines)),
                    "truncated": len(lines) > preview_lines,
                }

            except Exception as preview_error:
                logger.warning(f"Could not get preview for dataset {dataset_id}: {preview_error}")
                result_data["preview"] = {
                    "error": f"Preview unavailable: {str(preview_error)}",
                    "lines": None,
                }

        return GalaxyResult(
            data=result_data,
            success=True,
            message=f"Retrieved details for dataset '{dataset_info.get('name', dataset_id)}'",
        )

    except Exception as e:
        # If show_dataset failed, check if this might be a collection ID
        # by attempting to retrieve it as a collection
        try:
            collection_info = gi.dataset_collections.show_dataset_collection(
                dataset_id, instance_type="history"
            )
            # If we successfully retrieved it as a collection, that's the issue
            raise ValueError(
                f"The ID '{dataset_id}' is a dataset collection, not a dataset. "
                f"Collection name: '{collection_info.get('name', 'Unknown')}'. "
                "Use get_collection_details(collection_id) to inspect dataset "
                "collections and their members."
            ) from e
        except ValueError:
            # Re-raise the ValueError we just created above
            raise
        except Exception:
            # Not a collection either (show_dataset_collection failed),
            # so fall through to re-raise the original dataset error
            pass

        # Original error - not a collection
        if "404" in str(e):
            raise ValueError(
                f"Dataset ID '{dataset_id}' not found. "
                "Make sure the dataset exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to get dataset details for '{dataset_id}': {str(e)}") from e


@mcp.tool()
def get_collection_details(collection_id: str, max_elements: int = 100) -> GalaxyResult:
    """
    Get detailed information about a dataset collection and its members

    Dataset collections group multiple datasets together (e.g., paired-end reads,
    sample lists). This tool shows the collection structure and member datasets.

    Args:
        collection_id: Galaxy dataset collection ID - a hexadecimal hash string
                      (e.g., 'a1b2c3d4e5f6g7h8', typically 16 characters)
        max_elements: Maximum number of collection elements to return (default: 100)
                     Set lower for large collections to avoid overwhelming output

    Returns:
        GalaxyResult with collection metadata and elements in data field.
        Use get_dataset_details(dataset_id) to get full details for individual datasets.
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get collection details using bioblend
        collection_info = gi.dataset_collections.show_dataset_collection(
            collection_id, instance_type="history"
        )

        # Extract and normalize collection metadata
        collection_metadata = {
            "id": collection_info.get("id"),
            "name": collection_info.get("name"),
            "collection_type": collection_info.get("collection_type"),
            "element_count": collection_info.get("element_count", 0),
            "populated": collection_info.get("populated", True),
            "state": collection_info.get("state", "unknown"),
        }

        # Extract and normalize elements
        raw_elements = collection_info.get("elements", [])
        total_element_count = len(raw_elements)

        # Limit elements to max_elements
        elements_to_return = raw_elements[:max_elements]
        elements_truncated = total_element_count > max_elements

        normalized_elements = []
        for idx, element in enumerate(elements_to_return):
            element_obj = element.get("object", {})
            normalized_element = {
                "element_index": idx,
                "element_identifier": element.get("element_identifier", ""),
                "element_type": element.get("element_type", ""),
                "object_id": element_obj.get("id", ""),
                "name": element_obj.get("name", ""),
                "state": element_obj.get("state", ""),
                "extension": element_obj.get("extension", ""),
                "file_size": element_obj.get("file_size"),
            }
            normalized_elements.append(normalized_element)

        return GalaxyResult(
            data={
                "collection_id": collection_id,
                "history_content_type": "dataset_collection",
                "collection": collection_metadata,
                "elements": normalized_elements,
                "elements_truncated": elements_truncated,
                "note": (
                    "Use get_dataset_details(object_id) to get full details "
                    "for individual datasets in this collection."
                ),
            },
            success=True,
            message=f"Retrieved collection '{collection_metadata.get('name', collection_id)}'",
            count=len(normalized_elements),
        )

    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Collection ID '{collection_id}' not found. "
                "Make sure the collection exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to get collection details for '{collection_id}': {str(e)}") from e


@mcp.tool()
def download_dataset(
    dataset_id: str,
    file_path: str | None = None,
    use_default_filename: bool = True,
    require_ok_state: bool = True,
) -> GalaxyResult:
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
        GalaxyResult with download information in data field:
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

        # Check dataset state if required
        if require_ok_state and dataset_info.get("state") != "ok":
            raise ValueError(
                f"Dataset '{dataset_id}' is in state '{dataset_info.get('state')}', not 'ok'. "
                "Set require_ok_state=False to download anyway."
            )

        # Download the dataset
        result_path: str | bytes
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

            # Create suggested filename from dataset info
            filename = dataset_info.get("name", f"dataset_{dataset_id}")
            extension = dataset_info.get("extension", "")
            if extension and not filename.endswith(f".{extension}"):
                filename = f"{filename}.{extension}"

            download_path = None  # No file saved
            file_size = len(result_path) if isinstance(result_path, bytes | str) else None

        return GalaxyResult(
            data={
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
            },
            success=True,
            message=f"Downloaded dataset '{dataset_id}'",
        )

    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Dataset ID '{dataset_id}' not found. "
                "Make sure the dataset exists and you have permission to view it."
            ) from e
        raise ValueError(f"Failed to download dataset '{dataset_id}': {str(e)}") from e


@mcp.tool()
def upload_file(path: str, history_id: str | None = None) -> GalaxyResult:
    """
    Upload a local file to Galaxy for analysis.

    Galaxy automatically detects the file type (FASTQ, BAM, BED, etc.) and
    indexes the file appropriately. Large files are uploaded efficiently.

    RECOMMENDED WORKFLOW:
    1. Create a history: create_history("My Analysis")
    2. Upload your data files with this function
    3. Wait for upload to complete (check dataset state)
    4. Run tools on the uploaded data: run_tool()

    Args:
        path: Local file path to upload. Supports common bioinformatics formats:
              - Sequences: .fastq, .fasta, .fa, .fq, .fastq.gz
              - Alignments: .bam, .sam, .cram
              - Annotations: .bed, .gff, .gtf, .vcf
              - Tabular: .csv, .tsv, .txt
        history_id: Target history ID. If None, uses the most recent history.
                    Recommend always specifying for clarity.

    Returns:
        GalaxyResult with:
        - data.outputs: List of created datasets with IDs
        - data.jobs: Upload job information

    Example:
        >>> upload_file("/data/reads.fastq.gz", "abc123def456")
        GalaxyResult(
            data={
                "outputs": [{"id": "dataset789", "name": "reads.fastq.gz", "state": "queued"}],
                "jobs": [{"id": "job123", "state": "ok"}]
            },
            message="Uploaded file '/data/reads.fastq.gz'"
        )

    NEXT STEPS:
    - Wait for upload: Dataset state changes from "queued" -> "running" -> "ok"
    - Check status: get_history_contents(history_id) or get_dataset_details(dataset_id)
    - Run analysis: run_tool(history_id, tool_id, {"input": {"src": "hda", "id": dataset_id}})

    ERROR HANDLING:
    - "File not found": Check path exists and is readable
    - "Permission denied": Ensure file has read permissions
    - "Quota exceeded": User's Galaxy storage quota may be full
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

        # BioBlend accepts None for history_id and uses the most recently used history
        result = gi.tools.upload_file(path, history_id=history_id)  # type: ignore[arg-type]
        return GalaxyResult(
            data=result,
            success=True,
            message=f"Uploaded file '{path}'",
        )
    except Exception as e:
        raise ValueError(f"Failed to upload file: {str(e)}") from e


@mcp.tool()
def upload_file_from_url(
    url: str,
    history_id: str | None = None,
    file_type: str = "auto",
    dbkey: str = "?",
    file_name: str | None = None,
) -> GalaxyResult:
    """
    Upload a file from a URL to Galaxy

    Args:
        url: URL of the file to upload (e.g., 'https://example.com/data.fasta')
        history_id: Galaxy history ID where to upload the file - optional, uses current history
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters)
        file_type: Galaxy file format name (default: 'auto' for auto-detection)
                  Common types: 'fasta', 'fastq', 'bam', 'vcf', 'bed', 'tabular', etc.
        dbkey: Database key/genome build (default: '?', e.g., 'hg38', 'mm10', 'dm6')
        file_name: Optional name for the uploaded file in Galaxy (inferred from URL if not provided)

    Returns:
        GalaxyResult with upload status and information about the created dataset(s) in data field
    """
    ensure_connected()

    try:
        # Prepare kwargs for put_url
        kwargs = {
            "file_type": file_type,
            "dbkey": dbkey,
        }
        if file_name:
            kwargs["file_name"] = file_name

        result = galaxy_state["gi"].tools.put_url(url, history_id=history_id, **kwargs)
        return GalaxyResult(
            data=result,
            success=True,
            message=f"Uploaded file from URL '{url}'",
        )
    except Exception as e:
        raise ValueError(
            format_error(
                "Upload file from URL",
                e,
                {
                    "url": url,
                    "history_id": history_id,
                    "file_type": file_type,
                    "dbkey": dbkey,
                    "file_name": file_name,
                },
            )
        ) from e


@mcp.tool()
def get_invocations(
    invocation_id: str | None = None,
    workflow_id: str | None = None,
    history_id: str | None = None,
    limit: int | None = None,
    view: str = "collection",
    step_details: bool = False,
) -> GalaxyResult:
    """
    View workflow invocations in Galaxy

    Args:
        invocation_id: Specific workflow invocation ID to view - a hexadecimal hash string
                      (e.g., 'a1b2c3d4e5f6789a', typically 16 characters, optional)
        workflow_id: Filter invocations by workflow ID - a hexadecimal hash string
                    (e.g., 'b2c3d4e5f6789abc', typically 16 characters, optional)
        history_id: Filter invocations by history ID - a hexadecimal hash string
                   (e.g., '1cd8e2f6b131e5aa', typically 16 characters, optional)
        limit: Maximum number of invocations to return (optional, default: no limit)
        view: Level of detail to return - 'element' for detailed or 'collection' for summary
             (default: 'collection')
        step_details: Include details on individual workflow steps
                     (only applies when view is 'element', default: False)

    Returns:
        GalaxyResult with workflow invocation information in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # If invocation_id is provided, get details of a specific invocation
        if invocation_id:
            invocation = gi.invocations.show_invocation(invocation_id)
            return GalaxyResult(
                data=invocation,
                success=True,
                message=f"Retrieved invocation '{invocation_id}'",
            )

        # Otherwise get a list of invocations with optional filters
        invocations = gi.invocations.get_invocations(
            workflow_id=workflow_id,
            history_id=history_id,
            limit=limit,
            view=view,
            step_details=step_details,
        )
        return GalaxyResult(
            data=invocations,
            success=True,
            message=f"Retrieved {len(invocations)} workflow invocations",
            count=len(invocations),
        )
    except Exception as e:
        raise ValueError(f"Failed to get workflow invocations: {str(e)}") from e


@lru_cache(maxsize=1)
def get_manifest_json() -> list[dict[str, Any]]:
    response = requests.get("https://iwc.galaxyproject.org/workflow_manifest.json")
    response.raise_for_status()
    manifest = response.json()
    return manifest


@mcp.tool()
def get_iwc_workflows() -> GalaxyResult:
    """
    Fetch all workflows from the IWC (Interactive Workflow Composer)

    Returns:
        GalaxyResult with workflow manifest in data field
    """
    try:
        manifest = get_manifest_json()
        # Collect workflows from all manifest entries
        all_workflows = []
        for entry in manifest:
            if "workflows" in entry:
                all_workflows.extend(entry["workflows"])

        return GalaxyResult(
            data=all_workflows,
            success=True,
            message=f"Retrieved {len(all_workflows)} workflows from IWC",
            count=len(all_workflows),
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch IWC workflows: {str(e)}") from e


@mcp.tool()
def search_iwc_workflows(query: str) -> GalaxyResult:
    """
    Search for workflows in the IWC manifest

    Args:
        query: Search query (matches against name, description, and tags)

    Returns:
        GalaxyResult with matching workflows in data field
    """
    try:
        # Get the full manifest
        iwc_result = get_iwc_workflows.fn()
        manifest = iwc_result.data

        # Filter workflows based on the search query
        results = []
        query = query.lower()

        for workflow in manifest:
            # Check if query matches name, description or tags (case-insensitive)
            definition = workflow.get("definition", {})
            name = definition.get("name", "")
            description = definition.get("annotation", "")
            tags = definition.get("tags", [])

            # Lowercase for matching
            name_lower = name.lower()
            description_lower = description.lower()
            tags_lower = [tag.lower() for tag in tags]

            if (
                query in name_lower
                or query in description_lower
                or (tags_lower and any(query in tag for tag in tags_lower))
            ):
                results.append(
                    {
                        "trsID": workflow["trsID"],
                        "name": name,
                        "description": description,
                        "tags": tags,
                    }
                )

        return GalaxyResult(
            data=results,
            success=True,
            message=f"Found {len(results)} IWC workflows matching '{query}'",
            count=len(results),
        )
    except Exception as e:
        raise ValueError(f"Failed to search IWC workflows: {str(e)}") from e


@mcp.tool()
def import_workflow_from_iwc(trs_id: str) -> GalaxyResult:
    """
    Import a workflow from IWC to the user's Galaxy instance

    Args:
        trs_id: TRS ID of the workflow in the IWC manifest

    Returns:
        GalaxyResult with imported workflow information in data field
    """
    state = ensure_connected()
    gi: GalaxyInstance = state["gi"]

    try:
        # Get the workflow manifest
        iwc_result = get_iwc_workflows.fn()
        manifest = iwc_result.data

        # Find the specified workflow
        workflow = None
        for wf in manifest:
            if wf.get("trsID") == trs_id:
                workflow = wf
                break

        if not workflow:
            raise ValueError(
                f"Workflow with trsID '{trs_id}' not found in IWC manifest. "
                "Check the trsID format and that it exists in the IWC. "
                "You can search workflows using search_iwc_workflows() first."
            )

        # Extract the workflow definition
        workflow_definition = workflow.get("definition")
        if not workflow_definition:
            raise ValueError(
                f"No definition found for workflow with trsID '{trs_id}'. "
                "The workflow exists but has no valid definition. "
                "This may be a problem with the IWC manifest."
            )

        # Import the workflow into Galaxy
        imported_workflow = gi.workflows.import_workflow_dict(workflow_definition)
        return GalaxyResult(
            data=imported_workflow,
            success=True,
            message=f"Successfully imported workflow '{trs_id}'",
        )
    except Exception as e:
        raise ValueError(f"Failed to import workflow from IWC: {str(e)}") from e


@mcp.tool()
def list_workflows(
    workflow_id: str | None = None, name: str | None = None, published: bool = False
) -> GalaxyResult:
    """
    List workflows available in the Galaxy instance

    Args:
        workflow_id: Specific workflow ID to get (optional) - a hexadecimal hash string
        name: Filter workflows by name (optional)
        published: Include published workflows (default: False, shows only user workflows)

    Returns:
        GalaxyResult with list of workflows in data field
    """
    ensure_connected()

    try:
        workflows = galaxy_state["gi"].workflows.get_workflows(
            workflow_id=workflow_id, name=name, published=published
        )
        return GalaxyResult(
            data=workflows,
            success=True,
            message=f"Found {len(workflows)} workflows",
            count=len(workflows),
        )
    except Exception as e:
        raise ValueError(
            format_error(
                "List workflows",
                e,
                {"workflow_id": workflow_id, "name": name, "published": published},
            )
        ) from e


@mcp.tool()
def get_workflow_details(workflow_id: str, version: int | None = None) -> GalaxyResult:
    """
    Get detailed information about a specific workflow

    Args:
        workflow_id: ID of the workflow to get details for - a hexadecimal hash string
        version: Specific version of the workflow (optional, uses latest if not specified)

    Returns:
        GalaxyResult with workflow information including steps, inputs, and parameters in data field
    """
    ensure_connected()

    try:
        workflow = galaxy_state["gi"].workflows.show_workflow(
            workflow_id=workflow_id, version=version
        )
        return GalaxyResult(
            data=workflow,
            success=True,
            message=f"Retrieved details for workflow '{workflow.get('name', workflow_id)}'",
        )
    except Exception as e:
        raise ValueError(
            format_error(
                "Get workflow details", e, {"workflow_id": workflow_id, "version": version}
            )
        ) from e


@mcp.tool()
def invoke_workflow(
    workflow_id: str,
    inputs: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    history_id: str | None = None,
    history_name: str | None = None,
    inputs_by: str = "step_index",
    parameters_normalized: bool = False,
) -> GalaxyResult:
    """
    Invoke (run) a workflow with specified inputs and parameters

    Args:
        workflow_id: ID of the workflow to invoke - a hexadecimal hash string
        inputs: Mapping of workflow inputs to datasets. Format:
               {'step_index': {'id': 'dataset_id', 'src': 'hda'}} where src can be:
               - 'hda' for HistoryDatasetAssociation
               - 'hdca' for HistoryDatasetCollectionAssociation
               - 'ldda' for LibraryDatasetDatasetAssociation
               - 'ld' for LibraryDataset
        params: Tool parameter overrides as a nested dictionary
        history_id: ID of history to store workflow outputs (optional)
        history_name: Name for new history to create (ignored if history_id provided)
        inputs_by: How to identify workflow inputs - 'step_index', 'step_uuid', or 'name'
        parameters_normalized: Whether parameters are already in normalized format

    Returns:
        GalaxyResult with workflow invocation information including invocation ID in data field
    """
    ensure_connected()

    try:
        invocation = galaxy_state["gi"].workflows.invoke_workflow(
            workflow_id=workflow_id,
            inputs=inputs,
            params=params,
            history_id=history_id,
            history_name=history_name,
            inputs_by=inputs_by,
            parameters_normalized=parameters_normalized,
        )
        return GalaxyResult(
            data=invocation,
            success=True,
            message=f"Invoked workflow '{workflow_id}'",
        )
    except Exception as e:
        raise ValueError(
            format_error(
                "Invoke workflow",
                e,
                {
                    "workflow_id": workflow_id,
                    "history_id": history_id,
                    "history_name": history_name,
                    "inputs_by": inputs_by,
                },
            )
        ) from e


@mcp.tool()
def cancel_workflow_invocation(invocation_id: str) -> GalaxyResult:
    """
    Cancel a running workflow invocation

    Args:
        invocation_id: ID of the workflow invocation to cancel - a hexadecimal hash string

    Returns:
        GalaxyResult with cancellation status and updated invocation information in data field
    """
    ensure_connected()

    try:
        result = galaxy_state["gi"].workflows.cancel_invocation(invocation_id)
        return GalaxyResult(
            data={"cancelled": True, "invocation": result},
            success=True,
            message=f"Cancelled workflow invocation '{invocation_id}'",
        )
    except Exception as e:
        raise ValueError(
            format_error("Cancel workflow invocation", e, {"invocation_id": invocation_id})
        ) from e


def run_http_server(
    *,
    host: str | None = None,
    port: int | None = None,
    transport: str | None = None,
    path: str | None = None,
) -> None:
    """Run the MCP server over HTTP-based transport."""
    resolved_host = host or os.environ.get("GALAXY_MCP_HOST", "0.0.0.0")
    resolved_port = port if port is not None else int(os.environ.get("GALAXY_MCP_PORT", "8000"))
    resolved_transport = (
        transport or os.environ.get("GALAXY_MCP_TRANSPORT") or "streamable-http"
    ).lower()
    if resolved_transport not in {"streamable-http", "sse"}:
        raise ValueError(
            f"Unsupported transport '{resolved_transport}'. Choose 'streamable-http' or 'sse'."
        )
    # Type-safe cast after validation
    http_transport = cast(Literal["streamable-http", "sse"], resolved_transport)

    resolved_path = path or os.environ.get("GALAXY_MCP_HTTP_PATH")
    if resolved_path is None and resolved_transport == "streamable-http":
        resolved_path = "/"
    if resolved_path is not None and not resolved_path.startswith("/"):
        resolved_path = f"/{resolved_path}"

    logger.info(
        "Starting Galaxy MCP server over %s at %s:%s%s",
        http_transport,
        resolved_host,
        resolved_port,
        resolved_path or "",
    )
    mcp.run(
        transport=http_transport,
        host=resolved_host,
        port=resolved_port,
        path=resolved_path,
    )


if __name__ == "__main__":
    selected_transport = os.environ.get("GALAXY_MCP_TRANSPORT", "stdio").lower()
    if selected_transport in {"streamable-http", "sse"}:
        run_http_server(transport=selected_transport)
    else:
        mcp.run()
