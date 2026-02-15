# Galaxy MCP Server
import concurrent.futures
import logging
import os
import threading
from functools import lru_cache
from typing import Any

import requests
from bioblend.galaxy import GalaxyInstance
from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP

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

# Create an MCP server
mcp: FastMCP = FastMCP("Galaxy")

# Galaxy client state
galaxy_state: dict[str, Any] = {
    "url": os.environ.get("GALAXY_URL"),
    "api_key": os.environ.get("GALAXY_API_KEY"),
    "gi": None,
    "connected": False,
}


# Initialize Galaxy client if environment variables are set
if galaxy_state["url"] and galaxy_state["api_key"]:
    try:
        galaxy_url = (
            galaxy_state["url"] if galaxy_state["url"].endswith("/") else f"{galaxy_state['url']}/"
        )
        galaxy_state["url"] = galaxy_url
        galaxy_state["gi"] = GalaxyInstance(url=galaxy_url, key=galaxy_state["api_key"])
        galaxy_state["connected"] = True
        logger.info(f"Galaxy client initialized from environment variables (URL: {galaxy_url})")
    except Exception as e:
        logger.warning(f"Failed to initialize Galaxy client from environment variables: {e}")
        logger.warning("You'll need to use connect() to establish a connection.")


def ensure_connected():
    """Helper function to ensure Galaxy connection is established"""
    if not galaxy_state["connected"] or not galaxy_state["gi"]:
        raise ValueError(
            "Not connected to Galaxy. "
            "Please run connect() first with your Galaxy URL and API key. "
            "Example: connect(url='https://your-galaxy.org', api_key='your-key')"
        )


@mcp.tool(
    name="connect",
    description="Connect to a Galaxy server using a URL and API key. Uses environment variables or a .env file when values are not provided.",
    tags={"api", "connection", "auth"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "connected": {
                "type": "boolean",
                "description": "True if API key was validated and current user retrieved."
            },
            "user": {
                "type": "object",
                "description": "User information."
            }
        },
        "required": ["connected", "user"]
    }
)
def connect(url: str | None = None, api_key: str | None = None) -> dict[str, Any]:
    """
    Connect to Galaxy server

    Args:
        url: Galaxy server URL (optional, uses GALAXY_URL env var if not provided)
        api_key: Galaxy API key (optional, uses GALAXY_API_KEY env var if not provided)

    Returns:
        Connection status and user information
    """
    try:
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

        return {"connected": True, "user": user_info}
    except Exception as e:
        # Reset state on failure
        galaxy_state["url"] = None
        galaxy_state["api_key"] = None
        galaxy_state["gi"] = None
        galaxy_state["connected"] = False

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


@mcp.tool(
    name="search_tools",
    description="Search Galaxy for tools matching a given name.",
    tags={"tools", "search", "lookup"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema= {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                },
                "description": "Tools whose names match the query string, returned as an array."
            }
        },
        "required": ["tools"]
    }
)
def search_tools(query: str) -> dict[str, Any]:
    """
    Search for tools in Galaxy

    Args:
        query: Search query (tool name to filter on)

    Returns:
        List of tools matching the query
    """
    ensure_connected()

    try:
        # The get_tools method is used with name filter parameter
        tools = galaxy_state["gi"].tools.get_tools(name=query)
        return {"tools": tools}
    except Exception as e:
        raise ValueError(format_error("Search tools", e, {"query": query})) from e


@mcp.tool(
    name="get_tool_details",
    description="Retrieve detailed information about a specific Galaxy tool identified by tool_id.",
    tags={"galaxy", "tools", "metadata", "details"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "model_class": {
                "type": "string",
                "description": "Galaxy model class. Typically 'Tool'."
            },
            "id": {
                "type": "string",
                "description": "Unique Galaxy tool identifier."
            },
            "name": {
                "type": "string",
                "description": "Human-readable tool name"
            },
            "version": {
                "type": "string",
                "description": "Version of the tool."
            },
            "description": {
                "type": "string",
                "description": "Tool description."
            },
            "icon": {
                "type": ["string", "null"],
                "description": "Icon URL, if available."
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of labels associated with the tool."
            },
            "edam_operations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of EDAM operation terms."
            },
            "edam_topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of EDAM topic terms."
            },
            "hidden": {
                "type": "string",
                "description": "Hidden flag, if any."
            },
            "is_workflow_compatible": {
                "type": "boolean",
                "description": "Whether the tool is workflow compatible."
            },
            "xrefs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "External references associated with the tool."
            },
            "tool_shed_repository": {
                "type": "object",
                "description": "Tool shed repository information."
            },
            "inputs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Tool input parameter definitions. Included when io_details is true."
            },
            "outputs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Tool output definitions. Included when io_details is true."
            },
            "panel_section_id": {
                "type": "string",
                "description": "Tool panel section identifier."
            },
            "panel_section_name": {
                "type": "string",
                "description": "Tool panel section name."
            },
            "form_style": {
                "type": "string",
                "description": "Form style used for tool parameters."
            }
        }
    }
)
def get_tool_details(tool_id: str, io_details: bool = False) -> dict[str, Any]:
    """
    Get detailed information about a specific tool

    Args:
        tool_id: ID of the tool
        io_details: Whether to include input/output details

    Returns:
        Tool details
    """
    ensure_connected()

    try:
        # Get detailed information about the tool
        tool_info = galaxy_state["gi"].tools.show_tool(tool_id, io_details=io_details)
        return tool_info
    except Exception as e:
        raise ValueError(
            format_error("Get tool details", e, {"tool_id": tool_id, "io_details": io_details})
        ) from e


@mcp.tool(
    name="get_tool_citations",
    description="Retrieve citation information for a specific Galaxy tool, tool identified by tool ID.",
    tags={"tools", "citations"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Human-readable tool name."
            },
            "tool_version": {
                "type": "string",
                "description": "Version of the tool."
            },
            "citations": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Citation metadata objects associated with the tool, returned as an array. May be empty."
            }
        },
        "required": ["tool_name", "tool_version", "citations"]
    }
)
def get_tool_citations(tool_id: str) -> dict[str, Any]:
    """
    Get citation information for a specific tool

    Args:
        tool_id: ID of the tool

    Returns:
        Tool citation information
    """
    ensure_connected()

    try:
        # Get the tool information which includes citations
        tool_info = galaxy_state["gi"].tools.show_tool(tool_id)

        # Extract citation information
        citations = tool_info.get("citations", [])

        return {
            "tool_name": tool_info.get("name", tool_id),
            "tool_version": tool_info.get("version", "unknown"),
            "citations": citations,
        }
    except Exception as e:
        raise ValueError(format_error("Get tool citations", e, {"tool_id": tool_id})) from e


@mcp.tool(
    name="run_tool",
    description="Execute a Galaxy tool in an existing history using a tool ID and structured input parameters.",
    tags={"tools", "execution", "jobs"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Galaxy job objects created by the tool execution."
            },
            "outputs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Output dataset objects produced by the tool execution."
            },
            "implicit_collections": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Implicit collection outputs, if any."
            },
            "output_collections": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Explicit collection outputs, if any."
            }
        }
    }
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
    ensure_connected()

    try:
        # Run the tool with provided inputs
        result = galaxy_state["gi"].tools.run_tool(history_id, tool_id, inputs)
        return result
    except Exception as e:
        raise ValueError(
            format_error(
                "Run tool", e, {"history_id": history_id, "tool_id": tool_id, "inputs": inputs}
            )
        ) from e


@mcp.tool(
    name="get_tool_panel",
    description="Retrieve the Galaxy tool panel hierarchy, including sections and their nested tools",
    tags={"tools", "panel", "toolbox"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "tool_panel": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "Array of tool panel items. Each item is either a tool, or a section containing nested tools."
                )
            }
        },
        "required": ["tool_panel"]
    }
)
def get_tool_panel() -> dict[str, Any]:
    """
    Get the tool panel structure (toolbox)

    Returns:
        Tool panel hierarchy
    """
    ensure_connected()

    try:
        # Get the tool panel structure
        tool_panel = galaxy_state["gi"].tools.get_tool_panel()
        return {"tool_panel": tool_panel}
    except Exception as e:
        raise ValueError(format_error("Get tool panel", e)) from e


@mcp.tool(
    name="create_history",
    description="Create a new history in Galaxy.",
    tags={"history", "creation"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "model_class": {
                "type": "string",
                "description":
                "Galaxy model class of the object. Typically 'History'."
            },
            "id": {
                "type": "string",
                "description": "Unique Galaxy history identifier."
            },
            "name": {
                "type": "string",
                "description": "Human-readable name of the history."
            },
            "deleted": {
                "type": "boolean",
                "description": "Whether the history is deleted."
            },
            "purged": {
                "type": "boolean",
                "description": "Whether the history is purged."
            },
            "archived": {
                "type": "boolean",
                "description": "Whether the history is archived."
            },
            "url": {
                "type": "string",
                "description": "URL of the history."
            },
            "published": {
                "type": "boolean",
                "description": "Whether the history is published."
            },
            "count": {
                "type": "number",
                "description": "Number of datasets in the history."
            },
            "annotation": {
                "type": ["string", "null"],
                "description": "Annotation text for the history."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags associated with the history."
            },
            "update_time": {
                "type": "string",
                "description": "Timestamp of the last update."
            },
            "contents_url": {
                "type": "string",
                "description": "URL for the contents of the history."
            },
            "size": {
                "type": "number",
                "description": "Size of the history."
            },
            "user_id": {
                "type": "string",
                "description": "User ID of the history owner."
            },
            "create_time": {
                "type": "string",
                "description": "Creation timestamp of the history."
            },
            "state": {
                "type": "string",
                "description": "Current state of the history."
            },
            "state_ids": {
                "type": "object",
                "description": "Mapping of dataset states to list of dataset IDs."
            },
            "state_details": {
                "type": "object",
                "description": "Mapping of dataset states to counts of datasets."
            }
        },
        "required": ["model_class", "id", "name", "deleted", "purged", "archived", "url", "published", 
                     "count", "annotation", "tags", "update_time", "contents_url", "size", "user_id", 
                     "create_time", "state", "state_ids", "state_details"]
    }
)
def create_history(history_name: str) -> dict[str, Any]:
    """
    Create a new history in Galaxy

    Args:
        history_name: Human-readable name for the new history (e.g., 'RNA-seq Analysis')

    Returns:
        Dictionary containing the created history details including the new history ID hash
    """
    ensure_connected()
    return galaxy_state["gi"].histories.create_history(history_name)


@mcp.tool(
    name="filter_tools_by_dataset",
    description="Identify Galaxy tools suitable for one or more dataset types",
    tags={"tools", "dataset", "filter"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "recommended_tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Galaxy tool ID."},
                        "name": {"type": "string", "description": "Tool name."},
                        "description": {"type": "string", "description": "Tool description."},
                        "versions": {"type": "array", "items": {"type": "string"}, "description": "List of available tool versions."}
                    },
                    "required": ["id", "name"]
                },
                "description": "List of tool objects identified as suitable for the specified dataset types. May be empty if no tools match."
            },
            "count": {"type": "number", "description": "Total number of recommended tools."}
        },
        "required": ["recommended_tools", "count"]
    }
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

    ensure_connected()

    lock = threading.Lock()

    dataset_keywords = [dt.lower() for dt in dataset_type]

    try:
        tool_panel = galaxy_state["gi"].tools.get_tool_panel()

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
                tool_details = galaxy_state["gi"].tools.show_tool(tool_id, io_details=True)
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
    name="get_server_info",
    description="Retrieve Galaxy server metadata including version, base URL, and configuration details.",
    tags={"server", "info", "metadata"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string", 
                "description": "Base URL of the Galaxy server."
            },
            "version": {
                "type": "object", 
                "description": "Galaxy server version."
            },
            "config": {
                "type": "object", 
                "description": "Server configuration details."
            }
        },
        "required": ["url", "version", "config"]
    }
)
def get_server_info() -> dict[str, Any]:
    """
    Get Galaxy server information including version, URL, and configuration details

    Returns:
        Server information including version, URL, and other configuration details
    """
    ensure_connected()

    try:
        # Get server configuration info
        config_info = galaxy_state["gi"].config.get_config()

        # Get server version info
        version_info = galaxy_state["gi"].config.get_version()

        # Build comprehensive server info response
        server_info = {
            "url": galaxy_state["url"],
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
    name="get_user",
    description="Retrieve metadata for the current user, including disk usage, quota, and account settings.",
    tags={"user", "metadata", "account"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "total_disk_usage": {
                "type": "number", 
                "description": "Total disk usage, in bytes."
            },
            "nice_total_disk_usage": {
                "type": "string", 
                "description": "Human-readable representation of total disk usage."
            },
            "quota_percent": {
                "type": "number", 
                "description": "Percentage of the user's allocated quota currently in use."
            },
            "id": {
                "type": "string", 
                "description": "Unique Galaxy user identifier."
            },
            "username": {
                "type": "string", 
                "description": "Username of the current user."
            },
            "email": {
                "type": "string", 
                "description": "Email associated with the user account."
            },
            "deleted": {
                "type": "boolean", 
                "description": "Whether the user account is deleted."
            },
            "is_admin": {
                "type": "boolean", 
                "description": "Whether the user has administrator privileges."
            },
            "purged": {
                "type": "boolean", 
                "description": "Whether the user account has been purged."
            },
            "preferences": {
                "type": "object", 
                "description": "User-specific preference settings, returned as a dictionary."
            },
            "preferred_object_store_id": {
                "type": ["string", "null"], 
                "description": "ID of the user's preferred object store."
            },
            "quota": {
                "type": "string", 
                "description": "Human-readable representation of the user's total quota."
            },
            "quota_bytes": {
                "type": "number", 
                "description": "Total quota allocated to the user, in bytes."
            }
        },
        "required": ["total_disk_usage", "id", "username", "email", "quota", "quota_bytes"]
    }
)
def get_user() -> dict[str, Any]:
    """
    Get current user information

    Returns:
        Current user details
    """
    ensure_connected()

    try:
        user_info = galaxy_state["gi"].users.get_current_user()
        return user_info
    except Exception as e:
        raise ValueError(f"Failed to get user: {str(e)}") from e


@mcp.tool(
    name="get_histories",
    description="Retrieve a list of Galaxy histories, optionally filtered by name and paginated.",
    tags={"history", "list", "pagination", "filtered"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "histories": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of Galaxy history objects."
            },
            "pagination": {
                "type": "object",
                "description": "Pagination metadata.",
                "properties": {
                    "total_items": {
                        "type": "integer", 
                        "description": "Total matching histories, ignoring pagination."
                    },
                    "returned_items": {
                        "type": "integer", 
                        "description": "Number of histories returned in this call."
                    },
                    "paginated": {
                        "type": "boolean",
                        "description": "Whether pagination is applied."
                    },
                    "limit": {
                        "type": ["integer", "null"], 
                        "description": "Maximum number of histories per page, null if all returned."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset used."
                    },
                    "current_page": {
                        "type": "integer", 
                        "description": "Current page number based on offset and limit."
                    },
                    "total_pages": {
                        "type": "integer", 
                        "description": "Total number of pages available based on limit."
                    },
                    "has_next": {
                        "type": "boolean", 
                        "description": "Whether there is a next page available."
                    },
                    "has_previous": {
                        "type": "boolean", 
                        "description": "Whether there is a previous page available."
                    },
                    "next_offset": {
                        "type": ["integer", "null"], 
                        "description": "Offset for next page, if any."
                    },
                    "previous_offset": {
                        "type": ["integer", "null"], 
                        "description": "Offset for previous page, if any."
                    },
                    "helper_text": {
                        "type": ["string", "null"],
                        "description": "Human-readable guidance for pagination."
                    }
                },
                "required": ["total_items", "returned_items"]
            }
        },
        "required": ["histories", "pagination"]
    }
)
def get_histories(
    limit: int | None = None, offset: int = 0, name: str | None = None
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
    ensure_connected()

    try:
        # Get histories with pagination and optional filtering
        histories = galaxy_state["gi"].histories.get_histories(
            limit=limit, offset=offset, name=name
        )

        # If pagination is used, get total count for metadata
        if limit is not None:
            # Get total count without pagination
            all_histories = galaxy_state["gi"].histories.get_histories(name=name)
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

        return {"histories": histories, "pagination": pagination}
    except Exception as e:
        raise ValueError(
            f"Failed to get histories: {str(e)}. "
            "Check your connection to Galaxy and that you have "
            "permission to view histories."
        )


@mcp.tool(
    name="list_history_ids",
    description="Get a simplified list of Galaxy history IDs and names.",
    tags={"histories", "id", "list"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "description": "Array of objects containing history IDs and names.",
        "properties": {
            "histories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "History ID."},
                        "name": {"type": "string", "description": "History name."}
                    },
                    "required": ["id", "name"]
                }
            }
        },
        "required": ["histories"]
    }
)
def list_history_ids() -> dict[str, Any]:
    """
    Get a simplified list of history IDs and names for easy reference

    Returns:
        List of dictionaries containing 'id' and 'name' fields
    """
    ensure_connected()

    try:
        histories = galaxy_state["gi"].histories.get_histories()
        if not histories:
            return {"histories": []}
        # Extract just the id and name for convenience
        simplified = [{"id": h["id"], "name": h.get("name", "Unnamed")} for h in histories]
        return {"histories": simplified}
    except Exception as e:
        raise ValueError(f"Failed to list history IDs: {str(e)}") from e


@mcp.tool(
    name="get_history_details",
    description="Retrieve basic metadata and summary count of a Galaxy history without returning datasets.",
    tags={"history", "metadata", "summary"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "history": {
                "type": "object",
                "description": "Basic metadata of the history, including name, ID and state."
            },
            "contents_summary": {
                "type": "object",
                "properties": {
                    "total_items": {"type": "number", "description": "Number of datasets in the history."},
                    "note": {"type": "string", "description": "Instruction to get actual datasets using get_history_contents."}
                },
                "required": ["total_items", "note"],
            }
        },
        "required": ["history", "contents_summary"]
    }
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
    ensure_connected()

    try:
        logger.info(f"Getting details for history ID: {history_id}")

        # Get history details
        history_info = galaxy_state["gi"].histories.show_history(history_id, contents=False)
        logger.info(f"Successfully retrieved history info: {history_info.get('name', 'Unknown')}")

        # Get total count by calling without limit
        all_contents = galaxy_state["gi"].histories.show_history(history_id, contents=True)
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
    name="get_history_contents",
    description=(
        "Retrieve paginated datasets from a specific Galaxy history. "
        "Supports filtering by visibility/deleted status and ordering by multiple fields."
    ),
    tags={"history", "datasets", "content"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "history_id": {
                "type": "string",
                "description": "History identifier used for this query."
            },
            "contents": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of dataset objects in this page of results."
            },
            "pagination": {
                "type": "object",
                "description": "Pagination metadata",
                "properties": {
                    "total_items": {
                        "type": "integer", 
                        "description": "Total number of datasets matching filters."
                    },
                    "returned_items": {
                        "type": "integer", 
                        "description": "Number of datasets returned."
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "Maximum number of histories per page."
                    },
                    "offset": {
                        "type": "integer", 
                        "description": "Number of items skipped from the beginning."
                    },
                    "current_page": {
                        "type": "integer", 
                        "description": "Current page number based on offset and limit"
                    },
                    "total_pages": {
                        "type": "integer", 
                        "description": "Total number of pages available based on limit."
                        },
                    "has_next": {
                        "type": "boolean", 
                        "description": "Whether there is a next page available."
                    },
                    "has_previous": {
                        "type": "boolean", 
                        "description": "Whether there is a previous page available"
                    },
                    "next_offset": {
                        "type": ["integer", "null"], 
                        "description": "Offset for next page, if any."
                    },
                    "previous_offset": {
                        "type": ["integer", "null"], 
                        "description": "Offset for previous page, null if first page"
                    },
                    "helper_text": {
                        "type": "string", 
                        "description": "Human-readable guidance for pagination."
                    }
                },
                "required": ["total_items", "returned_items", "limit", "offset", "current_page", 
                             "total_pages", "has_next", "has_previous", "helper_text"]
            }
        },
        "required": ["history_id", "contents", "pagination"]
    }
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
    ensure_connected()

    try:
        logger.info(
            f"Getting contents for history ID: {history_id} "
            f"(limit={limit}, offset={offset}, order={order})"
        )

        # Use datasets API for better ordering support
        contents = galaxy_state["gi"].datasets.get_datasets(
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
        all_contents = galaxy_state["gi"].datasets.get_datasets(
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
    name="get_job_details",
    description="Retrieve information about the job that created a specific history item.",
    tags={"galaxy", "jobs", "datasets"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "job": {
                "type": "object",
                "description": "Job metadata including tool, state, and timestamps"
            },
            "dataset_id": {
                "type": "string",
                "description": "Dataset ID for which job details were retrieved."
            },
            "job_id": {
                "type": "string",
                "description": "Job ID of the job that created the dataset."
            }
        },
        "required": ["job", "dataset_id", "job_id"]
    }
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
    ensure_connected()

    try:
        # Get dataset provenance to find the creating job
        try:
            provenance = galaxy_state["gi"].histories.show_dataset_provenance(
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
                dataset_details = galaxy_state["gi"].datasets.show_dataset(dataset_id)
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
        url = f"{galaxy_state['url']}api/jobs/{job_id}"
        headers = {"x-api-key": galaxy_state["api_key"]}
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
    name="get_dataset_details",
    description="Returns dataset metadata and an optional content preview.",
    tags={"datasets", "metadata", "preview"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "object",
                "description": "Metadata of the dataset including name, size, state, id, and other attributes."
            },
            "dataset_id": {
                "type": "string",
                "description": "Galaxy dataset ID for which details were retrieved."
            },
            "preview": {
                "type": "object",
                "description": "Optional content preview of the dataset if include_preview=True and dataset state is 'ok'.",
                "properties": {
                    "lines": {
                        "type": "string", 
                        "description": "Preview lines as a single string."
                    },
                    "total_lines": {
                        "type": "integer", 
                        "description": "Total number of lines in the dataset."
                    },
                    "preview_lines": {
                        "type": "integer", 
                        "description": "Number of lines returned in the preview."
                    },
                    "truncated": {
                        "type": "boolean", 
                        "description": "Whether the preview was truncated."
                    },
                    "error": {
                        "type": "string", 
                        "description": "Error message if preview could not be retrieved."
                    }
                },
                "required": ["lines", "total_lines", "preview_lines", "truncated"]
            }
        },
        "required": ["dataset", "dataset_id"]
    }
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
    ensure_connected()

    try:
        # Get dataset details using bioblend
        dataset_info = galaxy_state["gi"].datasets.show_dataset(dataset_id)

        result = {"dataset": dataset_info, "dataset_id": dataset_id}

        # Add content preview if requested and dataset is in 'ok' state
        if include_preview and dataset_info.get("state") == "ok":
            try:
                # Get dataset content for preview
                content = galaxy_state["gi"].datasets.download_dataset(
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
    name="download_dataset",
    description=(
        "Download a Galaxy dataset to the local filesystem or to memory. "
        "If `file_path` is provided, the content is saved there. "
        "Otherwise, content is returned in memory with a suggested filename."
    ),
    tags={"dataset", "download", "file path"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "Dataset ID; use for follow-up actions."
            },
            "file_path": {
                "type": ["string", "null"],
                "description": "Local path where dataset was saved, or null if in memory."
            },
            "suggested_filename": {
                "type": ["string", "null"],
                "description": "Recommended filename if content downloaded to memory."
            },
            "content_available": {
                "type": "boolean",
                "description": "Whether the dataset content was successfully downloaded."
            },
            "file_size": {
                "type": ["integer", "null"],
                "description": "Size of downloaded content in bytes."
            },
            "note": {
                "type": "string",
                "description": "Explanation of whether content was saved to file or returned in memory."
            },
            "dataset_info": {
                "type": "object",
                "description": "Metadata of the dataset",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name"},
                    "extension": {"type": "string", "description": "Dataset file extension"},
                    "state": {"type": "string", "description": "Processing state of the dataset"},
                    "genome_build": {"type": ["string", "null"], "description": "Associated genome build."},
                    "file_size": {"type": ["integer", "null"], "description": "Reported size in Galaxy"}
                },
                "required": ["name", "extension", "state", "genome_build", "file_size"]
            }
        },
        "required": ["dataset_id", "content_available", "dataset_info", "note"]
    }
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
    ensure_connected()

    try:
        # Get dataset info first to check state and get metadata
        dataset_info = galaxy_state["gi"].datasets.show_dataset(dataset_id)

        # Check dataset state if required
        if require_ok_state and dataset_info.get("state") != "ok":
            raise ValueError(
                f"Dataset '{dataset_id}' is in state '{dataset_info.get('state')}', not 'ok'. "
                "Set require_ok_state=False to download anyway."
            )

        # Download the dataset
        if file_path:
            # Download to specific path
            result_path = galaxy_state["gi"].datasets.download_dataset(
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
            result_path = galaxy_state["gi"].datasets.download_dataset(
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
    name="upload_file",
    description="Upload a local file to a Galaxy history. Returns details about created datasets.",
    tags={"upload", "file", "dataset", "history"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "outputs": {
                "type": "array",
                "description": "List of datasets created from the uploaded file."
            },
            "output_collections": {
                "type": "array",
                "description": "List of output collections created by the upload, if any."
            },
            "jobs": {
                "type": "array",
                "description": "Jobs triggered by the upload."
            },
            "implicit_collections": {
                "type": "array",
                "description": "Implicit collections generated by the upload, if any."
            },
            "produces_entry_points": {
                "type": "boolean",
                "description": "Whether the upload produces entry oints usable as workflow inputs."
            }
        },
        "required": ["outputs", "jobs", "output_collections", "implicit_collections", "produces_entry_points"]
    }
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
    ensure_connected()

    try:
        if not os.path.exists(path):
            abs_path = os.path.abspath(path)
            raise ValueError(
                f"File not found: '{path}' (absolute: '{abs_path}'). "
                "Check that the file exists and you have read permissions."
            )

        result = galaxy_state["gi"].tools.upload_file(path, history_id=history_id)
        return result
    except Exception as e:
        raise ValueError(f"Failed to upload file: {str(e)}") from e


@mcp.tool(
    name="upload_file_from_url",
    description=(
        "Upload a file to a Galaxy history by fetching it from a remote URL. "
        "Galaxy downloads the content server-side and creates one or more datasets. "
        "If no history ID is provided, the file is uploaded to the current history."
    ),
    tags={"upload", "url", "dataset", "history"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "outputs": {
                "type": "array",
                "description": "List of datasets created from the uploaded URL."
            },
            "output_collections": {
                "type": "array",
                "description": "List of output collections created by the upload, if any."
            },
            "jobs": {
                "type": "array",
                "description": "Jobs triggered to fetch the remote file."
            },
            "implicit_collections": {
                "type": "array",
                "description": "Implicit dataset collections automatically generated by the upload, if any."
            },
            "produces_entry_points": {
                "type": "boolean",
                "description": (
                    "Whether the upload produces entry points usable as workflow inputs."
                )
            }
        }
    }
)
def upload_file_from_url(
    url: str,
    history_id: str | None = None,
    file_type: str = "auto",
    dbkey: str = "?",
    file_name: str | None = None,
) -> dict[str, Any]:
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
        Dictionary containing upload status and information about the created dataset(s)
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
        return result
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


@mcp.tool(
    name="get_invocations",
    description=(
        "Retrieve workflow invocation information from Galaxy. "
        "Can return a specific invocation by ID or list workflow invocations "
        "optionally filtered by workflow or history. "
        "Supports summary or detailed views, with optional per-step details."
    ),
    tags={"workflow", "invocation", "execution", "filter"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "invocation": {
                "type": "object",
                "description": (
                    "Detailed information for a single workflow invocation. "
                    "Present only when invocation_id is provided."
                )
            },
            "invocations": {
                "type": "array",
                "description": (
                    "List of workflow invocations matching the provided filters. "
                    "Present when invocation_id is not specified."
                )
            }
        }
    }
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
        Dictionary containing workflow invocation information, execution status, and step details
    """
    ensure_connected()

    try:
        # If invocation_id is provided, get details of a specific invocation
        if invocation_id:
            invocation = galaxy_state["gi"].invocations.show_invocation(invocation_id)
            return {"invocation": invocation}

        # Otherwise get a list of invocations with optional filters
        invocations = galaxy_state["gi"].invocations.get_invocations(
            workflow_id=workflow_id,
            history_id=history_id,
            limit=limit,
            view=view,
            step_details=step_details,
        )
        return {"invocations": invocations}
    except Exception as e:
        raise ValueError(f"Failed to get workflow invocations: {str(e)}") from e


@lru_cache(maxsize=1)
def get_manifest_json() -> list[dict[str, Any]]:
    response = requests.get("https://iwc.galaxyproject.org/workflow_manifest.json")
    response.raise_for_status()
    manifest = response.json()
    return manifest


@mcp.tool(
    name="get_iwc_workflows",
    description="Fetch all workflows published in the Interactive Workflow Composer (IWC).",
    tags={"iwc", "workflow", "registry", "manifest"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "workflows": {
                "type": "array",
                "description": "List of workflows defined in the IWC manifest."
            }
        },
        "required": ["workflows"],
    }
)
def get_iwc_workflows() -> dict[str, Any]:
    """
    Fetch all workflows from the IWC (Interactive Workflow Composer)

    Returns:
        Complete workflow manifest from IWC
    """
    try:
        manifest = get_manifest_json()
        # Collect workflows from all manifest entries
        all_workflows = []
        for entry in manifest:
            if "workflows" in entry:
                all_workflows.extend(entry["workflows"])

        return {"workflows": all_workflows}
    except Exception as e:
        raise ValueError(f"Failed to fetch IWC workflows: {str(e)}") from e


@mcp.tool(
    name="search_iwc_workflows",
    description=(
        "Search workflows published in the Interactive Workflow Composer (IWC) "
        "using a free-text query. Matches are performed against workflow names, "
        "descriptions, and tags in a case-insensitive manner."
    ),
    tags={"iwc", "workflow", "search", "discovery", "registry", "query"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "workflows": {
                "type": "array",
                "description": "List of workflows matching the search query.",
                "items": {
                    "type": "object",
                    "properties": {
                        "trsID": {
                            "type": "string",
                            "description": "Tool Registry Service (TRS) identifier for the workflow."
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable workflow name."
                        },
                        "description": {
                            "type": "string",
                            "description": "Workflow description or annotation."
                        },
                        "tags": {
                            "type": "array",
                            "description": "Tags associated with the workflow.",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["trsID", "name", "description", "tags"]
                }
            },
            "count": {
                "type": "integer",
                "description": "Number of workflows matching the query."
            }
        }
    }
)
def search_iwc_workflows(query: str) -> dict[str, Any]:
    """
    Search for workflows in the IWC manifest

    Args:
        query: Search query (matches against name, description, and tags)

    Returns:
        List of matching workflows
    """
    try:
        # Get the full manifest
        manifest = get_iwc_workflows.fn()["workflows"]

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

        return {"workflows": results, "count": len(results)}
    except Exception as e:
        raise ValueError(f"Failed to search IWC workflows: {str(e)}") from e


@mcp.tool(
    name="import_workflow_from_iwc",
    description=(
        "Import a workflow from the Interactive Workflow Composer (IWC) "
        "into the user's Galaxy instance using its TRS identifier. "
        "The workflow definition is retrieved from the IWC manifest and "
        "registered as a new workflow in Galaxy."
    ),
    tags={"iwc", "workflow", "import", "trs", "registry"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "imported_workflow": {
                "type": "object",
                "description": "Metadata of the workflow imported into Galaxy."
            }
        }
    }
)
def import_workflow_from_iwc(trs_id: str) -> dict[str, Any]:
    """
    Import a workflow from IWC to the user's Galaxy instance

    Args:
        trs_id: TRS ID of the workflow in the IWC manifest

    Returns:
        Imported workflow information
    """
    ensure_connected()

    try:
        # Get the workflow manifest
        manifest = get_iwc_workflows.fn()["workflows"]

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
        imported_workflow = galaxy_state["gi"].workflows.import_workflow_dict(workflow_definition)
        return {"imported_workflow": imported_workflow}
    except Exception as e:
        raise ValueError(f"Failed to import workflow from IWC: {str(e)}") from e


@mcp.tool(
    name="list_workflows",
    description=(
        "List workflows available in the connected Galaxy instance. "
        "Can return all accessible workflows or filter by workflow ID or name. "
        "Optionally includes published workflows."
    ),
    tags={"workflow", "list", "metadata", "filtered"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "workflows": {
                "type": "array",
                "description": "List of workflows accessible in the Galaxy instance.",
                "items": {
                    "type": "object",
                    "description": "Workflow metadata including ID, name, and additional attributes."
                }
            }
        }
    }
)
def list_workflows(
    workflow_id: str | None = None, name: str | None = None, published: bool = False
) -> dict[str, Any]:
    """
    List workflows available in the Galaxy instance

    Args:
        workflow_id: Specific workflow ID to get (optional) - a hexadecimal hash string
        name: Filter workflows by name (optional)
        published: Include published workflows (default: False, shows only user workflows)

    Returns:
        Dictionary containing list of workflows with their IDs, names, and metadata
    """
    ensure_connected()

    try:
        workflows = galaxy_state["gi"].workflows.get_workflows(
            workflow_id=workflow_id, name=name, published=published
        )
        return {"workflows": workflows}
    except Exception as e:
        raise ValueError(
            format_error(
                "List workflows",
                e,
                {"workflow_id": workflow_id, "name": name, "published": published},
            )
        ) from e


@mcp.tool(
    name="get_workflow_details",
    description=(
        "Retrieve detailed information for a specific workflow in Galaxy, including steps," 
        "inputs, outputs, and parameters."
        "An optional version may be specified; otherwise the latest version is returned."
    ),
    tags={"workflow", "metadata", "steps", "inputs", "outputs"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "workflow": {
                "type": "object",
                "properties": {
                    "model_class": {"type": "string", "description": "Class type of the workflow object."},
                    "id": {"type": "string", "description": "Unique Galaxy workflow ID."},
                    "name": {"type": "string", "description": "Name of the workflow."},
                    "create_time": {"type": "string", "description": "Timestamp when the workflow was created."},
                    "update_time": {"type": "string", "description": "Timestamp of the last workflow update."},
                    "annotations": {"type": ["string", "null"], "description": "Annotations attached to the workflow."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags associated with the workflow."},
                    "steps": {
                        "type": "object",
                        "description": "Workflow steps with inputs, outputs, tools, and configuration.",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "description": "Step ID in the workflow."},
                                "annotation": {"type": ["string", "null"], "description": "Description of the step."},
                                "type": {"type": "string", "description": "Step type, e.g., 'tool', 'data_collection_input', 'parameter_input'."},
                                "tool_id": {"type": ["string", "null"], "description": "Galaxy tool ID used in this step, if applicable."},
                                "tool_version": {"type": ["string", "null"], "description": "Tool version used in this step."},
                                "tool_inputs": {"type": "object", "description": "Input parameters provided to the tool."},
                                "input_steps": {"type": "object", "description": "Connections from previous workflow steps."}
                            }
                        }
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Workflow inputs.",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Input label for display."},
                                "value": {"type": "string", "description": "Default or assigned value of the input."},
                                "uuid": {"type": "string", "description": "UUID of the input in the workflow definition."}
                            }
                        }
                    }
                }
            }
        },
        "required": ["workflow"]
    }
)
def get_workflow_details(workflow_id: str, version: int | None = None) -> dict[str, Any]:
    """
    Get detailed information about a specific workflow

    Args:
        workflow_id: ID of the workflow to get details for - a hexadecimal hash string
        version: Specific version of the workflow (optional, uses latest if not specified)

    Returns:
        Dictionary containing detailed workflow information including steps, inputs, and parameters
    """
    ensure_connected()

    try:
        workflow = galaxy_state["gi"].workflows.show_workflow(
            workflow_id=workflow_id, version=version
        )
        return {"workflow": workflow}
    except Exception as e:
        raise ValueError(
            format_error(
                "Get workflow details", e, {"workflow_id": workflow_id, "version": version}
            )
        ) from e


@mcp.tool(
    name="invoke_workflow",
    description=(
        "Run a Galaxy workflow with specified structured inputs."
    ),
    tags={"workflow", "execution", "invoke", "run", "jobs"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "invocation": {
                "type": "object",
                "description": "Workflow invocation metadata including invocation ID, state, and associated history."
            }
        }
    }
)
def invoke_workflow(
    workflow_id: str,
    inputs: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    history_id: str | None = None,
    history_name: str | None = None,
    inputs_by: str = "step_index",
    parameters_normalized: bool = False,
) -> dict[str, Any]:
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
        Dictionary containing workflow invocation information including invocation ID
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
        return {"invocation": invocation}
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


@mcp.tool(
    name="cancel_workflow_invocation",
    tags={"workflow", "invocation", "cancel", "execution", "jobs"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    },
    output_schema={
        "type": "object",
        "properties": {
            "cancelled": {
                "type": "boolean",
                "description": "Whether the cancellation request was successful."
            },
            "invocation": {
                "type": "object",
                "description": "Updated workflow invocation metadata after cancellation."
            }
        }
    }
)
def cancel_workflow_invocation(invocation_id: str) -> dict[str, Any]:
    """
    Cancel a running workflow invocation

    Args:
        invocation_id: ID of the workflow invocation to cancel - a hexadecimal hash string

    Returns:
        Dictionary containing cancellation status and updated invocation information
    """
    ensure_connected()

    try:
        result = galaxy_state["gi"].workflows.cancel_invocation(invocation_id)
        return {"cancelled": True, "invocation": result}
    except Exception as e:
        raise ValueError(
            format_error("Cancel workflow invocation", e, {"invocation_id": invocation_id})
        ) from e


if __name__ == "__main__":
    mcp.run()
