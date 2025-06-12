# Galaxy MCP Server
import concurrent.futures
import logging
import os
import threading
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


@mcp.tool()
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


@mcp.tool()
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
        raise ValueError(
            f"Failed to search tools with query '{query}': {str(e)}. "
            "Check that the Galaxy instance is accessible and the tool name is correct."
        ) from e


@mcp.tool()
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
            f"Failed to get tool details for ID '{tool_id}': {str(e)}. "
            "Verify the tool ID is correct and the tool is installed on this "
            "Galaxy instance."
        ) from e


@mcp.tool()
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
        raise ValueError(f"Failed to get tool citations: {str(e)}") from e


@mcp.tool()
def run_tool(history_id: str, tool_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Run a tool in Galaxy

    Args:
        history_id: ID of the history where to run the tool
        tool_id: ID of the tool to run
        inputs: Tool input parameters and datasets

    Returns:
        Information about the tool execution
    """
    ensure_connected()

    try:
        # Run the tool with provided inputs
        result = galaxy_state["gi"].tools.run_tool(history_id, tool_id, inputs)
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


@mcp.tool()
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
        raise ValueError(f"Failed to get tool panel: {str(e)}") from e


@mcp.tool()
def create_history(history_name: str) -> dict[str, Any]:
    """
    Create a new history in Galaxy.
    :param history_name: Name for the new history.
    :return: Created history ID.
    """
    ensure_connected()
    return galaxy_state["gi"].histories.create_history(history_name)


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def get_histories() -> list[dict[str, Any]]:
    """
    Get list of user histories

    Returns:
        List of histories, where each history is a dictionary containing
        'id', 'name', and other fields
    """
    ensure_connected()

    try:
        histories = galaxy_state["gi"].histories.get_histories()
        return histories
    except Exception as e:
        raise ValueError(
            f"Failed to get histories: {str(e)}. "
            "Check your connection to Galaxy and that you have "
            "permission to view histories."
        )


@mcp.tool()
def list_history_ids() -> list[dict[str, str]]:
    """
    Get a simplified list of history IDs and names for easy reference

    Returns:
        List of dictionaries containing 'id' and 'name' fields
    """
    ensure_connected()

    try:
        histories = galaxy_state["gi"].histories.get_histories()
        if not histories:
            return []
        # Extract just the id and name for convenience
        simplified = [{"id": h["id"], "name": h.get("name", "Unnamed")} for h in histories]
        return simplified
    except Exception as e:
        raise ValueError(f"Failed to list history IDs: {str(e)}") from e


@mcp.tool()
def get_history_details(history_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific history, including datasets

    Args:
        history_id: ID of the history (the 'id' field from get_histories(),
            not the entire history object)

    Returns:
        History details with datasets
    """
    ensure_connected()

    try:
        # Check if the history_id looks like a dictionary string
        if history_id.startswith("{") and history_id.endswith("}"):
            # Try to parse it and extract the actual ID
            import json

            try:
                history_dict = json.loads(history_id)
                history_id = history_dict.get("id")
                logger.warning(
                    f"Received full history object instead of ID, extracting ID: {history_id}"
                )
            except json.JSONDecodeError as json_error:
                logger.error(f"Invalid history_id format: {history_id}")
                raise ValueError(
                    "Invalid history_id: expected a history ID string, "
                    "got what looks like a malformed dictionary"
                ) from json_error

        logger.info(f"Getting details for history ID: {history_id}")

        # Get history details
        history_info = galaxy_state["gi"].histories.show_history(history_id, contents=False)
        logger.info(f"Successfully retrieved history info: {history_info.get('name', 'Unknown')}")

        # Get history contents (datasets)
        contents = galaxy_state["gi"].histories.show_history(history_id, contents=True)
        logger.info(f"Successfully retrieved {len(contents)} items from history")

        return {"history": history_info, "contents": contents}
    except Exception as e:
        logger.error(f"Failed to get history details for ID '{history_id}': {str(e)}")
        if "404" in str(e) or "No route" in str(e):
            raise ValueError(
                f"History ID '{history_id}' not found. Make sure to pass just "
                "the ID string from get_histories(), not the entire history object."
            ) from e
        raise ValueError(f"Failed to get history details for ID '{history_id}': {str(e)}") from e


@mcp.tool()
def get_job_details(job_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific job

    Args:
        job_id: ID of the job (not the entire job object)

    Returns:
        Job details with tool information
    """
    ensure_connected()

    try:
        # Check if job_id looks like a dictionary string and extract the ID
        if job_id.startswith("{") and job_id.endswith("}"):
            import json

            try:
                job_dict = json.loads(job_id)
                job_id = job_dict.get("id")
                logger.warning(f"Received full job object instead of ID, extracting ID: {job_id}")
            except json.JSONDecodeError as json_error:
                raise ValueError(
                    "Invalid job_id: expected a job ID string, "
                    "got what looks like a malformed dictionary"
                ) from json_error

        # Get job details using the Galaxy API directly
        # (Bioblend doesn't have a direct method for this)
        url = f"{galaxy_state['url']}api/jobs/{job_id}"
        headers = {"x-api-key": galaxy_state["api_key"]}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        job_info = response.json()

        return {"job": job_info}
    except Exception as e:
        if "404" in str(e):
            raise ValueError(
                f"Job ID '{job_id}' not found. Make sure to pass just "
                "the ID string, not the entire job object."
            ) from e
        raise ValueError(f"Failed to get job details: {str(e)}") from e


@mcp.tool()
def upload_file(path: str, history_id: str | None = None) -> dict[str, Any]:
    """
    Upload a local file to Galaxy

    Args:
        path: Path to local file
        history_id: Target history ID (optional)

    Returns:
        Upload status
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


@mcp.tool()
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
        invocation_id: Specific invocation ID to view details (optional)
        workflow_id: Filter invocations by workflow ID (optional)
        history_id: Filter invocations by history ID (optional)
        limit: Maximum number of invocations to return (optional)
        view: Level of detail to return, either 'element' or 'collection'
            (default: collection)
        step_details: Include details on individual steps (only if view is 'element')

    Returns:
        Workflow invocation(s) information
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


@mcp.tool()
def get_iwc_workflows() -> dict[str, Any]:
    """
    Fetch all workflows from the IWC (Interactive Workflow Composer)

    Returns:
        Complete workflow manifest from IWC
    """
    try:
        response = requests.get("https://iwc.galaxyproject.org/workflow_manifest.json")
        response.raise_for_status()
        workflows = response.json()[0]["workflows"]
        return {"workflows": workflows}
    except Exception as e:
        raise ValueError(f"Failed to fetch IWC workflows: {str(e)}") from e


@mcp.tool()
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
            # Check if query matches name, description or tags
            name = workflow.get("definition", {}).get("name", "").lower()
            description = workflow.get("definition", {}).get("annotation", "").lower()
            tags = [tag.lower() for tag in workflow.get("definition", {}).get("tags", [])]

            if (
                query in name
                or query in description
                or (tags and any(query in tag for tag in tags))
            ):
                results.append(workflow)

        return {"workflows": results, "count": len(results)}
    except Exception as e:
        raise ValueError(f"Failed to search IWC workflows: {str(e)}") from e


@mcp.tool()
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


if __name__ == "__main__":
    mcp.run()
