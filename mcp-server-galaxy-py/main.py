# Galaxy MCP Server
import os
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from bioblend.galaxy import GalaxyInstance
import requests
from dotenv import load_dotenv, find_dotenv
import concurrent.futures
import threading
from itertools import islice


# Try to load environment variables from .env file
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")

# Create an MCP server
mcp = FastMCP("Galaxy", dependencies=["bioblend", "requests"])

# Galaxy client state
galaxy_state = {
    "url": os.environ.get("GALAXY_URL"),
    "api_key": os.environ.get("GALAXY_API_KEY"),
    "gi": None,
    "connected": False,
}


# Initialize Galaxy client if environment variables are set
if galaxy_state["url"] and galaxy_state["api_key"]:
    galaxy_url = (
        galaxy_state["url"]
        if galaxy_state["url"].endswith("/")
        else f"{galaxy_state['url']}/"
    )
    galaxy_state["url"] = galaxy_url
    galaxy_state["gi"] = GalaxyInstance(url=galaxy_url, key=galaxy_state["api_key"])
    galaxy_state["connected"] = True
    print(f"Galaxy client initialized from environment variables (URL: {galaxy_url})")


def ensure_connected():
    """Helper function to ensure Galaxy connection is established"""
    if not galaxy_state["connected"] or not galaxy_state["gi"]:
        raise ValueError("Not connected to Galaxy. Use connect command first.")


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

        raise ValueError(f"Failed to connect to Galaxy: {str(e)}")


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
        raise ValueError(f"Failed to search tools: {str(e)}")


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
        raise ValueError(f"Failed to get tool details: {str(e)}")


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
        raise ValueError(f"Failed to run tool: {str(e)}")


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
        raise ValueError(f"Failed to get tool panel: {str(e)}")

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
            if any(kw in name for kw in dataset_keywords) or any(kw in description for kw in dataset_keywords):
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
                    elif isinstance(fmt, str):
                        if fmt and any(kw in fmt.lower() for kw in dataset_keywords):
                            return tool
                return None
            except Exception as e:    
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
            slim_tools.append({
                "id": tool.get("id", ""),
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "versions": tool.get("versions", [])
            })
        return {"recommended_tools": slim_tools, "count": len(slim_tools)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_datesets_by_history(history_id: str) -> dict[str, Any]:
    """
    Get datasets from a specific history

    Args:
        history_id: ID of the history to fetch datasets from

    Returns:
        List of datasets in the specified history
    """
    ensure_connected()

    try:
        datasets = galaxy_state["gi"].histories.show_history(history_id, contents=True)
        return datasets
    except Exception as e:
        raise ValueError(f"Failed to get datasets by history: {str(e)}")


@mcp.tool()
def read_dataset_head(
    dataset_path: Optional[str] = None,
    dataset_id: Optional[str] = None,
    nrows: int = 5,
) -> list[str]:
    """
    Read the header (first few lines) of a dataset in Galaxy.

    Either 'dataset_path' or 'dataset_id' must be provided.

    Args:
        dataset_path (Optional[str]): Local path to the dataset file.
        dataset_id (Optional[str]): The ID of the dataset to read from Galaxy.
        nrows (int): Number of lines to read from the top of the dataset.

    Returns:
        List[str]: List of the first nrows lines from the dataset.

    Raises:
        ValueError: If neither dataset_path nor dataset_id is provided, or if an error 
                    occurs during dataset retrieval or file reading.
    """
    if nrows < 1:
        raise ValueError("nrows must be greater than 0.")

    if not dataset_path and not dataset_id:
        raise ValueError("Either dataset_path or dataset_id must be provided.")

    if dataset_path:
        try:
            with open(dataset_path, "r") as file:
                return list(islice(file, nrows))
        except Exception as e:
            raise ValueError(f"Error reading local dataset at '{dataset_path}': {e}")

    if dataset_id:
        ensure_connected()
        try:
            dataset = galaxy_state["gi"].datasets.show_dataset(dataset_id=dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID '{dataset_id}' not found in Galaxy.")
            
            # Download the dataset content from Galaxy.
            file_content = galaxy_state["gi"].datasets.download_dataset(dataset_id)
            return file_content.splitlines()[:nrows]
        except Exception as e:
            raise ValueError(f"Failed to read dataset from Galaxy for ID '{dataset_id}': {e}")


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
        raise ValueError(f"Failed to get user: {str(e)}")


@mcp.tool()
def get_histories() -> dict[str, Any]:
    """
    Get list of user histories

    Returns:
        List of histories
    """
    ensure_connected()

    try:
        histories = galaxy_state["gi"].histories.get_histories()
        return histories
    except Exception as e:
        raise ValueError(f"Failed to get histories: {str(e)}")


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
            raise ValueError(f"File not found: {path}")

        result = galaxy_state["gi"].tools.upload_file(path, history_id=history_id)
        return result
    except Exception as e:
        raise ValueError(f"Failed to upload file: {str(e)}")


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
        view: Level of detail to return, either 'element' or 'collection' (default: collection)
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
        raise ValueError(f"Failed to get workflow invocations: {str(e)}")


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
        raise ValueError(f"Failed to fetch IWC workflows: {str(e)}")


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
        manifest = get_iwc_workflows()["workflows"]

        # Filter workflows based on the search query
        results = []
        query = query.lower()

        for workflow in manifest:
            # Check if query matches name, description or tags
            name = workflow.get("definition", {}).get("name", "").lower()
            description = workflow.get("definition", {}).get("annotation", "").lower()
            tags = [
                tag.lower() for tag in workflow.get("definition", {}).get("tags", [])
            ]

            if (
                query in name
                or query in description
                or (tags and any(query in tag for tag in tags))
            ):
                results.append(workflow)

        return {"workflows": results, "count": len(results)}
    except Exception as e:
        raise ValueError(f"Failed to search IWC workflows: {str(e)}")


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
        manifest = get_iwc_workflows()["workflows"]

        # Find the specified workflow
        workflow = None
        for wf in manifest:
            if wf.get("trsID") == trs_id:
                workflow = wf
                break

        if not workflow:
            raise ValueError(f"Workflow with trsID {trs_id} not found in IWC manifest")

        # Extract the workflow definition
        workflow_definition = workflow.get("definition")
        if not workflow_definition:
            raise ValueError(f"No definition found for workflow with trsID {trs_id}")

        # Import the workflow into Galaxy
        imported_workflow = galaxy_state["gi"].workflows.import_workflow_dict(
            workflow_definition
        )
        return {"imported_workflow": imported_workflow}
    except Exception as e:
        raise ValueError(f"Failed to import workflow from IWC: {str(e)}")


if __name__ == "__main__":
    mcp.run()
