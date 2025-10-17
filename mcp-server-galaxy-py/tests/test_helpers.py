"""Test helpers for FastMCP2 functions"""

# Import all the wrapped functions from server
from galaxy_mcp.server import (
    create_history,
    download_dataset,
    ensure_connected,
    filter_tools_by_dataset,
    galaxy_state,
    get_dataset_details,
    get_histories,
    get_history_contents,
    get_history_details,
    get_invocations,
    get_job_details,
    get_server_info,
    get_user,
    import_workflow_from_iwc,
    iwc_workflows,
    run_tool,
    search_dataset_collections,
    search_datasets,
    search_histories,
    search_invocations,
    search_jobs,
    search_libraries,
    search_library_datasets,
    search_tools,
    search_workflows,
    upload_file,
)


# FastMCP2 wraps functions in FunctionTool objects - extract the underlying functions
# for testing purposes
def get_function(tool_or_function):
    """Extract the underlying function from a FastMCP2 FunctionTool if needed"""
    if hasattr(tool_or_function, "fn"):
        return tool_or_function.fn
    return tool_or_function


# Create function aliases for testing
create_history_fn = get_function(create_history)
download_dataset_fn = get_function(download_dataset)
filter_tools_by_dataset_fn = get_function(filter_tools_by_dataset)
get_dataset_details_fn = get_function(get_dataset_details)
get_histories_fn = get_function(get_histories)
get_history_contents_fn = get_function(get_history_contents)
get_history_details_fn = get_function(get_history_details)
get_job_details_fn = get_function(get_job_details)
get_server_info_fn = get_function(get_server_info)
get_user_fn = get_function(get_user)
get_invocations_fn = get_function(get_invocations)
import_workflow_from_iwc_fn = get_function(import_workflow_from_iwc)
iwc_workflows_fn = get_function(iwc_workflows)
run_tool_fn = get_function(run_tool)
search_histories_fn = get_function(search_histories)
search_tools_fn = get_function(search_tools)
search_workflows_fn = get_function(search_workflows)
search_datasets_fn = get_function(search_datasets)
search_dataset_collections_fn = get_function(search_dataset_collections)
search_libraries_fn = get_function(search_libraries)
search_library_datasets_fn = get_function(search_library_datasets)
search_jobs_fn = get_function(search_jobs)
search_invocations_fn = get_function(search_invocations)
upload_file_fn = get_function(upload_file)

# Re-export non-wrapped items
__all__ = [
    "create_history_fn",
    "download_dataset_fn",
    "filter_tools_by_dataset_fn",
    "get_dataset_details_fn",
    "get_histories_fn",
    "get_history_contents_fn",
    "get_history_details_fn",
    "get_job_details_fn",
    "get_server_info_fn",
    "get_user_fn",
    "get_invocations_fn",
    "import_workflow_from_iwc_fn",
    "iwc_workflows_fn",
    "run_tool_fn",
    "search_histories_fn",
    "search_tools_fn",
    "search_workflows_fn",
    "search_datasets_fn",
    "search_dataset_collections_fn",
    "search_libraries_fn",
    "search_library_datasets_fn",
    "search_jobs_fn",
    "search_invocations_fn",
    "upload_file_fn",
    "galaxy_state",
    "ensure_connected",
]
