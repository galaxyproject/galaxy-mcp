"""Test helpers for FastMCP2 functions"""

# Import all the wrapped functions from server
from galaxy_mcp.server import (
    cancel_workflow_invocation,
    connect,
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
    get_iwc_workflows,
    get_job_details,
    get_server_info,
    get_tool_citations,
    get_tool_details,
    get_tool_panel,
    get_user,
    get_workflow_details,
    import_workflow_from_iwc,
    invoke_workflow,
    list_history_ids,
    list_workflows,
    run_tool,
    search_iwc_workflows,
    search_tools,
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
cancel_workflow_invocation_fn = get_function(cancel_workflow_invocation)
connect_fn = get_function(connect)
create_history_fn = get_function(create_history)
download_dataset_fn = get_function(download_dataset)
filter_tools_by_dataset_fn = get_function(filter_tools_by_dataset)
get_dataset_details_fn = get_function(get_dataset_details)
get_histories_fn = get_function(get_histories)
get_history_contents_fn = get_function(get_history_contents)
get_history_details_fn = get_function(get_history_details)
get_invocations_fn = get_function(get_invocations)
get_iwc_workflows_fn = get_function(get_iwc_workflows)
get_job_details_fn = get_function(get_job_details)
get_server_info_fn = get_function(get_server_info)
get_tool_citations_fn = get_function(get_tool_citations)
get_tool_details_fn = get_function(get_tool_details)
get_tool_panel_fn = get_function(get_tool_panel)
get_user_fn = get_function(get_user)
get_workflow_details_fn = get_function(get_workflow_details)
import_workflow_from_iwc_fn = get_function(import_workflow_from_iwc)
invoke_workflow_fn = get_function(invoke_workflow)
list_history_ids_fn = get_function(list_history_ids)
list_workflows_fn = get_function(list_workflows)
run_tool_fn = get_function(run_tool)
search_iwc_workflows_fn = get_function(search_iwc_workflows)
search_tools_fn = get_function(search_tools)
upload_file_fn = get_function(upload_file)

# Re-export non-wrapped items
__all__ = [
    "cancel_workflow_invocation_fn",
    "connect_fn",
    "create_history_fn",
    "download_dataset_fn",
    "filter_tools_by_dataset_fn",
    "get_dataset_details_fn",
    "get_histories_fn",
    "get_history_contents_fn",
    "get_history_details_fn",
    "get_invocations_fn",
    "get_iwc_workflows_fn",
    "get_job_details_fn",
    "get_server_info_fn",
    "get_tool_citations_fn",
    "get_tool_details_fn",
    "get_tool_panel_fn",
    "get_user_fn",
    "get_workflow_details_fn",
    "import_workflow_from_iwc_fn",
    "invoke_workflow_fn",
    "list_history_ids_fn",
    "list_workflows_fn",
    "run_tool_fn",
    "search_iwc_workflows_fn",
    "search_tools_fn",
    "upload_file_fn",
    "galaxy_state",
    "ensure_connected",
]
