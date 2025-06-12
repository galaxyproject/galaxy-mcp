"""Test helpers for FastMCP2 functions"""

# Import all the wrapped functions from server
from galaxy_mcp.server import (
    connect,
    create_history,
    ensure_connected,
    filter_tools_by_dataset,
    galaxy_state,
    get_histories,
    get_history_details,
    get_invocations,
    get_iwc_workflows,
    get_job_details,
    get_server_info,
    get_tool_citations,
    get_tool_details,
    get_tool_panel,
    get_user,
    import_workflow_from_iwc,
    list_history_ids,
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
connect_fn = get_function(connect)
create_history_fn = get_function(create_history)
filter_tools_by_dataset_fn = get_function(filter_tools_by_dataset)
get_histories_fn = get_function(get_histories)
get_history_details_fn = get_function(get_history_details)
get_iwc_workflows_fn = get_function(get_iwc_workflows)
get_job_details_fn = get_function(get_job_details)
get_server_info_fn = get_function(get_server_info)
get_tool_citations_fn = get_function(get_tool_citations)
get_tool_details_fn = get_function(get_tool_details)
get_tool_panel_fn = get_function(get_tool_panel)
get_user_fn = get_function(get_user)
get_invocations_fn = get_function(get_invocations)
import_workflow_from_iwc_fn = get_function(import_workflow_from_iwc)
list_history_ids_fn = get_function(list_history_ids)
run_tool_fn = get_function(run_tool)
search_iwc_workflows_fn = get_function(search_iwc_workflows)
search_tools_fn = get_function(search_tools)
upload_file_fn = get_function(upload_file)

# Re-export non-wrapped items
__all__ = [
    "connect_fn",
    "create_history_fn",
    "filter_tools_by_dataset_fn",
    "get_histories_fn",
    "get_history_details_fn",
    "get_iwc_workflows_fn",
    "get_job_details_fn",
    "get_server_info_fn",
    "get_tool_citations_fn",
    "get_tool_details_fn",
    "get_tool_panel_fn",
    "get_user_fn",
    "get_invocations_fn",
    "import_workflow_from_iwc_fn",
    "list_history_ids_fn",
    "run_tool_fn",
    "search_iwc_workflows_fn",
    "search_tools_fn",
    "upload_file_fn",
    "galaxy_state",
    "ensure_connected",
]
