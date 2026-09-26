# Surface parity

Generated -- do not edit by hand. Regenerate with `pnpm parity:report` from `galaxy-agent-tools/`, which is also what CI checks this file against.

Two surfaces expose the same Galaxy operations, and this is every way they disagree. The Python column is the checked-in surface manifest (`mcp-server-galaxy-py/tests/testdata/mcp-surface.json`); the TypeScript column is what a client is really advertised by `@galaxyproject/galaxy-mcp`. Compared: which tools exist, what parameters they take, their types, requiredness and declared defaults, whether a tool says it changes anything, and what it says it needs from the server. Not compared: result shapes, wording, value constraints, what is inside an object, and everything else -- so a difference can be real and have no row here.

Every difference carries the status and the reason recorded in `galaxy-agent-tools/packages/galaxy-mcp/test/fixtures/accepted-divergences.json`, which is also where the statuses themselves are explained. A status says how well a difference is understood, not that it is acceptable.

## Differences by status

| Status | Differences |
| --- | --- |
| `intentional` | `1` |
| `pending-port` | `0` |
| `pending-decision` | `0` |
| `unreviewed-gap` | `7` |
| **total** | `8` |

`unreviewed-gap` is the status nobody has ruled on yet. The check holds the registry to the 7 it declares, so the count cannot drift from the number; raising that number is an edit somebody has to make in the diff, and it is meant to come down, never up.

## Tools

A row per tool, then a row per parameter the surfaces disagree about. `--` means that surface does not have it. A tool's own row says what it advertises about changing things and about the Galaxy it needs; a parameter's row says what each surface declares it to be, in the terms the comparison compares.

| Tool | Parameter | Python | TypeScript | Difference | Status | Why |
| --- | --- | --- | --- | --- | --- | --- |
| `cancel_workflow_invocation` |  | `write (tag)` | `write (hint)` |  |  |  |
| `connect` |  | `write (tag)` | -- | `missing-ts-tool` | `intentional` | TS takes the Galaxy URL and key when the server is built, so there is no per-session connect call to expose. |
| `create_history` |  | `write (tag)` | `write (hint)` |  |  |  |
| `create_page` |  | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` |  |  |  |
| `create_user_tool` |  | `write (tag)` | `write (hint)` |  |  |  |
| `delete_user_tool` |  | `write (tag)` | `write (hint)` |  |  |  |
| `download_dataset` |  | `read (tag)` | `write (hint)` | `mutability-mismatch` | `unreviewed-gap` | Python tags the tool read; the TS op advertises readOnlyHint false because it can write the bytes to a local path. One of the two is wrong about what read-only means for a tool that touches the caller's disk, and MCP clients gate approval on that hint. |
| `download_dataset` | `require_ok_state` | `type=boolean required=false default=true` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `download_dataset` | `use_default_filename` | `type=boolean required=false default=true` | -- | `missing-ts-param` | `unreviewed-gap` | Python can write next to the dataset's own name; TS only writes to the exact filePath it is given. |
| `get_collection_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_dataset_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_histories` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_history_contents` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_history_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_invocations` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_iwc_workflow_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_iwc_workflows` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_job_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_page` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_page_revision` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `get_server_info` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_citations` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_input_template` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_panel` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_run_examples` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_user` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_input_template` |  | `read (tag)` | `read (hint)` |  |  |  |
| `import_workflow_from_iwc` |  | `write (tag)` | `write (hint)` |  |  |  |
| `invoke_workflow` |  | `write (tag)` | `write (hint)` |  |  |  |
| `list_history_ids` |  | `read (tag)` | `read (hint)` |  |  |  |
| `list_page_revisions` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `list_pages` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `list_user_tools` |  | `read (tag)` | `read (hint)` |  |  |  |
| `list_workflows` |  | `read (tag)` | `read (hint)` |  |  |  |
| `recommend_biocontainer` |  | `read (tag)` | -- | `missing-ts-tool` | `unreviewed-gap` | Needs galaxy.tool_util's mulled recommender, which has no TS equivalent, so a port means reimplementing mulled name resolution rather than translating an op. |
| `recommend_iwc_workflows` |  | `read (tag)` | `read (hint)` |  |  |  |
| `revert_page_revision` |  | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` |  |  |  |
| `run_tool` |  | `write (tag)` | `write (hint)` |  |  |  |
| `run_tool` | `tool_version` | -- | `type=string required=false default=none` | `missing-py-param` | `unreviewed-gap` | TS accepts toolVersion; Python's run_tool takes only history_id, tool_id and inputs, so pinning a tool version is not expressible there. |
| `run_user_tool` |  | `write (tag)` | `write (hint)` |  |  |  |
| `search_iwc_workflows` |  | `read (tag)` | `read (hint)` |  |  |  |
| `search_tools_by_keywords` |  | `read (tag)` | `read (hint)` |  |  |  |
| `search_tools_by_name` |  | `read (tag)` | `read (hint)` |  |  |  |
| `update_history` |  | `write (tag)` | `write (hint)` |  |  |  |
| `update_page` |  | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` |  |  |  |
| `upload_file` |  | `write (tag)` | `write (hint)` |  |  |  |
| `upload_file_from_url` |  | `write (tag)` | `write (hint)` |  |  |  |
| `upload_file_from_url` | `dbkey` | `type=string required=false default="?"` | `type=string required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `upload_file_from_url` | `file_type` | `type=string required=false default="auto"` | `type=string required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
