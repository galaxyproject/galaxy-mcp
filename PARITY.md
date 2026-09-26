# Surface parity

Generated -- do not edit by hand. Regenerate with `pnpm parity:report` from `galaxy-agent-tools/`, which is also what CI checks this file against.

Two surfaces expose the same Galaxy operations, and this is every way they disagree. The Python column is the checked-in surface manifest (`mcp-server-galaxy-py/tests/testdata/mcp-surface.json`); the TypeScript column is what a client is really advertised by `@galaxyproject/galaxy-mcp`. Compared: which tools exist, what parameters they take, their types, requiredness and declared defaults, whether a tool says it changes anything, what it says it needs from the server, and -- where both surfaces state one -- whether the result is a sequence or an object, whether the page window sits in the envelope or in the data, and which top-level keys it carries. Not compared: wording, value constraints, what is inside a key, and everything else -- so a difference can be real and have no row here.

Every difference carries the status and the reason recorded in `galaxy-agent-tools/packages/galaxy-mcp/test/fixtures/accepted-divergences.json`, which is also where the statuses themselves are explained. A status says how well a difference is understood, not that it is acceptable.

## Differences by status

| Status | Differences |
| --- | --- |
| `intentional` | `5` |
| `pending-port` | `1` |
| `pending-decision` | `18` |
| `unreviewed-gap` | `7` |
| **total** | `31` |

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
| `download_dataset` | `file_path` | -- | -- | `result-shape` | `intentional` | Set only on the branch that was given a path to write to; the in-memory branch has no path to name, so the field is not promised. |
| `download_dataset` | `file_size` | -- | -- | `result-shape` | `intentional` | Set only on the in-memory branch, which is the one that measured the bytes. Mutually exclusive with file_path, so neither is promised. |
| `download_dataset` | `note` | -- | -- | `result-shape` | `pending-port` | Python's result carries a next-steps note and the TS op builds none. |
| `download_dataset` | `require_ok_state` | `type=boolean required=false default=true` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `download_dataset` | `use_default_filename` | `type=boolean required=false default=true` | -- | `missing-ts-param` | `unreviewed-gap` | Python can write next to the dataset's own name; TS only writes to the exact filePath it is given. |
| `get_collection_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_dataset_details` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Python nests the dataset under a `dataset` key beside `dataset_id`; the TS op returns Galaxy's dataset at the top level with `preview` added. Python's key list is not stated because `preview` is added conditionally after the literal. |
| `get_histories` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | The paged-listing convention above. Python's data is bioblend's return value, which the manifest generator cannot read, so only the TS side states a kind; the difference is the wrapper, not the rows. |
| `get_history_contents` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_history_contents` | `contents` | -- | -- | `result-shape` | `pending-decision` | The paged-listing convention above, where both surfaces wrap and name the rows differently: Python calls them `contents`, the TS op `items`. |
| `get_history_contents` | `history_id` | -- | -- | `result-shape` | `pending-decision` | Python repeats the history id beside the rows it was asked for; the TS op does not, because the caller passed it. |
| `get_history_contents` | `items` | -- | -- | `result-shape` | `pending-decision` | The TS name for Python's `contents`; see that entry. |
| `get_history_contents` | `pagination` | -- | -- | `result-shape` | `pending-decision` | The window inside the data, which the paged-listing convention above decides. Both surfaces also report it in the envelope. |
| `get_history_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_invocations` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_iwc_workflow_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_iwc_workflows` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `get_job_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_page` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_page_revision` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `get_server_info` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_server_info` | `unsupported_ops` | -- | -- | `result-shape` | `pending-decision` | The TS spelling of Python's unsupported_tools; see that entry. |
| `get_server_info` | `unsupported_tools` | -- | -- | `result-shape` | `pending-decision` | The same list under two names: Python calls it unsupported_tools, the TS op unsupported_ops. One name has to win and neither surface is obviously the reference for it. |
| `get_server_info` | `version_source` | -- | -- | `result-shape` | `pending-decision` | Where the enforced version came from, which the TS op reports and Python does not. Useful or noise is a decision; it is not a port either way. |
| `get_tool_citations` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_input_template` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_panel` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | The panel-wide tool_count and section_count the TS op adds, because how many tools a server has installed has no other answer in the surface. Python builds two different key sets by branch, so its field list is not stated and the two cannot be lined up field by field. |
| `get_tool_run_examples` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_user` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_input_template` |  | `read (tag)` | `read (hint)` |  |  |  |
| `import_workflow_from_iwc` |  | `write (tag)` | `write (hint)` |  |  |  |
| `invoke_workflow` |  | `write (tag)` | `write (hint)` |  |  |  |
| `list_history_ids` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `list_page_revisions` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` | `result-shape` | `intentional` | No known difference: both return the rows, and Python's data is the route's own body, which the generator cannot read. |
| `list_pages` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` | `result-shape` | `intentional` | No known difference in the rows: both return them and both report a window. Python's data is the route's own body, which the generator cannot read, so only the TS side can state that it is a list. |
| `list_user_tools` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `list_workflows` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `recommend_biocontainer` |  | `read (tag)` | -- | `missing-ts-tool` | `unreviewed-gap` | Needs galaxy.tool_util's mulled recommender, which has no TS equivalent, so a port means reimplementing mulled name resolution rather than translating an op. |
| `recommend_iwc_workflows` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | The paged-listing convention above, on a tool Python does not page at all: it returns the rows, the TS op wraps them and reports a window. |
| `revert_page_revision` |  | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` |  |  |  |
| `run_tool` |  | `write (tag)` | `write (hint)` |  |  |  |
| `run_tool` | `tool_version` | -- | `type=string required=false default=none` | `missing-py-param` | `unreviewed-gap` | TS accepts toolVersion; Python's run_tool takes only history_id, tool_id and inputs, so pinning a tool version is not expressible there. |
| `run_user_tool` |  | `write (tag)` | `write (hint)` |  |  |  |
| `search_iwc_workflows` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `search_tools_by_keywords` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `search_tools_by_name` |  | `read (tag)` | `read (hint)` | `result-shape` | `pending-decision` | Both surfaces page this listing and disagree about what `data` is: Python puts the rows there and the window beside them, the TS op puts `{items, pagination}` there and the window beside that as well. An agent reading `data` gets rows on one surface and a wrapper on the other. Which convention is right is one decision for every paged op, recorded here so it is made from the whole list rather than one tool at a time. |
| `update_history` |  | `write (tag)` | `write (hint)` |  |  |  |
| `update_page` |  | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` |  |  |  |
| `upload_file` |  | `write (tag)` | `write (hint)` |  |  |  |
| `upload_file_from_url` |  | `write (tag)` | `write (hint)` |  |  |  |
| `upload_file_from_url` | `dbkey` | `type=string required=false default="?"` | `type=string required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `upload_file_from_url` | `file_type` | `type=string required=false default="auto"` | `type=string required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
