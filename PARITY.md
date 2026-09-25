# Surface parity

Generated -- do not edit by hand. Regenerate with `pnpm parity:report` from `galaxy-agent-tools/`, which is also what CI checks this file against.

Two surfaces expose the same Galaxy operations, and this is every way they disagree. The Python column is the checked-in surface manifest (`mcp-server-galaxy-py/tests/testdata/mcp-surface.json`); the TypeScript column is what a client is really advertised by `@galaxyproject/galaxy-mcp`. Compared: which tools exist, what parameters they take, their types, requiredness and declared defaults, whether a tool says it changes anything, and what it says it needs from the server. Not compared: result shapes, wording, value constraints, what is inside an object, and everything else -- so a difference can be real and have no row here.

Every difference carries the status and the reason recorded in `galaxy-agent-tools/packages/galaxy-mcp/test/fixtures/accepted-divergences.json`, which is also where the statuses themselves are explained. A status says how well a difference is understood, not that it is acceptable.

## Differences by status

| Status | Differences |
| --- | --- |
| `intentional` | `1` |
| `pending-port` | `14` |
| `pending-decision` | `0` |
| `unreviewed-gap` | `17` |
| **total** | `32` |

`unreviewed-gap` is the status nobody has ruled on yet. The check holds the registry to the 17 it declares, so the count cannot drift from the number; raising that number is an edit somebody has to make in the diff, and it is meant to come down, never up.

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
| `get_collection_details` | `max_elements` | `type=integer required=false default=100` | `type=integer required=false default=none` | `default-mismatch` | `unreviewed-gap` | Python always truncates the element list to 100; TS truncates only when asked, so the same call can return a much larger payload. |
| `get_dataset_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_dataset_details` | `include_preview` | `type=boolean required=false default=true` | -- | `missing-ts-param` | `pending-port` | The TS op's own summary defers content preview to a later phase. |
| `get_dataset_details` | `preview_lines` | `type=integer required=false default=10` | -- | `missing-ts-param` | `pending-port` | The TS op's own summary defers content preview to a later phase. |
| `get_histories` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_history_contents` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_history_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_invocations` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_invocations` | `history_id` | `type=string required=false default=none` | -- | `missing-ts-param` | `pending-port` | The TS op is the detail-by-id read only; its own comment defers listing and filtering to a later version. |
| `get_invocations` | `invocation_id` | `type=string required=false default=none` | `type=string required=true default=none` | `required-mismatch` | `pending-port` | The TS op is the detail-by-id read only; its own comment defers listing and filtering to a later version. |
| `get_invocations` | `limit` | `type=integer required=false default=none` | -- | `missing-ts-param` | `pending-port` | The TS op is the detail-by-id read only; its own comment defers listing and filtering to a later version. |
| `get_invocations` | `step_details` | `type=boolean required=false default=false` | -- | `missing-ts-param` | `pending-port` | The TS op is the detail-by-id read only; its own comment defers listing and filtering to a later version. |
| `get_invocations` | `view` | `type=string required=false default="collection"` | -- | `missing-ts-param` | `pending-port` | The TS op is the detail-by-id read only; its own comment defers listing and filtering to a later version. |
| `get_invocations` | `workflow_id` | `type=string required=false default=none` | -- | `missing-ts-param` | `pending-port` | The TS op is the detail-by-id read only; its own comment defers listing and filtering to a later version. |
| `get_iwc_workflow_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_iwc_workflows` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_job_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_page` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_page` | `include_rendered` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `pending-port` | TS applies the same default inside run() but leaves it off the advertised schema, so an agent reading the tool cannot see it. The TS op passes `i.includeRendered ?? false` to stripRendered; declaring `.default(false)` on the input closes it. |
| `get_page_revision` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `get_server_info` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_citations` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_details` | `io_details` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `get_tool_input_template` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_panel` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_tool_run_examples` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_user` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_details` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_input_template` |  | `read (tag)` | `read (hint)` |  |  |  |
| `get_workflow_input_template` | `verbose` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `import_workflow_from_iwc` |  | `write (tag)` | `write (hint)` |  |  |  |
| `invoke_workflow` |  | `write (tag)` | `write (hint)` |  |  |  |
| `invoke_workflow` | `inputs` | `type=anyOf<object\|string> required=false default=none` | `type=object required=false default=none` | `type-mismatch` | `unreviewed-gap` | Python also accepts a JSON string, which is what agents often send; TS accepts an object only and rejects the string form. |
| `invoke_workflow` | `inputs_by` | `type=string required=false default="step_index"` | `type=string required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `invoke_workflow` | `parameters_normalized` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `invoke_workflow` | `params` | `type=anyOf<object\|string> required=false default=none` | `type=object required=false default=none` | `type-mismatch` | `unreviewed-gap` | Python also accepts a JSON string, which is what agents often send; TS accepts an object only and rejects the string form. |
| `list_history_ids` |  | `read (tag)` | `read (hint)` |  |  |  |
| `list_page_revisions` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `list_page_revisions` | `sort_desc` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `pending-port` | TS applies the same default inside run() but leaves it off the advertised schema, so an agent reading the tool cannot see it. The TS op sends `i.sortDesc ?? false` as the sort_desc query parameter; declaring `.default(false)` on the input closes it. |
| `list_pages` |  | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` |  |  |  |
| `list_pages` | `limit` | `type=integer required=false default=100` | `type=integer required=false default=none` | `default-mismatch` | `pending-port` | TS applies the same default inside run() but leaves it off the advertised schema, so an agent reading the tool cannot see it. Python advertises the 100-row cap; the TS op applies the same cap through its DEFAULT_LIMIT constant, so an agent deciding whether to pass limit has to guess what omitting it does. `.default(100)` closes this entry. It does not tell a caller whether more rows exist: the TS result carries no total_matches at all, which is a separate gap this comparator cannot see. |
| `list_pages` | `offset` | `type=integer required=false default=0` | `type=integer required=false default=none` | `default-mismatch` | `pending-port` | TS applies the same default inside run() but leaves it off the advertised schema, so an agent reading the tool cannot see it. The TS op sends `i.offset ?? 0`; declaring `.default(0)` on the input closes it. |
| `list_pages` | `show_published` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `pending-port` | TS applies the same default inside run() but leaves it off the advertised schema, so an agent reading the tool cannot see it. Both sides deliberately override Galaxy's own index default, which is on; the TS op sends `i.showPublished ?? false` but advertises nothing, so an agent cannot see which way the flag falls. `.default(false)` closes it. |
| `list_pages` | `show_shared` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `pending-port` | TS applies the same default inside run() but leaves it off the advertised schema, so an agent reading the tool cannot see it. The TS op sends `i.showShared ?? false`; declaring `.default(false)` on the input closes it. |
| `list_user_tools` |  | `read (tag)` | `read (hint)` |  |  |  |
| `list_user_tools` | `active` | `type=boolean required=false default=true` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS applies the same default in run() but does not declare it in the advertised schema, so an agent reading the tool cannot see it. |
| `list_workflows` |  | `read (tag)` | `read (hint)` |  |  |  |
| `list_workflows` | `published` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `default-mismatch` | `unreviewed-gap` | TS sends Galaxy's show_published query parameter only when `published` is given, and Galaxy's own default for it is false, the value Python always sends; only the declaration differs. |
| `list_workflows` | `workflow_id` | `type=string required=false default=none` | -- | `missing-ts-param` | `unreviewed-gap` | Python's list_workflows doubles as a fetch-by-id; TS splits that into get_workflow_details. Probably right, but nobody has confirmed it. |
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
