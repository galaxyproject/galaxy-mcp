# Surface parity

Generated -- do not edit by hand. Regenerate with `pnpm parity:report` from `typescript/`, which is also what CI checks this file against.

Three surfaces expose the same Galaxy operations, and this is every way they disagree. The Python column is the checked-in surface manifest (`contract/mcp-surface.json`); the TypeScript column is what a client is really advertised by `@galaxyproject/galaxy-mcp`; the Built-in column is the MCP server Galaxy itself serves (`lib/galaxy/webapps/galaxy/api/mcp.py`, behind `enable_mcp_server`), read from a snapshot of that module. Compared: which tools exist, what parameters they take, their types, requiredness and declared defaults, whether a tool takes parameters it does not declare, whether it says it changes anything, and what it says it needs from the server. Not compared: result shapes, wording, value constraints, what is inside an object, and everything else -- so a difference can be real and have no row here.

Every difference carries the status and the reason recorded in `contract/accepted-divergences.json`, which is also where the statuses themselves are explained. A status says how well a difference is understood, not that it is acceptable.

The third column describes Galaxy `v26.1.1-1293-g8300420556b` (`8300420556b` on `debug-request-endpoint`, committed 2026-08-31, captured 2026-09-26). That column is a snapshot taken by hand, because reading it needs a Galaxy checkout and CI has none: it is as current as the commit it names and no more. Refresh it with `builtin_surface.py`, which says how.

Two differences with that server are recorded once here rather than as a row per tool, because they are true of all of them: every built-in tool takes the API key as an argument, where both surfaces here take the credential once per session; and every built-in tool is registered without tags or a read-only hint, so a client is told nothing about which of them change state and MCP's own default is to assume they all might. That is why the Built-in column reads `write (mcp default)` throughout.

## Differences by status

### Python and TypeScript

| Status | Differences |
| --- | --- |
| `intentional` | `2` |
| `pending-port` | `0` |
| `pending-decision` | `0` |
| `unreviewed-gap` | `0` |
| **total** | `2` |

`unreviewed-gap` is the status nobody has ruled on yet. The check holds the registry to the 0 it declares, so the count cannot drift from the number; raising that number is an edit somebody has to make in the diff, and it is meant to come down, never up.

### Python and Galaxy's built-in server

| Status | Differences |
| --- | --- |
| `intentional` | `1` |
| `pending-port` | `0` |
| `pending-decision` | `0` |
| `unreviewed-gap` | `54` |
| **total** | `55` |

These are counted apart and ratcheted apart -- at 54 -- because they are not this repository's to close on its own: each one is a rename, an addition or a removal somebody has to agree with galaxyproject/galaxy. Nothing here has been ruled on yet.

## What each surface is missing

Every tool any surface has, against the surfaces that do not have it. A name here is either a tool somebody has not written yet or the same tool under another name; the rows further down say which, one tool at a time.

**Missing from Python** (4): `get_invocation_details`, `get_job_status`, `list_file_source_templates`, `list_user_file_sources`

**Missing from TypeScript** (5): `connect`, `get_invocation_details`, `get_job_status`, `list_file_source_templates`, `list_user_file_sources`

**Missing from Built-in** (7): `get_iwc_workflows`, `get_tool_input_template`, `get_workflow_input_template`, `list_jobs`, `recommend_iwc_workflows`, `update_history`, `upload_file`

## Tools

A row per tool, then a row per parameter the surfaces disagree about. `--` means that surface does not have it. A tool's own row says what it advertises about changing things and about the Galaxy it needs; a parameter's row says what each surface declares it to be, in the terms the comparison compares. A difference found against the built-in server says so in its kind.

`Cases` is how many golden result envelopes that tool has under `contract/envelopes` -- calls the Python server answered, replayed through both surfaces here and compared key for key. It is not part of the comparison above, which is about what a tool declares; it is how much of what a tool ANSWERS anybody checks. A `0` fails the parity check for a tool the Python server serves, unless the registry's `fixtures.excluded` list names it; a tool only another surface has has nothing here to generate cases from, so its `0` says only that.

| Tool | Parameter | Cases | Python | TypeScript | Built-in | Difference | Status | Why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cancel_workflow_invocation` |  | `2` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `connect` |  | `0` | `write (tag)` | -- | `write (mcp default)` | `missing-ts-tool` | `intentional` | TS takes the Galaxy URL and key when the server is built, so there is no per-session connect call to expose. |
| `connect` | `api_key` |  | `type=string required=false default=none` | -- | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours takes the key as an argument, because connect is the tool that establishes a session; the built-in's api_key is the per-call credential the comparison drops everywhere. So this row is the one place the dropped parameter is a real difference in what the tool does. |
| `connect` | `url` |  | `type=string required=false default=none` | -- | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours points the session at a Galaxy; the built-in is served BY one, so there is no URL to take. |
| `create_history` |  | `2` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `create_history` | `history_name` |  | `type=string required=true default=none` | `type=string required=true default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | One required string under two names: `history_name` here, `name` there. |
| `create_history` | `name` |  | -- | -- | `type=string required=true default=none` | `vs built-in: missing-py-param` | `unreviewed-gap` | One required string under two names: `name` there, `history_name` here. |
| `create_page` |  | `6` | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` | `write (mcp default)` |  |  |  |
| `create_user_tool` |  | `6` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `delete_user_tool` |  | `5` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `download_dataset` |  | `0` | `read (tag)` | `write (hint)` | `write (mcp default)` | `mutability-mismatch` | `intentional` | Both surfaces write the bytes to a local path when they are given one, and both say so; they disagree because the two flags are about different things. Python's read tag is a statement about Galaxy: the tags exist to gate GALAXY_MCP_INCLUDE/EXCLUDE_TAGS, and the tool is two GETs that create, change and delete nothing on the server -- the file Python writes to file_path is outside what the tag speaks to. The MCP readOnlyHint has no such scope: the SDK defines readOnlyHint true as a tool that does not modify its environment, and with filePath the TS op overwrites whatever file is at that path, so it advertises false and clients that gate approval on the hint ask before it runs. Dannon's call (2026-09-26): each flag is correct about its own scope, so neither side moves and the mismatch is what the two scopes look like from the comparator. |
| `download_dataset` | `file_path` |  | `type=string required=false default=none` | `type=string required=false default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours fetches the dataset and can write it to a path on the caller's disk. The built-in returns no bytes at all: its download_dataset hands back a download URL built from galaxy_infrastructure_url, alongside the dataset's name, extension, size and state. So this row is a real difference in what the two tools do rather than a parameter somebody forgot: a server-side tool can point a client at the bytes, and that is all it can do. |
| `download_dataset` | `require_ok_state` |  | `type=boolean required=false default=true` | `type=boolean required=false default=true` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Both refuse a dataset that is not ok; only ours can be told not to. AgentOperationsManager.download_dataset returns an error and no download URL whenever the state is anything but `ok`, unconditionally -- so what the built-in is missing is the escape hatch, not the guard. Reconciling this one is a question of whether an agent should ever be able to ask for a dataset that is still running or failed, not of adding a check there. |
| `download_dataset` | `use_default_filename` |  | `type=boolean required=false default=true` | `type=boolean required=false default=true` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours can name the file after the dataset; Galaxy's has no local file to name. |
| `get_collection_details` |  | `6` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_collection_details` | `max_elements` |  | `type=integer required=false default=100` | `type=integer required=false default=100` | `type=integer required=false default=500` | `vs built-in: default-mismatch` | `unreviewed-gap` | Both cap the elements returned and the caps differ: 100 here, 500 there. |
| `get_dataset_details` |  | `8` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_dataset_details` | `include_preview` |  | `type=boolean required=false default=true` | `type=boolean required=false default=true` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours can put the first N lines of the dataset's text in the answer, read through get_content_as_text; Galaxy's serialises the `detailed` view, which carries `peek` -- the stored few-line snippet Galaxy renders for text-like datatypes -- but no way to ask for the content itself or to leave the peek out. |
| `get_dataset_details` | `preview_lines` |  | `type=integer required=false default=10` | `type=integer required=false default=10` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | How many lines ours reads; Galaxy's `peek` is a fixed snippet with no line count to choose. |
| `get_histories` |  | `10` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_histories` | `limit` |  | `type=integer required=false default=none` | `type=integer required=false default=none` | `type=integer required=false default=50` | `vs built-in: default-mismatch` | `unreviewed-gap` | Ours declares no default and returns every history; Galaxy's pages at 50. |
| `get_history_contents` |  | `6` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_history_contents` | `deleted` |  | `type=boolean required=false default=false` | `type=boolean required=false default=false` | `type=boolean required=false default=none` | `vs built-in: default-mismatch` | `unreviewed-gap` | Ours defaults to excluding deleted items; Galaxy's passes None through and lets the API decide. |
| `get_history_contents` | `visible` |  | `type=boolean required=false default=true` | `type=boolean required=false default=true` | `type=boolean required=false default=none` | `vs built-in: default-mismatch` | `unreviewed-gap` | Ours defaults to visible items only; Galaxy's passes None through and lets the API decide. |
| `get_history_details` |  | `4` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_invocation_details` |  | `0` | -- | -- | `write (mcp default)` | `vs built-in: missing-py-tool` | `unreviewed-gap` | Only Galaxy's. The same reading exists here inside get_invocations, which returns one invocation's detail when it is given invocation_id. |
| `get_invocations` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_invocations` | `invocation_id` |  | `type=string required=false default=none` | `type=string required=false default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours takes an invocation id and returns that one invocation's detail; Galaxy's splits that into get_invocation_details. |
| `get_invocations` | `limit` |  | `type=integer required=false default=none` | `type=integer required=false default=none` | `type=integer required=false default=50` | `vs built-in: default-mismatch` | `unreviewed-gap` | Ours declares no default; Galaxy's pages at 50. |
| `get_invocations` | `offset` |  | -- | -- | `type=integer required=false default=0` | `vs built-in: missing-py-param` | `unreviewed-gap` | Galaxy's pages with limit+offset; ours takes limit only. |
| `get_invocations` | `step_details` |  | `type=boolean required=false default=false` | `type=boolean required=false default=false` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours can ask for per-step detail on the listing, and Galaxy's server cannot get it at all. get_invocation_details is not the substitute: it builds InvocationSerializationParams(view="element") and leaves step_details at its False default (lib/galaxy/agents/operations.py:617), and a step's jobs, outputs and output_collections are only filled in when the step itself is serialised with view="element" (WorkflowInvocationStep.to_dict, lib/galaxy/model/__init__.py:10917). So what it returns is the invocation's own element view -- state, inputs, input_step_parameters, the invocation's labelled outputs, and a list of steps carrying state, order_index and a bare job_id or implicit_collection_jobs_id -- but no per-step outputs, output_collections or job detail. Ours passes the flag through to the same index endpoint, which serialises the listing with it, so get_invocations(view="element", step_details=True) does return each step's jobs and outputs; and for one invocation by id it sends step_details to GET /api/invocations/{id}, so get_invocations(invocation_id=..., step_details=True) returns each step's jobs too. |
| `get_invocations` | `view` |  | `type=string required=false default="collection"` | `type=string required=false default="collection"` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours chooses how much of each invocation to return; Galaxy's has one shape. |
| `get_iwc_workflow_details` |  | `4` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_iwc_workflows` |  | `6` | `read (tag)` | `read (hint)` | -- | `vs built-in: missing-builtin-tool` | `unreviewed-gap` | Only here, and it is a browse listing rather than a dump: limit/offset over projected summaries, cut short again when a page would not fit the output budget. Galaxy's server can rank the manifest against a query (search_iwc_workflows) and fetch one entry whole (get_iwc_workflow_details), but nothing there pages the catalogue without a query. |
| `get_job_details` |  | `6` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_job_status` |  | `0` | -- | -- | `write (mcp default)` | `vs built-in: missing-py-tool` | `unreviewed-gap` | Only Galaxy's, and what it adds is an entry point rather than runtime. get_job_status calls jobs_service.show(full=False) (lib/galaxy/agents/operations.py:318), which is view_show_job -> Job.to_dict("element"): id, state, exit_code, create_time, update_time, tool_id, tool_version, galaxy_version, command_version, history_id, plus params and input/output ids. No runtime and no job metrics -- metrics need full=true and an admin -- and update_time minus create_time is not a runtime, because it includes the time the job sat queued. The gap is the id: a job is reachable here only through get_job_details(dataset_id), which resolves the job from the dataset's provenance and then reads the same non-full job, so an agent holding a job id from run_tool has nowhere to take it. |
| `get_page` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_page_revision` |  | `4` | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` | `write (mcp default)` |  |  |  |
| `get_page_revision` | `include_rendered` |  | -- | -- | `type=boolean required=false default=false` | `vs built-in: missing-py-param` | `unreviewed-gap` | Galaxy's can return the rendered revision beside the editable one; ours returns the editable text and says which source it came from. |
| `get_server_info` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_tool_citations` |  | `3` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_tool_details` |  | `3` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_tool_input_template` |  | `3` | `read (tag)` | `read (hint)` | -- | `vs built-in: missing-builtin-tool` | `unreviewed-gap` | Only here. Galaxy's server has no pre-flight skeleton for run_tool, so an agent there shapes `inputs` from get_tool_details(io_details=True) by hand. |
| `get_tool_panel` |  | `8` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_tool_panel` | `limit` |  | `type=integer required=false default=100` | `type=integer required=false default=100` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours pages the panel, because the whole tree is far more than a model can read; Galaxy's returns the whole thing. |
| `get_tool_panel` | `offset` |  | `type=integer required=false default=0` | `type=integer required=false default=0` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | The other half of that window; Galaxy's returns the whole panel. |
| `get_tool_panel` | `section_id` |  | `type=string required=false default=none` | `type=string required=false default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours drills into one section by id; Galaxy's has no drill-in. |
| `get_tool_panel` | `view` |  | -- | -- | `type=string required=false default=none` | `vs built-in: missing-py-param` | `unreviewed-gap` | Galaxy's can ask for an admin-configured named panel view; ours always reads the standard panel. |
| `get_tool_run_examples` |  | `4` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_user` |  | `4` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_workflow_details` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `get_workflow_input_template` |  | `8` | `read (tag)` | `read (hint)` | -- | `vs built-in: missing-builtin-tool` | `unreviewed-gap` | Only here. Same gap before invoke_workflow: no template, no run guide, no input validation on that side. |
| `import_workflow_from_iwc` |  | `4` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `invoke_workflow` |  | `4` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `invoke_workflow` | `inputs` |  | `type=anyOf<object\|string> required=false default=none` | `type=anyOf<object\|string> required=false default=none` | `type=object required=false default=none` | `vs built-in: type-mismatch` | `unreviewed-gap` | Ours also accepts the JSON text of the object, because that is what a CLI flag and some clients send; Galaxy's takes the object only. |
| `invoke_workflow` | `inputs_by` |  | `type=string required=false default="step_index"` | `type=string required=false default="step_index"` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours lets the caller say how inputs are keyed (step index, step id, name); Galaxy's uses the API default. |
| `invoke_workflow` | `parameters` |  | -- | -- | `type=object required=false default=none` | `vs built-in: missing-py-param` | `unreviewed-gap` | One parameter map under two names: `parameters` there, `params` here. |
| `invoke_workflow` | `parameters_normalized` |  | `type=boolean required=false default=false` | `type=boolean required=false default=false` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours passes Galaxy's parameters_normalized flag through; Galaxy's does not expose it. |
| `invoke_workflow` | `params` |  | `type=anyOf<object\|string> required=false default=none` | `type=anyOf<object\|string> required=false default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | One parameter map under two names: `params` here, `parameters` there. |
| `list_file_source_templates` |  | `0` | -- | -- | `write (mcp default)` | `vs built-in: missing-py-tool` | `unreviewed-gap` | Only Galaxy's: the catalog of remote file-source plugin templates (Dropbox, S3, Zenodo, ...). Nothing here exposes file sources at all. |
| `list_history_ids` |  | `8` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `list_history_ids` | `offset` |  | `type=integer required=false default=0` | `type=integer required=false default=0` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours pages with limit+offset; Galaxy's takes limit only. |
| `list_jobs` |  | `7` | `read (tag)` | `read (hint)` | -- | `vs built-in: missing-builtin-tool` | `intentional` | Only here, and on purpose: a paged listing of GET /api/jobs filtered by history, state and update time, which is what an agent reconciling its own record of a history against Galaxy needs (Loom's reconcile asks for the jobs updated since a time, in one history, a page at a time). Galaxy's server reads one job at a time -- get_job_details by dataset, get_job_status by job -- and has no index; an agent there would have to walk the history's datasets and ask about each one. |
| `list_page_revisions` |  | `3` | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` | `write (mcp default)` |  |  |  |
| `list_pages` |  | `6` | `read (tag), requires >=26.1` | `read (hint), requires >=26.1` | `write (mcp default)` |  |  |  |
| `list_user_file_sources` |  | `0` | -- | -- | `write (mcp default)` | `vs built-in: missing-py-tool` | `unreviewed-gap` | Only Galaxy's: the user's own configured file-source instances. Nothing here exposes file sources at all. |
| `list_user_tools` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `list_user_tools` | `limit` |  | `type=integer required=false default=25` | `type=integer required=false default=25` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours pages the list; Galaxy's returns every user-defined tool. |
| `list_user_tools` | `offset` |  | `type=integer required=false default=0` | `type=integer required=false default=0` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | The other half of that window; Galaxy's has none. |
| `list_workflows` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `list_workflows` | `name` |  | `type=string required=false default=none` | `type=string required=false default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours filters by name; Galaxy's filters with `search`, which runs its structured workflow query over names, tags and owners. |
| `list_workflows` | `published` |  | `type=boolean required=false default=false` | `type=boolean required=false default=false` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | One flag under two names: `published` here, `show_published` there, same default. |
| `list_workflows` | `search` |  | -- | -- | `type=string required=false default=none` | `vs built-in: missing-py-param` | `unreviewed-gap` | Galaxy's hands the term to WorkflowManager.index_query, which parses it: a bare word matches the name, a tag or the owner, and `name:`/`tag:`/`user:`/`is:` narrow it. A word that appears only in a workflow's annotation is not found. Ours filters name only, through `name`. |
| `list_workflows` | `show_published` |  | -- | -- | `type=boolean required=false default=false` | `vs built-in: missing-py-param` | `unreviewed-gap` | One flag under two names: `show_published` there, `published` here, same default. |
| `list_workflows` | `show_shared` |  | -- | -- | `type=boolean required=false default=true` | `vs built-in: missing-py-param` | `unreviewed-gap` | Galaxy's can leave out workflows shared with the user; ours always includes them. |
| `recommend_biocontainer` |  | `6` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `recommend_iwc_workflows` |  | `9` | `read (tag)` | `read (hint)` | -- | `vs built-in: missing-builtin-tool` | `unreviewed-gap` | Only here, and what Galaxy's server is missing is BM25, not free-text ranking as such: its search_iwc_workflows takes a natural-language query -- its own examples are sentences -- and ranks the manifest through iwc.search_workflows. That score is a count (lib/galaxy/agents/iwc.py:155): one point per query token found anywhere in name, annotation, tags, the 300-character readme summary and tool names, with no IDF, no term frequency and no length normalisation, so a common word weighs as much as a rare one and integer ties are ordinary. Ours scores with rank_bm25's Okapi over name (counted twice, to weight it), annotation, tags, the whole readme and tool names, minus a stop-word list, so a rare term separates workflows a token count ties and a long readme does not win for being long. It is not one-way: ours tokenises runs of two or more letters, so a term like hg38 tokenises to nothing, where Galaxy's alphanumeric tokeniser keeps it. Both drop zero scores and return match_score. |
| `revert_page_revision` |  | `2` | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` | `write (mcp default)` |  |  |  |
| `run_tool` |  | `13` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `run_tool` | `tool_version` |  | `type=string required=false default=none` | `type=string required=false default=none` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours can ask for a specific tool version (posted through to Galaxy, which may still fall back to an installed one); Galaxy's runs whatever version the toolbox resolves for the id and takes no version. |
| `run_user_tool` |  | `5` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `search_iwc_workflows` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `search_iwc_workflows` | `limit` |  | `type=integer required=false default=20` | `type=integer required=false default=20` | `type=integer required=false default=10` | `vs built-in: default-mismatch` | `unreviewed-gap` | Both cap the ranking and the caps differ: 20 here, 10 there. |
| `search_iwc_workflows` | `offset` |  | `type=integer required=false default=0` | `type=integer required=false default=0` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours pages the results; Galaxy's returns the top `limit` and no more. |
| `search_tools_by_keywords` |  | `5` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `search_tools_by_keywords` | `limit` |  | `type=integer required=false default=50` | `type=integer required=false default=50` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours pages the matches; Galaxy's returns them all. |
| `search_tools_by_keywords` | `offset` |  | `type=integer required=false default=0` | `type=integer required=false default=0` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | The other half of that window; Galaxy's has none. |
| `search_tools_by_name` |  | `7` | `read (tag)` | `read (hint)` | `write (mcp default)` |  |  |  |
| `search_tools_by_name` | `limit` |  | `type=integer required=false default=25` | `type=integer required=false default=25` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | Ours pages the matches; Galaxy's returns them all. |
| `search_tools_by_name` | `offset` |  | `type=integer required=false default=0` | `type=integer required=false default=0` | -- | `vs built-in: missing-builtin-param` | `unreviewed-gap` | The other half of that window; Galaxy's has none. |
| `update_history` |  | `6` | `write (tag)` | `write (hint)` | -- | `vs built-in: missing-builtin-tool` | `unreviewed-gap` | Only here. Galaxy's server can create a history and read it, but cannot rename, annotate, tag, publish or delete one. |
| `update_page` |  | `3` | `write (tag), requires >=26.1` | `write (hint), requires >=26.1` | `write (mcp default)` |  |  |  |
| `upload_file` |  | `0` | `write (tag)` | `write (hint)` | -- | `vs built-in: missing-builtin-tool` | `unreviewed-gap` | Only here. Galaxy's server uploads only from a URL; there is no tool for a local path (the tus upload). |
| `upload_file_from_url` |  | `0` | `write (tag)` | `write (hint)` | `write (mcp default)` |  |  |  |
| `upload_file_from_url` | `history_id` |  | `type=string required=false default=none` | `type=string required=false default=none` | `type=string required=true default=none` | `vs built-in: required-mismatch` | `unreviewed-gap` | Galaxy's requires the target history; ours takes it optionally and uploads into the current one when it is left out. |
