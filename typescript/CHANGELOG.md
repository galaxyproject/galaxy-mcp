# Changelog

`@galaxyproject/galaxy-ops`, `@galaxyproject/galaxy-cli` and
`@galaxyproject/galaxy-mcp` share a version and are published together, so one
entry covers all three; where something only affects one surface, it says which.

## 0.3.1 (unreleased)

### `get_job_details` by job id, with the full record (#TBD)

The op takes `jobId` as an alternative to `datasetId` -- exactly one of the two, refused as
`validation` (exit 64) when neither or both arrive -- and a `full` flag that sends `?full=true`
to `GET /api/jobs/{job_id}` on either path. A job id Galaxy answers 400 or 404 to is reported
as `not_found` (exit 66) with one sentence, because the agent holding the id is asking whether
the run still exists and cannot act on which of the two it was. `datasetId` is no longer
required, so on the CLI it moves from a positional to `--dataset-id`, and `GetJobDetailsResult.dataset_id`
is `null` when the job was asked for by id. Six golden cases pin it on both surfaces; every
answer through a dataset is unchanged.

### `get_invocations` sends `step_details` for one invocation (#152)

Given an `invocationId`, the op ignored `stepDetails` and Galaxy answered with every step's
`jobs` list empty, which is what the Python server did too -- both now send `step_details`
to `GET /api/invocations/{id}` when it is asked for, and the request is unchanged when it is
not. A new golden case, `get_invocations/single_step_details`, is answered only when the flag
arrives, so either surface dropping it again fails the replay.

## 0.3.0 (2026-10-04)

Breaking on both surfaces. 0.2.0 was tagged but never reached npm, so this release
carries both entries.

### Releases are staged and approved

The release workflow now stages each package on npm instead of publishing it, and the
trusted publishers are configured stage-only. A pushed `ts-v*` tag puts the three
packages in the registry where nobody can install them; a maintainer approves each one
with 2FA before it goes live.

### A failed call answers the way the Python server's failed call answers (#145)

Measured before anything was designed: drive the Python server through an in-memory
FastMCP client (fastmcp 3.4.2, `mask_error_details` false by default), call a tool under a
mocked Galaxy that refuses, and the client gets

```json
{ "content": [{ "type": "text", "text": "Error calling tool 'get_page': Get page failed: 404 Client Error: Not Found for url: https://galaxy.example/api/pages/p1 (Resource not found - check IDs and URLs). Context: page_id=p1" }], "isError": true }
```

No envelope. No `structuredContent`. One text block, `isError`, and a sentence that is
three layers deep: FastMCP's wrapper, then the tool's own action and context, then --
quoted verbatim in the middle of it -- the text bioblend or requests raised.

- **Breaking (MCP):** a failure is now an MCP error result. `isError` is true, the single
  text block is that sentence, and there is nothing to parse. A client that read
  `JSON.parse(text)` on a failure and looked at `success: false`, `message` or `errorKind`
  now has to read `isError` and the text. The MCP SDK surfaces the block as
  `result.content[0].text`; a client using `raise_on_error`-style helpers gets an
  exception carrying the same string.
- `errorKind` and the `GalaxyError` classes have not moved. They are still on the error a
  library caller catches from `@galaxyproject/galaxy-ops`, and still in the CLI's json.
  Only the MCP wire changed.
- **The CLI's json failure shape is the CLI's own**, and stays
  `{"success": false, "message": "...", "errorKind": "..."}` -- no data, no count, no
  pagination. The Python server has no command line, so there is nothing to be in parity
  with: the exit code has to be read off something, `errorKind` is what it is read off (see
  the exit-code table), and what parity means here is the SENTENCE. `message` is now the
  Python tool's sentence, word for word, without FastMCP's `Error calling tool '<name>': `
  wrapper -- which belongs to the MCP surface and not to the tool.
- **Every tool's failure sentence is that server's**, which means writing its client
  libraries' prose too. Three shapes reach a caller and which one appears depends on how
  the tool asked, not on what went wrong: a bioblend GET reports
  `GET: error 404: b'<body>', 0 attempts left: <body>` -- the body twice, once as a Python
  `bytes` repr -- a bioblend write reports `Unexpected HTTP status code: 404: <body>`, and
  a request made with requests directly reports `404 Client Error: Not Found for url:
  <url>`. Around that goes `<Action> failed:`, the hint for 401, 403, 404 and 500, and the
  context dict rendered as `. Context: k=v, k2=v2`. Each operation declares its own action,
  context and shape; `python-failure.ts` writes the library prose.
- **Breaking:** `update_history` refusing an update with no fields in it, and a tool run
  Galaxy refuses with a 400 -- over its inputs or over its credentials, on `run_tool` and
  `run_user_tool` alike -- now report `errorKind: "validation"` where they reported
  `"connection"`. The CLI exit code for those changes from 69 to 64. Both are usage errors:
  an agent told "connection" backs off and retries a call that can never succeed. A refusal
  on any other status keeps the kind its status was classified with, advice or no advice: a
  500 that happens to mention credentials is still a server that fell over. The CLI replay
  asserts the exact exit code for every one of the 65 failure cases now, rather than that it
  is not zero.
- **Breaking:** these sentences changed beyond the reformatting above, because one surface
  was refusing what the other answered, or refusing it somewhere else:
  - `get_user` no longer raises on the anonymous reply `/api/users/current` gives an
    unauthenticated session. The record is handed back as it arrived and the sentence names
    the user `'unknown'`, which is what the other server says. **Library callers:** the
    exported `CurrentUser` type says so now -- `id`, `email` and `username` are optional,
    because the endpoint answers `DetailedUserModel | AnonUserModel` and the anonymous model
    has none of the three. `user.username.toLowerCase()` used to compile and then throw; it
    is a compile error now, and code reading any of the three has to check first.
  - `create_page` no longer refuses a standalone report with no title or slug, and
    `update_page` no longer refuses an edit with nothing in it. Neither refusal exists over
    there: the request goes out and Galaxy answers with its own reason.
  - `list_history_ids` raises `Failed to list history IDs: 'id'` for a history record with
    no id, where it used to report the id as `""`. That sentence is a KeyError's text,
    quoted the way the tool quotes it.
  - `get_dataset_details` asks the collections API about an id it could not read as a
    dataset, and says `The ID '<id>' is a dataset collection, not a dataset` with the
    collection's name -- rather than reporting it as not found.
  - `get_job_details` holds a provenance failure back instead of raising it: the dataset's
    own record names the job that made it, so a history whose provenance is out of reach
    still answers. The held failure is reported only if the fallback fails too, or finds no
    job. A 404 anywhere in it keeps the tool's own sentence, which does not claim whether
    the dataset or the permission was missing.
  - `get_tool_panel`'s unknown section, the version guard's refusal (which now says
    "Nothing was sent to Galaxy."), the five IWC tools' manifest failures and the two
    trsID refusals, `create_user_tool`'s three argument refusals (a mistyped container now
    names Python's type: `got int: 3`), and `get_invocations`' 200-with-an-error-body all
    read as that server reads.
- **Breaking:** every parameter the Python server declares a null branch for now accepts an
  explicit `null` and treats it as unset, where thirteen of them across seven tools refused
  it outright -- `update_history`'s five fields, `create_page`'s five, `update_page`'s two,
  `get_job_details.history_id`, `get_tool_run_examples.tool_version`,
  `get_workflow_input_template.history_id`, `list_pages.history_id`, `list_pages.search`
  and `run_tool.tool_version`. Nothing that Python declares as a plain string or integer
  takes one. `update_history`'s message names only the fields it actually sent, so a field
  passed as null is not reported as updated by the call that left it out.
- **New requests, and two clauses that come with them.** `run_tool` reads the tool's schema
  before it submits -- only when an input looks like a dataset reference, which is the
  condition that decides whether the request happens at all -- and looks up the caller's
  stored credentials for the tool, sending them with the run. Its success message gains
  ` (with credentials)` when it found some, and
  ` (inputs not pre-checked: <why>)` when the schema could not be read, with all four
  reasons worded as over there. `run_user_tool` gets the second clause from the
  representation it already holds. A refused run gets the enriched explanation: the
  parameter list read back, a structural example from one of the tool's own tests, the
  warning about Galaxy's misleading wording, and -- for a refusal that mentions credentials
  -- what to check about those instead. The credentials advice is `run_tool`'s alone, as it is
  over there: a refused `run_user_tool` gets the input explanation whatever Galaxy's reply
  mentions, because a user-defined tool has no credentials path to configure.
- **Not ported, and it shows in one place:** the input checker itself. Walking Galaxy's
  parameter model -- conditionals, repeats, sections, datatype compatibility -- is its own
  piece of work. So where the other server refuses a run on its own evidence, this one
  submits it and lets Galaxy answer; and the two clauses that name which inputs the schema
  proves wrong, and which keys it does not model, are absent from the enriched refusal.
  Everything that does not depend on the check is there, including the requests.
- A failed request that never got a reply -- a refused connection, a DNS failure, an abort
  -- is now a typed failure with the shape of a Python one rather than an exception escaping
  as a bug. The text inside it is this runtime's and not requests', and no case pins either.
- **Seventy-one new cases** on both replay suites, which is a failure case for all 42 open
  tools plus the refusals that make no request: 385 files, 223 cases, of which 65 are
  failures.
  The generator drives the real server through an in-memory MCP client for those, because a
  failure has no envelope to write down -- what it records is the `CallToolResult` the wire
  carries, plus the tool's own sentence with FastMCP's wrapper stripped off.
- **Breaking:** an off-schema request whose reply carried no body is decided by the status
  now, where the guard asked whether there was an error body to read. openapi-fetch parses a
  failed reply's body as text, which for an empty body is the empty string -- falsy -- so
  `delete_user_tool` answered `{"deactivated": true}` with `success: true` for a DELETE
  Galaxy refused with a 403 or a 404 and no body. It now fails with the same sentence any
  other refused request gets. A 204 No Content is still a success, which is what a DELETE
  usually answers with.
- The Python server moved in one place, because it contradicted itself: the three raw
  requests over the unprivileged-tools API did not check the status Galaxy answered with,
  so `delete_user_tool` reported `deactivated: true` for a DELETE that 404'd,
  `list_user_tools` sliced an error body as a page and reported
  `List user tools failed: slice(0, 25, None)`, and `run_user_tool` read one as a tool
  record and said no such tool existed. All three now fail like their siblings. See
  `mcp-server-galaxy-py/CHANGELOG.md`.

**The `#145` is a guess** -- #144 is the merge this branch starts from -- so correct it
when the PR is opened.

### The MCP and CLI envelopes are the Python server's (#140)

A prompt written against the Python MCP server reads `data[0]` and
`pagination.next_offset`. Against these packages it got an object with the page
inside it under `items`, a pagination block in camelCase under different names,
a second copy of that block travelling inside `data`, and no `count` at all. Two
servers, the same tool names, two answers. The envelope is now that server's,
key for key, on the MCP text block and on `galaxy-cli --format json` alike.

- **Breaking:** a listing's `data` is the page itself, not `{ items, pagination }`.
  `data[0]` is the first row. The exceptions are the tools where the Python
  server returns an object too: `get_tool_panel` answers with `{ entries }` or
  `{ section_id, section_name, tools }`, and `get_history_contents` with
  `{ history_id, contents }` -- in each case without the pagination block that
  used to travel inside `data` as well.
- **Breaking:** the pagination keys are the Python server's names. Old to new:
  `total` -> `total_items`, `returned` -> `returned_items`,
  `hasNext` -> `has_next`, `hasPrevious` -> `has_previous`,
  `nextOffset` -> `next_offset`, `previousOffset` -> `previous_offset`,
  `helperText` -> `helper_text`. `limit` and `offset` keep their names. Every
  field is always present, `null` where there is nothing to say, rather than
  absent -- which is what the pydantic model on the other side serialises to.
- **Breaking:** `trimmedForSize` is gone from the wire. The Python server has no
  such field and folds the fact into the helper text, which this side already
  matched word for word: a cut page's `helper_text` says "This page was cut short
  to fit the output budget, not because there is nothing more." That sentence is
  how a cut page is told apart, and in numbers it is `has_next` true while
  `returned_items` is under the `limit` asked for. A short page on its own says
  nothing -- an ordinary last page is short too, and reports `has_next: false` --
  so comparing `returned_items` with `limit` and stopping there will call the end
  of a walk a cut. `recommend_iwc_workflows` has no pagination block at all -- a
  ranking has no offset to walk -- so its cut is reported at the end of `message`
  instead.
- **New:** `count`, the number of rows in this answer, for the tools the Python
  server counts: every listing, plus `get_tool_run_examples` (test cases),
  `get_tool_citations` (citations), `get_collection_details` (elements returned,
  after truncation), `get_workflow_input_template` (input slots),
  `get_invocations` (the list, not a single invocation), `list_pages`,
  `list_page_revisions`, and `get_history_details` -- which counts everything in
  the history, and now pays for the same second request the Python tool pays for
  to get it.
- `list_pages` reports a real `total_items`, taken from the `total_matches`
  response header, where before it returned only the window it had asked for.
  Its block comes from the shared describer, like every other listing's.
- `get_invocations` no longer sends a pagination block carrying just the `limit`.
  Galaxy windows that index itself and reports no total, so there was never a
  window to describe; the Python tool sends none either.
- `list_user_tools` sends "user tools" in its helper text where it sent "tools",
  which is the noun the Python tool uses. The noun changes on the way out: a
  direct `run()` caller still gets "tools" in `Paged.pagination.helperText`.
- **Breaking for anyone importing them:** `runWithEnvelope` now returns
  `GalaxyResult<unknown>` (a projection may emit something other than what `run`
  returned), `GalaxyResult` has gained `count`, `Pagination` is the wire shape
  above, and `Operation.project` returns the whole envelope body rather than just
  a message and a window. It also takes an optional third argument, the
  collector `runWithEnvelope` puts on the context for that one call, through
  which an op hands its projection a fact its return value has no room for -- a
  total off a response header, a count from a second request. Optional at both
  ends: a context built by hand carries none and an existing projection compiles
  unchanged.
- **Breaking:** `get_history_contents` checks its window before it asks Galaxy
  anything, where it used to check nothing at all. The whole rule, which is
  `validatePagination`'s and which most of the other listings were already under
  (the list is below): the `limit` must be a whole number ("limit must be a whole number (got
  1.5)") and at least 1 ("limit must be at least 1 (got 0)"), and the `offset`
  must be a whole number and 0 or greater, both refused with the one sentence
  ("offset must be 0 or greater (got -1)", "offset must be 0 or greater (got
  0.5)"). None of those windows had a page to describe: `limit: 0` reported more
  to come with a next offset equal to the one asked for, so a walk sat where it
  was for ever; a negative offset sliced from the end of the list and then
  claimed a next page back inside what it had just returned; and a fractional one
  is not a row count. There is still no ceiling on the limit, because neither
  server caps this tool.

  A fraction is where the two servers agree by different means, and it is worth
  knowing which: `_validate_pagination` over there never sees one, because the
  tool is typed `int` and pydantic refuses `1.5` at the boundary with "Input
  should be a valid integer, got a number with a fractional part". These surfaces
  declare `z.number().int()` for the same parameters, so a fractional window
  never reached the operation from the MCP or CLI wire either -- what is new is
  the library path, where `getHistoryContents({ historyId, limit: 1.5 })` used to
  return a page and now throws. Both sides refuse the same windows; only the
  sentence and the shape of the refusal differ, which is the failure envelope
  named at the end of this entry.

  Not every listing is under that rule, and the earlier claim that they all were
  was too broad. Ten check the window themselves, the same ten on both servers:
  `get_history_contents`, `get_iwc_workflows`, `get_tool_panel`,
  `list_history_ids`, `list_user_tools`, `list_workflows`,
  `recommend_iwc_workflows` (a limit and no offset -- a ranking has no window to
  walk), `search_iwc_workflows`, `search_tools_by_keywords` and
  `search_tools_by_name`. `list_pages` hands its `limit` and `offset` to Galaxy
  instead of slicing a page itself, so the window is Galaxy's to judge, and
  `get_invocations` likewise passes its limit along and takes no offset at all.
  And `get_histories` still validates nothing on either side: `limit: null` there
  means "every history" rather than a page size, so the floor is not its rule,
  and on that no-limit branch a negative offset is read differently by each
  server -- a divergence older than this entry, left alone deliberately rather
  than settled by a rule nobody has chosen yet.
- **Unchanged, with four exceptions:** every operation's `run()` result,
  `Paged<T>` and its camelCase `PaginationInfo`. A TypeScript caller importing an
  op directly, and code mode with it, sees exactly what it saw before -- the same
  wire-versus-library split as the parameter-name change above -- except for the
  four places where the library was not doing what the Python server does and so
  had to move with it:
  - `getHistoryContents` throws for the windows above instead of returning a page.
  - `recommendIwcWorkflows` tokenises intents and readmes by Unicode word
    boundaries, as `re` does and as JavaScript's `\b` does not. An intent of
    "café" has no searchable term in it rather than the term `caf`, a readme
    saying "protéomique" no longer indexes `prot`, and a ranking over text that is
    not plain ASCII can come back in a different order or not at all. Plain ASCII
    text, which is nearly all of the IWC manifest, ranks exactly as before.
  - `searchToolsByName`, `searchToolsByKeywords` and `searchIwcWorkflows` fold
    case with the Python server's `str.lower()` rather than `toLowerCase`, both
    the mapping and the final-sigma rule (below). A query of a capital letter
    Unicode gave a case mapping after that server's edition no longer matches a
    tool named with the small one -- the other server never made that match --
    and a capital sigma is final or not according to which characters that
    server calls cased, so a tool named U+1C8A followed by U+03A3 lowercases to
    an ordinary sigma and a query of an ordinary sigma finds it.
  - `cleanReadmeSummary` is `_clean_readme_summary`'s output now, which moves
    `readme_summary` on the four operations that enrich a manifest entry
    (`get_iwc_workflows`, `get_iwc_workflow_details`, `recommend_iwc_workflows`,
    `search_iwc_workflows`) and the summary line of a workflow guide. Whitespace
    is Python's set -- a leading byte order mark no longer hides a markdown
    heading, a next-line character now does -- and the 300-character cut is
    measured and taken in code points, so a summary carrying an emoji keeps it
    whole rather than ending in half of one.
- The `message` of the nine listings the output budget can cut --
  `get_iwc_workflows`, `get_tool_panel`, `list_history_ids`, `list_user_tools`,
  `list_workflows`, `recommend_iwc_workflows`, `search_iwc_workflows`,
  `search_tools_by_keywords`, `search_tools_by_name` -- is now the Python
  server's sentence word for word. That is not cosmetic: the budget is measured
  on the whole envelope, `message` included, so prose of a different length cuts
  the page at a different row. A page of two histories weighing exactly 50,000
  bytes with one sentence and 50,011 with the other returned two rows on one
  surface and one on the other. That includes a tool's early returns:
  `recommend_iwc_workflows` says "No workflows in IWC manifest" when the IWC
  manifest is empty and "No searchable terms in query" when the intent tokenises
  to nothing, rather than reporting that nothing matched -- three different
  reasons to get an empty ranking, and only one of them is the query being too
  broad. `list_history_ids` says "No histories found" for an account with none.
- **The Unicode data is the Python server's, by table rather than by rule.** A
  word boundary, the whitespace a summary is split on and a lowercased needle are
  all decided by Unicode data, and the two runtimes read different editions of it
  -- Node 22 is on 17.0, CPython 3.12 reports 15.0.0, and 9,661 code points are
  word characters to the first and not to the second. Spelling `\w` out as
  `[\p{L}\p{N}_]` therefore still put a word boundary in a different place on
  each side: an intent of "rnaseq" followed by one of those letters was five
  ranked workflows on one server and "No searchable terms in query" on the other.
  So the tokeniser and the text helpers read a table instead, generated from the
  contract interpreter's own answer for every code point in the space
  (`mcp-server-galaxy-py/tests/testdata/python-unicode-15.0.0.json`, written by
  `uv run python -m tests.python_unicode`) and copied into
  `packages/galaxy-ops/src/python-unicode-data.ts`, with a test comparing the two
  copies row for row. Whitespace needed no new table: `isPySpace` was already a
  written-down copy of that interpreter's 29 code points and still matches it
  exactly, and it is now compared against the file too. Lowercasing takes two
  tables and a rule, because `str.lower()` is two things: a mapping, shipped
  whole (1,433 code points, U+0130 to two of them), and one context-sensitive
  case -- a capital sigma is a FINAL sigma when a cased character stands behind
  it and none in front, with only case-ignorable characters in between. Which
  characters those are is Unicode data again, so both sets are read off the
  contract interpreter (4,259 cased code points in 150 ranges, 2,707
  case-ignorable in 437) and the rule is applied over them. Nothing in
  `pyLower` calls `toLowerCase` or a Unicode property class any more: a list of
  exceptions only covers the code points whose own lowercase differs, and in a
  name ending U+1C8A followed by a capital sigma neither character's own
  lowercase differs at all -- the name still ends in a final sigma under this
  runtime's tables and an ordinary one under that server's, which is one search
  result against none. The tables are held
  to the interpreter at both ends: it checks that the rule over them reproduces
  its `lower()` for every code point in the space in four contexts each, and
  the checked-in sample of its own output
  (`python-lower-probes-15.0.0.json`, 6,392 strings) is what this side's copy
  is compared against. Moving to an interpreter with newer tables is a
  deliberate regeneration and a note here, not a silent change of answer.
- **Not aligned yet, and not claimed to be:** the `message` text of every other
  tool, which the two servers still word differently, and the failure envelope --
  Python raises and FastMCP turns that into an MCP error, while these surfaces
  answer with `{ success: false, message, errorKind }`. Both are named as the
  next result-shape rows rather than quietly left out.

Backed by golden fixtures rather than by reading both sides: the Python suite
generates what its tools emit for a set of calls, with the Galaxy replies they
were answered with (`uv run python -m tests.envelope_fixtures`), and the MCP
server and the CLI each replay all 89 of those cases against those replies and
compare keys, `data`, `count` and `pagination` exactly, plus `message` for the
nine listings above. Nothing is skipped on either surface, the pages the budget
cut included -- which is also why `galaxy-cli` measures the budget against the
compact line the other surfaces measure and prints indented afterwards, rather
than measuring its own indentation and cutting a shorter page than MCP would
for the same call.

### What is inside `data` is the Python server's (#142)

The envelope PR settled the shape around the payload -- `data`, `success`,
`message`, `count`, `pagination` -- and left the payload itself alone. Three
tools got a `count` and no golden case because their `data` did not match at
all, and five smaller differences were written down and not acted on. This is
that list.

- **Breaking:** `get_history_details` answers with `{ history, contents_summary }`
  rather than the bare history record. `data.history` is what Galaxy sent;
  `data.contents_summary` is `{ total_items, note }`, the number of items in the
  history -- deleted and hidden included, which is why it costs a second request
  -- and the sentence the Python tool puts beside it. `run()` still returns the
  record itself.
- **Breaking:** `get_collection_details` answers with the Python tool's
  normalised object -- `{ collection_id, history_content_type, collection,
  elements, elements_truncated, note }` -- rather than Galaxy's record with its
  `elements` sliced. Each element is flattened to `{ element_index,
  element_identifier, element_type, object_id, name, state, extension,
  file_size }`, one level deep: a nested collection keeps its `object_id` and
  comes back with the dataset fields empty rather than recursed into. `run()`
  still returns Galaxy's record, truncated.
- **Breaking:** `get_tool_run_examples` always states `requested_version`. It is
  `null` when the caller named no version, where the key used to be absent.
- **Breaking:** `list_history_ids` calls a history with no name `"Unnamed"`,
  which is the word the other server uses, rather than the empty string. A name
  Galaxy sent as `null` stays null and an empty one stays empty -- the word
  stands in for a missing field, not for a blank one.
- **Breaking:** `list_pages` sends the same pagination block every other listing
  sends. It now carries `helper_text` instead of `null`, and `next_offset`
  advances by the rows that came back rather than by the `limit` asked for, so a
  short page no longer points past rows nobody has seen. The Python tool moved
  first: it was the last listing there still hand-building its own block.
- `get_workflow_input_template` builds the same slots as the other builder.
  `step_uuid` is stated as `null` for a step with no uuid rather than left out; a
  `step_index` Galaxy sent as null skips the step instead of falling through to
  its id; a tool step whose `type` happens to read as an input discriminator is
  no longer templated as an input; and an empty `parameter_type` on a param falls
  through to the step's.
- A workflow's steps are walked in a stated order on both servers, because a JSON
  object has none the two languages agree on: numeric key ascending for keys that
  are non-negative integers written canonically -- no leading zeros, no sign --
  then the remaining keys in the order they arrived. That changes `tools_used` on
  the four IWC operations, the `inputs` and `outputs` of
  `get_iwc_workflow_details`, and the slot order of `get_workflow_input_template`
  for any workflow whose step keys do not arrive sorted. The same rule says which
  key is a step index, which retires a second skew: `1e2` and `0x10` used to be
  read as indexes on one side and skipped on the other.
- `get_iwc_workflow_details` reads a step's `label` and `annotation` and an
  output's `label` the way the other server reads them: a value that is present
  is the value, `null` and `""` included, and only an absent key falls back to
  `Input N`, `Step N`, the `output_name` or `""`. A `.ga` export states the labels
  it does not have as `null`, so `name: null` is now the honest answer where it
  used to read `Step 7`.
- `searchToolsByName`, `searchToolsByKeywords` and `searchIwcWorkflows` test for
  a substring by code point, as Python's `in` does, rather than by UTF-16 code
  unit as `String.prototype.includes` does. A query that is half of a surrogate
  pair no longer matches the other half of an astral character. It takes a lone
  surrogate in the query to reach, which the JSON-RPC wire does carry.
- Not changed, deliberately: `get_tool_panel` gains no `tool_count` or
  `section_count` for the whole server. The Python tool has no such keys -- its
  overview answers with `entries`, and a section entry's `tool_count` is the
  tools directly inside that section, a sub-section neither counted nor walked
  through. Two fixtures on a nested panel pin that.

Twenty-one more golden cases, 89 in all across twenty tools, and the three tools
the envelope PR could only count are now compared in full on both surfaces. Still
not aligned: the `message` text of every tool but the nine listings, and the
failure envelope. One new wrinkle in that second one -- a lone-surrogate query is
a call the Python server cannot answer at all, because its message echoes the
query and `pydantic_core` refuses to serialise a lone surrogate, where these
surfaces answer with an empty page.

### A golden case for every open tool (#143)

The envelope and payload rounds compared twenty tools in full and left twenty-two
with no fixture at all, which is where a difference hides. Every open tool now has
golden cases generated from the Python server and replayed on both surfaces, and
the differences they turned up are below.

- **Breaking:** `run_tool` submits and returns, and answers with Galaxy's own
  submission record. It used to POST `/api/jobs`, poll `/api/tool_requests/{id}`
  and then wait for every job it spawned, answering with
  `{ toolRequestId, jobs, implicitCollections, state: "ok" }` once they were all
  terminal. The Python tool POSTs `/api/tools` and hands back what Galaxy said --
  `{ outputs, output_collections, jobs, implicit_collections, ... }` with the jobs
  in their starting state -- and a call that blocks for the length of a
  bioinformatics job is a different tool from one that queues it. Poll the jobs
  with `get_job_details`. The queue-and-wait path is still here and is now
  exported as `executeToolRequest`, with its `ToolRun` result, for a caller who
  wants it. The inputs go with it: `/api/tools` takes Galaxy's legacy format, as
  the Python tool sends it, so a parameter inside a section, conditional or
  repeat is one flat key joined with `|` (`advanced|threshold`), and the nested
  `{ advanced: { threshold } }` this op used to describe is read by the legacy
  parser as nothing at all -- the default applies. A `{ __class__: "Batch" }`
  wrapper is likewise the other path's spelling. The schema text and summary now
  say so.
- **Breaking:** `invoke_workflow` hands back every invocation a batch run made.
  Galaxy answers a batch with a list, and this took the first element of it and
  dropped the rest; the Python tool passes the list through, so now this does too.
  A single invocation is unchanged. `InvokeWorkflowResult` (the op's output type)
  is `InvocationResult | InvocationResult[]`, and the summary line names every id.
- **Breaking:** `get_server_info` answers the way the Python tool does, in three
  ways. The list of things this server is too old to run is `unsupported_tools`
  rather than `unsupported_ops`, and it is sorted by name. `config` is the sixteen
  fields that tool lifts out of `/api/configuration` by name -- `brand`,
  `logo_url`, `welcome_url`, `support_url`, `citation_url`, `terms_url`,
  `allow_user_creation`, `allow_user_deletion`, `enable_quotas`,
  `ftp_upload_site`, `wiki_url`, `screencasts_url`, `library_import_dir`,
  `user_library_import_dir`, `allow_library_path_paste` and
  `enable_unique_workflow_defaults` -- rather than Galaxy's whole configuration; a
  field Galaxy did not mention reads `null`, and an unmentioned `brand` reads
  `"Galaxy"`. And `version_source` is gone from the wire: the other server sends
  no such key, so a caller could not rely on it. Whether the version was supplied
  rather than fetched was still said in the summary line at this point; the entry
  below moves it onto the library result instead, because the other server's
  sentence does not say it either.
  `UnsupportedOp` is now
  `UnsupportedTool` and `ServerConfigSummary` is exported beside it.
- **Breaking:** `get_page`, `create_page` and `update_page` drop the expanded
  `content` on an HTML page too. They dropped it only for a page that had a
  `content_editor` to keep instead, which left an HTML page -- where Galaxy fills
  content_editor on the markdown path only -- answering with its rendered body
  where the Python tool answers without it. `content` is the expanded form either
  way, so a caller editing it and sending it back bakes the expansion into the
  page; `get_page`'s `includeRendered` is how to ask for it on purpose.
- **Breaking:** `get_dataset_details` answers with `{ dataset, dataset_id }` and
  puts `preview` beside them, rather than spreading Galaxy's record at the top
  level with the preview mixed in. `data.dataset` is what Galaxy sent and
  `data.dataset_id` is the id that was asked for, so it is there even for a record
  that carries no id -- and a Galaxy that grew a `preview` field of its own could
  no longer collide with ours. The preview itself has not moved.
- **Breaking:** `get_user` answers with Galaxy's whole user record rather than
  `{ id, email, username }`. A DetailedUserModel carries the disk usage, the
  quota, the tags in use and the stored preferences, and the Python tool passes
  all of it through. The three named fields are still there and are still what
  the anonymous-response guard insists on, so a caller reading only those is
  unaffected; `CurrentUser` has gained an index signature for the rest.

Fifty-three more golden cases, 143 in all across all forty-two open tools, so
every tool either surface serves is now replayed on both and compared key for key:
the key set, `data`, `success`, `count` and `pagination`, and `message` for the
nine budgeted listings. `PARITY.md` gains a `Cases` column saying how many each
tool has, and the parity check fails on an open tool that has none. Still not
aligned at this point: the `message` text of every tool but those nine -- which
is the entry below -- and the failure envelope.

### Every tool's `message` is the Python server's sentence (#144)

`message` is the first thing an agent reads and the only part of the answer
written for it rather than for a parser, and until now it was the one field the
two servers were free to word their own way. Nine of the forty-two open tools
matched, because their sentence is measured against the output budget and decides
where a page is cut; the other thirty-three said something of their own. Every
tool's sentence is now the Python server's, byte for byte, on the MCP text block
and on `galaxy-cli --format json` alike, and all 152 golden cases compare it
exactly rather than checking that it is a non-empty string.

- **Breaking:** the sentence changed for thirty tools. A caller matching on the
  old text will stop matching. In one line each, old to new:
  `cancel_workflow_invocation` "Cancelled invocation X" -> "Cancelled workflow
  invocation 'X'"; `create_history` "Created history ID (NAME)" -> "Created
  history 'NAME'", naming the name that was asked for rather than the record;
  `create_page` "Created page ID (TITLE)" -> "Created page 'ID'";
  `create_user_tool` "Created user tool UUID" -> "Created user-defined tool
  'NAME'", from the representation that was sent; `delete_user_tool`
  "Deactivated user tool UUID" -> "Deactivated user-defined tool 'UUID'";
  `get_collection_details` "Collection ID (N elements)" -> "Retrieved collection
  'NAME'"; `get_dataset_details` "Dataset ID state=S" -> "Retrieved details for
  dataset 'NAME'"; `get_histories` "N of M histories" -> "Retrieved N of M
  histories", and "Retrieved N histories" when no limit was given;
  `get_history_contents` "N of M item(s)" -> "Retrieved N items from history";
  `get_history_details` "History ID state=S" -> "Retrieved details for history
  'NAME'"; `get_invocations` "Invocation ID state=S" -> "Retrieved invocation
  'ID'" for one, and the plural is written for a list of one; `get_job_details`
  "Job J for dataset D" -> "Retrieved job details for dataset 'D'"; `get_page`
  "Page ID (TITLE)" -> "Retrieved page 'ID'"; `get_page_revision` "Revision R of
  page P (SOURCE, content_editor from X)" -> "Retrieved revision 'R' of page
  'P'"; `get_server_info` "Galaxy at URL (version V)" -> "Retrieved server info
  for URL"; `get_tool_citations` "N citation(s) for NAME" -> "Retrieved N
  citations for tool 'ID'"; `get_tool_details` "Tool ID (NAME vV)" -> "Retrieved
  details for tool 'ID'"; `get_tool_input_template` "Input template for ID (N
  top-level param(s))" -> "Built an input template for tool 'ID'. Replace
  placeholders (e.g. <dataset_id>) and pass the result as `inputs` to run_tool.";
  `get_tool_run_examples` "N test case(s) for ID" -> "Retrieved N test cases for
  tool 'ID'"; `get_user` "Authenticated as NAME <EMAIL>" -> "Retrieved user info
  for 'NAME'"; `get_workflow_details` "Workflow ID (NAME)" -> "Retrieved details
  for workflow 'NAME'"; `get_workflow_input_template` "N input slot(s) for
  workflow W" -> "Built an input template for workflow 'W' (N slot(s), source:
  style=run). Fill inputs_template and invoke with
  inputs_by='step_index|step_uuid'."; `import_workflow_from_iwc` "Imported
  workflow ID (NAME)" -> "Successfully imported workflow 'TRS_ID'";
  `invoke_workflow` "Invoked workflow W (invocation I)" -> "Invoked workflow 'W'";
  `list_page_revisions` "N revision(s) for page P" -> "Retrieved N revisions for
  page 'P'"; `list_pages` "N page(s)" -> "Retrieved N pages";
  `revert_page_revision` "Reverted page P to revision R as NEW" -> "Reverted page
  'P' to revision 'R'"; `run_tool` "Submitted T to history H (N job(s))" ->
  "Started tool 'T' in history 'H'"; `run_user_tool` "Submitted user tool UUID to
  history H" -> "Started user tool 'TOOL_ID' (UUID: UUID) in history 'H'";
  `update_page` "Updated page P (FIELDS)" -> "Updated page 'P'".
- **Breaking:** `get_server_info`'s wire answer no longer says whether the version
  it reports was supplied by the caller or fetched from the server -- the other
  server has no such field and its sentence does not say it, and nobody on the
  MCP or CLI surface can supply a version anyway. The one caller who can, a
  library caller handing `createGalaxyContext` a `serverVersion`, reads it on the
  op's own result: `ServerInfo.version_source` is `"server"`, `"supplied"` or
  `"unknown"`, and comes off in the projection to the wire, the way
  `trimmedForSize` does. A version that was never asked of the server is not
  reported as if it had been.
- **New in the sentence:** `run_tool` reports the version that RAN when a version
  was asked for, read off the jobs Galaxy answered with -- " at version 0.74", or
  " at version 0.74 (not the 0.72 requested)", or " at an unreported version (0.72
  requested)" when the jobs name none or disagree. Asking Galaxy for a version
  does not get you that version; its toolbox falls back to an installed one, and
  only the reply says which ran. `get_workflow_input_template` says which of its
  two sources the template was built from, and `run_user_tool` names the Galaxy
  tool id behind the uuid.
- **Not reachable here:** the Python `run_tool` and `run_user_tool` can add two
  more clauses, "(with credentials)" when a stored credential context went with
  the run and "(inputs not pre-checked: ...)" when their schema preflight could
  not be made. Neither piece of work exists on this surface, so neither clause is
  produced; the gap is the one the golden-fixtures entry above already recorded.
- Eight golden cases were added for message branches nothing exercised: a name
  read with `dict.get` falling back to the id in three tools, a created page whose
  reply carries no id, a representation whose name is stated as null, two of
  `run_tool`'s version branches, and a keyword search with two keywords, whose
  sentence joins them. 152 cases in all.
- The nine budgeted listings are untouched: their sentences already matched, so
  no page is cut at a different row than before.

## 0.2.0 (tagged, never published -- shipped in 0.3.0)

Breaking, and the first release since the packages went up on npm. Everything in
`0.1.0` that a caller could reach still works the same way except where this
entry says otherwise.

### MCP tools take the Python parameter names

The Python MCP server and this one exposed the same tools under different
parameter names -- `get_history_details` wanted `history_id` from one and
`historyId` from the other -- so a `tools/call` payload was only ever right
against the server it was written for. `galaxy-mcp` now advertises and accepts
the Python spelling for every parameter of every tool.

- **Breaking (`galaxy-mcp`):** every tool is now closed to arguments it does not
  declare (`additionalProperties: false`, as all 46 of the Python server's tool
  schemas do), so an unrecognised name is refused rather than dropped -- and a
  camelCase one is answered with the name to use: `Unrecognized parameter:
  "historyId" -- did you mean "history_id"?`. Dropping it was the dangerous
  answer: `upload_file_from_url` with the old `historyId` would have uploaded
  into the default history. That holds for every name a caller can write,
  including `__proto__`, which the MCP SDK's own argument parser discards before
  a tool's schema ever sees it and which is refused a step earlier for that
  reason. The argument list is read once, where the call arrives, and the keys
  counted there are counted off the same copy the SDK goes on to parse -- so a
  caller that hands over a live object rather than JSON text cannot answer one
  thing to the count and another to the parser. What is counted is the own
  string keys that parser drops: `__proto__`, and any property that is not
  enumerable. A symbol key is not one a caller can write as JSON text and is not
  in that count, so an enumerable one is refused by the parser itself as an
  invalid key and a hidden one is dropped as it always was.
- A `tools/call` that sends no `arguments` field at all is accepted for a tool
  whose parameters are all optional, which is what the Python server does;
  `get_user` was refused before.
- The server tells a client the version this package was published as, rather
  than `0.0.0`.
- Keys *inside* an object-valued argument -- a workflow's `inputs`, a tool's
  parameters, a user tool's `representation` -- are the caller's own data and are
  passed to Galaxy exactly as written. One key was not, in `0.1.0`, on the two
  surfaces that validate a call against these schemas: the five parameters that
  take an object (`run_tool.inputs`, `run_user_tool.inputs`,
  `create_user_tool.representation`, `invoke_workflow.inputs` and
  `invoke_workflow.params`) were declared as zod records, a record parses by
  copying the object into a fresh one, and zod skips a key named `__proto__`
  while copying -- so a `create_user_tool` whose representation carried one
  succeeded and posted a document its caller had not written, over MCP and from
  the CLI alike. An op imported straight from `@galaxyproject/galaxy-ops` runs
  without its schema being parsed at all and never lost the key. Those five are
  copied the way a record copies now, with that key carried across instead of
  dropped, which is what the Python server's `dict[str, Any]` does with it. That
  one key is the whole of it: everything else a record does with a key still
  holds -- an own property that is not enumerable is dropped, a symbol key is
  refused, a getter is read once and what it returned is what is stored -- and
  the object itself is read once, where it arrives, so what Galaxy is posted is
  what that one read found. One thing those five refuse where `0.1.0` took it: a
  value that presents itself as a record without being one -- a Proxy around a
  class instance that answers `Object` when it is asked for its `constructor` --
  is refused with `expected record, received <Name>`, because the question is
  asked of the copy this surface took and that copy is built on the prototype
  the value really had (a class instance plain and simple was refused before and
  is refused now). The schema each of them advertises has not changed by a byte.
- `@galaxyproject/galaxy-ops` still takes camelCase: an op imported from
  TypeScript is called exactly as it was. `galaxy-cli`'s flags are kebab-cased
  from those same TypeScript names and have not moved (`--tool-version`,
  `--history-id`).
- A tool's own help names a parameter the way the surface reading it takes that
  parameter -- `section_id` over MCP, `--section-id` from the CLI -- so following
  what a tool says about itself works on the surface you read it from.

### Listings page, cap and budget themselves (#127)

The history, tool, workflow and IWC listings return one page at a time, with the
defaults, ceilings and byte budget the Python server uses -- including where it
uses none.

- **Breaking:** these listings return a page rather than everything they find.
  Ten of them return it as `{ items, pagination }` where they used to hand back
  a bare array: `get_histories`, `get_history_contents`, `list_history_ids`,
  `list_workflows`, `list_user_tools`, `search_tools_by_name`,
  `search_tools_by_keywords`, `get_iwc_workflows`, `search_iwc_workflows` and
  `recommend_iwc_workflows`, whose `pagination` carries no offset because a
  ranking is one result rather than a window. `get_tool_panel` was an object
  before and has two shapes now: `{ entries, pagination }` for the overview of
  the sections, and `{ section_id, section_name, tools, pagination }` for one
  section's tools.
- Each of the eleven names its `limit` default in its own help except
  `get_histories`, which has none -- unset still returns every history Galaxy
  sends. Nine of them refuse a `limit` over their ceiling before the call goes
  out, and cut a page that would still be too large for a model to read, saying
  so in `pagination` (`trimmedForSize`, and a sentence in `helperText`).
  `get_histories` and `get_history_contents` are the two with neither a ceiling
  nor an output budget -- the Python server caps and budgets neither of them
  either -- so a page from those two goes out at whatever size it comes to.
- `get_histories` and `get_history_contents` fetch the whole list and then
  count, filter and slice here rather than asking Galaxy to, and
  `get_history_contents` sorts here too, so their totals are real totals.
- **Breaking:** the name filters on `get_histories` and `list_workflows` match
  the whole name, as bioblend does, rather than a case-insensitive substring.
- **Breaking:** `get_tool_panel` answers with an overview of the sections and
  drills into one with `section_id` (`--section-id`), instead of returning the
  whole panel. `get_iwc_workflows` hands back the same enriched summaries
  `search_iwc_workflows` does rather than raw manifest entries, because one raw
  entry runs to hundreds of KB; the full record is one
  `get_iwc_workflow_details` call away.
- A refused window reports `errorKind: "validation"` (CLI exit code 64).

### `get_invocations` lists as well as fetches (#128)

- **Breaking:** `invocation_id` is optional. Given one, the tool returns that
  invocation's detail as before; given anything else -- a workflow, a history, or
  nothing at all -- it returns a list of summaries, so the return is now a union
  of the two.
- **Breaking (`galaxy-cli`):** `get_invocations` takes `--invocation-id` rather
  than a positional argument, because it no longer has one required input.
- `invoke_workflow` accepts `inputs` and `params` as JSON strings as well as
  objects, which is what a CLI flag and some MCP clients actually send.
- **Breaking:** when `invoke_workflow`'s preflight refuses an input Galaxy would
  have rejected -- a VCF dataset handed to a FASTQ input -- it reports
  `errorKind: "validation"` and the CLI exits 64, where it reported
  `errorKind: "connection"` and exited 69. Nothing reached Galaxy in either case,
  and the message is word for word the one it was; what changes is what a caller
  branching on the kind or the exit status does about it. `"connection"` told an
  agent to back off and retry, which for a wrong argument loops for ever. A
  non-object `inputs` or `params` is `"validation"` for the same reason.

### Arguments are decoded the way the other server decodes them (#130)

FastMCP validates a tool call with pydantic in its non-strict mode, so the Python
server has always taken `"5"` for an integer and `"yes"` for a boolean. Both
surfaces here now do the same, at argument validation, without changing what the
schemas advertise.

- `"5"`, `" 5 "`, `"05"`, `"5.0"`, `"1_000"`, `5.0` and `true` are integers;
  `"yes"`, `"off"`, `"True"`, `1` and `0` are booleans -- the same table pydantic
  accepts, including its whitespace rules. Anything outside it is still refused.
- **Breaking:** `get_collection_details` truncates the element list whether or
  not you asked it to, which is what the Python tool has always done.
  `max_elements` declares its default of 100, and where omitting it used to hand
  back every element a collection had -- 101 of 101 -- it now hands back 100.
  Ask for more by asking.
- **Breaking:** the four schemas still built on `z.coerce.number()` --
  `get_collection_details`'s `max_elements`, `get_workflow_details`'s `version`
  and `list_pages`'s `limit` and `offset` -- are plain integers now, so what they
  take is the table above rather than whatever JavaScript's `Number()` makes of
  it. `"5"` and `true` still arrive as 5 and 1, and `false` still arrives as 0 on
  all four, because that is what pydantic makes of it. `"0x10"`, `"1e2"` and
  `[1]` are refused on all four where they used to become 16, 100 and 1. `""`
  and `[]` used to become 0 on `version` and `offset` and are refused now; on
  `max_elements` and `limit` they were already refused, because the 0 they
  coerced to failed those two schemas' positive floor. `null` is the one that
  changed meaning: `version: null` says "no version" rather than version 0,
  `offset: null` is refused where it used to be 0, and `max_elements: null` and
  `limit: null` were refused before and are refused now. (`list_pages` arrived
  after 0.1.0, so no published version ever coerced or floored its two.)
- **Breaking:** the floors those four carried go with the coercion, because the
  Python schemas declare bare integers and validate neither -- so a value this
  surface used to refuse now runs. A negative `version` or `offset` goes out to
  Galaxy as written. `max_elements: 0`, which `false` also decodes to, is
  accepted and hands back an element list with nothing in it; `list_pages`'s
  `limit: 0` is accepted and asks Galaxy for an empty page. Both of those were
  refused before. The nine capped listings validate their own window and still
  refuse one, with `errorKind: "validation"`.
- Ten other parameters declare in the advertised schema the default their op was
  already applying, so an agent reading a tool can see it; what those ten do is
  unchanged.
- An argument list that arrives as a live object rather than as JSON text is
  decoded even when this realm did not parse it: whether a value is an argument
  list at all is zod's own question now, the one the SDK's parser asks before it
  takes the object, and not a test of the prototype's identity -- so `"5"` in an
  object a `vm` context parsed becomes 5 the way it does from any other peer. A
  Date, a Map and a class instance are still not argument lists, and a call
  carrying one is still refused rather than run on its defaults. The container
  holding the call is read the way the SDK reads it too: a `params` that is a
  class instance, or that keeps `name` on a prototype, is a call the SDK runs, so
  it is a call this decodes and inspects -- and every field of it is read once,
  at the transport, so what is inspected is what runs. A caller sending JSON text
  is unaffected by any of this, while an in-process caller handing live objects
  over an `InMemoryTransport` has each of the fields those schemas declare, and
  each own enumerable key of the containers holding them, read exactly once, at
  the boundary, and what was read is what runs -- an object that answers
  differently when asked again has nothing further to say about those. What sits
  inside `_meta` or `task`, or on a prototype, is passed on rather than copied
  and is read by the SDK and by zod as often as they read it.

### `get_dataset_details` previews the dataset by default (#134)

- **Breaking:** `get_dataset_details` asks Galaxy for the dataset's own text
  peek (`GET /api/datasets/{id}/get_content_as_text`) whenever the dataset is in
  the `ok` state, and returns it as `preview` beside the dataset payload -- the
  first ten lines by default -- where `0.1.0` returned the payload alone and
  made no second request. `include_preview: false` (`includePreview` from
  galaxy-ops, `--no-include-preview` from the CLI) restores the single request;
  `preview_lines` sets the slice. The preview has the three shapes the Python
  tool returns: the text, `lines: null` with an `error` and Galaxy's own
  truncation flag for a datatype Galaxy has no text for, and `lines: null` with
  an `error` alone when the peek could not be taken -- a failed peek is reported
  in `preview`, never as a failure of the call.
- `download_dataset` declares `require_ok_state` (default true) and takes
  `use_default_filename` (default true) the way the Python tool takes it, which
  is to say it changes nothing: the flag is inert on both servers, and neither
  writes a file the caller did not name. Its description no longer claims the
  bytes come back "in memory" without `file_path`; what comes back then is
  metadata (`file_size`, `suggested_filename`, `content_available`), and
  `file_path` is the only way to get the content.
- `upload_file_from_url` declares the `file_type` (`"auto"`) and `dbkey` (`"?"`)
  defaults it was already applying.

### `recommend_biocontainer`, and a command line for list parameters (#136)

- `recommend_biocontainer` resolves a verified `quay.io/biocontainers` image for
  a list of conda packages (`samtools=1.17`, `bwa`), the way the Python server's
  tool of the same name does, and is the container to give `create_user_tool`
  instead of a guessed tag. It is a port of Galaxy's mulled recommender:
  the mulled-v2 name and version hashes, PEP 440 ordering with the legacy
  fallback, exact-or-newest selection, a five-minute cache, and Python's error
  model reply for reply -- which quay.io answers mean "no container", which
  become a failed lookup with the reason in `notes`, which fall back to the
  newest build along the way, and which are an error -- pinned by tests against
  Galaxy's own cases rather than restated here. It reaches quay.io directly,
  which is the second host besides Galaxy
  these packages talk to (the IWC manifest is the other). Where Node's runtime
  answers differently from CPython's on inputs a registry does not send --
  codec tables beyond UTF-8/16/32 and Latin-1, the JSON decoder's own error
  text and limits, socket timing at the edge of the twelve-second budget -- the
  difference is listed at the top of `galaxy-ops/src/mulled.ts` rather than
  ported.
- **Breaking (`galaxy-cli`):** a list-typed parameter is a repeatable flag.
  `search_tools_by_keywords` took its keywords as a positional that could only
  ever arrive as one string, which its schema refused, so the command had no
  working form; it takes `--keywords bwa concatenate` (or `--keywords bwa
  --keywords concatenate`) now, `update_history --tags` works the same way, and
  `recommend_biocontainer --packages samtools=1.17 bwa` runs end to end.

### Also since 0.1.0

- Pages: `list_pages`, `get_page`, `create_page`, `update_page`,
  `list_page_revisions`, `get_page_revision` and `revert_page_revision` (#112).
- Operations that need a newer Galaxy than the oldest these tools speak to
  declare it, and are refused before anything is sent: `errorKind: "version"`
  over MCP, exit code 76 from the CLI, and the bound in the tool description.
  `get_server_info` reports what the connected server cannot run (#114).
- `galaxy-ops` has an entry a browser can run. The default one registers every
  operation, and `download_dataset`, `upload_file` and `recommend_biocontainer`
  need a filesystem or `node:crypto`, so a bundler pulling the package in for
  the rest got the Node builtins with them. A bundler resolving the `browser`
  export condition, or an import of `@galaxyproject/galaxy-ops/browser`, now
  gets everything but those three and no `node:` import in the bundle.
- `galaxy-ops`'s `typescript` peer dependency is optional and accepts `>=5.5`
  rather than `>=6`, so a project on TypeScript 5 can use the emitted
  declarations, which need nothing newer.

## 0.1.0 (2026-06-26)

First release of the three packages on npm.
