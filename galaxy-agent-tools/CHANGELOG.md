# Changelog

`@galaxyproject/galaxy-ops`, `@galaxyproject/galaxy-cli` and
`@galaxyproject/galaxy-mcp` share a version and are published together, so one
entry covers all three; where something only affects one surface, it says which.

## 0.3.0 (unreleased)

Breaking on both surfaces. 0.2.0 has not been published, so one release will
carry both entries.

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
server and the CLI each replay all 68 of those cases against those replies and
compare keys, `data`, `count` and `pagination` exactly, plus `message` for the
nine listings above. Nothing is skipped on either surface, the pages the budget
cut included -- which is also why `galaxy-cli` measures the budget against the
compact line the other surfaces measure and prints indented afterwards, rather
than measuring its own indentation and cutting a shorter page than MCP would
for the same call.

## 0.2.0 (unreleased)

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
