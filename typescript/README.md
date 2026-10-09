# Galaxy MCP -- TypeScript

A TypeScript toolkit for driving the [Galaxy](https://galaxyproject.org/)
bioinformatics platform from the command line and from AI agents. It exposes
Galaxy's core operations -- histories, datasets, tools, workflows, invocations,
pages, the IWC catalog -- two ways, both built on a single shared core:

- **`galaxy-cli`** -- a command-line tool. One subcommand per operation, with
  table / JSON / plain-text output and meaningful exit codes. Good for scripts,
  CI, and poking at a Galaxy server by hand.
- **`galaxy-mcp`** -- a [Model Context Protocol](https://modelcontextprotocol.io)
  server (Node). Exposes the same operations as MCP tools over stdio so an
  MCP-aware assistant (Claude Desktop, etc.) can use them.

This is the TypeScript sibling of the Python MCP server in
[`../python`](../python). The operation set is kept in
lockstep with it, so a tool called `get_histories` here behaves like
`get_histories` there.

> The packages are published on npm under the
> [`@galaxyproject`](https://www.npmjs.com/org/galaxyproject) scope -- install
> them (below), or build from source to develop. All three share a version and
> are released together; [CHANGELOG.md](CHANGELOG.md) is what changed, including
> what breaks.

## Layout

A pnpm workspace with three packages:

| Package | What it is |
| --- | --- |
| `packages/galaxy-ops` | The framework-free core: typed Galaxy client, the operation registry, error model, and orchestration (tool runs, job polling). |
| `packages/galaxy-cli` | The `galaxy-cli` command-line surface. |
| `packages/galaxy-mcp` | The `galaxy-mcp` MCP server surface. |

The two surfaces are thin: each iterates the same operation registry, so a new
operation shows up in both the CLI and the MCP server with no surface-specific
code.

## Requirements

- Node.js `>=22.19`
- [pnpm](https://pnpm.io) `9.12` (`corepack enable` will provide it)
- A Galaxy server URL and an API key (Galaxy: **User -> Preferences -> Manage API Key**)

## Install

```bash
# the CLI -- install globally for a `galaxy-cli` command, or run via npx:
npm install -g @galaxyproject/galaxy-cli
npx @galaxyproject/galaxy-cli --help

# the MCP server:
npx @galaxyproject/galaxy-mcp
```

The core is a library you can depend on directly:

```bash
npm install @galaxyproject/galaxy-ops
```

### From source (for development)

```bash
cd typescript
pnpm install
pnpm -r build        # compiles each package to dist/
```

The built entry points are `packages/galaxy-cli/dist/index.js` and
`packages/galaxy-mcp/dist/index.js`; run them with `node <path>`.

## Connecting to Galaxy

Both tools need a **base URL** and an **API key**. The CLI looks for them in this
order (first match wins); the MCP server uses environment variables only:

1. CLI flags: `--url <url>` and `--api-key <key>`
2. Environment: `GALAXY_URL` and `GALAXY_API_KEY`
3. A `.env` file in the current directory (same two variable names)
4. A [planemo](https://planemo.readthedocs.io) profile: `--profile <name>` reads
   `~/.planemo.yml` (uses that profile's `galaxy_url` and `galaxy_user_key`, or
   `galaxy_admin_key`). A top-level `galaxy_url`/`galaxy_user_key` in that file is
   treated as the `default` profile.

```bash
export GALAXY_URL=https://usegalaxy.org/
export GALAXY_API_KEY=your-api-key
```

If neither a URL nor a key can be found, the CLI exits with a usage error
explaining what to set.

## Using the CLI

```
galaxy-cli [global options] <command> [arguments]
```

List commands and get help at any level:

```bash
galaxy-cli --help
galaxy-cli run_tool --help
```

### Global options

| Option | Description |
| --- | --- |
| `--url <url>` | Galaxy base URL |
| `--api-key <key>` | Galaxy API key |
| `--profile <name>` | planemo profile name from `~/.planemo.yml` |
| `--format <fmt>` | Output format: `table` (default), `json`, or `text` |
| `--quiet` | Suppress the status/summary line on stderr |
| `--timeout <ms>` | Poll timeout for blocking operations (e.g. `run_tool`) |

### How command arguments work

Each command's arguments come from that operation's inputs:

- **Required values are positional.** e.g. `get_history_details <historyId>`
- **Optional values are flags.** e.g. `get_histories --limit 10 --name rnaseq`
  (camelCase inputs become kebab-case flags: `toolVersion` -> `--tool-version`)
- **Booleans are bare flags.** e.g. `get_history_contents <historyId> --deleted`
- **Lists are repeatable flags,** required or not. e.g.
  `recommend_biocontainer --packages samtools=1.17 --packages bwa`, or the same
  thing as `--packages samtools=1.17 bwa`
- **Structured inputs take JSON** -- inline or from a file with `@`:
  `run_tool cat1 <historyId> --inputs '{"input1":{"src":"hda","id":"abc123"}}'`
  or `--inputs @inputs.json`

### Output and exit codes

`--format table` (default, also accepts `text`) prints a compact table for lists
and a key/value block for single objects, with the status line on stderr.
`--format json` prints the full result envelope
(`{ data, success, message, count, pagination }`, the same one the MCP surface
sends -- see [The result envelope](#the-result-envelope)) pretty-printed -- use
this for scripting. A failure prints
`{ success: false, message, errorKind }` instead, which is this surface's own
shape: the MCP surface sends an MCP error there instead of an envelope. `--quiet`
drops the stderr status line.

The process exit code reflects the outcome, following `sysexits.h` conventions:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `64` | Usage error (bad flags, failed input validation, a run Galaxy refused with a 400) |
| `66` | Not found |
| `65` | Tool request rejected |
| `69` | Connection / server unavailable |
| `70` | Software error or job failure |
| `76` | The connected Galaxy is too old for the operation |
| `77` | Authentication failure (bad/missing API key) |

### Examples

```bash
CLI="galaxy-cli"   # or: CLI="npx @galaxyproject/galaxy-cli"

# Who am I?
$CLI get_user

# List histories as JSON, filtered by name
$CLI --format json get_histories --name rnaseq

# Create a history and upload a file into it
$CLI create_history "My new analysis"
$CLI upload_file ./reads.fastq.gz --history-id <historyId>

# Find a tool, then inspect how to call it
$CLI search_tools_by_keywords --keywords fastqc
$CLI get_tool_input_template toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74

# Run a tool and wait for it to finish (10 min timeout)
$CLI --timeout 600000 run_tool cat1 <historyId> --inputs @inputs.json

# Workflows: see the expected inputs, then invoke
$CLI get_workflow_input_template <workflowId>
$CLI invoke_workflow <workflowId> --inputs @wf-inputs.json --history-name "WF run"

# Download a result to disk
$CLI download_dataset <datasetId> --file-path ./result.txt
```

> Tip: for tool and workflow runs, call `get_tool_input_template` /
> `get_workflow_input_template` first -- they return a ready-to-fill input
> skeleton describing each expected input, so you can shape `--inputs` correctly.

## Using the MCP server

`galaxy-mcp` speaks MCP over **stdio** and reads its connection from the
environment. It registers every operation as an MCP tool (read-only operations
are flagged with `readOnlyHint`).

Tool parameters are spelled the way the Python MCP server spells them, so one
`tools/call` payload works against either server:

```json
{
  "method": "tools/call",
  "params": {
    "name": "get_history_details",
    "arguments": { "history_id": "f2db41e1fa331b3e" }
  }
}
```

Keys inside an object-valued argument -- a workflow's `inputs`, a tool's
parameters, a user tool's `representation` -- are your own data and go to Galaxy
as written. Every tool is closed to names it does not declare, so an argument
nobody asked for is refused rather than ignored, and a camelCase one is answered
with the name to use instead. The camelCase names `galaxy-ops` uses in TypeScript
are therefore not parameters of the MCP server (they were until 0.2.0); the CLI's
flags come off those TypeScript names and are unchanged. A tool's own help names
a parameter the way the surface you are reading it from takes it -- `section_id`
over MCP, `--section-id` from the CLI -- so a tool's advice about itself is
advice you can follow.

Run it directly to sanity-check:

```bash
GALAXY_URL=https://usegalaxy.org/ GALAXY_API_KEY=your-api-key \
  npx @galaxyproject/galaxy-mcp
# -> "galaxy-mcp: connected on stdio"
```

More usefully, register it with an MCP client. For example, in Claude Desktop's
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "galaxy": {
      "command": "npx",
      "args": ["-y", "@galaxyproject/galaxy-mcp"],
      "env": {
        "GALAXY_URL": "https://usegalaxy.org/",
        "GALAXY_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Available operations

Both surfaces expose the same set -- CLI command names and MCP tool names are
identical. Operations marked *(write)* change state -- on the server, or, for
`download_dataset` with a `filePath`, on the disk of the machine the caller runs
on; the rest are read-only. The MCP `readOnlyHint` follows this marker, because
the hint is about whether a tool modifies its environment rather than about
Galaxy alone.

The history, tool, workflow and IWC listings return one page at a time. `limit`
and `offset` default to a page sized for what a model can actually read, each
operation's help says what its default is, and the result carries a `pagination`
block with the offset to ask for next. Nine of them also have a ceiling -- the
same one the Python server enforces -- and refuse a bigger `limit` before the
call goes out; a page that would still be too many bytes is cut to fit and says
so. Three do not fit that description: `get_histories` has no default limit and
no ceiling, so leaving `limit` off returns everything there is;
`get_history_contents` has a default but no ceiling, and neither of those two has a
byte budget either, exactly as on the Python side; and `recommend_iwc_workflows`
takes a `limit` but no `offset`, because a ranking is cut from the bottom rather
than paged through. The two history listings count their own items and filter,
sort and slice them here rather than asking Galaxy to, which is what the Python
tools do and is why they can report a real total. The Pages operations sit half
in: `list_pages` takes `limit` and `offset` and reports the total the server
counted, on a `total_matches` header, with the same helper sentence the other
listings carry; `list_page_revisions` returns every revision at once.

### The result envelope

Both surfaces answer with the Python MCP server's envelope, key for key. Over
MCP it is the text content block; from the CLI it is what `--format json`
prints.

```json
{
  "data": [{ "id": "f2db41e1fa331b3e", "name": "RNA-seq" }],
  "success": true,
  "message": "Retrieved 1 of 40 histories",
  "count": 1,
  "pagination": {
    "total_items": 40,
    "returned_items": 1,
    "limit": 1,
    "offset": 10,
    "has_next": true,
    "has_previous": true,
    "next_offset": 11,
    "previous_offset": 9,
    "helper_text": "Showing 1 of 40 histories (offset 10). Use offset=11 for the next page."
  }
}
```

`data` is the answer itself -- for a listing, the page, not a wrapper around it.
`message` is the Python server's sentence for that tool, word for word, which is
why it reads as prose rather than as a field dump.
`count` is how many rows are in this answer, for the tools that count. `count`
and `pagination` are always there, `null` where the tool has neither, so a
missing key never has to be told apart from "this tool does not page". A page
cut short to fit the output budget says so in `helper_text`; a ranking, which
has no pagination block, says it in `message`.

A TypeScript caller importing an operation from `@galaxyproject/galaxy-ops`
directly gets none of this: `run()` returns its own typed data, and a listing
returns `Paged<T>` with camelCase pagination, unchanged.

#### When a call fails

There is no envelope. The Python server raises, FastMCP turns the exception into an
MCP error result, and this surface sends the same thing: `isError` true and one text
block carrying `Error calling tool '<name>': ` and then the tool's own sentence.

```
Error calling tool 'get_page': Get page failed: 404 Client Error: Not Found for url: https://galaxy.example/api/pages/p1 (Resource not found - check IDs and URLs). Context: page_id=p1
```

Nothing in there is JSON, so read `isError` and the text rather than parsing it. The
sentence is that server's, which includes the prose of the client library it used: a
bioblend GET quotes Galaxy's reply twice, once as a Python `bytes` repr, and a
request made with requests names the URL it could not read. Around it goes the
tool's action, the hint for 401, 403, 404 and 500, and its context dict.

`galaxy-cli --format json` prints its own shape instead, because a command line has
to say something an exit code can be read off:

```json
{ "success": false, "message": "Get page failed: 404 Client Error: ...", "errorKind": "not_found" }
```

`message` is the same sentence without FastMCP's wrapper; `errorKind` is what the
exit code comes from (see [Output and exit codes](#output-and-exit-codes)). A TypeScript caller catches a
`GalaxyError` carrying the library's own short message and its `kind`.

### Connection
| Operation | What it does |
| --- | --- |
| `get_user` | Current authenticated user (id, email, username) |
| `get_server_info` | Connected Galaxy's URL, version, public configuration, and the operations it is too old to run |

### Histories
| Operation | What it does |
| --- | --- |
| `get_histories` | List your histories (id, name, counts); optional name filter |
| `list_history_ids` | Compact id + name list of histories |
| `get_history_details` | One history by id (name, state, counts) |
| `get_history_contents` | List the datasets and collections in a history |
| `create_history` *(write)* | Create a new history |
| `update_history` *(write)* | Update history metadata (name, annotation, tags, deleted, published) |

### Datasets & collections
| Operation | What it does |
| --- | --- |
| `get_dataset_details` | Dataset metadata by id (state, extension, name), with an optional content preview from Galaxy's bounded text route |
| `get_collection_details` | A dataset collection by id, with its elements |
| `get_job_details` | A job by `--job-id`, or the job that produced a `--dataset-id` (one of the two); `--full` adds its stdout and stderr, job messages, dependencies and metrics |
| `list_jobs` | A page of jobs from Galaxy's job index, filtered by history, state and update time, newest first; no total, so a short page is the last one. An unknown history is an empty page, not a 404 |
| `download_dataset` *(write)* | Download a dataset's content; with `--file-path` it writes the bytes to that local path, overwriting what is there -- the caller's disk, not the server, which is why the write marker is here while Python's tag says read. Without `--file-path` you get the metadata and the byte count and no content, so `--file-path` is the only way to the bytes |
| `upload_file` *(write)* | Upload a local file via the tus resumable-upload protocol |
| `upload_file_from_url` *(write)* | Upload a file from a URL via the classic upload tool |

### Tools
| Operation | What it does |
| --- | --- |
| `search_tools_by_name` | Search tools by name, id, or description substring |
| `search_tools_by_keywords` | Search tools by keywords (name, description, input extensions) |
| `get_tool_details` | A tool's metadata by id (name, version, description); `--tool-version` describes one installed version and is refused if Galaxy answers with another |
| `get_tool_panel` | The tool panel's sections with their tool counts; name one with `--section-id` to list its tools |
| `get_tool_citations` | Citations for a tool by id |
| `get_tool_run_examples` | Test-data examples (inputs/outputs) for a tool |
| `get_tool_input_template` | A ready-to-fill inputs skeleton for a tool (call before `run_tool`) |
| `run_tool` *(write)* | Run a tool and wait until its jobs reach a terminal state |
| `recommend_biocontainer` | Resolve a verified `quay.io/biocontainers` image for a set of conda packages -- how to pick the `container` for `create_user_tool` instead of guessing one |

`recommend_biocontainer` asks quay.io, which is the second host besides Galaxy these operations
reach -- the other is `iwc.galaxyproject.org`, which the IWC operations fetch their manifest
from -- so outbound access has to allow both. It answers from a five-minute cache, failures
included. Its budget for a silent registry is a little wider than Galaxy's Python tool, which
allows twelve seconds to connect and then a fresh twelve for the first byte: `fetch` cannot see
the connect phase separately, so this allows twenty-four seconds to the first byte and then
twelve between chunks. Three edges follow. A server that connects at once and then says nothing
fails in Python at twelve seconds and here at twenty-four. A connection that takes between
Node's own ten-second connect limit and Python's twelve fails here and succeeds there. And a
server whose header lines keep arriving past the doubled window answers in Python, whose budget
is renewed by every read, and is abandoned here. All three need a registry that is up but silent
or dawdling for over ten seconds, and either way the answer is a failed lookup rather than a
wrong image -- so we take the wider budget rather than pin the package to one runtime's HTTP
internals. Those three edges and the rest are listed together as the accepted divergences at the
top of `packages/galaxy-ops/src/mulled.ts`, each with the input that differs. They come from four
places and stay there by decision rather than by oversight: the codec tables, where only UTF-8,
UTF-16, UTF-32 and latin-1 answer as Python's codecs do; the JSON decoder, whose wording,
positions, depth limit and error ordering on a body that will not parse are V8's; the HTTP
timing above; and the degenerate responses on which Galaxy's own recommender raises, where this
does not promise the same exception -- one such response is skipped here, another raises on both
sides under different names.

### User-defined tools
| Operation | What it does |
| --- | --- |
| `list_user_tools` | List the current user's user-defined tools |
| `get_user_tool` | One user-defined tool by uuid, with its full representation; a value that is not a uuid is refused as not found before anything is sent |
| `create_user_tool` *(write)* | Create a user-defined tool from a tool representation |
| `delete_user_tool` *(write)* | Deactivate a user-defined tool by uuid (soft delete) |
| `run_user_tool` *(write)* | Run a user-defined tool (lookup, then POST to the tools API) |

### Workflows & invocations
| Operation | What it does |
| --- | --- |
| `list_workflows` | List stored workflows; optional name + published filter |
| `get_workflow_details` | One stored workflow by id (name, steps, inputs) |
| `get_workflow_input_template` | A ready-to-fill input template + run guide, at a given stored version (call before `invoke_workflow`) |
| `invoke_workflow` *(write)* | Invoke a workflow with inputs/parameters (validates inputs first) |
| `get_invocations` | One invocation by id (state, steps), or the invocations of a workflow or history; pages with `limit` (at most 100, Galaxy's cap) + `offset`, sorts by `create_time` or `update_time`, and `includeTerminal: false` keeps only the ones still running |
| `cancel_workflow_invocation` *(write)* | Cancel a running workflow invocation |

### IWC (Intergalactic Workflow Commission) catalog
| Operation | What it does |
| --- | --- |
| `get_iwc_workflows` | Browse curated IWC workflows, a page of summaries at a time (full record: `get_iwc_workflow_details`) |
| `get_iwc_workflow_details` | Full details (inputs, outputs, readme) for an IWC workflow by TRS id |
| `search_iwc_workflows` | Search curated IWC workflows by substring |
| `recommend_iwc_workflows` | Rank IWC workflows by relevance to a free-text intent (BM25) |
| `import_workflow_from_iwc` *(write)* | Import an IWC curated workflow into the connected Galaxy |

### Pages (notebooks & reports)
| Operation | What it does |
| --- | --- |
| `list_pages` | List pages (markdown notebooks and reports); filter to one history's notebooks |
| `get_page` | One page with its editable `content_editor` markdown |
| `create_page` *(write)* | Create a standalone report, or a notebook attached to a history |
| `update_page` *(write)* | Update a page's content, creating a new revision, or just its title, which does not |
| `list_page_revisions` | A page's revision history, newest or oldest first |
| `get_page_revision` | One revision: the editable `content_editor`, the expanded `content`, and which of the two the editable text came from |
| `revert_page_revision` *(write)* | Restore an earlier revision by writing it back as a new one |

Six of the seven declare Galaxy 26.1 or newer (see
[Version requirements](#version-requirements)) -- each one because 26.0 cannot actually serve
it, not as a blanket rule for the set:

| Operation | On a 26.0 server |
| --- | --- |
| `get_page` | **Works** -- 26.0 already returns `content_editor`, so it is not gated |
| `list_pages` | Endpoint works, but `--history-id` is silently ignored and you get *every* page |
| `create_page` *(write)* | Makes a report, never a notebook, and answers without `content_editor` |
| `update_page` *(write)* | Cannot change content at all: 26.0's update payload has no content field |
| `list_page_revisions`, `get_page_revision`, `revert_page_revision` | No such endpoint |

`list_pages` is gated for the filter rather than the endpoint: answering a request for one
history's notebooks with every page on the server is worse than refusing.

`content_editor` on a *revision* is newer still and arrives after 26.1; where a server does
not send it the ops fill it from that revision's `content`, which has its embeds already
expanded, and say which it was in `content_editor_source`.

### Version requirements

A few operations need a Galaxy newer than the oldest one these tools speak to. Each
declares its own minimum; `galaxy-cli <command> --help` and the MCP tool description
say what it is, and the check happens *before* anything is sent -- so a write can
never half-succeed against a server that was never going to accept it. An operation is
gated only where the server really cannot serve it; one that merely returns less on an
older Galaxy says so in its description instead.

```bash
$ galaxy-cli list_page_revisions f2db41e1fa331b3e
list_page_revisions needs Galaxy 26.1 or newer; this server reports 26.0   # on stderr
$ echo $?
76
```

The explanation goes to stderr, so `--quiet` leaves you with exit code 76 alone. Over MCP
the same refusal arrives as an MCP error carrying the Python server's sentence, which ends
"Nothing was sent to Galaxy."

The version comes from `/api/version`, asked once per connection. A server that will
not answer leaves its version unknown, and an unknown version refuses nothing: the
operation is attempted and stands or falls on its own. `get_server_info` reports what
the connected server cannot run in `unsupported_ops`, and says in `version_known`
whether an empty list means anything.

Run `galaxy-cli <command> --help` for the exact arguments of any operation.

## Development

```bash
pnpm -r typecheck    # tsc --noEmit (strict)
pnpm -r test         # vitest
pnpm depcruise       # enforce the surface -> core import boundary
pnpm -r build        # tsup -> dist/
pnpm parity:report   # regenerate PARITY.md (see below)
```

### Parity with the Python server

Lockstep with the Python MCP server is a check rather than a good intention:
`pnpm -r test` compares what this package advertises against that server's
generated surface manifest -- which tools exist, what parameters they take, their
types, requiredness and declared defaults, whether a tool takes parameters it does
not declare, whether it says it changes anything, and what it says it needs from
the Galaxy it runs against. Anything the
two disagree about has to be listed in
`../contract/accepted-divergences.json` with a status and a
reason, and an entry the surfaces no longer support fails the check too, so the
list cannot quietly rot. The `unreviewed-gap` status -- the comparator found it and
nobody has read both sides -- is ratcheted: the registry pins how many of those it
may hold, and one more needs a reviewed status and a reason rather than a bigger
number. [`PARITY.md`](../PARITY.md) in the repository root is that whole state as a
table, one row per tool and one per parameter the two disagree about; regenerate it
with `pnpm parity:report` when a change moves parity. CI holds the checked-in copy
to the generator and prints the same table into the job summary, so the movement
arrives in the diff instead of waiting for somebody to go looking.

### The third column: Galaxy's own MCP server

Galaxy ships an MCP server of its own -- `lib/galaxy/webapps/galaxy/api/mcp.py`,
served when `enable_mcp_server` is set -- its own implementation, in process over
Galaxy's operations manager, where these tools go through the REST API. `PARITY.md`
has it as a third column, compared against the Python
surface the same way, with its differences recorded in the `builtin` section of the
registry and ratcheted apart from the two above: they are not this repository's to
close, they are a list to take to galaxyproject/galaxy.

That column comes from a snapshot rather than a live read, because reading it needs
a Galaxy checkout and CI has none. It is refreshed by hand, and it says which Galaxy
commit it describes:

```bash
export GALAXY_ROOT=/path/to/galaxy         # a Galaxy checkout; only ever read
cd /path/to/galaxy-mcp                     # this repository's root, not this package
"$GALAXY_ROOT/.venv/bin/python" -B python/tests/builtin_surface.py
cd typescript
pnpm parity:report                         # then regenerate the table
```

Four things that one-liner would have got wrong. The assignment is its own line
because a `VAR=x cmd` prefix does not reach the `$VAR` in its own arguments -- the
shell expands those first, so the one-liner runs `/.venv/bin/python`. The two
commands want different directories, and only the second one wants this package's.
The interpreter is named rather than found: it has to be one that can import Galaxy,
which in a checkout set up by `run.sh` or `make setup` is the `.venv` under the
checkout, and anywhere else is whatever environment has Galaxy installed -- nothing
goes looking, so a checkout without a `.venv` fails with "no such file" before the
generator starts. And `-B` keeps the run from leaving `__pycache__` in the checkout,
which importing Galaxy otherwise does; the generator also sets
`sys.dont_write_bytecode`, and the flag is the half that holds whatever else is on
the way in.

The generator needs no Galaxy server, no database and no credentials: it imports
the module, hands it a stub app and reads what the tools declare. The Galaxy
checkout is only ever read.

## License

See [LICENSE](../LICENSE) in the repository root.
