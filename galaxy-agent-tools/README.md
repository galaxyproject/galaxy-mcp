# galaxy-agent-tools

A TypeScript toolkit for driving the [Galaxy](https://galaxyproject.org/)
bioinformatics platform from the command line and from AI agents. It exposes
Galaxy's core operations -- histories, datasets, tools, workflows, invocations,
the IWC catalog -- two ways, both built on a single shared core:

- **`galaxy-cli`** -- a command-line tool. One subcommand per operation, with
  table / JSON / plain-text output and meaningful exit codes. Good for scripts,
  CI, and poking at a Galaxy server by hand.
- **`galaxy-mcp`** -- a [Model Context Protocol](https://modelcontextprotocol.io)
  server (Node). Exposes the same operations as MCP tools over stdio so an
  MCP-aware assistant (Claude Desktop, etc.) can use them.

This is the TypeScript sibling of the Python MCP server in
[`../mcp-server-galaxy-py`](../mcp-server-galaxy-py). The operation set is kept in
lockstep with it, so a tool called `get_histories` here behaves like
`get_histories` there.

> The packages are published on npm under the
> [`@galaxyproject`](https://www.npmjs.com/org/galaxyproject) scope -- install
> them (below), or build from source to develop.

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
cd galaxy-agent-tools
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
- **Structured inputs take JSON** -- inline or from a file with `@`:
  `run_tool cat1 <historyId> --inputs '{"input1":{"src":"hda","id":"abc123"}}'`
  or `--inputs @inputs.json`

### Output and exit codes

`--format table` (default, also accepts `text`) prints a compact table for lists
and a key/value block for single objects, with the status line on stderr.
`--format json` prints the full result envelope (`{ data, success, message, ... }`)
pretty-printed -- use this for scripting. `--quiet` drops the stderr status line.

The process exit code reflects the outcome, following `sysexits.h` conventions:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `64` | Usage error (bad flags / failed input validation) |
| `66` | Not found |
| `65` | Tool request rejected |
| `69` | Connection / server unavailable |
| `70` | Software error or job failure |
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
$CLI search_tools_by_keywords fastqc
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
identical. Operations marked *(write)* create or change state on the server;
the rest are read-only.

### Connection
| Operation | What it does |
| --- | --- |
| `get_user` | Current authenticated user (id, email, username) |
| `get_server_info` | Connected Galaxy's URL, version, and public configuration |

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
| `get_dataset_details` | Dataset metadata by id (state, extension, name) |
| `get_collection_details` | A dataset collection by id, with its elements |
| `get_job_details` | Job that produced a given dataset |
| `download_dataset` *(write)* | Download a dataset's content, optionally to a local file |
| `upload_file` *(write)* | Upload a local file via the tus resumable-upload protocol |
| `upload_file_from_url` *(write)* | Upload a file from a URL via the classic upload tool |

### Tools
| Operation | What it does |
| --- | --- |
| `search_tools_by_name` | Search tools by name, id, or description substring |
| `search_tools_by_keywords` | Search tools by keywords (name, description, input extensions) |
| `get_tool_details` | A tool's metadata by id (name, version, description) |
| `get_tool_panel` | The full Galaxy tool panel (nested sections) |
| `get_tool_citations` | Citations for a tool by id |
| `get_tool_run_examples` | Test-data examples (inputs/outputs) for a tool |
| `get_tool_input_template` | A ready-to-fill inputs skeleton for a tool (call before `run_tool`) |
| `run_tool` *(write)* | Run a tool and wait until its jobs reach a terminal state |

### User-defined tools
| Operation | What it does |
| --- | --- |
| `list_user_tools` | List the current user's user-defined tools |
| `create_user_tool` *(write)* | Create a user-defined tool from a tool representation |
| `delete_user_tool` *(write)* | Deactivate a user-defined tool by uuid (soft delete) |
| `run_user_tool` *(write)* | Run a user-defined tool (lookup, then POST to the tools API) |

### Workflows & invocations
| Operation | What it does |
| --- | --- |
| `list_workflows` | List stored workflows; optional name + published filter |
| `get_workflow_details` | One stored workflow by id (name, steps, inputs) |
| `get_workflow_input_template` | A ready-to-fill input template + run guide (call before `invoke_workflow`) |
| `invoke_workflow` *(write)* | Invoke a workflow with inputs/parameters (validates inputs first) |
| `get_invocations` | A workflow invocation by id (state, steps) |
| `cancel_workflow_invocation` *(write)* | Cancel a running workflow invocation |

### IWC (Intergalactic Workflow Commission) catalog
| Operation | What it does |
| --- | --- |
| `get_iwc_workflows` | Fetch all workflows from the IWC manifest (raw) |
| `get_iwc_workflow_details` | Full details (inputs, outputs, readme) for an IWC workflow by TRS id |
| `search_iwc_workflows` | Search curated IWC workflows by substring |
| `recommend_iwc_workflows` | Rank IWC workflows by relevance to a free-text intent (BM25) |
| `import_workflow_from_iwc` *(write)* | Import an IWC curated workflow into the connected Galaxy |

Run `galaxy-cli <command> --help` for the exact arguments of any one.

## Development

```bash
pnpm -r typecheck    # tsc --noEmit (strict)
pnpm -r test         # vitest
pnpm depcruise       # enforce the surface -> core import boundary
pnpm -r build        # tsup -> dist/
```

## License

See [LICENSE](../LICENSE) in the repository root.
