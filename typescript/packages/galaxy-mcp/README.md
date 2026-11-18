# @galaxyproject/galaxy-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server (Node) that
exposes [Galaxy](https://galaxyproject.org/) agent operations -- histories,
datasets, tools, workflows, pages, and the IWC catalog -- as MCP tools over stdio or
Streamable HTTP. Built
on [`@galaxyproject/galaxy-ops`](https://www.npmjs.com/package/@galaxyproject/galaxy-ops).

## Install / run

```bash
GALAXY_URL=https://usegalaxy.org/ GALAXY_API_KEY=your-api-key \
  npx @galaxyproject/galaxy-mcp
# -> "galaxy-mcp: connected on stdio"
```

Requires Node.js `>=22.19`. The server reads `GALAXY_URL` and `GALAXY_API_KEY`
from the environment and speaks MCP over stdio.

## Use with an MCP client

For example, in Claude Desktop's `claude_desktop_config.json`:

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

Every operation is registered as an MCP tool; read-only operations are flagged
with `readOnlyHint`.

## Serving over HTTP

```bash
GALAXY_URL=https://usegalaxy.org/ GALAXY_API_KEY=your-api-key \
  npx @galaxyproject/galaxy-mcp --transport streamable-http --port 8000
# -> "galaxy-mcp: listening on http://127.0.0.1:8000/"
```

The flags and environment variables are the [Python server's](https://github.com/galaxyproject/galaxy-mcp/tree/main/python#serving-over-http):
`--transport` / `GALAXY_MCP_TRANSPORT`, `--host` / `GALAXY_MCP_HOST` (default
`127.0.0.1`), `--port` / `GALAXY_MCP_PORT` (default `8000`) and `--path` /
`GALAXY_MCP_HTTP_PATH` (default `/`). Requests are stateless: each POST is answered
on its own, so there is no MCP session, `GET` and `DELETE` are not served, and the
server sends nothing outside the response to a request. None of the tools need more.

There is no OAuth on this server yet, so every caller acts with the configured
`GALAXY_API_KEY`. It applies the same guards the Python server applies without OAuth:

- **Loopback by default.** Binding to any other address is refused unless
  `--allow-unauthenticated` (`GALAXY_MCP_ALLOW_UNAUTHENTICATED=1`) says something
  else controls who can reach the listener, such as a container port published only
  to loopback or an authenticating reverse proxy.
- **Host and Origin checks.** Requests must carry a loopback `Host`, and browser
  requests are only answered for pages served from the same machine. Add names with
  `GALAXY_MCP_ALLOWED_HOSTS` and `GALAXY_MCP_ALLOWED_ORIGINS` (comma-separated; `*`
  disables the check).
- **No server-filesystem access.** `upload_file` and `download_dataset` with
  `file_path` would use paths on the machine running the server, so they are refused
  over HTTP. A trusted single-user deployment can turn them back on with
  `GALAXY_MCP_ALLOW_LOCAL_FILES=1`.

The older HTTP+SSE transport the Python server also offers (`--transport sse`) is deprecated
in MCP and not provided here.

## Cards (MCP Apps)

With `GALAXY_MCP_APPS=1` the server also offers cards for
[MCP Apps](https://github.com/modelcontextprotocol/ext-apps) hosts, over either transport.
They are off by default, which keeps the tool list identical to the Python server's.

| Tool | Card |
| --- | --- |
| `show_histories` | The user's histories. Returns what `get_histories` returns. |
| `show_history` | One history's datasets and collections. Returns what `get_history_contents` returns. |
| `show_datasets` | Only the datasets an agent chose, in its order, with an optional caption shown as the assistant's note. |

Reads stay card-free. An agent looks things up with `get_*` and calls a `show_*` tool only when the
user wants to see something, at most once per reply. To show "the latest successful datasets", for
example, it reads with `get_history_contents(order="create_time-dsc")`, picks the datasets whose
state is `ok`, and passes their ids to `show_datasets`. Galaxy supplies every name and state. An id
that is not in the history, or that the user cannot access, is shown as such.

A card shows five rows, and Show more reveals rows it already holds before it fetches the next page.
Show less hides rows again, and Collapse leaves only the header with the card's scope and count. Rows
open into their details within the card, and Full screen appears where the host offers it. A card is a
self-contained page with no external resources. It reads only through tool calls the host relays,
so it never holds a Galaxy credential. Hosts without MCP Apps just see the `show_*` tools' results.

## Tool arguments

Parameters are spelled the way the [Python MCP server](https://github.com/galaxyproject/galaxy-mcp/tree/main/python)
spells them -- `history_id`, `tool_id`, `io_details` -- so one `tools/call` payload
works against either server:

```json
{
  "method": "tools/call",
  "params": {
    "name": "get_history_details",
    "arguments": { "history_id": "f2db41e1fa331b3e" }
  }
}
```

Keys *inside* an object-valued argument -- a workflow's `inputs`, a user tool's
`representation` -- are your own data and are sent to Galaxy exactly as written.

Every tool is closed to arguments it does not declare, exactly as the Python
server's schemas are, so a name nobody asked for is refused rather than dropped:

```
Unrecognized parameter: "historyId" -- did you mean "history_id"?
```

Since 0.2.0 the camelCase names used by
[`@galaxyproject/galaxy-ops`](https://www.npmjs.com/package/@galaxyproject/galaxy-ops)
are therefore not parameters of this server. Importing an op from that package
still uses camelCase, and `galaxy-cli`'s flags are unchanged.

A tool's description names its parameters in the spelling this server accepts
them, so a tool that tells you to pass `section_id` is telling you something you
can pass.

## Documentation

See the [TypeScript workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/typescript#readme)
for the full list of operations exposed as tools.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
