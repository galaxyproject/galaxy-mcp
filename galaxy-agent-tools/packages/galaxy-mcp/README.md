# @galaxyproject/galaxy-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server (Node) that
exposes [Galaxy](https://galaxyproject.org/) agent operations -- histories,
datasets, tools, workflows, pages, and the IWC catalog -- as MCP tools over stdio. Built
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

## Tool arguments

Parameters are spelled the way the [Python MCP server](https://github.com/galaxyproject/galaxy-mcp/tree/main/mcp-server-galaxy-py)
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

A tool is described as the Python server describes the same tool, its parameters
included, so a client is told the same thing by either server. Those descriptions
name parameters in the spelling this server accepts them, so a tool that tells you
to pass `section_id` is telling you something you can pass.

## Documentation

See the [galaxy-agent-tools workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/galaxy-agent-tools#readme)
for the full list of operations exposed as tools.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
