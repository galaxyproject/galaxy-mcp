# @galaxyproject/galaxy-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server (Node) that
exposes [Galaxy](https://galaxyproject.org/) agent operations -- histories,
datasets, tools, workflows, and the IWC catalog -- as MCP tools over stdio. Built
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

## Documentation

See the [galaxy-agent-tools workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/galaxy-agent-tools#readme)
for the full list of operations exposed as tools.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
