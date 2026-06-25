# @galaxyproject/galaxy-ops

The framework-free TypeScript core for Galaxy agent operations. It provides a
typed [Galaxy](https://galaxyproject.org/) API client (built on `openapi-fetch`
and the official `@galaxyproject/galaxy-api-client` types), an operation
registry, a typed error model, and run/poll orchestration for tool runs and
workflow invocations.

This package has no dependency on any CLI or MCP framework -- it's the shared
core behind [`@galaxyproject/galaxy-cli`](https://www.npmjs.com/package/@galaxyproject/galaxy-cli)
and [`@galaxyproject/galaxy-mcp`](https://www.npmjs.com/package/@galaxyproject/galaxy-mcp),
and you can use it directly to script Galaxy from TypeScript.

## Install

```bash
npm install @galaxyproject/galaxy-ops
```

Requires Node.js `>=22.19`.

## Usage

```ts
import { createGalaxyContext, getUser, getHistories } from "@galaxyproject/galaxy-ops";

const ctx = createGalaxyContext({
  baseUrl: "https://usegalaxy.org/",
  apiKey: process.env.GALAXY_API_KEY!,
});

const me = await getUser({}, ctx);
console.log(`Hello, ${me.username}`);

const histories = await getHistories({ limit: 10 }, ctx);
```

Each operation is a small function `(input, ctx) => Promise<data>` that returns
typed data or throws a typed `GalaxyError`. For the surface-style envelope
(`{ data, success, message, errorKind }`) used by the CLI and MCP server, wrap a
registered op with `runWithEnvelope`; iterate `allOperations` to enumerate the
full set.

## Documentation

See the [galaxy-agent-tools workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/galaxy-agent-tools#readme)
for the full operation list and design.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
