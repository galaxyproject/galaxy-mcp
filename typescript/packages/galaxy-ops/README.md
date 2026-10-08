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
typed data or throws a typed `GalaxyError`. A listing returns `Paged<T>` --
`{ items, pagination }`, camelCase -- which is the library shape and is not what
goes over a wire.

For the surface envelope used by the CLI and MCP server, wrap a registered op
with `runWithEnvelope`; iterate `allOperations` to enumerate the full set. That
envelope is the Python MCP server's, key for key:
`{ data, success, message, count, pagination }`, where a listing's `data` is the
bare page and `pagination` is spelled the way that server spells it
(`total_items`, `returned_items`, `has_next`, `next_offset`, `helper_text`, and
so on), always present and `null` where the tool has none. On a failure it is
`{ data: undefined, success: false, message, errorKind }`.

An operation may declare a minimum Galaxy version (`requires: { galaxy: ">=26.1" }`).
That is checked before the operation runs -- through the direct call above as much as
through `runWithEnvelope` -- and a server that is too old is refused with a
`GalaxyVersionError` before any request is sent. The version is read from
`/api/version` once per context, or taken from the `serverVersion` you pass to
`createGalaxyContext`; a version that cannot be read refuses nothing.

## In a browser

The default entry registers every operation, and three of them need something a
browser does not have: `download_dataset` and `upload_file` read and write local
files, and `recommend_biocontainer` hashes with `node:crypto`. A bundler that
resolves the `browser` export condition -- most do, for a web target -- gets the
browser entry instead, which is the same surface minus those three operations and
the mulled helpers behind the third, and which pulls in no Node builtin. Import
the subpath directly if your bundler does not resolve that condition, or if you
would rather the choice were visible in the source:

```ts
import { createGalaxyContext, getUser } from "@galaxyproject/galaxy-ops/browser";
```

Nothing else moves: the operations that are there are the same objects with the
same types, and `allOperations` is that shorter list.

The `typescript` peer dependency is optional and asks for `>=5.5` -- the emitted
declarations use nothing newer, and a project that does not typecheck against
them needs no TypeScript at all.

## Documentation

See the [TypeScript workspace README](https://github.com/galaxyproject/galaxy-mcp/tree/main/typescript#readme)
for the full operation list and design.

## License

[MIT](https://github.com/galaxyproject/galaxy-mcp/blob/main/LICENSE)
