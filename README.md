# Galaxy MCP Server

This project provides a Model Context Protocol (MCP) server for interacting with the Galaxy bioinformatics platform. It enables AI assistants and other clients to connect to Galaxy instances, search and execute tools, manage workflows, and access other features of the Galaxy ecosystem.

## Project Overview

The repository holds two independent implementations of the same Galaxy operation set:

- [`mcp-server-galaxy-py/`](mcp-server-galaxy-py/) -- the Python MCP server, published to
  PyPI as `galaxy-mcp`. It reaches Galaxy through BioBlend and carries the larger tool
  surface. See the [Python README](mcp-server-galaxy-py/README.md).
- [`galaxy-agent-tools/`](galaxy-agent-tools/) -- a TypeScript pnpm workspace offering the
  same operations two ways: a `galaxy-cli` command-line tool and a `galaxy-mcp` (Node) MCP
  server, both built on a shared framework-free core. See the
  [galaxy-agent-tools README](galaxy-agent-tools/README.md).

The two are meant to stay in step: an operation keeps its name and its meaning across
both. They are still separate codebases with separate release trains, though, so each
README lists the operations that surface actually has -- read the one you are using.

## Key Features

- **Galaxy Connection**: Connect to any Galaxy instance with a URL and API key
- **OAuth Login (optional)**: Offer browser-based sign-in that exchanges credentials for temporary Galaxy API keys
- **Server Information**: Retrieve comprehensive server details including version, configuration, and capabilities
- **Tools Management**: Search the tool catalog, inspect a tool's inputs, and execute Galaxy tools
- **User-Defined Tools**: Create, list, run, and deactivate unprivileged user-defined tools
- **Workflow Integration**: Find and import workflows from the Intergalactic Workflow Commission (IWC), then invoke them and follow their invocations
- **History Operations**: Manage Galaxy histories, datasets, and collections, and inspect the jobs behind them
- **File Management**: Upload files to Galaxy from local storage or from a URL, and download results back
- **Pages**: Read and write Galaxy-flavored markdown pages -- history-attached notebooks and standalone reports -- including their revision history
- **Verified Container Recommendation (optional)**: Resolve a real `quay.io/biocontainers` image for a set of conda packages instead of guessing one -- see [Optional extras](#optional-extras)
- **Mock-based test suite**: The default suite runs entirely against mocked Galaxy responses, so it needs no server; a separate suite against a live Galaxy is opt-in and skips when no credentials are configured

## Optional extras

### `container-recommend`

Enables the `recommend_biocontainer` tool, which resolves a verified
`quay.io/biocontainers` image for a set of conda packages. Authoring a user-defined
tool means naming a container, and a hallucinated image tag is the most common way a
UDT passes validation and then dies at run time with `manifest unknown`. This wraps
`galaxy.tool_util.deps.mulled.recommend` -- the same resolver Galaxy's own custom-tool
agent uses (added in Galaxy 26.1, galaxyproject/galaxy#22981) -- so the image is
checked against quay.io rather than invented.

It is an extra rather than a hard dependency because `galaxy-tool-util` pulls in lxml
and conda-package-streaming, and every other tool on this server works without it. The
tool is **only registered when the extra is installed**, so a stock install simply
doesn't advertise it.

```bash
uvx --from 'galaxy-mcp[container-recommend]' galaxy-mcp
# or, for a local checkout:
cd mcp-server-galaxy-py && uv sync --extra container-recommend
```

The same resolver is also available as a standalone CLI once installed:
`mulled-recommend samtools=1.17`.

### `code-mode`

Collapses the whole tool catalog into three meta-tools -- `search`, `get_schema`, and
`run_galaxy_tool` -- so an agent pays for a tool's schema only when it decides to use it.
The extra pulls in `pydantic-monty`, the sandboxed interpreter that runs the submitted
code. Without the extra the server still starts, but asking for code mode is an error
rather than a silent downgrade.

```bash
uvx --from 'galaxy-mcp[code-mode]' galaxy-mcp --discovery-mode code
```

See [Tool discovery mode](mcp-server-galaxy-py/README.md#tool-discovery-mode-experimental)
in the Python README for what the trade-off buys you.

## Quick Start

The `galaxy-mcp` CLI ships with both stdio (local) and HTTP transports. Choose the setup that
matches your client:

```bash
# Stdio transport (default) – great for local development tools
uvx galaxy-mcp

# HTTP transport with OAuth (for remote/browser clients)
# Generate the session secret ONCE (`openssl rand -hex 32`) and store it. Every
# restart and every replica must use the same value, or tokens issued earlier --
# or by another replica -- stop decrypting.
export GALAXY_URL="https://usegalaxy.org.au/"          # Target Galaxy instance
export GALAXY_MCP_PUBLIC_URL="https://mcp.example.com" # Public base URL for OAuth redirects
export GALAXY_MCP_SESSION_SECRET="<your stored secret>"
uvx galaxy-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

When running over stdio you can provide long-lived credentials via environment variables:

```bash
export GALAXY_URL="https://usegalaxy.org/"
export GALAXY_API_KEY="your-api-key"
```

For OAuth flows the server exchanges user credentials for short-lived Galaxy API keys on demand, so
you typically leave `GALAXY_API_KEY` unset.

For non-OAuth HTTP clients, `connect(url=..., api_key=...)` stores Galaxy credentials per MCP
session rather than globally. Clients normally preserve MCP sessions by default, which allows
multiple users to share the same MCP server while keeping their Galaxy credentials isolated.

### Alternative Installation

```bash
# Install from PyPI
pip install galaxy-mcp

# Run (stdio by default)
galaxy-mcp

# Or from source using uv
cd mcp-server-galaxy-py
uv sync
uv run galaxy-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

## Container Usage

Images are published to the GitHub Container Registry as
[`ghcr.io/galaxyproject/galaxy-mcp`](https://github.com/galaxyproject/galaxy-mcp/pkgs/container/galaxy-mcp).
Use `:latest` (default) or pin a release, e.g. `:1.9.0`.

The published image defaults to stdio transport (no HTTP listener):

```bash
docker run --rm -it \
  -e GALAXY_URL="https://usegalaxy.org/" \
  -e GALAXY_API_KEY="your-api-key" \
  ghcr.io/galaxyproject/galaxy-mcp
```

For OAuth + HTTP:

```bash
# Generate once and keep it -- do NOT inline `openssl rand` here, or each
# container start mints a key that invalidates every token issued before it.
export GALAXY_MCP_SESSION_SECRET="<your stored secret>"

docker run --rm -it -p 8000:8000 \
  -e GALAXY_URL="https://usegalaxy.org.au/" \
  -e GALAXY_MCP_TRANSPORT="streamable-http" \
  -e GALAXY_MCP_PUBLIC_URL="https://mcp.example.com" \
  -e GALAXY_MCP_SESSION_SECRET \
  ghcr.io/galaxyproject/galaxy-mcp
```

## Connect to Claude Desktop
- Ensure that GalaxyMCP runs with `uvx galaxy-mcp`
- Add `export GALAXY_URL=https://usegalaxy.org` to your .bashrc (or equiv)
- Download and install [claude desktop](https://www.claude.com/download)
- Go to Settings -> Developer -> Edit Config
- Add this to `claude_desktop_config.json`
```
{
  "mcpServers": {
    "galaxy-mcp": {
      "command": "uvx",
      "args": ["galaxy-mcp"],
      "env": {
        "GALAXY_URL": "https://usegalaxy.org",
        "GALAXY_API_KEY": "SECRETS"
      }
    }
  }
}
```
- Under the developer menu, you should now see `galaxy-mcp` as running (you may need to restart Claude desktop)
- Prompt Claude with "can you connect to galaxy"
- If you have not provided the optional env config you'll be asked for connection details which you can provide like "Use my Galaxy API key: XXXXXXX"
- Talk to Claude to work with your galaxy instance, e.g. "give a summary with my histories"

## Development Guidelines

See the [Python implementation README](mcp-server-galaxy-py/README.md) for specific instructions and documentation.

## License

[MIT](LICENSE)
