# Galaxy MCP Server

This project provides a Model Context Protocol (MCP) server for interacting with the Galaxy bioinformatics platform. It enables AI assistants and other clients to connect to Galaxy instances, search and execute tools, manage workflows, and access other features of the Galaxy ecosystem.

## Project Overview

This repository contains a Python-based MCP server implementation that provides comprehensive integration with Galaxy's API through BioBlend.

Note: There is also a work-in-progress TypeScript implementation available in a separate branch of this repository.

## Key Features

- **Galaxy Connection**: Connect to any Galaxy instance with a URL and API key
- **OAuth Login (optional)**: Offer browser-based sign-in that exchanges credentials for temporary Galaxy API keys
- **Server Information**: Retrieve comprehensive server details including version, configuration, and capabilities
- **Tools Management**: Search, view details, and execute Galaxy tools
- **Workflow Integration**: Access and import workflows from the Interactive Workflow Composer (IWC)
- **History Operations**: Manage Galaxy histories and datasets
- **File Management**: Upload files to Galaxy from local storage
- **`gxy` CLI**: Direct terminal access to all Galaxy operations with JSON output
- **Comprehensive Testing**: Full test suite with mock-based testing for reliability

## Quick Start

The `galaxy-mcp` CLI ships with both stdio (local) and HTTP transports. Choose the setup that
matches your client:

```bash
# Stdio transport (default) – great for local development tools
uvx galaxy-mcp

# HTTP transport with OAuth (for remote/browser clients)
export GALAXY_URL="https://usegalaxy.org.au/"          # Target Galaxy instance
export GALAXY_MCP_PUBLIC_URL="https://mcp.example.com" # Public base URL for OAuth redirects
export GALAXY_MCP_SESSION_SECRET="$(openssl rand -hex 32)"
uvx galaxy-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

When running over stdio you can provide long-lived credentials via environment variables:

```bash
export GALAXY_URL="https://usegalaxy.org/"
export GALAXY_API_KEY="your-api-key"
```

For OAuth flows the server exchanges user credentials for short-lived Galaxy API keys on demand, so
you typically leave `GALAXY_API_KEY` unset.

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

The published image defaults to stdio transport (no HTTP listener):

```bash
docker run --rm -it \
  -e GALAXY_URL="https://usegalaxy.org/" \
  -e GALAXY_API_KEY="your-api-key" \
  galaxyproject/galaxy-mcp
```

For OAuth + HTTP:

```bash
docker run --rm -it -p 8000:8000 \
  -e GALAXY_URL="https://usegalaxy.org.au/" \
  -e GALAXY_MCP_TRANSPORT="streamable-http" \
  -e GALAXY_MCP_PUBLIC_URL="https://mcp.example.com" \
  -e GALAXY_MCP_SESSION_SECRET="$(openssl rand -hex 32)" \
  galaxyproject/galaxy-mcp
```

## `gxy` Command-Line Interface

The package also installs `gxy`, a command-line tool for interacting with Galaxy directly from your terminal. It outputs JSON by default (pipe to `jq`), or use `--pretty` for human-readable formatting.

```bash
# Install
pip install galaxy-mcp    # or: uvx --from galaxy-mcp gxy --help

# Set credentials
export GALAXY_URL="https://usegalaxy.org/"
export GALAXY_API_KEY="your-api-key"

# Search for tools
gxy tools search fastqc

# List your histories
gxy --pretty history list --limit 5

# Discover IWC workflows using natural language
gxy iwc recommend "differential expression from RNA-seq"

# Import a workflow and run it
gxy iwc import "#workflow/github.com/iwc-workflows/rnaseq-pe/main"
gxy workflow invoke WORKFLOW_ID -i '{"0": {"src": "hda", "id": "DATASET_ID"}}'

# Upload data
gxy dataset upload reads.fastq.gz
```

### Available Command Groups

| Group | Commands | Description |
|-------|----------|-------------|
| `tools` | `search`, `keywords`, `details`, `examples`, `citations`, `panel`, `run` | Search and run Galaxy tools |
| `history` | `list`, `ids`, `create`, `details`, `contents` | Manage histories |
| `dataset` | `details`, `download`, `upload`, `upload-url`, `job` | Dataset operations |
| `collection` | `details` | Dataset collections |
| `workflow` | `list`, `details`, `invoke`, `invocations`, `cancel` | Manage workflows |
| `iwc` | `list`, `search`, `details`, `recommend`, `import` | IWC workflow discovery |
| `server` | `info` | Server information |
| `user` | `info` | User information |

Use `gxy <group> --help` for details on any command group.

### Configuration

`gxy` reads credentials from (in priority order):

1. Command-line options (`--url`, `--api-key`)
2. Environment variables (`GALAXY_URL`, `GALAXY_API_KEY`)
3. Config file `~/.galaxy-mcp/config.toml`
4. Planemo profiles (`~/.planemo/profiles/`)

Config file example:

```toml
[default]
url = "https://usegalaxy.org/"
api_key = "your-api-key"

[usegalaxy-eu]
url = "https://usegalaxy.eu/"
api_key = "eu-api-key"
```

Use a named profile with `--profile`:

```bash
gxy --profile usegalaxy-eu history list
```

If you already have planemo profiles set up for external Galaxy instances, `gxy` can use those too -- just pass the planemo profile name with `--profile`.

## Connect to Claude Desktop
- Ensure that GalaxyMCP runs with `uvx galaxy-mcp`
- Add `export GALAXY_SERVER=https://usegalaxy.org` to your .bashrc (or equiv)
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
