# Galaxy MCP Server - Python Implementation

<!-- mcp-name: io.github.galaxyproject/galaxy-mcp -->

This is the Python implementation of the Galaxy MCP server, providing a Model Context Protocol server for interacting with Galaxy instances.

## Features

- Complete Galaxy API integration through BioBlend
- Optional OAuth login flow for HTTP deployments
- Intergalactic Workflow Commission (IWC) integration
- FastMCP 3 server with remote deployment support
- Session-aware tool visibility: every tool is tagged, and a middleware can hide tags a
  session cannot use (see [Filtering the catalog by tag](#filtering-the-catalog-by-tag))
- Type-annotated Python codebase

## Requirements

- Python 3.10+
- FastMCP 3.1+

## Installation

### From PyPI (Recommended)

```bash
# Install from PyPI
pip install galaxy-mcp

# Or using uv (recommended)
uvx galaxy-mcp
```

### From Source

```bash
# Clone the repository
git clone https://github.com/galaxyproject/galaxy-mcp.git
cd galaxy-mcp/mcp-server-galaxy-py

# Install with uv (recommended)
uv sync --all-extras
```

## Configuration

At minimum the server needs to know which Galaxy instance to target:

```bash
export GALAXY_URL="https://usegalaxy.org.au/"
```

How you authenticate depends on your transport:

- **Stdio / long-lived sessions** – provide an API key:

  ```bash
  export GALAXY_API_KEY="your-api-key"
  ```

- **HTTP / OAuth** – configure the public URL that users reach and a signing secret for session
  tokens. The server mints short-lived Galaxy API keys on behalf of each user.

  `GALAXY_MCP_SESSION_SECRET` is required whenever `GALAXY_MCP_PUBLIC_URL` is set; the server
  refuses to start without it. Generate it once with `openssl rand -hex 32` and store it — the
  same value must be used on every restart and every replica, since it is the key that encrypts
  session tokens. A fresh secret invalidates every token issued under the old one.

  ```bash
  export GALAXY_MCP_PUBLIC_URL="https://mcp.example.com"
  export GALAXY_MCP_SESSION_SECRET="<your stored secret>"
  ```

  Optionally set `GALAXY_MCP_CLIENT_REGISTRY` to control where OAuth client registrations are stored.

  For non-OAuth HTTP clients, `connect(url=..., api_key=...)` stores Galaxy credentials per MCP
  session rather than globally. Clients normally preserve MCP sessions by default, which allows
  multiple users to share the same MCP server while keeping their Galaxy credentials isolated.

You can also steer the transport with `GALAXY_MCP_TRANSPORT` (`stdio`, `streamable-http`, or `sse`).
All variables can be placed in a `.env` file for convenience.

## Usage

### Quick Start with `uvx`

```bash
# Local stdio transport (no network listener)
uvx galaxy-mcp

# Remote/browser clients with HTTP + OAuth
export GALAXY_URL="https://usegalaxy.org.au/"
export GALAXY_MCP_PUBLIC_URL="https://mcp.example.com"
export GALAXY_MCP_SESSION_SECRET="$(openssl rand -hex 32)"
uvx galaxy-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

### Installed CLI

```bash
pip install galaxy-mcp
galaxy-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

If `--transport` is omitted the server defaults to stdio and reads/writes MCP messages via stdin/stdout.

### Working from a checkout

```bash
uv sync
uv run galaxy-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed tool usage patterns.

### Tool discovery mode (experimental)

Galaxy MCP registers one `@mcp.tool` per operation. The client fetches that catalog once per session, but it carries every one of those definitions into the model's context on each turn -- expensive for agents that only ever use a handful. `--discovery-mode code` (also honored via `GALAXY_MCP_DISCOVERY_MODE=code`) collapses the catalog into three meta-tools:

- `search` -- BM25 search over tool names and descriptions
- `get_schema` -- fetch the full schema for specific tools
- `run_galaxy_tool` -- execute any tool by name

```bash
pip install 'galaxy-mcp[code-mode]'   # pulls in pydantic-monty sandbox
galaxy-mcp --discovery-mode code
```

The `code-mode` extra installs `pydantic-monty`, the sandboxed Python interpreter that backs `run_galaxy_tool`. Without it the server still starts in `full` mode but raises a clear error if `--discovery-mode code` is requested.

Default is `full`, which keeps the existing catalog unchanged. CodeMode is useful when you want to keep the agent's context lean and are willing to trade a few extra turns (search -> schema -> execute) per tool call. It's built on FastMCP's experimental `CodeMode` transform, so the API may shift before it stabilizes.

The server also ships agent-facing usage guidance via the MCP `instructions` field (returned during the initial handshake). It explains the typical workflow, the difference between MCP tools and Galaxy tools (e.g. FastQC isn't an MCP tool -- find it via `search_tools_by_name`), and -- when code mode is active -- how to use `run_galaxy_tool` and `call_tool`. Agents that respect the `instructions` field will read this without you having to prompt them.

### Filtering the catalog by tag

Every tool is registered with three tags: a family (`connection`, `user`, `histories`, `datasets`, `jobs`, `tools`, `workflows`, `iwc`, `pages`), an access level (`read` or `write`), and a tier (`core`, `extended`, or `niche`). `ToolVisibilityMiddleware` in [`middleware.py`](src/galaxy_mcp/middleware.py) filters both `tools/list` and direct calls against those tags, so a hidden tool cannot be invoked by name either.

Two environment variables drive it, each a comma-separated tag list:

```bash
export GALAXY_MCP_INCLUDE_TAGS="core,histories"   # show only tools carrying one of these
export GALAXY_MCP_EXCLUDE_TAGS="write"            # then drop any tool carrying one of these
```

The middleware also hides the `admin` and `user_tools` tags from callers that cannot use them, probing the connected Galaxy for `is_admin` and for `enable_unprivileged_tools`. Mind the cache scope: the middleware is constructed once when the server module loads, so those answers are cached for the life of the process, not per session -- the admin answer keyed by URL and API key, the `user_tools` answer by URL alone, and so shared across every key on that server. No tool currently carries either tag, so this half is inert until tools are labelled for it.

The code-mode meta-tools are added by FastMCP's transform and carry no tags at all, so tag filtering and `--discovery-mode code` do not combine: with `GALAXY_MCP_INCLUDE_TAGS` set, all three meta-tools are hidden.

## Available MCP Tools

Every tool returns a `GalaxyResult`: `data`, `success` and `message` always, `count` on
most list operations, and `pagination` on the ones that actually page. Names and behavior
are the contract; arguments live in each tool's own description, which is what the MCP
client sees.

### Connection and account

- `connect`: Point the session at a Galaxy instance and validate the credentials
- `get_server_info`: Version, URL, and public configuration of the connected Galaxy
- `get_user`: The authenticated user

### Histories

- `get_histories`: List histories, optionally filtered by name
- `list_history_ids`: A compact id-and-name list, for when you just need an id
- `get_history_details`: One history's metadata and item counts, without its contents
- `get_history_contents`: The datasets and collections inside a history
- `create_history`: Create a history
- `update_history`: Rename, annotate, tag, delete, or publish a history

### Datasets, collections, and jobs

- `get_dataset_details`: Dataset metadata, optionally with a short content preview
- `get_collection_details`: A dataset collection and its elements
- `get_job_details`: The job that produced a dataset, with its state and parameters
- `upload_file`: Upload a local file into a history
- `upload_file_from_url`: Have Galaxy fetch a file from a URL into a history
- `download_dataset`: Fetch a dataset's content, optionally writing it to disk

### Galaxy tools

- `search_tools_by_name`: Substring search over tool name, id, and description
- `search_tools_by_keywords`: Match keywords against tool names, descriptions, and the file extensions a tool accepts as input
- `get_tool_details`: A tool's metadata, optionally including its full input schema
- `get_tool_panel`: The tool panel as Galaxy organizes it, section by section
- `get_tool_citations`: How to cite a tool
- `get_tool_run_examples`: The tool's own XML test definitions -- real, working invocations
- `get_tool_input_template`: A ready-to-fill `inputs` skeleton plus a compact schema; call this before `run_tool` when the shape is unclear
- `run_tool`: Run a Galaxy tool in a history
- `recommend_biocontainer`: Resolve a verified `quay.io/biocontainers` image for a set of conda packages. Registered **only** when the `container-recommend` extra is installed

### User-defined tools

User-defined tools are unprivileged, containerized tools a user creates themselves. They
are addressed by UUID rather than by tool id, and `run_user_tool` resolves that UUID before
submitting the run, so they need their own run tool even though the run itself goes through
the same Galaxy tools API as a catalog tool.

- `list_user_tools`: The current user's user-defined tools
- `create_user_tool`: Create one from a tool representation
- `delete_user_tool`: Deactivate one, so it stops loading into the toolbox
- `run_user_tool`: Run one in a history

### Workflows and invocations

- `list_workflows`: Stored workflows, optionally filtered by name or published state
- `get_workflow_details`: One workflow's steps and inputs, at a given version
- `get_workflow_input_template`: A ready-to-fill input template plus a run guide; call this before `invoke_workflow`
- `invoke_workflow`: Run a workflow, validating the inputs against its steps first
- `get_invocations`: Invocations, by invocation, workflow, or history
- `cancel_workflow_invocation`: Cancel a running invocation

### IWC catalog

The Intergalactic Workflow Commission publishes a curated, versioned workflow catalog.
These tools read that public manifest; only the import touches your Galaxy.

- `get_iwc_workflows`: The whole IWC manifest
- `search_iwc_workflows`: Substring search across the catalog
- `recommend_iwc_workflows`: Rank catalog workflows against a free-text description of what you want to do
- `get_iwc_workflow_details`: Everything about one workflow before you commit to importing it
- `import_workflow_from_iwc`: Import a catalog workflow into the connected Galaxy

### Pages

Pages are Galaxy-flavored markdown documents. One attached to a history is a Notebook; a
standalone one is a Report. Embedded datasets are referenced by encoded id.

- `list_pages`: Pages, optionally filtered by history or search term
- `get_page`: A page and its latest revision, editable markdown and optionally rendered
- `create_page`: Create a page
- `update_page`: Update a page; a content change records a new revision
- `list_page_revisions`: A page's revision history
- `get_page_revision`: One revision's content
- `revert_page_revision`: Restore an earlier revision as a new one

## Testing

The suite is mock-based: it stands up a `GalaxyInstance` mock and never talks to a real
server, so it runs anywhere in seconds. Test dependencies come from the `dev` extra, which
`uv sync --all-extras` already installs.

```bash
# Run everything
uv run pytest

# One file
uv run pytest tests/test_history_operations.py

# Coverage has to be asked for -- see the config note below
uv run pytest --cov=galaxy_mcp --cov-report=term-missing

# Type checking is a separate gate -- neither pytest nor pre-commit runs it
uv run mypy src/galaxy_mcp
```

Two pytest configs exist and `pytest.ini` wins, which pytest says out loud on every run
(`ignoring pytest config in pyproject.toml`). The `addopts` in `pyproject.toml` therefore
never apply, and the ones in `pytest.ini` sit under a `[tool:pytest]` header that
`pytest.ini` does not read. A bare `uv run pytest` gets no `-v` and no coverage; pass the
flags yourself, as CI does.

### How the tests are put together

- `conftest.py` builds the shared `mock_galaxy_instance` fixture and resets the server's
  module-level connection and cache state between tests.
- `test_helpers.py` unwraps the registered tools back to plain functions, so most tests
  call a tool directly and assert on the `GalaxyResult` it returns.
- `test_job_operations.py` mocks at the HTTP layer with `responses` instead, where the
  code under test goes around BioBlend.
- `tests/mcp_session.py` drives the tools through an in-memory FastMCP client. Only the
  live suite uses it today. That path matters because a tool called without an MCP request
  context has no session id, and the session-scoped connection store then has nowhere to
  put the client -- which is what made the live suite look broken for months.

### The live suite

`tests/test_real_integration.py` runs against a real Galaxy and is **skipped by default**.
It needs `GALAXY_TEST_API_KEY` set to a valid key and `GALAXY_TEST_URL` (default
`http://localhost:8080`) answering on `/api/version`; if either is missing the whole module
is skipped. A green `pytest` run therefore says nothing about these tests -- check the skip
count before reading it as coverage.

```bash
export GALAXY_TEST_URL="http://localhost:8080"
export GALAXY_TEST_API_KEY="a-real-key"
uv run pytest tests/test_real_integration.py
```

## Development

### Code Style Guidelines

- Use Python 3.10+ features
- Employ type hints where appropriate
- Follow PEP 8 style guidelines
- Use ruff for code formatting and linting
- All code should pass type checking with mypy

### Development Setup

```bash
# Install every dependency, including the dev and optional extras
make install

# Set up pre-commit hooks (required for contributing)
uv run pre-commit install
```

Pre-commit hooks will automatically format your code and run linting checks when you commit. All contributors should install these hooks to maintain consistent code quality.

### Development Commands

We use a Makefile for consistent development commands:

```bash
# Show all available commands
make help

# Install dependencies
make install       # Install all dependencies

# Code quality
make lint          # Run the pre-commit hooks (formatting and lint -- no type checking)

# Testing
make test          # Type-check with mypy, then run tests with coverage

# Building
make clean         # Clean build artifacts
make build         # Build distribution packages

# Running
make run           # Run the MCP server
make dev           # Run the FastMCP dev inspector
```

### Using uv directly

All commands can also be run directly with uv:

```bash
# Install dependencies
uv sync --all-extras

# Format and lint code
uv run pre-commit run --all-files

# Type check
uv run mypy src/galaxy_mcp

# Run tests with coverage
uv run pytest --cov=galaxy_mcp --cov-report=html

# Update dependencies
uv lock --upgrade
```

### Cross-version Testing

Test across multiple Python versions using tox:

```bash
# Test on all supported Python versions
tox

# Test on specific version
tox -e py312

# Run only linting
tox -e lint

# Run type checking
tox -e type
```

### Pre-commit Hooks

The project uses pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit hooks (one-time setup)
uv run pre-commit install

# Run pre-commit manually on all files
uv run pre-commit run --all-files

# Skip pre-commit for a single commit (not recommended)
git commit --no-verify
```

Pre-commit runs automatically on `git commit` and includes:

- Code formatting with ruff
- Linting with ruff
- Trailing whitespace removal
- File cleanup (EOF, YAML/JSON/TOML validation)
- Large file detection
- Merge conflict detection

It does **not** type-check. CI runs `mypy src/galaxy_mcp` as its own job, so run that
yourself before pushing.

## License

[MIT](../LICENSE)
