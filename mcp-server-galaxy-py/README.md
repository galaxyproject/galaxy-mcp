# Galaxy MCP Server - Python Implementation

This is the Python implementation of the Galaxy MCP server, providing a Model Context Protocol server for interacting with Galaxy instances.

## Features

- Complete Galaxy API integration through BioBlend
- Interactive Workflow Composer (IWC) integration
- FastMCP2 server with remote deployment support
- Type-annotated Python codebase

## Requirements

- Python 3.10+
- FastMCP 2.3.0+

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

The server requires Galaxy credentials to connect to an instance. You can provide these via environment variables:

```bash
export GALAXY_URL=<galaxy_url>
export GALAXY_API_KEY=<galaxy_api_key>
```

Alternatively, create a `.env` file in the project root with these variables.

## Usage

### Quick Start with uvx

The fastest way to run the Galaxy MCP server is using `uvx`:

```bash
# Run the server directly without installation
uvx galaxy-mcp

# Run with FastMCP2 dev tools
uvx --from galaxy-mcp fastmcp dev src/galaxy_mcp/server.py

# Run as remote server
uvx --from galaxy-mcp fastmcp run src/galaxy_mcp/server.py --transport sse --port 8000
```

### As a standalone MCP server

```bash
# Install and run the MCP server
pip install galaxy-mcp
galaxy-mcp

# The server will wait for MCP protocol messages on stdin
```

### With MCP clients

```bash
# Use with FastMCP2 CLI tools
fastmcp dev src/galaxy_mcp/server.py
fastmcp run src/galaxy_mcp/server.py

# Use with other MCP-compatible clients
your-mcp-client galaxy-mcp
```

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed usage patterns and common examples.

## Available MCP Tools

The Python implementation provides the following MCP tools:

- `connect`: Establish connection to a Galaxy instance
- `search_tools`: Find Galaxy tools by name
- `get_tool_details`: Retrieve detailed tool information
- `run_tool`: Execute a Galaxy tool with parameters
- `get_tool_panel`: Retrieve the Galaxy tool panel structure
- `get_user`: Get current user information
- `get_histories`: List available Galaxy histories
- `list_history_ids`: Get simplified list of history IDs and names
- `get_history_details`: Get detailed information about a specific history
- `upload_file`: Upload local files to Galaxy
- `list_workflows`: List available workflows in Galaxy instance
- `get_workflow_details`: Get detailed information about a specific workflow
- `invoke_workflow`: Execute/run a workflow with specified inputs
- `cancel_workflow_invocation`: Cancel a running workflow invocation
- `get_invocations`: View workflow executions
- `get_iwc_workflows`: Access Interactive Workflow Composer workflows
- `search_iwc_workflows`: Search IWC workflows by keywords
- `import_workflow_from_iwc`: Import an IWC workflow to Galaxy

## Testing

The project includes a comprehensive test suite using pytest with mock-based testing.

### Running Tests

```bash
# Install test dependencies
uv pip install -r requirements-test.txt

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=main --cov-report=html

# Run specific test file
uv run pytest tests/test_history_operations.py

# Run tests with verbose output
uv run pytest -v
```

### Test Structure

Tests are organized by functionality:

- `test_connection.py` - Galaxy connection and authentication
- `test_history_operations.py` - History-related operations
- `test_dataset_operations.py` - Dataset upload/download
- `test_tool_operations.py` - Tool search and execution
- `test_workflow_operations.py` - Workflow import and invocation
- `test_integration.py` - End-to-end scenarios

See [tests/README.md](tests/README.md) for more details on the testing strategy.

## Development

### Code Style Guidelines

- Use Python 3.10+ features
- Employ type hints where appropriate
- Follow PEP 8 style guidelines
- Use ruff for code formatting and linting
- All code should pass type checking with mypy

### Development Setup

```bash
# Install development dependencies
make install-dev

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
make lint          # Format code and run all checks

# Testing
make test          # Run tests with coverage

# Building
make clean         # Clean build artifacts
make build         # Build distribution packages

# Running
make run           # Run the MCP server
make dev           # Run FastMCP2 dev server
```

### Using uv directly

All commands can also be run directly with uv:

```bash
# Install dependencies
uv sync --all-extras

# Format and lint code
uv run pre-commit run --all-files

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

## License

[MIT](../LICENSE)
