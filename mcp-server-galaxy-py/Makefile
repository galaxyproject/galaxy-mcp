.PHONY: help install lint test clean build upload run dev

# Default target
help:
	@echo "Galaxy MCP Development Commands"
	@echo "================================"
	@echo "  make install       Install all dependencies"
	@echo "  make lint          Format code and run all checks"
	@echo "  make test          Run tests with coverage"
	@echo "  make clean         Clean build artifacts"
	@echo "  make build         Build package"
	@echo "  make run           Run the MCP server"
	@echo "  make dev           Run MCP dev server"

# Install all dependencies
install:
	uv sync --all-extras

# Format code and run all checks
lint:
	uv run pre-commit run --all-files --show-diff-on-failure

# Run tests with coverage and type checking
test:
	uv run mypy src/galaxy_mcp
	uv run pytest --cov=galaxy_mcp --cov-report=html --cov-report=term-missing

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Build package
build: clean
	uv run python -m build


# Run the MCP server
run:
	uv run galaxy-mcp

# Run FastMCP2 dev server
dev:
	uv run --from . fastmcp dev src/galaxy_mcp/server.py
