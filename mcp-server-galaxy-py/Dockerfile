# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm

# Add metadata labels
LABEL maintainer="Galaxy Team" \
      org.opencontainers.image.title="Galaxy MCP" \
      org.opencontainers.image.description="Galaxy Model Context Protocol Server - Python" \
      org.opencontainers.image.source="https://github.com/galaxyproject/galaxy-mcp"

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# Define default environment variables
ENV GALAXY_INSTANCE="https://usegalaxy.org" \
    GALAXY_API_KEY=""

COPY --from=uv /root/.local /root/.local
COPY --from=uv --chown=app:app /app/.venv /app/.venv

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Create and set ownership for data directory
RUN mkdir -p /data && chown -R app:app /data

# Define volume for persistent data
VOLUME ["/data"]

# Switch to non-root user
USER app

# Expose service port
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD mcp-server-galaxy --health-check || exit 1

# Use tini as init to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Update command to use environment variables instead of DB path
CMD ["mcp-server-galaxy", "--galaxy-url", "${GALAXY_INSTANCE}", "--api-key", "${GALAXY_API_KEY}"]
