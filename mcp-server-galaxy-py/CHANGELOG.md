# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Over the HTTP transports, `connect` only reaches the Galaxy in `GALAXY_URL` and any base URLs
  the operator lists in the new `GALAXY_MCP_EXTRA_ALLOWED_URLS`, matched exactly. Before this, an
  authenticated HTTP caller could pass any URL and a junk key, have the server request it, and read
  the reply back out of the connection error. `GALAXY_URL` is now required for HTTP use. stdio is
  unchanged: its caller is the operator.
- A failed connection over HTTP reports the HTTP status, not the body the remote host sent.
- Galaxy GET requests no longer follow redirects (bioblend already refused them on writes).
- A `.env` reloaded after startup can no longer send its `GALAXY_API_KEY` to a `GALAXY_URL` it
  replaced.

## [1.11.0] - 2026-10-05

### Security

Fixes [GHSA-8f3c-j762-q9q4](https://github.com/galaxyproject/galaxy-mcp/security/advisories/GHSA-8f3c-j762-q9q4).
galaxy-mcp is first of all a local server for the person running it, and several tools
assumed the caller owns the process. Over the HTTP transports that assumption did not hold.
Reported by Syed Anas Mohiuddin (items 1 and 2).

1. `upload_file(path)` read any path the server process could read into the caller's history.
2. HTTP transports bound `0.0.0.0` by default with no authentication unless OAuth was
   configured, and fell back to the operator's environment credentials.
3. `download_dataset(file_path=...)` wrote dataset contents to any path the process could write.
4. The CORS preflight handler reflected any `Origin`, and `Host` was never validated, so a web
   page open in the user's browser could drive a local unauthenticated HTTP server.
5. `connect(url=...)` without an `api_key` sent the environment's `GALAXY_API_KEY` to the
   caller-supplied URL.

What changed, and what to do if it affects you:

- `upload_file` and `download_dataset(file_path=...)` refuse over HTTP. Use
  `upload_file_from_url` and in-memory downloads, or set `GALAXY_MCP_ALLOW_LOCAL_FILES=1` on
  a trusted single-user deployment. Stdio is unchanged.
- HTTP binds `127.0.0.1` by default. A non-loopback bind without OAuth refuses to start unless
  you pass `--allow-unauthenticated` (`GALAXY_MCP_ALLOW_UNAUTHENTICATED=1`). The container
  image binds `0.0.0.0`, so running it over HTTP without OAuth now needs that flag.
- Without OAuth, requests must carry a loopback `Host`, and browser requests are only answered
  for loopback origins. Add names with `GALAXY_MCP_ALLOWED_HOSTS` / `GALAXY_MCP_ALLOWED_ORIGINS`.
  With OAuth the bearer token is the gate and behaviour is unchanged.
- `connect(url=...)` only pairs the environment's `GALAXY_API_KEY` with the configured
  `GALAXY_URL`; any other server needs an explicit `api_key`. With OAuth enabled the
  environment's credentials are never lent to a request, and an OAuth provider that fails to
  initialise now stops startup instead of running without auth.
- New `SECURITY.md`; the README has a "Serving over HTTP" section covering all of the above.

### Fixed

- The three raw requests over the unprivileged-tools API now check the status Galaxy answered
  with, as every other raw request in the server already did. `delete_user_tool` reported
  `deactivated: true` for a DELETE that 404'd or 403'd, `list_user_tools` sliced an error body
  as a page and reported the result as `List user tools failed: slice(0, 25, None)`, and
  `run_user_tool` read one as a tool record and answered "No user-defined tool found with
  UUID ...". All three now fail with the sentence `format_error` builds for a refused request.
- `get_invocations(invocation_id=..., step_details=True)` now sends `step_details` to Galaxy.
  The single-invocation path went through bioblend's `show_invocation`, which has no such
  parameter, so the flag was dropped and Galaxy answered with every step's `jobs` list empty --
  an invocation still running looked like one with nothing left to do. Without `step_details`
  the request is unchanged.

## [1.4.0] - 2026-04-22

### Added

- User-defined tool support: `create_user_tool`, `list_user_tools`, `delete_user_tool`, and `run_user_tool` for managing and executing unprivileged tools (PR #40)
- Galaxy tool credentials integration -- auto-discover and supply user credentials when running tools (PRs #41, #44)
- `User-Agent` header on all GalaxyInstance requests (`galaxy-mcp/{version} bioblend/{version}`) so Galaxy can identify agent traffic

### Changed

- Shared `GalaxyInstance` is now thread-safe, enabling concurrent MCP calls (PR #38)
- Upgraded to fastmcp >=3.0.0; IWC functions and integration tests updated for the new API (PR #35)
- Added upper-bound version pins to all production dependencies (PR #36)

### Fixed

- IWC workflow functions no longer break under fastmcp >= 3.0.0

## [1.3.0] - 2026-01-28

### Added

- `recommend_iwc_workflows` MCP tool for semantic search of IWC workflows using BM25 ranking
- `get_iwc_workflow_details` MCP tool for comprehensive workflow info before importing
- `GalaxyResult` and `PaginationInfo` structured response models for consistent API responses
- Browser-based OAuth authentication for Galaxy instances (PR #18)
- Dataset collection support with robust collection detection (PR #26)
- Real integration tests against live Galaxy instances

### Changed

- Enriched `search_iwc_workflows` with additional metadata (readme_summary, step_count, authors, categories, tools_used)
- Improved tool search to support substring matching in name, ID, and description (PR #27)
- Added `rank-bm25` dependency for semantic workflow search

## [1.2.0] - 2025-12-19

### Added

- `search_tools` MCP tool for searching Galaxy tools by name or keyword

### Changed

- Fixed release workflow to checkout correct tag during deployment

## [1.1.0] - 2025-06-23

### Fixed

- `download_dataset` now handles read-only filesystem environments gracefully by downloading to memory when no file path is specified
- Fixed Makefile `make dev` command to work with current uv version by removing unsupported `--from` flag

### Improved

- Enhanced `download_dataset` documentation with explicit LLM guidance for filesystem access requirements
- Added clear messaging about memory vs. filesystem download modes with suggested filename support
- Updated function signature to provide better error handling and user feedback

## [1.0.0] - 2025-06-22

### Added

- `get_dataset_details` MCP tool for comprehensive dataset metadata with optional content preview
- `download_dataset` MCP tool to download datasets to local filesystem with flexible naming options
- Binary file support in dataset content preview with hexadecimal display
- Enhanced parameter documentation with detailed Galaxy ID format examples and comprehensive descriptions

### Changed

- **BREAKING**: `get_job_details` now accepts `dataset_id` instead of `job_id` as primary parameter, using dataset provenance to find creating job
- Upgraded from FastMCP 1.0 to FastMCP2 with remote deployment support via SSE transport
- Simplified parameter validation by removing complex JSON string parsing across all functions
- Improved API consistency with plain string-only identifier parameters
- Consolidated dependency management to use uv with lock file
- Enhanced test coverage for job operations and dataset functionality

### Fixed

- Job lookup now works with the more commonly available dataset IDs instead of job IDs
- Fallback mechanism for job details when provenance data is unavailable
- Better error handling for dataset state validation in download operations

## [0.2.1] - 2025-06-11

### Added

- `filter_tools_by_dataset` MCP tool to recommend Galaxy tools based on dataset types/keywords
- Comprehensive test suite for the new filtering functionality

### Changed

- Improved error handling consistency across all MCP tools

## [0.2.0] - 2025-05-21

### Added

- `get_server_info` MCP tool to retrieve comprehensive Galaxy server information including version, URL, and configuration details

## [0.1.0] - 2025-01-16

### Added

- Initial release of galaxy-mcp
- MCP server implementation for Galaxy bioinformatics platform
- Connection and authentication with Galaxy instances
- History management (create, list, get details)
- Tool operations (search, run)
- Dataset operations (upload, download)
- Workflow operations (import from IWC, list invocations)
- Comprehensive test suite
- Command-line interface via `galaxy-mcp` command
- Environment variable support for configuration
