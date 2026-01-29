# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
