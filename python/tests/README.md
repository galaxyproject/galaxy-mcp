# Galaxy MCP Server Tests

This directory contains integration tests for the Galaxy MCP server.

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_connection.py` - Tests for Galaxy connection and authentication
- `test_history_operations.py` - Tests for history-related operations
- `test_dataset_operations.py` - Tests for dataset operations
- `test_tool_operations.py` - Tests for tool search and execution
- `test_workflow_operations.py` - Tests for workflow operations
- `test_integration.py` - End-to-end integration tests
- `test_oauth.py` - Tests for OAuth session handling and public endpoints

## Running Tests

### Install test dependencies

```bash
pip install -r requirements-test.txt
```

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_history_operations.py
```

### Run with coverage

```bash
pytest --cov=main --cov-report=html
```

### Run only unit tests

```bash
pytest -m unit
```

### Run only integration tests

```bash
pytest -m integration
```

## Test Strategy

1. **Unit Tests**: Test individual functions with mocked dependencies
2. **Integration Tests**: Test complete workflows and interactions
3. **Mock Galaxy API**: Use mocked BioBlend responses to avoid requiring a live Galaxy instance

## Writing New Tests

1. Create test files following the naming pattern `test_*.py`
2. Use the fixtures from `conftest.py` for common setup
3. Mock external dependencies (Galaxy API, file system, etc.)
4. Test both success and error cases
5. Use descriptive test names that explain what is being tested

## Continuous Integration

These tests can be integrated into CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
    test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v3
            - uses: actions/setup-python@v4
              with:
                  python-version: "3.11"
            - run: pip install -r requirements.txt -r requirements-test.txt
            - run: pytest --cov=main
```
