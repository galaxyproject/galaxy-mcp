# Galaxy MCP Server Integration Testing Strategy

## Overview

The best approach for writing integration tests for the Galaxy MCP server is to use mock-based testing that simulates Galaxy API responses without requiring a live Galaxy instance. This enables fast, reliable, and isolated testing.

## Key Components

1. **Test Framework**: pytest with mock support
2. **Mocking Library**: unittest.mock for simulating Galaxy API responses
3. **Test Structure**: Organized by functionality (connection, history, datasets, tools, workflows)
4. **Isolation**: Each test is isolated with reset fixtures and environment variables

## Implementation Details

### Test Infrastructure

- **conftest.py**: Contains shared fixtures for:
    - Mock Galaxy instance with pre-configured responses
    - Test environment variables
    - Auto-reset of galaxy state between tests
    - Mock MCP context for tool testing

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_connection.py       # Galaxy connection and authentication
├── test_history_operations.py   # History-related operations
├── test_dataset_operations.py   # Dataset upload/download
├── test_tool_operations.py      # Tool search and execution
├── test_workflow_operations.py  # Workflow import and invocation
├── test_oauth.py                # OAuth flow and HTTP public routes
└── test_integration.py         # End-to-end scenarios
```

### Key Testing Patterns

1. **Mock Galaxy API Responses**:

    ```python
    mock_galaxy_instance.histories.get_histories.return_value = [
        {"id": "test_history_1", "name": "Test History 1"}
    ]
    ```

2. **Test State Isolation**:

    ```python
    with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
        # Test code here
    ```

3. **Error Validation**:
    ```python
    with pytest.raises(ValueError, match="Expected error message"):
        function_under_test(invalid_input)
    ```

## Running Tests

```bash
# Install test dependencies
uv pip install -r requirements-test.txt

# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=main --cov-report=html

# Run specific test file
uv run pytest tests/test_history_operations.py
```

## Benefits

1. **No Galaxy Dependency**: Tests run without a live Galaxy instance
2. **Fast Execution**: Mock responses are instant
3. **Reliable**: No network or external service issues
4. **Complete Control**: Can simulate any API response or error condition
5. **CI/CD Ready**: Easy to integrate into automated pipelines

## Future Enhancements

1. Add property-based testing with Hypothesis
2. Include performance benchmarks
3. Add integration with actual test Galaxy instance (optional)
4. Expand test coverage to 80%+
5. Add contract tests for Galaxy API compatibility
