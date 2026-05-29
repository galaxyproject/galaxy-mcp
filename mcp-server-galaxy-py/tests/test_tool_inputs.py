import bioblend

from galaxy_mcp.tool_inputs import is_input_related_error


def _conn_err(status, body="boom"):
    return bioblend.ConnectionError(
        f"Unexpected HTTP status code: {status}", body=body, status_code=status
    )


def test_400_is_input_related():
    assert is_input_related_error(_conn_err(400, "Required parameter(s) kwd not provided.")) is True


def test_typeerror_is_input_related():
    assert is_input_related_error(TypeError("bad inputs")) is True


def test_auth_and_notfound_are_not_input_related():
    assert is_input_related_error(_conn_err(401)) is False
    assert is_input_related_error(_conn_err(403)) is False
    assert is_input_related_error(_conn_err(404)) is False
    assert is_input_related_error(_conn_err(500)) is False


def test_plain_exception_is_not_input_related():
    assert is_input_related_error(RuntimeError("network down")) is False


def test_valueerror_is_not_input_related():
    assert is_input_related_error(ValueError("bad inputs")) is False
