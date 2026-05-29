"""Pure helpers for diagnosing and scaffolding Galaxy tool inputs.

No bioblend client, no network, no global state -- everything here is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching schemas via a Galaxy client, registering MCP tools) lives in
server.py.
"""


def is_input_related_error(exc: Exception) -> bool:
    """True when a tool-run failure is plausibly caused by the provided inputs.

    Keys off the structured bioblend error (HTTP 400 == Galaxy rejected the
    tool form/parameters) rather than substring-scanning the message. Galaxy's
    masked-TypeError 'kwd not provided' bug also surfaces as a 400. A bare
    TypeError (e.g. bioblend choking while building the request from a
    malformed inputs dict) counts too. Auth (401/403), 404, and 5xx do not.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 400
    return isinstance(exc, TypeError)
