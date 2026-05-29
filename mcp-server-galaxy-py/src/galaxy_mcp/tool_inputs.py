"""Pure helpers for diagnosing and scaffolding Galaxy tool inputs.

No bioblend client, no network, no global state -- everything here is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching schemas via a Galaxy client, registering MCP tools) lives in
server.py.
"""

from typing import Any


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


def _summarize_param(p: dict[str, Any]) -> dict[str, Any]:
    ptype = p.get("type") or p.get("model_class")
    out: dict[str, Any] = {"name": p.get("name"), "type": ptype}
    if p.get("optional") is not None:
        out["optional"] = p.get("optional")
    if ptype == "repeat":
        out["repeat_key_hint"] = f"{p.get('name')}_0|<param>"
        out["children"] = [_summarize_param(c) for c in p.get("inputs", [])]
    elif ptype == "section":
        out["section_key_hint"] = f"{p.get('name')}|<param>"
        out["children"] = [_summarize_param(c) for c in p.get("inputs", [])]
    elif ptype == "conditional":
        tp = p.get("test_param") or {}
        out["selector"] = {
            "name": tp.get("name"),
            "type": tp.get("type"),
            "choices": _option_values(tp),
            "key_hint": f"{p.get('name')}|{tp.get('name')}",
        }
        out["cases"] = [
            {
                "when": case.get("value"),
                "params": [_summarize_param(c) for c in case.get("inputs", [])],
            }
            for case in p.get("cases", [])
        ]
    elif ptype == "select":
        out["choices"] = _option_values(p)
    return out


def _option_values(p: dict[str, Any]) -> list[Any]:
    # Galaxy options are [label, value, selected] triples.
    values = []
    for o in (p.get("options") or [])[:25]:
        values.append(o[1] if isinstance(o, (list, tuple)) and len(o) > 1 else o)
    return values


def summarize_tool_inputs(tool_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact a tool's io_details schema into a model-friendly parameter list.

    Preserves the nesting that matters for building flattened input keys
    (repeats -> ``name_0|param``, conditionals -> ``name|selector``, sections
    -> ``name|param``) without the full Galaxy schema noise.
    """
    if not isinstance(tool_info, dict):
        return []
    return [_summarize_param(p) for p in tool_info.get("inputs", [])]
