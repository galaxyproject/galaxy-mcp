"""Pure helpers for diagnosing and scaffolding Galaxy tool inputs.

No bioblend client, no network, no global state -- everything here is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching schemas via a Galaxy client, registering MCP tools) lives in
server.py.
"""

import json
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
        if _options_truncated(tp):
            out["selector"]["choices_truncated"] = True
        out["cases"] = [
            {
                "when": case.get("value"),
                "params": [_summarize_param(c) for c in case.get("inputs", [])],
            }
            for case in p.get("cases", [])
        ]
    elif ptype == "select":
        out["choices"] = _option_values(p)
        if _options_truncated(p):
            out["choices_truncated"] = True
    return out


_MAX_OPTIONS = 25


def _option_values(p: dict[str, Any]) -> list[Any]:
    # Galaxy options are [label, value, selected] triples; cap to keep summaries compact.
    options = p.get("options") or []
    return [
        o[1] if isinstance(o, (list, tuple)) and len(o) > 1 else o for o in options[:_MAX_OPTIONS]
    ]


def _options_truncated(p: dict[str, Any]) -> bool:
    # Signal when the cap dropped choices so a model doesn't read a clipped list as
    # the full set. Mirrors the `*_truncated` fields used elsewhere (see server.py).
    return len(p.get("options") or []) > _MAX_OPTIONS


def _placeholder(p: dict[str, Any]) -> Any:
    ptype = p.get("type")
    if ptype == "data":
        return {"src": "hda", "id": "<dataset_id>"}
    if ptype == "data_collection":
        return {"src": "hdca", "id": "<collection_id>"}
    if ptype == "select":
        choices = _option_values(p)
        return choices[0] if choices else "<choice>"
    if ptype == "boolean":
        return False
    if ptype == "integer":
        return 0
    if ptype == "float":
        return 0.0
    return "<value>"


def _fill_param(p: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
    name = p.get("name")
    if name is None:
        return
    key = f"{prefix}{name}"
    ptype = p.get("type")
    if ptype == "repeat":
        for child in p.get("inputs", []):
            _fill_param(child, prefix=f"{key}_0|", out=out)
    elif ptype == "section":
        for child in p.get("inputs", []):
            _fill_param(child, prefix=f"{key}|", out=out)
    elif ptype == "conditional":
        tp = p.get("test_param") or {}
        tp_name = tp.get("name")
        cases = p.get("cases", [])
        first = cases[0] if cases else None
        sel_value = first.get("value") if first else "<choice>"
        if tp_name:
            out[f"{key}|{tp_name}"] = sel_value
        if first:
            for child in first.get("inputs", []):
                _fill_param(child, prefix=f"{key}|", out=out)
    else:
        out[key] = _placeholder(p)


def build_input_template(tool_info: dict[str, Any]) -> dict[str, Any]:
    """Build a ready-to-fill flattened ``inputs`` skeleton from a tool schema.

    Data params -> ``{"src": "hda", "id": "<dataset_id>"}``; selects -> a valid
    choice; conditionals -> the first case's selector + that branch's params;
    repeats -> one ``name_0|...`` instance (duplicate with ``name_1|...`` to add
    more); sections -> ``name|...``.
    """
    out: dict[str, Any] = {}
    if not isinstance(tool_info, dict):
        return out
    for p in tool_info.get("inputs", []):
        _fill_param(p, prefix="", out=out)
    return out


def summarize_tool_inputs(tool_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact a tool's io_details schema into a model-friendly parameter list.

    Preserves the nesting that matters for building flattened input keys
    (repeats -> ``name_0|param``, conditionals -> ``name|selector``, sections
    -> ``name|param``) without the full Galaxy schema noise.
    """
    if not isinstance(tool_info, dict):
        return []
    return [_summarize_param(p) for p in tool_info.get("inputs", [])]


def format_input_mismatch_error(
    *,
    original_error: str,
    tool_id: str,
    schema_summary: list[dict[str, Any]] | None,
    example: Any | None,
) -> str:
    """Assemble a truthful, actionable error for a likely input-shape mismatch.

    Preserves the original error verbatim and offers the schema as help; it does
    NOT assert a specific cause (we cannot reliably tell a wrong name from a
    missing value from a bad dataset id).
    """
    lines = [
        original_error,
        "",
        f"This most likely means the `inputs` you provided do not match the parameter "
        f"schema for tool '{tool_id}'. It is not a sign that the MCP or the Galaxy "
        f"version is incompatible. (Galaxy sometimes reports input problems as a "
        f'misleading "Required parameter(s) kwd not provided in request" error -- '
        f"ignore that wording.)",
    ]
    if schema_summary is not None:
        lines += [
            "",
            f"Expected input parameters for '{tool_id}' (build flattened keys like "
            f"`section|param`, `cond|selector`, `repeat_0|param`):",
            json.dumps(schema_summary, indent=2, default=str),
        ]
    if example is not None:
        lines += [
            "",
            "Structural example from a tool test -- NOT runnable: the dataset IDs below "
            "will not exist in your history. Copy the shape, not the values:",
            json.dumps(example, indent=2, default=str),
        ]
    if schema_summary is None and example is None:
        lines += [
            "",
            "Call get_tool_details(tool_id, io_details=True) (or "
            "get_tool_input_template(tool_id)) to see the parameter schema, then rebuild "
            "`inputs` and retry.",
        ]
    else:
        lines += ["", "Rebuild `inputs` to match the schema above and call the tool again."]
    return "\n".join(lines)
