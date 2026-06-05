"""Pure helpers for workflow input templates and invoke-time validation.

No bioblend client, no network, no global state -- every function is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching style=run / .ga / datatypes via a Galaxy client, registering MCP
tools) lives in server.py. Mirrors tool_inputs.py.
"""

import json
from typing import Any


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Best-effort int coercion; returns ``default`` instead of raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list:
    """Coerce a possibly-scalar field to a list. A bare string becomes a single-element list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_readme_summary(readme: str, max_length: int = 300) -> str:
    """Extract a clean summary from a readme, stripping markdown headers."""
    if not readme:
        return ""
    lines = readme.split("\n")
    clean_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            continue
        if not clean_lines and not line.strip():
            continue
        clean_lines.append(line)
    text = " ".join(clean_lines)
    text = " ".join(text.split())
    if len(text) > max_length:
        text = text[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return text


def _options_from_triples(raw: Any) -> list[dict]:
    """style=run param options come as [label, value, selected] triples."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append({"label": item[0], "value": item[1]})
    return out


def _options_from_restrictions(raw: Any) -> list[dict]:
    """.ga enumerated params carry a flat list of allowed string values."""
    return [{"label": v, "value": v} for v in raw] if isinstance(raw, list) else []


_SORT_SENTINEL = 10**9  # sorts non-numeric step keys to the end without crashing

_INPUT_TYPE_MAP = {
    "data_input": "data",
    "data_collection_input": "data_collection",
    "parameter_input": "parameter",
}
_SRC_MAP = {"data": "hda", "data_collection": "hdca", "parameter": None}
_FALLBACK_LABEL = {
    "data": "Input dataset",
    "data_collection": "Input dataset collection",
    "parameter": "Input parameter",
}


def _coerce_state(tool_state: Any) -> dict[str, Any]:
    if isinstance(tool_state, str):
        try:
            return json.loads(tool_state)
        except (ValueError, TypeError):
            return {}
    return tool_state if isinstance(tool_state, dict) else {}


def subtype_satisfies(supplied_ext: str, accepted_exts: list[str], mapping: dict[str, Any]) -> bool:
    """True if a dataset of ``supplied_ext`` is acceptable where ``accepted_exts`` are required.

    ``accepted_exts == []`` means the slot accepts any datatype. Uses Galaxy's
    datatype class hierarchy: supplied satisfies accepted if some accepted
    extension's class is in the supplied extension's class ancestry (so ``bed``
    satisfies ``tabular``). Unknown supplied/accepted extensions are treated
    permissively -- we only reject a *provable* mismatch.
    """
    if not accepted_exts:
        return True
    ext_to_class = mapping.get("ext_to_class_name", {})
    class_to_classes = mapping.get("class_to_classes", {})
    supplied_class = ext_to_class.get(supplied_ext)
    if supplied_class is None:
        return True  # cannot prove a mismatch for an unknown extension
    ancestry = class_to_classes.get(supplied_class, {})
    for accepted in accepted_exts:
        accepted_class = ext_to_class.get(accepted)
        if accepted_class is None:
            return True  # unknown accepted extension -> don't claim a mismatch
        if accepted_class == supplied_class or accepted_class in ancestry:
            return True
    return False


def normalize_ga_steps(definition: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a .ga / IWC-manifest workflow ``definition`` into input slots.

    Reads ``definition["steps"]`` (dict keyed by string order_index), keeps only
    data_input / data_collection_input / parameter_input steps, and parses each
    step's ``tool_state`` for the declared constraints. Absent ``format`` means
    "no restriction" (empty accepted_formats).
    """
    steps = definition.get("steps", {})
    slots: list[dict[str, Any]] = []
    # Sort numeric keys first (ascending), then non-numeric keys at the end.
    # Non-numeric keys are skipped below; the sentinel just keeps sort() from crashing.
    for key, step in sorted(
        steps.items(), key=lambda kv: (_safe_int(kv[0], _SORT_SENTINEL), kv[0])
    ):
        index = _safe_int(key)
        if index is None:
            continue  # non-numeric key -- skip; .ga files should never have these
        input_type = _INPUT_TYPE_MAP.get(step.get("type", ""))
        if input_type is None:
            continue
        state = _coerce_state(step.get("tool_state"))
        label = step.get("label") or f"{_FALLBACK_LABEL[input_type]} (step {index})"
        slots.append(
            _make_slot(
                step_index=index,
                step_uuid=step.get("uuid"),
                label=label,
                input_type=input_type,
                accepted_formats=_as_list(state.get("format")),
                acceptable_extensions=[],
                collection_type=state.get("collection_type"),
                parameter_type=state.get("parameter_type"),
                optional=bool(state.get("optional", False)),
                options=_options_from_restrictions(state.get("restrictions")),
            )
        )
    return slots


def _make_slot(
    *,
    step_index: int,
    step_uuid: str | None,
    label: str,
    input_type: str,
    accepted_formats: list,
    acceptable_extensions: list,
    collection_type: str | None,
    parameter_type: str | None,
    optional: bool,
    options: list,
) -> dict[str, Any]:
    """Build the slot contract dict shared by both normalizers."""
    return {
        "step_index": step_index,
        "step_uuid": step_uuid,
        "label": label,
        "input_type": input_type,
        "src": _SRC_MAP[input_type],
        "accepted_formats": accepted_formats,
        "acceptable_extensions": acceptable_extensions,
        "collection_type": collection_type,
        "parameter_type": parameter_type,
        "optional": optional,
        "options": options,
    }


# style=run step "step_type" -> our input_type. Galaxy uses the same
# data_input/data_collection_input/parameter_input discriminators here.
def normalize_run_model(run_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a style=run workflow model into input slots (the slot contract).

    style=run is the webapp's own run-form serialization; for data inputs its
    ``extensions`` already reflect Galaxy's downstream-consumer resolution. We
    consume it rather than reconstruct from connections.
    """
    raw_steps = run_dict.get("steps")
    if isinstance(raw_steps, dict):
        # non-numeric keys get sorted to the end and skipped below
        step_iter = [
            raw_steps[k] for k in sorted(raw_steps, key=lambda k: (_safe_int(k, _SORT_SENTINEL), k))
        ]
    else:
        step_iter = list(raw_steps or [])
    slots: list[dict[str, Any]] = []
    for step in step_iter:
        input_type = _INPUT_TYPE_MAP.get(step.get("step_type") or step.get("type", ""))
        if input_type is None:
            continue
        # style=run can expose step_index, order_index, or id; all three may be absent
        # or may be a hash string on unusual exports -- skip rather than crash.
        raw_idx = step.get("step_index", step.get("order_index", step.get("id")))
        index = _safe_int(raw_idx)
        if index is None:
            continue
        # The run model nests the actual param under "inputs"[0] for input steps.
        param = (step.get("inputs") or [{}])[0]
        ctype = param.get("collection_type")
        if not ctype:
            ctypes = param.get("collection_types")
            ctype = ctypes[0] if isinstance(ctypes, list) and ctypes else None
        # style=run uses "step_label" for the step name; param["label"] is a fallback.
        label = (
            step.get("step_label")
            or param.get("label")
            or f"{_FALLBACK_LABEL[input_type]} (step {index})"
        )
        # parameter_type: style=run puts it on param first, then step. The .ga path
        # reads it only from tool_state -- different source schemas, intentional asymmetry.
        slots.append(
            _make_slot(
                step_index=index,
                step_uuid=step.get("uuid"),
                label=label,
                input_type=input_type,
                accepted_formats=_as_list(param.get("extensions")),
                acceptable_extensions=_as_list(param.get("acceptable_extensions")),
                collection_type=ctype,
                parameter_type=param.get("parameter_type") or step.get("parameter_type"),
                optional=bool(param.get("optional", False)),
                options=_options_from_triples(param.get("options")),
            )
        )
    return slots


def _has_runtime_value(obj: Any) -> bool:
    """True if ``obj`` contains a bare RuntimeValue marker (not ConnectedValue)."""
    if isinstance(obj, dict):
        if obj.get("__class__") == "RuntimeValue":
            return True
        return any(_has_runtime_value(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_runtime_value(v) for v in obj)
    return False


def find_legacy_warnings(definition: dict[str, Any]) -> list[dict[str, str]]:
    """Warn on tool steps carrying unconnected RuntimeValue params (the real
    legacy-run-form trigger). Bare ``parameter_input`` steps are formal inputs
    and are NOT flagged.
    """
    warnings: list[dict[str, str]] = []
    for key, step in sorted(
        definition.get("steps", {}).items(),
        key=lambda kv: (_safe_int(kv[0], _SORT_SENTINEL), kv[0]),
    ):
        if step.get("type") != "tool":
            continue
        if _has_runtime_value(_coerce_state(step.get("tool_state"))):
            name = step.get("label") or step.get("tool_id") or f"step {key}"
            warnings.append(
                {
                    "kind": "legacy_runtime_value",
                    "message": (
                        f"Tool step '{name}' has a RuntimeValue parameter set at runtime "
                        f"(legacy run-form pattern); the workflow may not run cleanly via the API."
                    ),
                }
            )
    return warnings


def _collection_type_compatible(supplied: str | None, required: str | None) -> bool:
    """Direct or map-over compatibility. None required == accept any shape."""
    if not required or not supplied:
        return True
    if supplied == required:
        return True
    # map-over: a list:paired collection can feed a 'paired' (or 'list:paired') slot.
    # Compare colon-delimited segments, not a raw string suffix.
    return supplied.split(":")[-1] == required or supplied.endswith(":" + required)


def _ext_accepted(supplied_ext: str, slot: dict[str, Any], mapping: dict[str, Any]) -> bool:
    """True if a dataset of supplied_ext is acceptable for this slot.

    Prefer Galaxy's own converter+subclass-aware acceptable_extensions (exact GUI
    parity, available from style=run); fall back to subclass closure on the
    declared accepted_formats (the .ga path, which lacks the converter set).
    """
    acc = slot.get("acceptable_extensions") or []
    if acc:
        return supplied_ext in acc
    return subtype_satisfies(supplied_ext, slot["accepted_formats"], mapping)


def validate_inputs(
    slots: list[dict[str, Any]],
    supplied: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Three-tier preflight. ``supplied`` keyed by str(step_index); data entries
    carry ``ext``; collection entries carry ``collection_type`` and optional
    ``element_extensions``; parameter entries are scalars. Rejects are provable
    structural/datatype mismatches; warnings are inferred/uncertain.
    """
    rejects: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    # Detect duplicate step_index values before collapsing -- silent collision
    # would validate only the last slot, hiding the schema problem.
    seen: set[str] = set()
    for s in slots:
        idx = str(s["step_index"])
        if idx in seen:
            warnings.append(
                {
                    "step_index": idx,
                    "message": (
                        f"Workflow has multiple input slots sharing step_index {idx};"
                        f" only one will be validated."
                    ),
                }
            )
        seen.add(idx)
    by_index = {str(s["step_index"]): s for s in slots}

    for key, value in supplied.items():
        slot = by_index.get(str(key))
        if slot is None:
            warnings.append(
                {
                    "step_index": key,
                    "message": (
                        f"Supplied input for step {key} has no matching workflow input slot."
                    ),
                }
            )
            continue
        itype = slot["input_type"]
        is_ref = isinstance(value, dict) and "src" in value

        if itype == "parameter":
            if is_ref:
                rejects.append(
                    {
                        "step_index": slot["step_index"],
                        "label": slot["label"],
                        "reason": (
                            "Parameter input given a dataset/collection reference;"
                            " expected a scalar value."
                        ),
                    }
                )
            continue

        if not is_ref:
            rejects.append(
                {
                    "step_index": slot["step_index"],
                    "label": slot["label"],
                    "reason": f"{itype} input expects a {{'src','id'}} reference.",
                }
            )
            continue

        if itype == "data":
            if value["src"] != "hda":
                rejects.append(
                    {
                        "step_index": slot["step_index"],
                        "label": slot["label"],
                        "reason": (
                            f"Slot expects a single dataset (hda);"
                            f" got a collection ({value['src']})."
                        ),
                    }
                )
                continue
            ext = value.get("ext")
            if ext and not _ext_accepted(ext, slot, mapping):
                rejects.append(
                    {
                        "step_index": slot["step_index"],
                        "label": slot["label"],
                        "reason": (
                            f"Dataset datatype '{ext}' is not accepted here "
                            f"(expects: {', '.join(slot['accepted_formats'])})."
                        ),
                    }
                )
            elif not ext and slot["accepted_formats"]:
                warnings.append(
                    {
                        "step_index": slot["step_index"],
                        "message": (
                            f"Could not determine datatype of the dataset for '{slot['label']}'; "
                            f"slot expects {', '.join(slot['accepted_formats'])}."
                        ),
                    }
                )

        elif itype == "data_collection":
            if value["src"] != "hdca":
                rejects.append(
                    {
                        "step_index": slot["step_index"],
                        "label": slot["label"],
                        "reason": f"Slot expects a dataset collection (hdca); got {value['src']}.",
                    }
                )
                continue
            if not _collection_type_compatible(
                value.get("collection_type"), slot["collection_type"]
            ):
                rejects.append(
                    {
                        "step_index": slot["step_index"],
                        "label": slot["label"],
                        "reason": (
                            f"Collection type '{value.get('collection_type')}' is incompatible "
                            f"with the slot's '{slot['collection_type']}'."
                        ),
                    }
                )
                continue
            for el_ext in value.get("element_extensions") or []:
                if not _ext_accepted(el_ext, slot, mapping):
                    rejects.append(
                        {
                            "step_index": slot["step_index"],
                            "label": slot["label"],
                            "reason": (
                                f"Collection element datatype '{el_ext}' is not accepted here "
                                f"(expects: {', '.join(slot['accepted_formats'])})."
                            ),
                        }
                    )
                    break

    # warn on missing required slots (don't block -- the model may add them next call)
    for slot in slots:
        if not slot["optional"] and str(slot["step_index"]) not in supplied:
            warnings.append(
                {
                    "step_index": slot["step_index"],
                    "message": (
                        f"Required input '{slot['label']}'"
                        f" (step {slot['step_index']}) not supplied yet."
                    ),
                }
            )
    return {"rejects": rejects, "warnings": warnings}


def _placeholder_for(slot: dict[str, Any]) -> Any:
    if slot["input_type"] == "data":
        return {"src": "hda", "id": "<dataset_id>"}
    if slot["input_type"] == "data_collection":
        return {"src": "hdca", "id": "<collection_id>"}
    return "<value>"


def build_workflow_input_template(
    slots: list[dict[str, Any]], warnings: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Assemble the model-facing template: a ready-to-fill skeleton keyed by
    step_index, the per-slot constraint summary, the invoke key hint, and any
    legacy warnings.
    """
    display_slots = [{k: v for k, v in s.items() if k != "acceptable_extensions"} for s in slots]
    return {
        "inputs_template": {str(s["step_index"]): _placeholder_for(s) for s in slots},
        "slots": display_slots,
        "inputs_by": "step_index|step_uuid",
        "warnings": warnings or [],
    }
