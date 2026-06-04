"""Pure helpers for workflow input templates and invoke-time validation.

No bioblend client, no network, no global state -- every function is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching style=run / .ga / datatypes via a Galaxy client, registering MCP
tools) lives in server.py. Mirrors tool_inputs.py.
"""

import json
from typing import Any

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
    for key, step in sorted(steps.items(), key=lambda kv: int(kv[0])):
        input_type = _INPUT_TYPE_MAP.get(step.get("type", ""))
        if input_type is None:
            continue
        index = int(key)
        state = _coerce_state(step.get("tool_state"))
        label = step.get("label") or f"{_FALLBACK_LABEL[input_type]} (step {index})"
        slots.append(
            {
                "step_index": index,
                "step_uuid": step.get("uuid"),
                "label": label,
                "input_type": input_type,
                "src": _SRC_MAP[input_type],
                "accepted_formats": list(state.get("format") or []),
                "collection_type": state.get("collection_type"),
                "parameter_type": state.get("parameter_type"),
                "optional": bool(state.get("optional", False)),
            }
        )
    return slots


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
        step_iter = [raw_steps[k] for k in sorted(raw_steps, key=lambda k: int(k))]
    else:
        step_iter = list(raw_steps or [])
    slots: list[dict[str, Any]] = []
    for step in step_iter:
        input_type = _INPUT_TYPE_MAP.get(step.get("step_type") or step.get("type", ""))
        if input_type is None:
            continue
        index = int(step.get("step_index", step.get("order_index", step.get("id"))))
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
        slots.append(
            {
                "step_index": index,
                "step_uuid": step.get("uuid"),
                "label": label,
                "input_type": input_type,
                "src": _SRC_MAP[input_type],
                "accepted_formats": list(param.get("extensions") or []),
                "collection_type": ctype,
                "parameter_type": param.get("parameter_type") or step.get("parameter_type"),
                "optional": bool(param.get("optional", False)),
            }
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
    for key, step in sorted(definition.get("steps", {}).items(), key=lambda kv: int(kv[0])):
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
