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
