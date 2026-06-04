"""Pure helpers for workflow input templates and invoke-time validation.

No bioblend client, no network, no global state -- every function is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching style=run / .ga / datatypes via a Galaxy client, registering MCP
tools) lives in server.py. Mirrors tool_inputs.py.
"""

from typing import Any


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
