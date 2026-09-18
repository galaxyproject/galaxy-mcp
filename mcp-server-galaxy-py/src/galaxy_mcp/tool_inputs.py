"""Pure helpers for diagnosing and scaffolding Galaxy tool inputs.

No bioblend client, no network, no global state -- everything here is a pure
function of its arguments so it is trivially unit-testable. The I/O wiring
(fetching schemas via a Galaxy client, registering MCP tools) lives in
server.py.

check_tool_inputs refuses only what Galaxy refuses, with one deliberate
exception: a library dataset (``ldda``) on a ``data_collection`` parameter. Galaxy
accepts that and starts the job, because ``_patch_library_inputs``
(webapps/galaxy/services/tools.py) copies the dataset into the history before the
parameter is read, blind to the parameter's type; nothing matches afterwards, the
parameter resolves to None, and the tool runs with that collection input empty.
Saying so beats letting the caller's input disappear.

The other case worth knowing about is a repeat instance Galaxy would not reach --
``rep_5|x`` with no ``rep_0`` -- and it is NOT refused. ``_populate_state_legacy``
(tools/parameters/__init__.py) fills a repeat from index 0 and breaks at the first
index no supplied key mentions, so that instance is dropped and the job runs
without it, with no error and no warning. A value sitting there that is otherwise
fine is left alone, exactly as Galaxy leaves it. A value that is wrong for its
parameter is refused for being wrong, and the message then also says to number the
instances from 0, since the caller is about to edit that key anyway.
"""

import json
import re
from typing import Any, NamedTuple


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


# Galaxy's {"src": ..., "id": ...} reference kinds. workflow_inputs.validate_inputs
# tests these src strings inline; this table is the shared, named version of the same
# idea, and it knows that a library dataset is still a single dataset.
DATASET_SRCS = frozenset({"hda", "ldda", "ld"})
COLLECTION_SRCS = frozenset({"hdca"})
# A "dce" is one element of a collection, which may itself be a dataset or a
# sub-collection; nothing here can tell which, so it is never used as evidence.
ELEMENT_SRCS = frozenset({"dce"})


def classify_src(src: Any) -> str:
    """Bucket a Galaxy input src into dataset / collection / element / unknown.

    Unknown is not an error: Galaxy grows new src values, and a check that cannot
    name what it is looking at must not reject it.
    """
    if not isinstance(src, str):
        return "unknown"
    if src in DATASET_SRCS:
        return "dataset"
    if src in COLLECTION_SRCS:
        return "collection"
    if src in ELEMENT_SRCS:
        return "element"
    return "unknown"


def describe_reference(src: Any) -> str:
    """Name a supplied reference the way the workflow validator names slots."""
    kind = classify_src(src)
    if kind == "dataset":
        return f"a single dataset ({src})"
    if kind == "collection":
        return f"a dataset collection ({src})"
    if kind == "element":
        return f"a collection element ({src})"
    return f"src '{src}'"


def schema_describes_tool(tool_id: str, tool_info: Any) -> bool:
    """True when this schema is for the exact tool version that will be submitted.

    A lookup for an unversioned id resolves to whatever version is installed, and
    a lookup for a version that is NOT installed can come back describing a
    different one. Checking inputs against the wrong version's parameters is how a
    preflight invents a mismatch that isn't there, so when the returned schema does
    not match what will run, the caller skips the check instead.

    Only for a schema that was looked up by id. A caller holding the definition it
    is about to submit -- run_user_tool, with a user tool's own representation --
    has nothing to compare and does not ask.
    """
    if not isinstance(tool_info, dict):
        return False
    schema_id = tool_info.get("id")
    if not isinstance(schema_id, str) or not schema_id:
        return False
    # Galaxy expands an unversioned id to the installed version's full id.
    return schema_id == tool_id or schema_id.startswith(f"{tool_id}/")


_REPEAT_INSTANCE = re.compile(r"^(?P<name>.+)_\d+$")
_REPEAT_WILDCARD = "_#"


# A tool's parameters arrive here in one of two shapes, and the check has to read
# both. Galaxy's io_details nests a group's children under "inputs" and a
# conditional's branches under "cases"; a user-defined tool's stored representation
# is a UserToolSource, where the same things are "parameters", "test_parameter" and
# "whens" (galaxy/tool_util_models/yaml_parameters.py). Reading only the first shape
# makes the check a no-op for every nested parameter of a user tool.
def _children(param: dict[str, Any]) -> Any:
    """The nested parameter list of a repeat, a section or a conditional branch."""
    children = param.get("inputs")
    return children if isinstance(children, list) else param.get("parameters")


def _selector(param: dict[str, Any]) -> dict[str, Any]:
    """A conditional's test parameter."""
    test_param = param.get("test_param")
    if not isinstance(test_param, dict):
        test_param = param.get("test_parameter")
    return test_param if isinstance(test_param, dict) else {}


def _branches(param: dict[str, Any]) -> Any:
    """A conditional's branches. Each carries its own children."""
    cases = param.get("cases")
    return cases if isinstance(cases, list) else param.get("whens")


def _case_value(case: dict[str, Any]) -> Any:
    """The selector value that reaches this branch (io_details, then UserToolSource)."""
    value = case.get("value")
    return case.get("discriminator") if value is None else value


class _Guard(NamedTuple):
    """One conditional a parameter sits inside, and the case that reaches it.

    Galaxy populates only the selected case (``_populate_state_legacy``), so a value
    under any other case is never looked at and must never be refused. Which case is
    selected is another matter: see _live_candidates.
    """

    depth: int  # segments of the supplied key that precede this conditional's children
    selector: str  # the test param's own name
    kind: Any  # the test param's declared type
    case: Any  # this branch's declared case value, verbatim
    cases: tuple[str, ...]  # every case value the conditional declares as a string
    position: int  # which of the conditional's cases this is
    case_count: int  # how many cases the conditional declares


# The guards a leaf param sits behind, recorded on a copy of it so an index entry is
# still the schema's own parameter dict to everything that only reads its type.
_GUARDS = "__guards__"


def _walk_params(
    params: Any,
    prefix: str,
    out: dict[str, list[dict[str, Any]]],
    guards: tuple[_Guard, ...] = (),
) -> None:
    """Collect every reachable leaf param under its flattened legacy key.

    Repeat instances collapse to a single ``name_#|child`` entry because any index
    is legal. Conditionals contribute every case, not just the first, so a key that
    means different things in different branches ends up with several candidates --
    each tagged with the case that reaches it, because only one case is ever read.
    """
    if not isinstance(params, list):
        return
    for p in params:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not isinstance(name, str) or not name:
            continue
        key = f"{prefix}{name}"
        ptype = p.get("type")
        if ptype == "repeat":
            _walk_params(_children(p), f"{key}{_REPEAT_WILDCARD}|", out, guards)
        elif ptype == "section":
            _walk_params(_children(p), f"{key}|", out, guards)
        elif ptype == "conditional":
            test_param = _selector(p)
            tp_name = test_param.get("name")
            if isinstance(tp_name, str) and tp_name:
                _record(out, f"{key}|{tp_name}", test_param, guards)
            # Every case value the conditional spells out as a string; a selector has
            # to be one of these verbatim before it decides anything.
            case_values = tuple(
                value
                for case in _branches(p) or []
                if isinstance(case, dict) and isinstance(value := _case_value(case), str)
            )
            cases = [case for case in _branches(p) or [] if isinstance(case, dict)]
            for position, case in enumerate(cases):
                nested = guards
                if isinstance(tp_name, str) and tp_name:
                    nested = (
                        *guards,
                        _Guard(
                            len(key.split("|")),
                            tp_name,
                            test_param.get("type"),
                            _case_value(case),
                            case_values,
                            position,
                            len(cases),
                        ),
                    )
                _walk_params(_children(case), f"{key}|", out, nested)
        else:
            _record(out, key, p, guards)


def _record(
    out: dict[str, list[dict[str, Any]]],
    key: str,
    param: dict[str, Any],
    guards: tuple[_Guard, ...],
) -> None:
    out.setdefault(key, []).append({**param, _GUARDS: guards} if guards else param)


def index_tool_params(tool_info: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map every flattened input key a tool accepts to the param(s) it can mean."""
    out: dict[str, list[dict[str, Any]]] = {}
    if isinstance(tool_info, dict):
        _walk_params(tool_info.get("inputs"), "", out)
    return out


# What the caller's inputs say about one conditional a parameter sits under.
_NAMED = "named"  # the caller named this parameter's own case
_ELSEWHERE = "elsewhere"  # the caller named a different case of this conditional
_UNKNOWN = "unknown"  # nothing the caller said decides it


def _resolve_guard(guard: _Guard, key: str, inputs: dict[str, Any]) -> str:
    """What the supplied inputs settle about this conditional, and nothing more.

    One rule: a selector the caller supplied, as a string, that is literally one of
    the case values a *select* declares. Anything else is unknown.

    Everything that used to live here was an attempt to model how Galaxy turns a
    selector into a case, and every attempt was wrong somewhere. ``from_json`` puts a
    boolean through ``string_as_bool`` before ``to_param_dict_string`` maps it back to
    truevalue/falsevalue, so a supplied "enabled" selects the FALSE branch. A user
    tool's default comes from ``checked`` (``tool_util/parser/util.boolean_is_checked``),
    which is False unless stated, not from the ``value`` a representation carries. And
    ``GET /api/unprivileged_tools/{uuid}`` is declared
    ``response_model_exclude_defaults=True``, so the field any such guess needs is the
    first thing missing from what we are given. Guessing wrong refuses a run Galaxy
    would have accepted, so this reads only what the caller said outright.
    """
    if guard.kind != "select":
        return _UNKNOWN
    selector_key = "|".join([*key.split("|")[: guard.depth], guard.selector])
    selected = inputs.get(selector_key)
    if not isinstance(selected, str) or selected not in guard.cases:
        return _UNKNOWN
    return _NAMED if selected == guard.case else _ELSEWHERE


def _live_candidates(
    candidates: list[dict[str, Any]], key: str, inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    """The candidate params a verdict on this key may rest on.

    Galaxy populates one case of each conditional and reads nothing under the others
    (``_populate_state_legacy``), so a parameter under a case the run will not take
    must never be refused. When the caller named the case, that settles it.

    When nobody did, one claim is still available: that the value is wrong under
    every case of the conditional it sits in, so whichever case runs, it is wrong.
    That claim needs the whole conditional in view, which means exactly one
    conditional may be unresolved. **Nested conditionals with no named selector make
    no claim at all**: counting each one's cases separately says a key is covered
    when it is declared under a/x, a/y and b/x, and a run that resolves to b/y then
    never reads it. The combinations are the thing that would have to be enumerated,
    and enumerating them is how a preflight starts refusing runs that work, so it
    stays quiet instead.
    """
    surviving = []
    for param in candidates:
        chain = param.get(_GUARDS) or ()
        states = [_resolve_guard(guard, key, inputs) for guard in chain]
        if _ELSEWHERE not in states:
            surviving.append((param, chain, states))

    named = [param for param, _chain, states in surviving if all(s == _NAMED for s in states)]
    if named:
        return named

    unresolved: set[tuple[int, str]] = set()
    seen: set[int] = set()
    case_count = 0
    for _param, chain, states in surviving:
        unknown = [guard for guard, state in zip(chain, states, strict=True) if state == _UNKNOWN]
        if len(unknown) != 1:
            return []
        unresolved.add((unknown[0].depth, unknown[0].selector))
        seen.add(unknown[0].position)
        case_count = unknown[0].case_count
    # One conditional, and every case of it declaring this key. A case that does not
    # declare it is a case where the value is never looked at, so they do not all
    # agree and nothing can be said.
    if len(unresolved) != 1 or len(seen) < case_count:
        return []
    return [param for param, _chain, _states in surviving]


def _index_paths(index: dict[str, list[dict[str, Any]]]) -> set[str]:
    """Every path the schema contains, including the partial ones on the way down."""
    paths: set[str] = set()
    for key in index:
        segments = key.split("|")
        for i in range(1, len(segments) + 1):
            paths.add("|".join(segments[:i]))
    return paths


def _resolve_key(key: str, paths: set[str]) -> list[str]:
    """The schema paths a supplied key can mean, with repeat instances generalized.

    Which segments a repeat produced cannot be read off the key: a section or a
    conditional may be declared ``region_1`` just as legitimately as an instance of
    a repeat named ``region`` is written ``region_1``. So each segment is resolved
    against the paths the schema really has, and a segment is only generalized where
    the schema offers that form. Generalizing blind instead is how a real key ends up
    reported to the caller as a typo.

    A schema that offers BOTH forms for one segment -- a repeat named ``region``
    beside a section named ``region_1`` -- yields both paths, and the caller decides
    what to do with the ambiguity (check_tool_inputs only speaks up when every
    candidate agrees).
    """
    resolved = [""]
    for segment in key.split("|"):
        wildcard = _REPEAT_INSTANCE.sub(rf"\g<name>{_REPEAT_WILDCARD}", segment)
        forms = [segment] if wildcard == segment else [segment, wildcard]
        step = [
            candidate
            for prefix in resolved
            for form in forms
            if (candidate := f"{prefix}|{form}" if prefix else form) in paths
        ]
        if not step:
            return []
        resolved = step
    return resolved


def _lookup_param(key: str, index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Resolve a supplied key, generalizing repeat instances where the schema has one.

    The wildcard matches any index, including one Galaxy would never read: it fills
    a repeat from index 0 and breaks at the first index no supplied key mentions
    (``_populate_state_legacy``, galaxy/tools/parameters/__init__.py), so
    ``rep_5|x`` sent without a ``rep_0|...`` is dropped and the job runs without it.
    Resolving it anyway is what lets the value be checked like any other. Being at
    an index Galaxy will not reach is never itself a reason to refuse; it only adds
    the advice to renumber to a refusal the value has earned on its own.
    """
    if key in index:
        return index[key]
    return [
        p for resolved in _resolve_key(key, _index_paths(index)) for p in index.get(resolved, [])
    ]


def _walk_repeat_floors(params: Any, prefix: str, floors: dict[str, int]) -> None:
    for p in params if isinstance(params, list) else []:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not isinstance(name, str) or not name:
            continue
        key = f"{prefix}{name}"
        ptype = p.get("type")
        if ptype == "repeat":
            floors[key] = max(_as_index(p.get("min")), _as_index(p.get("default")))
            _walk_repeat_floors(_children(p), f"{key}{_REPEAT_WILDCARD}|", floors)
        elif ptype == "section":
            _walk_repeat_floors(_children(p), f"{key}|", floors)
        elif ptype == "conditional":
            for case in _branches(p) or []:
                if isinstance(case, dict):
                    _walk_repeat_floors(_children(case), f"{key}|", floors)


def _as_index(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def index_repeat_floors(tool_info: Any) -> dict[str, int]:
    """The lowest index Galaxy might stop at, per repeat, keyed by wildcard path.

    Galaxy creates every instance below ``max(min, default)`` whether the caller
    mentioned it or not, and only starts looking for a gap at that floor
    (``_populate_state_legacy``). A repeat declaring ``min="2"`` therefore does read
    a lone ``rep_2|x``, and telling that caller to renumber would be wrong. Both keys
    reach us: Repeat.dict_collection_visible_keys puts min and default in io_details.
    """
    floors: dict[str, int] = {}
    if isinstance(tool_info, dict):
        _walk_repeat_floors(tool_info.get("inputs"), "", floors)
    return floors


def _repeat_instances(
    key: str, index: dict[str, list[dict[str, Any]]]
) -> list[tuple[str, str, int]]:
    """The repeat instances a supplied key names, as ``(path, wildcard path, index)``.

    A segment counts as a repeat instance only when resolving the key against the
    schema had to generalize it, which is the one piece of evidence available that
    it is not a section or a param that happens to be named that way. ``path`` keeps
    the parent's concrete instance, because Galaxy scopes an inner repeat to the
    outer instance it sits in; the wildcard path is what the floor table is keyed by.
    """
    if key in index:
        return []
    segments = key.split("|")
    for resolved in _resolve_key(key, _index_paths(index)):
        if resolved not in index:
            continue
        general = resolved.split("|")
        return [
            (
                "|".join([*segments[:i], general[i].removesuffix(_REPEAT_WILDCARD)]),
                "|".join([*general[:i], general[i].removesuffix(_REPEAT_WILDCARD)]),
                int(segments[i].rsplit("_", 1)[1]),
            )
            for i in range(len(segments))
            if segments[i] != general[i]
        ]
    return []


def _supplied_repeat_indexes(
    inputs: dict[str, Any], index: dict[str, list[dict[str, Any]]]
) -> dict[str, set[int]]:
    """Every repeat instance index the caller supplied, grouped by repeat path."""
    seen: dict[str, set[int]] = {}
    for key in inputs:
        if isinstance(key, str):
            for path, _wildcard, position in _repeat_instances(key, index):
                seen.setdefault(path, set()).add(position)
    return seen


def _is_unreachable_repeat_key(
    key: str,
    index: dict[str, list[dict[str, Any]]],
    supplied: dict[str, set[int]],
    floors: dict[str, int],
) -> bool:
    """True when Galaxy would stop filling the repeat before it reached this key."""
    for path, wildcard, position in _repeat_instances(key, index):
        floor = floors.get(wildcard, 0)
        if position >= floor and not set(range(floor, position + 1)) <= supplied.get(path, set()):
            return True
    return False


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


def is_reference(value: Any) -> bool:
    """A Galaxy {"src": ..., "id": ...} input value, as workflow_inputs tests it.

    ``src`` has to be a string to mean anything; a caller that sends something else
    has sent Galaxy nonsense, and this code should not try to interpret it.
    """
    return isinstance(value, dict) and isinstance(value.get("src"), str)


def _is_batch_wrapper(value: Any) -> bool:
    """Galaxy's meta-parameter wrapper, which expand_meta_parameters detects by "values".

    Anything wrapped this way -- map-over, multi-run, cartesian -- is expanded before
    a parameter ever sees it, so its contents are none of this checker's business.
    """
    return isinstance(value, dict) and "values" in value


def _takes_one_dataset(param: dict[str, Any]) -> bool:
    """True when this data param accepts exactly one dataset.

    Galaxy's io_details always reports ``multiple``, but a user-defined tool's
    stored representation carries only what its author wrote. Absent means single:
    galaxy-tool-util's DataParameterModel declares ``multiple: bool = False``, so a
    param that never mentions it is a one-dataset param and an hdca will fail there.
    """
    return param.get("type") == "data" and not param.get("multiple")


# What Galaxy really does with each dataset src on a data_collection param. All three
# are refused, but for three different reasons, and only one of them is an error the
# caller would ever see (verified in galaxyproject/galaxy):
#  - hda  -> DataCollectionToolParameter.from_json calls src_id_to_item_collection,
#            which raises "Expected to find collection, but got single dataset wrapper".
#  - ldda -> _patch_library_inputs (webapps/galaxy/services/tools.py) copies every
#            top-level ldda into the history before any parameter sees it, ignoring
#            the parameter's type. The resulting HDA matches no from_json branch, so
#            the parameter resolves to None and the job starts without the collection.
#            Only for a library dataset the caller can read: _patch_library_dataset
#            returns nothing otherwise, the ldda survives, and it 400s like an hda.
#  - ld   -> src_id_to_item has no "ld" entry and raises "Unknown input source ld
#            passed to job submission API"; only the workflow API resolves ld.
_COLLECTION_PARAM_FATE = {
    "hda": "Galaxy answers 400 ('Expected to find collection, but got single dataset wrapper').",
    "ldda": (
        "Galaxy copies a readable library dataset into the history before the parameter "
        "sees it, so this parameter ends up empty: the run is accepted and the job starts "
        "without the collection rather than failing (one you cannot read 400s instead)."
    ),
    "ld": (
        "Galaxy's tool-run API does not resolve 'ld' at all -- only the workflow API does "
        "-- and answers 400 ('Unknown input source ld passed to job submission API')."
    ),
}


def _check_value(value: Any, param: dict[str, Any]) -> str | None:
    """Return a reason this value cannot work for this param, or None.

    Only failures Galaxy is known to refuse are named. Everything else -- an
    extension that does not match, a collection whose type differs from the
    param's, a bare id, a src this code does not recognise -- is left alone,
    because Galaxy accepts those and a false reject is worse than no check.
    """
    ptype = param.get("type")

    if (
        _takes_one_dataset(param)
        and is_reference(value)
        and not _is_batch_wrapper(value)
        and classify_src(value.get("src")) == "collection"
    ):
        return (
            f"expects {describe_reference('hda')}; got {describe_reference(value['src'])}. "
            'To run the tool once per element, pass {"batch": true, "values": '
            f'[{{"src": "{value["src"]}", "id": "..."}}]}}'
        )

    if ptype == "data" and param.get("multiple") and isinstance(value, list):
        srcs = {v["src"] for v in value if is_reference(v)}
        if "hdca" in srcs and len(srcs) > 1:
            return (
                "mixes a dataset collection (hdca) with other sources "
                f"({', '.join(sorted(srcs))}); when collections are supplied to a "
                "multiple-data parameter, every value must be a collection"
            )

    if (
        ptype == "data_collection"
        and is_reference(value)
        and not _is_batch_wrapper(value)
        and classify_src(value.get("src")) == "dataset"
    ):
        return (
            f"expects {describe_reference('hdca')}; "
            f"got {describe_reference(value['src'])}. "
            f"{_COLLECTION_PARAM_FATE.get(value['src'], _COLLECTION_PARAM_FATE['hda'])} "
            "Pass a collection id, or build one with a collection-creating tool"
        )

    return None


def schema_has_inputs(tool_info: Any) -> bool:
    """True when the schema actually carries an input list to check against.

    A tool with zero inputs is checkable and trivially fine; a schema fetched
    without ``io_details`` has no ``inputs`` key at all and is not checkable, and
    the two must not be confused -- reporting the second as "checked, all clear"
    is the silent-skip this preflight exists to avoid.
    """
    return isinstance(tool_info, dict) and isinstance(tool_info.get("inputs"), list)


# Keys Galaxy handles itself, outside a tool's own parameter list. Calling these
# unrecognised sends an agent to delete something that works, so they are dropped from
# the unrecognised list rather than reported. Sourced from Galaxy: dbkey and
# rerun_remap_job_id are read from the incoming inputs (tools/actions/__init__.py and
# tools/__init__.py), use_cached_job and send_email_notification off `inputs` in
# webapps/galaxy/services/tools.py, and __job_resource is the conditional
# jobs/__init__.py defines -- spliced into a tool's inputs at parse time when job
# resource params are configured, so it can legitimately appear in io_details. chromInfo
# is the odd one: Galaxy overwrites it with its own resolved value rather than reading
# the caller's, so it is a no-op input, which is still not a typo. A declared parameter
# is matched before this list is consulted, so a tool that names one of these itself is
# checked like any other.
GALAXY_MANAGED_INPUT_KEYS = frozenset(
    {
        "dbkey",
        "chromInfo",
        "use_cached_job",
        "send_email_notification",
        "rerun_remap_job_id",
        "__job_resource",
    }
)
_GALAXY_MANAGED_KEY_PREFIXES = ("__job_resource|",)


def is_galaxy_managed_key(key: str) -> bool:
    """True for an input key Galaxy handles itself rather than through the tool schema."""
    return key in GALAXY_MANAGED_INPUT_KEYS or key.startswith(_GALAXY_MANAGED_KEY_PREFIXES)


_REPEAT_NUMBERING_HINT = (
    "Also number this tool's repeat instances from 0 with no gaps: Galaxy fills a "
    "repeat starting at index 0 and stops at the first index no key mentions, so an "
    "instance past a gap is dropped without an error."
)


class ToolInputsUncheckableError(Exception):
    """The checker could not read this schema, so nothing was checked.

    Distinct from a clean verdict with no rejects. Swallowing the difference is how
    a caller ends up telling an agent the inputs were fine when they were never
    looked at.
    """


def check_tool_inputs(
    tool_info: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Preflight supplied tool inputs against a tool's io_details schema.

    Returns ``{"rejects": [...], "warnings": [...]}`` like
    workflow_inputs.validate_inputs; rejects carry ``param`` and ``reason``.
    Raises ToolInputsUncheckableError, and nothing else, when the schema defeats it --
    a checker that blows up must not block a run, but it must not pass for one
    either.
    """
    rejects: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(inputs, dict):
        return {"rejects": rejects, "warnings": warnings}

    try:
        index = index_tool_params(tool_info)
        supplied_repeats = _supplied_repeat_indexes(inputs, index)
        repeat_floors = index_repeat_floors(tool_info)
        for key, value in inputs.items():
            if not isinstance(key, str):
                continue
            candidates = _lookup_param(key, index)
            if not candidates:
                if not is_galaxy_managed_key(key):
                    warnings.append(
                        {"param": key, "message": f"'{key}' is not a parameter of this tool."}
                    )
                continue
            live = _live_candidates(candidates, key, inputs)
            if not live:
                # Nothing here can be called wrong: either the run will not read this
                # key at all, or which parameter it means cannot be established.
                continue
            # A key can still mean several params -- different cases, or the same
            # repeat reached two ways. Only speak up when all of them agree.
            reasons = [_check_value(value, p) for p in live]
            if all(reasons):
                reason = reasons[0]
                # Rides on a refusal the value earned; an unreachable index is not
                # one on its own, because Galaxy drops the key rather than failing.
                if _is_unreachable_repeat_key(key, index, supplied_repeats, repeat_floors):
                    reason = f"{reason}. {_REPEAT_NUMBERING_HINT}"
                rejects.append({"param": key, "reason": reason})
    except Exception as e:  # noqa: BLE001 -- reported as unchecked, never as all-clear
        raise ToolInputsUncheckableError(type(e).__name__) from e

    return {"rejects": rejects, "warnings": warnings}


# Where to send a caller for the shape they should have sent. A user-defined tool is
# scoped to its owner and never enters the global toolbox, so get_tool_input_template
# and get_tool_details 404 for one -- its shape only comes back with the tool itself.
SHAPE_HINT = "Call get_tool_input_template(tool_id) for the expected shape"
USER_TOOL_SHAPE_HINT = (
    "Read this tool's representation from list_user_tools() for the expected shape "
    "(a user-defined tool is not in the toolbox, so get_tool_input_template does not "
    "know it)"
)


def format_input_rejects(
    tool_id: str, rejects: list[dict[str, Any]], *, shape_hint: str = SHAPE_HINT
) -> str:
    """Render preflight rejects the way invoke_workflow renders its own."""
    lines = [f"  - {r['param']}: {r['reason']}" for r in rejects]
    return (
        f"Tool inputs failed validation; not submitting to '{tool_id}':\n"
        + "\n".join(lines)
        + f"\n\n{shape_hint}, then fix `inputs` and call the tool again."
    )


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
