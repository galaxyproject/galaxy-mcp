"""Helpers to expose structured Galaxy tool contracts and lint tool payloads."""

from __future__ import annotations

import re
from typing import Any


def _join_key(prefix: str, name: str) -> str:
    return f"{prefix}|{name}" if prefix else name


def _is_missing_value(value: Any) -> bool:
    return value in (None, "", [], {})


_REPEAT_SEGMENT_RE = re.compile(r"^(?P<base>.+_repeat)_\d+$")


def _normalize_repeat_segment(segment: str) -> str:
    match = _REPEAT_SEGMENT_RE.match(segment)
    if not match:
        return segment
    return str(match.group("base"))


def _normalize_repeat_key(key: str) -> str:
    return "|".join(_normalize_repeat_segment(part) for part in key.split("|"))


def _is_data_column_type(field_type: str) -> bool:
    lowered = field_type.strip().lower()
    return lowered == "data_column" or "datacolumn" in lowered


def _is_text_like_field_type(field_type: str) -> bool:
    lowered = field_type.strip().lower()
    if lowered in {"text", "textarea", "string", "str"}:
        return True
    return "texttoolparameter" in lowered or "textareatoolparameter" in lowered


def _looks_like_json_blob(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 2:
        return False
    if stripped[0] == "{" and stripped[-1] == "}":
        return True
    if stripped[0] == "[" and stripped[-1] == "]":
        return True
    return False


def _normalize_options(options: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(options, list):
        return normalized

    for option in options:
        label: Any = None
        value: Any = None
        selected = False

        if isinstance(option, dict):
            label = option.get("label", option.get("name", option.get("value")))
            value = option.get("value")
            selected = bool(option.get("selected", False))
        elif isinstance(option, (list, tuple)):
            if len(option) >= 1:
                label = option[0]
            if len(option) >= 2:
                value = option[1]
            else:
                value = label
            if len(option) >= 3:
                selected = bool(option[2])
        else:
            label = option
            value = option

        normalized.append(
            {
                "label": str(label) if label is not None else "",
                "value": value,
                "selected": selected,
            }
        )

    return normalized


def _infer_default_case_index(options: list[dict[str, Any]]) -> int | None:
    for idx, option in enumerate(options):
        if option.get("selected"):
            return idx
    return 0 if options else None


def _infer_case_value(case: dict[str, Any], options: list[dict[str, Any]], idx: int) -> Any:
    if "value" in case:
        return case.get("value")
    if idx < len(options):
        return options[idx].get("value")
    return idx


def _walk_tool_inputs(
    inputs: list[Any],
    *,
    prefix: str,
    fields: list[dict[str, Any]],
    known_keys: set[str],
    required_keys: set[str],
    conditional_groups: list[dict[str, Any]],
    case_required_bucket: set[str] | None = None,
    case_known_bucket: set[str] | None = None,
) -> None:
    for item in inputs:
        if not isinstance(item, dict):
            continue

        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            continue

        field_type = str(item.get("type") or item.get("model_class") or "unknown")
        key = _join_key(prefix, raw_name)
        optional = bool(item.get("optional", False))
        default_value = item.get("value", item.get("default_value"))
        has_default = not _is_missing_value(default_value)
        entry: dict[str, Any] = {
            "key": key,
            "name": raw_name,
            "label": item.get("label"),
            "type": field_type,
            "required": not optional,
            "has_default": has_default,
            "help": item.get("help"),
        }
        extensions_raw = item.get("extensions", item.get("format"))
        if isinstance(extensions_raw, list):
            extensions = [str(ext) for ext in extensions_raw if ext is not None]
            if extensions:
                entry["extensions"] = extensions
        elif isinstance(extensions_raw, str) and extensions_raw:
            entry["extensions"] = [extensions_raw]

        collection_types_raw = item.get("collection_types")
        if isinstance(collection_types_raw, list):
            collection_types = [str(col_type) for col_type in collection_types_raw if col_type]
            if collection_types:
                entry["collection_types"] = collection_types
        elif isinstance(collection_types_raw, str) and collection_types_raw:
            entry["collection_types"] = [collection_types_raw]

        if field_type == "conditional":
            test_param = item.get("test_param") if isinstance(item.get("test_param"), dict) else {}
            selector_name = test_param.get("name") if isinstance(test_param.get("name"), str) else "__current_case__"
            selector_key = _join_key(key, selector_name)
            selector_options = _normalize_options(test_param.get("options"))
            default_case = _infer_default_case_index(selector_options)

            known_keys.update({key, selector_key, f"{key}|__current_case__"})
            if case_known_bucket is not None:
                case_known_bucket.update({key, selector_key, f"{key}|__current_case__"})

            selector_field = {
                "key": selector_key,
                "name": selector_name,
                "label": test_param.get("label"),
                "type": str(test_param.get("type") or "select"),
                "required": True,
                "has_default": default_case is not None,
                "help": test_param.get("help"),
                "conditional_group": key,
                "is_selector": True,
                "choices": selector_options,
            }
            fields.append(selector_field)

            conditional_group: dict[str, Any] = {
                "key": key,
                "selector_key": selector_key,
                "selector_name": selector_name,
                "choices": selector_options,
                "default_case": default_case,
                "cases": [],
            }

            cases = item.get("cases") if isinstance(item.get("cases"), list) else []
            for idx, case in enumerate(cases):
                case_payload = case if isinstance(case, dict) else {}
                nested_inputs = (
                    case_payload.get("inputs") if isinstance(case_payload.get("inputs"), list) else []
                )
                case_required_keys: set[str] = set()
                case_known_keys: set[str] = set()
                _walk_tool_inputs(
                    nested_inputs,
                    prefix=key,
                    fields=fields,
                    known_keys=known_keys,
                    required_keys=required_keys,
                    conditional_groups=conditional_groups,
                    case_required_bucket=case_required_keys,
                    case_known_bucket=case_known_keys,
                )
                conditional_group["cases"].append(
                    {
                        "index": idx,
                        "value": _infer_case_value(case_payload, selector_options, idx),
                        "required_keys": sorted(case_required_keys),
                        "known_keys": sorted(case_known_keys),
                    }
                )

            conditional_groups.append(conditional_group)
            continue

        if field_type in {"repeat", "section"}:
            entry["is_container"] = True
            known_keys.add(key)
            if case_known_bucket is not None:
                case_known_bucket.add(key)
            fields.append(entry)
            nested = item.get("inputs") if isinstance(item.get("inputs"), list) else []
            _walk_tool_inputs(
                nested,
                prefix=key,
                fields=fields,
                known_keys=known_keys,
                required_keys=required_keys,
                conditional_groups=conditional_groups,
                case_required_bucket=case_required_bucket,
                case_known_bucket=case_known_bucket,
            )
            continue

        options = _normalize_options(item.get("options"))
        if options:
            entry["choices"] = options

        fields.append(entry)
        known_keys.add(key)
        if case_known_bucket is not None:
            case_known_bucket.add(key)

        should_require = (
            not optional and not has_default and field_type not in {"hidden", "hidden_data"}
        )
        if should_require:
            required_keys.add(key)
            if case_required_bucket is not None:
                case_required_bucket.add(key)


def build_tool_contract(tool_info: dict[str, Any]) -> dict[str, Any]:
    """Create an agent-friendly contract from Galaxy's raw tool details response."""

    raw_inputs = tool_info.get("inputs")
    inputs = raw_inputs if isinstance(raw_inputs, list) else []

    fields: list[dict[str, Any]] = []
    known_keys: set[str] = set()
    required_keys: set[str] = set()
    conditional_groups: list[dict[str, Any]] = []

    _walk_tool_inputs(
        inputs,
        prefix="",
        fields=fields,
        known_keys=known_keys,
        required_keys=required_keys,
        conditional_groups=conditional_groups,
    )

    return {
        "input_format_supported": ["legacy", "21.01"],
        "fields": fields,
        "known_keys": sorted(known_keys),
        "required_keys": sorted(required_keys),
        "conditional_groups": conditional_groups,
    }


def _lookup_input_value(inputs: dict[str, Any], key: str) -> Any:
    if key in inputs:
        return inputs[key]

    current: Any = inputs
    for part in key.split("|"):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _flatten_nested_keys(payload: dict[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for raw_key, value in payload.items():
        key = _join_key(prefix, str(raw_key))
        keys.add(key)
        if isinstance(value, dict):
            keys.update(_flatten_nested_keys(value, prefix=key))
    return keys


def _resolve_selected_case(
    conditional_group: dict[str, Any], inputs: dict[str, Any]
) -> tuple[int | None, bool]:
    selector_key = str(conditional_group.get("selector_key"))
    current_case_key = f"{conditional_group.get('key')}|__current_case__"

    selector_value = _lookup_input_value(inputs, selector_key)
    explicit_selector = selector_value is not None

    if explicit_selector:
        for case in conditional_group.get("cases", []):
            case_value = case.get("value")
            if selector_value == case_value or str(selector_value) == str(case_value):
                return int(case["index"]), True

        try:
            index_guess = int(selector_value)
        except Exception:
            index_guess = None
        if index_guess is not None:
            indexes = {int(case["index"]) for case in conditional_group.get("cases", [])}
            if index_guess in indexes:
                return index_guess, True
        return None, True

    case_value = _lookup_input_value(inputs, current_case_key)
    explicit_case = case_value is not None
    if explicit_case:
        try:
            index_guess = int(case_value)
        except Exception:
            return None, True
        indexes = {int(case["index"]) for case in conditional_group.get("cases", [])}
        if index_guess in indexes:
            return index_guess, True
        return None, True

    default_case = conditional_group.get("default_case")
    if isinstance(default_case, int):
        return default_case, False
    return None, False


def validate_payload_against_contract(
    contract: dict[str, Any],
    inputs: dict[str, Any],
    *,
    input_format: str = "legacy",
    require_explicit_conditionals: bool = True,
    strict_unknown_keys: bool = True,
) -> dict[str, Any]:
    """Lint payload keys against a structured tool contract."""

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    selected_conditionals: list[dict[str, Any]] = []

    top_level_keys = {str(k) for k in inputs.keys()}
    nested_keys = _flatten_nested_keys(inputs)
    submitted_keys = top_level_keys if input_format == "legacy" else nested_keys
    normalized_submitted_keys = {_normalize_repeat_key(key) for key in submitted_keys}
    submitted_keys_by_template: dict[str, list[str]] = {}
    for key in submitted_keys:
        template = _normalize_repeat_key(key)
        submitted_keys_by_template.setdefault(template, []).append(key)

    required_keys = set(contract.get("required_keys", []))
    missing_required = sorted(
        key for key in required_keys if key not in submitted_keys and key not in normalized_submitted_keys
    )
    if missing_required:
        errors.append(
            {
                "code": "missing_required_inputs",
                "field": None,
                "message": "Missing required input keys.",
                "details": {"missing_keys": missing_required},
            }
        )

    for conditional_group in contract.get("conditional_groups", []):
        group_key = str(conditional_group.get("key"))
        selector_key = str(conditional_group.get("selector_key"))
        current_case_key = f"{group_key}|__current_case__"

        has_explicit_selector = (
            _lookup_input_value(inputs, selector_key) is not None
            or _lookup_input_value(inputs, current_case_key) is not None
        )

        selected_case, selected_explicitly = _resolve_selected_case(conditional_group, inputs)
        if require_explicit_conditionals and not has_explicit_selector:
            errors.append(
                {
                    "code": "conditional_selector_missing",
                    "field": selector_key,
                    "message": (
                        f"Conditional '{group_key}' requires explicit selector '{selector_key}' "
                        "or '|__current_case__' to avoid silent default branches."
                    ),
                }
            )
            continue

        if selected_case is None:
            if has_explicit_selector:
                errors.append(
                    {
                        "code": "conditional_selector_invalid",
                        "field": selector_key,
                        "message": f"Conditional selector for '{group_key}' did not match any case.",
                    }
                )
            continue

        selected_conditionals.append(
            {
                "conditional": group_key,
                "selector_key": selector_key,
                "selected_case": selected_case,
                "selected_explicitly": selected_explicitly,
            }
        )

        group_cases = conditional_group.get("cases", [])
        case = next(
            (item for item in group_cases if int(item.get("index", -1)) == int(selected_case)),
            None,
        )
        if not isinstance(case, dict):
            continue

        missing_case_required = [
            key
            for key in case.get("required_keys", [])
            if key not in submitted_keys and key not in normalized_submitted_keys
        ]
        if missing_case_required:
            errors.append(
                {
                    "code": "conditional_required_missing",
                    "field": selector_key,
                    "message": (
                        f"Selected case {selected_case} for conditional '{group_key}' is missing "
                        "required inputs."
                    ),
                    "details": {"missing_keys": sorted(missing_case_required)},
                }
            )

    known_keys = set(contract.get("known_keys", []))
    unknown_keys = sorted(
        key
        for key in submitted_keys
        if key not in known_keys
        and _normalize_repeat_key(key) not in known_keys
        and not key.startswith("__")
    )
    if unknown_keys:
        issue = {
            "code": "unknown_input_keys",
            "field": None,
            "message": "Payload contains keys not present in tool schema.",
            "details": {"unknown_keys": unknown_keys},
        }
        if strict_unknown_keys:
            errors.append(issue)
        else:
            warnings.append(issue)

    for field in contract.get("fields", []):
        if not isinstance(field, dict):
            continue
        key = field.get("key")
        if not isinstance(key, str) or not key:
            continue
        choices = field.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        allowed_values = [choice.get("value") for choice in choices if isinstance(choice, dict)]
        allowed_str_values = {str(value) for value in allowed_values}
        allowed_display = sorted(allowed_str_values)

        candidate_keys = submitted_keys_by_template.get(key, [key])
        for candidate_key in candidate_keys:
            value = _lookup_input_value(inputs, candidate_key)
            if value is None:
                continue

            values_to_check = value if isinstance(value, list) else [value]
            invalid_values: list[Any] = []
            for current_value in values_to_check:
                if current_value in allowed_values or str(current_value) in allowed_str_values:
                    continue
                invalid_values.append(current_value)

            if invalid_values:
                errors.append(
                    {
                        "code": "invalid_select_value",
                        "field": candidate_key,
                        "message": (
                            f"Value '{invalid_values[0]}' not in allowed choices for '{candidate_key}'."
                        ),
                        "details": {
                            "invalid_values": invalid_values,
                            "allowed": allowed_display,
                        },
                    }
                )

    for field in contract.get("fields", []):
        if not isinstance(field, dict):
            continue
        key = field.get("key")
        if not isinstance(key, str) or not key:
            continue
        field_type = str(field.get("type") or "")
        if not _is_data_column_type(field_type):
            continue

        candidate_keys = submitted_keys_by_template.get(key, [key])
        for candidate_key in candidate_keys:
            value = _lookup_input_value(inputs, candidate_key)
            if value is None:
                continue

            if isinstance(value, bool):
                parsed = None
            elif isinstance(value, int):
                parsed = value
            elif isinstance(value, str) and value.strip().isdigit():
                parsed = int(value.strip())
            else:
                parsed = None

            if parsed is None or parsed < 1:
                errors.append(
                    {
                        "code": "invalid_data_column_value",
                        "field": candidate_key,
                        "message": (
                            f"Field '{candidate_key}' must be a 1-indexed integer column index."
                        ),
                        "details": {"value": value},
                    }
                )

    for field in contract.get("fields", []):
        if not isinstance(field, dict):
            continue
        key = field.get("key")
        if not isinstance(key, str) or not key:
            continue
        field_type = str(field.get("type") or "")
        if not _is_text_like_field_type(field_type):
            continue

        candidate_keys = submitted_keys_by_template.get(key, [key])
        for candidate_key in candidate_keys:
            value = _lookup_input_value(inputs, candidate_key)
            if isinstance(value, str) and _looks_like_json_blob(value):
                warnings.append(
                    {
                        "code": "json_blob_in_text_field",
                        "field": candidate_key,
                        "message": (
                            f"Field '{candidate_key}' looks like JSON in a text parameter. "
                            "Galaxy may escape characters (e.g., __oc__/__cc__/__dq__). "
                            "Consider using file/dataset input instead."
                        ),
                    }
                )

    return {
        "valid": len(errors) == 0,
        "input_format": input_format,
        "strict_unknown_keys": strict_unknown_keys,
        "errors": errors,
        "warnings": warnings,
        "missing_required_keys": missing_required,
        "unknown_keys": unknown_keys,
        "selected_conditionals": selected_conditionals,
    }


def build_agent_hints(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive agent-facing setup hints from a structured tool contract."""

    hints: list[dict[str, Any]] = []
    for field in contract.get("fields", []):
        if not isinstance(field, dict):
            continue
        key = field.get("key")
        field_type = str(field.get("type") or "")
        if not isinstance(key, str) or not key:
            continue

        if _is_data_column_type(field_type):
            hints.append(
                {
                    "field": key,
                    "code": "data_column_requires_index",
                    "hint": (
                        "Requires a 1-indexed integer column index (e.g., 1, 2, 3), "
                        "not a column name string."
                    ),
                }
            )
            continue

        if _is_text_like_field_type(field_type):
            hints.append(
                {
                    "field": key,
                    "code": "text_json_escape_risk",
                    "hint": (
                        "Text parameter. If you pass JSON text, Galaxy may escape characters "
                        "(__oc__/__cc__/__dq__). Prefer dataset/file input for structured data."
                    ),
                }
            )

    return hints
