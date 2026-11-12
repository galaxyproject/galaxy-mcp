import math
from typing import Any

MIB = 1024 * 1024
TERMINAL_STATES = {"ok", "error", "failed", "paused", "deleted", "stopped", "skipped"}
PEAK_MEMORY_NAMES = ("memory.peak", "memory.max_usage_in_bytes")
OOM_NAMES = (
    "memory.events.oom_kill",
    "memory.events.oom_group_kill",
    "memory.events.oom",
    "memory.failcnt",
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resource_requirements(representation: dict[str, Any]) -> dict[str, Any]:
    for requirement in representation.get("requirements") or []:
        if isinstance(requirement, dict) and requirement.get("type") == "resource":
            return requirement
    return {}


def _metrics_by_name(metrics: list[dict[str, Any] | None]) -> dict[tuple[str, str], float]:
    result = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        plugin = metric.get("plugin")
        name = metric.get("name")
        value = _number(metric.get("raw_value"))
        if isinstance(plugin, str) and isinstance(name, str) and value is not None:
            result[(plugin, name)] = value
    return result


def evaluate_resource_preflight(
    representation: dict[str, Any],
    job: dict[str, Any],
    metrics: list[dict[str, Any] | None],
    safety_factor: float = 1.5,
) -> dict[str, Any]:
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be a finite number at least 1")
    state = job.get("state")
    if state not in TERMINAL_STATES:
        raise ValueError(f"Job '{job.get('id')}' is not finished (state: {state})")
    values = _metrics_by_name(metrics)
    peak_memory = next(
        (values[("cgroup", name)] for name in PEAK_MEMORY_NAMES if ("cgroup", name) in values), None
    )
    runtime = values.get(("core", "runtime_seconds"))
    oom_events = max((values.get(("cgroup", name), 0) for name in OOM_NAMES), default=0)
    requirements = _resource_requirements(representation)
    declared = {
        key: _number(requirements.get(key))
        for key in ("ram_min", "ram_max", "timelimit")
    }
    recommendation = {
        "ram_min": math.ceil(peak_memory / MIB * safety_factor)
        if peak_memory is not None
        else None,
        "timelimit": math.ceil(runtime * safety_factor)
        if runtime is not None
        else None,
    }
    warnings = []
    if peak_memory is None:
        warnings.append(
            {
                "code": "peak_memory_unavailable",
                "message": "The server did not expose a cgroup peak-memory metric.",
            }
        )
    if runtime is None:
        warnings.append(
            {
                "code": "runtime_unavailable",
                "message": "The server did not expose the job runtime metric.",
            }
        )
    if state != "ok":
        warnings.append(
            {"code": "test_run_failed", "message": f"The test job finished in state '{state}'."}
        )
    if oom_events > 0:
        warnings.append(
            {
                "code": "oom_observed",
                "message": (
                    f"The test run recorded {oom_events:g} OOM or allocation-failure events."
                ),
            }
        )
    if not requirements:
        warnings.append(
            {
                "code": "resource_requirement_missing",
                "message": "The UDT has no resource requirement block.",
            }
        )
    if recommendation["ram_min"] is not None and (
        declared["ram_min"] is None or declared["ram_min"] < recommendation["ram_min"]
    ):
        warnings.append(
            {
                "code": "ram_min_underestimated",
                "message": (
                    f"Declare ram_min of at least {recommendation['ram_min']} MiB "
                    "for the tested scale."
                ),
            }
        )
    if (
        recommendation["ram_min"] is not None
        and declared["ram_max"] is not None
        and declared["ram_max"] < recommendation["ram_min"]
    ):
        warnings.append(
            {
                "code": "ram_range_conflict",
                "message": "The recommended ram_min exceeds the declared ram_max.",
            }
        )
    if recommendation["timelimit"] is not None and (
        declared["timelimit"] is None or declared["timelimit"] < recommendation["timelimit"]
    ):
        warnings.append(
            {
                "code": "timelimit_underestimated",
                "message": (
                    f"Declare timelimit of at least {recommendation['timelimit']} seconds "
                    "for the tested scale."
                ),
            }
        )
    return {
        "job_id": job.get("id"),
        "state": state,
        "observed": {
            "peak_memory_bytes": peak_memory,
            "runtime_seconds": runtime,
            "oom_events": oom_events,
        },
        "declared": declared,
        "recommendation": recommendation,
        "safety_factor": safety_factor,
        "warnings": warnings,
    }
