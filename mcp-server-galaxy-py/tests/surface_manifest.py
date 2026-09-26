"""Generate the checked-in snapshot of this server's MCP tool surface.

The snapshot is the contract the TypeScript ops in `galaxy-agent-tools` are
compared against. It is checked in rather than built on the fly so that a change
to a tool's parameters has to arrive as a reviewable diff, and so the two
languages cannot drift while a hand-maintained list of names keeps passing.

Regenerate with `uv run python -m tests.surface_manifest`.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

from fastmcp.tools import Tool

from galaxy_mcp import server
from galaxy_mcp.version import TOOL_REQUIREMENTS

MANIFEST_PATH = Path(__file__).parent / "testdata" / "mcp-surface.json"
REGENERATE_COMMAND = "uv run python -m tests.surface_manifest"

# Tools FastMCP registers only when an optional extra is present. Synthesizing
# them from the plain module-level function keeps the manifest identical on a
# machine that has the extra and one that does not, so the surface contract does
# not depend on how the generating environment was provisioned. When the extra
# is available, absence from the server is a registration bug, not a reason to
# invent a tool in the manifest.
CONDITIONAL_TOOLS: dict[str, dict[str, Any]] = {
    "recommend_biocontainer": {
        "extra": "container-recommend",
        "tags": ["extended", "read", "tools"],
        "available": lambda: server._container_recommender_available(),
    },
}


def _annotations(tool: Tool) -> dict[str, Any]:
    """The MCP annotations the tool advertises, empty when it advertises none.

    Recorded even though no tool sets one today: the read/write split lives in the
    tags, so a hint added later would change what a client is told about the tool
    while leaving every other field of this snapshot alone.
    """
    if tool.annotations is None:
        return {}
    return tool.annotations.model_dump(exclude_none=True)


# The largest integer a JSON number survives as itself. Anything past it is read
# back by the TypeScript side as a different number, so two declared defaults that
# differ would compare equal there and nobody would ever see it.
EXACT_INTEGER_LIMIT = 2**53 - 1


def _assert_numbers_survive_json(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, each in value.items():
            _assert_numbers_survive_json(each, f"{where}.{key}")
    elif isinstance(value, list):
        for index, each in enumerate(value):
            _assert_numbers_survive_json(each, f"{where}[{index}]")
    elif isinstance(value, bool):
        return
    elif isinstance(value, int) and abs(value) > EXACT_INTEGER_LIMIT:
        raise ValueError(
            f"{where} is {value}, which no JSON reader can hand back unchanged; "
            "the manifest is the contract the TypeScript ops are compared against, "
            "so a number it cannot carry has to stay out of the surface"
        )


# A result is a sequence of rows or an object with named keys. Which one, and the keys when they
# can be read, is what the TypeScript side declares too, so the two can be compared structurally
# rather than through a docstring.
SEQUENCE = "list"
OBJECT = "object"


def _dict_keys(node: ast.Dict) -> tuple[str, ...] | None:
    """The literal's keys, or None when any of them is computed."""
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return tuple(keys) if keys and len(keys) == len(node.keys) else None


def _lambda_parameters(scope: ast.AST) -> set[str]:
    """Names bound by a lambda inside `scope`.

    A tool's paged result is built by a continuation the paging helper calls back -- the page is
    that lambda's parameter, so it has no binding in the tool body. The parameter is the page of
    the sequence the helper was handed, which makes the result a sequence whatever the helper is
    called.
    """
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Lambda):
            args = node.args
            names.update(a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs])
    return names


def _bindings(scope: ast.AST, name: str) -> list[ast.expr]:
    """Every expression assigned to `name` anywhere in `scope`."""
    found: list[ast.expr] = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name) and t.id == name]
            found.extend(node.value for _ in targets)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                found.append(node.value)
    return found


def _is_slice(node: ast.expr) -> bool:
    return isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)


def _keyed_after(scope: ast.AST, name: str) -> bool:
    """Whether `name[...] = ...` adds a key somewhere, which a literal alone would not show."""
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                continue
            if target.value.id == name:
                return True
    return False


def _shape_of(
    expr: ast.expr, scope: ast.AST, seen: frozenset[str] = frozenset()
) -> dict[str, Any] | None:
    """What the expression is, structurally, or None when this generator cannot say.

    Saying nothing is the point of the None: a shape nobody stated is not a shape the comparison
    may assume, and an unread result is left to the op's own tests rather than guessed at here.
    """
    if isinstance(expr, ast.Dict):
        keys = _dict_keys(expr)
        return {"kind": OBJECT, "fields": sorted(keys)} if keys else {"kind": OBJECT}
    if isinstance(expr, (ast.List, ast.ListComp)) or _is_slice(expr):
        return {"kind": SEQUENCE}
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return None
        bound = _bindings(scope, expr.id)
        if not bound:
            return {"kind": SEQUENCE} if expr.id in _lambda_parameters(scope) else None
        shape = _agreed([_shape_of(b, scope, seen | {expr.id}) for b in bound])
        # A key added after the literal may be conditional, so the kind is a fact, the list is not.
        if shape and shape["kind"] == OBJECT and _keyed_after(scope, expr.id):
            return {"kind": OBJECT}
        return shape
    return None


def _agreed(shapes: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """One shape when they all say the same thing, the kind alone when only the keys differ."""
    if not shapes or any(s is None for s in shapes):
        return None
    kinds = {s["kind"] for s in shapes if s}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    fields = {tuple(s.get("fields", ())) for s in shapes if s}
    if len(fields) == 1:
        only = fields.pop()
        return {"kind": kind, "fields": list(only)} if only else {"kind": kind}
    # Two branches building different keys: the kind is still a fact, the field list is not.
    return {"kind": kind}


def _result(name: str) -> dict[str, Any] | None:
    """The shape of the tool's result, and whether it pages, read off the calls it returns.

    `paginated` says the envelope carries a page window beside the data, which is a fact about
    where the window lives rather than about the rows -- the two surfaces need not agree on it,
    and the report is the place that disagreement becomes visible.
    """
    fn = _SOURCE_FUNCTIONS.get(name)
    if fn is None:
        return None
    shapes: list[dict[str, Any] | None] = []
    paginated = False
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "GalaxyResult"):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "data" not in keywords:
            continue
        shapes.append(_shape_of(keywords["data"], fn))
        paginated = paginated or "pagination" in keywords
    shape = _agreed(shapes)
    if shape is None:
        return None
    return {**shape, "paginated": paginated}


def _source_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every tool-decorated function in server.py, by name."""
    tree = ast.parse(Path(server.__file__).read_text())
    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator.func if isinstance(decorator, ast.Call) else decorator
            if getattr(call, "attr", None) == "tool":
                out[node.name] = node
    return out


_SOURCE_FUNCTIONS = _source_functions()


def _entry(tool: Tool, conditional_on: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": tool.name, "tags": sorted(tool.tags)}
    # Recorded structurally rather than left to the description, so the TypeScript side's
    # `requires` can be compared against this one instead of against English.
    requires = TOOL_REQUIREMENTS.get(tool.name)
    if requires:
        entry["requires"] = {"galaxy": requires}
    if conditional_on:
        entry["conditionalOn"] = conditional_on
    entry["annotations"] = _annotations(tool)
    entry["inputSchema"] = tool.parameters
    result = _result(tool.name)
    if result is not None:
        entry["result"] = result
    _assert_numbers_survive_json(entry, tool.name)
    return entry


def build_manifest() -> dict[str, Any]:
    """Describe every tool the server can register, in a deterministic order."""
    # `code` discovery mode collapses the catalog into three meta-tools, which is a
    # different surface from the one this manifest describes.
    if server._discovery_mode != "full":
        raise RuntimeError(
            f"GALAXY_MCP_DISCOVERY_MODE={server._discovery_mode!r} changes the tool "
            "catalog; generate the manifest with the default 'full' mode."
        )

    # `run_middleware=False` on purpose: the tag filter is read from the environment,
    # and the snapshot describes the whole catalog rather than whatever one machine
    # happens to be showing. The filter can only hide tools, never change what one
    # takes. What it cannot do is decide between two registrations of one name --
    # over the wire FastMCP shows the highest version of each, and picking a
    # different one here would snapshot a tool nobody is served.
    listed = asyncio.run(server.mcp.list_tools(run_middleware=False))
    twice = sorted({t.name for t in listed if sum(1 for o in listed if o.name == t.name) > 1})
    if twice:
        raise ValueError(
            f"{', '.join(twice)} is registered more than once, and this snapshot has no way to "
            "say which registration a client is served; give the tools distinct names or teach "
            "the generator how the server chooses between them"
        )
    registered = {t.name: t for t in listed}
    entries = [
        _entry(tool, CONDITIONAL_TOOLS.get(name, {}).get("extra"))
        for name, tool in registered.items()
    ]
    for name, spec in CONDITIONAL_TOOLS.items():
        if name in registered:
            continue
        if spec["available"]():
            continue
        synthesized = Tool.from_function(getattr(server, name), tags=set(spec["tags"]))
        entries.append(_entry(synthesized, spec["extra"]))
    entries.sort(key=lambda e: str(e["name"]))

    return {
        "$comment": (
            "Generated snapshot of the Galaxy MCP tool surface -- do not edit by hand. "
            f"Regenerate with `{REGENERATE_COMMAND}`."
        ),
        "source": "mcp-server-galaxy-py/src/galaxy_mcp/server.py",
        "toolCount": len(entries),
        "tools": entries,
    }


def render(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def main() -> None:
    MANIFEST_PATH.write_text(render(build_manifest()), newline="\n")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
