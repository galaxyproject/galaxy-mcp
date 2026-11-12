"""The scenario-file format and the registry's `live` section, as pydantic models.

These models are the format: the loader validates every pack and the registry against
them, and `uv run python -m tests.live.model` writes the JSON Schema to
contract/scenarios/scenario.schema.json so an editor or a later TS runner can validate a
pack without this code. The offline check fails when that file is out of date.

Everything is `extra="forbid"`: a misspelt key in a hand-written JSON file would otherwise
be a scenario that silently checks less than its author thinks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

ID_PATTERN = r"^[a-z0-9_]+(\.[a-z0-9_]+)+$"
CAPABILITIES = ("network", "unprivileged-tools", "user-tool-role", "admin")
LIVE_KINDS = (
    "value-mismatch",
    "shape-mismatch",
    "failure-text-mismatch",
    "outcome-mismatch",
    "server-bug",
)
STATUSES = ("intentional", "pending-port", "pending-decision", "unreviewed-gap")
SERVERS = ("python", "typescript")

ServerName = Literal["python", "typescript"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _exactly_one(model: BaseModel, names: tuple[str, ...], what: str) -> None:
    present = [n for n in names if n in model.model_fields_set]
    if len(present) != 1:
        aliases = [model.__class__.model_fields[n].alias or n for n in names]
        raise ValueError(f"{what} needs exactly one of {aliases}, got {len(present)}")


# --- assertions --------------------------------------------------------------------------


class PathRef(_Strict):
    path: str


# A whole-string reference ("${saved}") stands in for a number until the run resolves it,
# so a bound can come from an earlier step.
Reference = Annotated[str, Field(pattern=r"^\$\{[^}]+\}$")]


class LengthSpec(_Strict):
    eq: int | None = None
    gt: int | None = None
    gte: int | None = None
    lt: int | None = None
    lte: int | None = None

    @model_validator(mode="after")
    def _some(self) -> LengthSpec:
        if not self.model_fields_set:
            raise ValueError("length needs at least one of eq/gt/gte/lt/lte")
        return self


class SortedSpec(_Strict):
    key: str | None = None
    order: Literal["asc", "desc"] = "asc"


ASSERTION_OPS = (
    "exists",
    "equals",
    "not_equals",
    "in_",
    "type",
    "contains",
    "not_contains",
    "matches",
    "keys_include",
    "keys_exclude",
    "keys_equal",
    "length",
    "gt",
    "gte",
    "lt",
    "lte",
    "has_item",
    "each",
    "sorted",
)


class Assertion(_Strict):
    """`{"path": "<dotted path>", "<op>": value}` with exactly one op."""

    path: str
    exists: bool | None = None
    equals: Any = None
    not_equals: Any = None
    in_: list[Any] | None = Field(default=None, alias="in")
    type: Literal["object", "list", "string", "number", "integer", "boolean", "null"] | None = None
    contains: str | None = None
    not_contains: str | None = None
    matches: str | None = None
    keys_include: list[str] | None = None
    keys_exclude: list[str] | None = None
    keys_equal: list[str] | None = None
    length: LengthSpec | None = None
    gt: float | PathRef | Reference | None = None
    gte: float | PathRef | Reference | None = None
    lt: float | PathRef | Reference | None = None
    lte: float | PathRef | Reference | None = None
    has_item: dict[str, Any] | None = None
    each: list[Assertion] | None = None
    sorted: SortedSpec | None = None

    @model_validator(mode="after")
    def _one_op(self) -> Assertion:
        _exactly_one(self, ASSERTION_OPS, "an assertion")
        if self.matches is not None:
            re.compile(self.matches)
        return self

    @property
    def op(self) -> str:
        return next(n for n in ASSERTION_OPS if n in self.model_fields_set)


# --- fixtures ----------------------------------------------------------------------------


class HistoryFixture(_Strict):
    name: str | None = None


class DatasetFixture(_Strict):
    history: str
    content: str
    ext: str = "txt"
    name: str | None = None


class CollectionElement(_Strict):
    name: str
    content: str | None = None
    ext: str = "txt"
    elements: list[CollectionElement] | None = None

    @model_validator(mode="after")
    def _leaf_or_nested(self) -> CollectionElement:
        if (self.content is None) == (self.elements is None):
            raise ValueError("a collection element has either content or elements, not both")
        return self


class CollectionFixture(_Strict):
    history: str
    type: Literal["list", "paired", "list:paired"]
    elements: list[CollectionElement] = Field(min_length=1)
    name: str | None = None

    @model_validator(mode="after")
    def _nesting(self) -> CollectionFixture:
        nested = self.type == "list:paired"
        for el in self.elements:
            if nested and not el.elements:
                raise ValueError("list:paired elements each need nested forward/reverse elements")
            if not nested and el.elements:
                raise ValueError(f"{self.type} elements take content, not nested elements")
        if self.type == "paired" and [e.name for e in self.elements] != ["forward", "reverse"]:
            raise ValueError("a paired collection has exactly the elements forward and reverse")
        return self


class WorkflowFixture(_Strict):
    file: str


class InvocationFixture(_Strict):
    workflow: str
    history: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    inputs_by: str | None = None


class PageFixture(_Strict):
    title: str | None = None
    content: str
    history: str | None = None


class UserToolFixture(_Strict):
    representation: dict[str, Any]


class LocalFileFixture(_Strict):
    name: str
    content: str


class TempPathFixture(_Strict):
    name: str


FIXTURE_KINDS = (
    "history",
    "dataset",
    "collection",
    "workflow",
    "invocation",
    "page",
    "user_tool",
    "local_file",
    "temp_path",
)

FIXTURE_BINDS: dict[str, tuple[str, ...]] = {
    "history": ("id", "name"),
    "dataset": ("id", "name", "hid", "history_id"),
    "collection": ("id", "name", "hid", "element_count"),
    "workflow": ("id", "name"),
    "invocation": ("id",),
    "page": ("id", "slug", "title"),
    "user_tool": ("uuid", "id"),
    "local_file": ("path", "name", "content", "size"),
    "temp_path": ("path",),
}


class Fixture(_Strict):
    as_: str = Field(alias="as", pattern=r"^[a-z][a-z0-9_]*$")
    history: HistoryFixture | None = None
    dataset: DatasetFixture | None = None
    collection: CollectionFixture | None = None
    workflow: WorkflowFixture | None = None
    invocation: InvocationFixture | None = None
    page: PageFixture | None = None
    user_tool: UserToolFixture | None = None
    local_file: LocalFileFixture | None = None
    temp_path: TempPathFixture | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> Fixture:
        _exactly_one(self, FIXTURE_KINDS, f"fixture {self.as_!r}")
        return self

    @property
    def kind(self) -> str:
        return next(k for k in FIXTURE_KINDS if k in self.model_fields_set)

    @property
    def spec(self) -> BaseModel:
        return getattr(self, self.kind)  # type: ignore[no-any-return]


# --- steps -------------------------------------------------------------------------------


class Until(_Strict):
    expect: list[Assertion] = Field(min_length=1)
    timeout: float = 180
    interval: float = 2


class Mask(_Strict):
    path: str
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        re.compile(v)
        return v


CREATABLE = ("history", "workflow", "page", "invocation", "user_tool")


class CallStep(_Strict):
    call: str
    args: dict[str, Any] = Field(default_factory=dict)
    outcome: Literal["success", "failure"] = "success"
    expect: list[Assertion] = Field(default_factory=list)
    save: dict[str, str] = Field(default_factory=dict)
    creates: dict[Literal["history", "workflow", "page", "invocation", "user_tool"], str] = Field(
        default_factory=dict
    )
    until: Until | None = None
    parity: Literal["shape", "exact", "none"] = "shape"
    parity_note: str | None = None
    mask: list[Mask] = Field(default_factory=list)
    divergences: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parity_note(self) -> CallStep:
        if self.parity == "none" and not self.parity_note:
            raise ValueError(f"{self.call}: parity none needs a parity_note saying why")
        for name in self.save:
            if not re.match(r"^[a-z][a-z0-9_]*$", name):
                raise ValueError(f"save name {name!r} must be lowercase identifier")
        return self


class WaitSpec(_Strict):
    dataset: str | None = None
    job: str | None = None
    invocation: str | None = None
    state: str | list[str] = "ok"
    timeout: float = 180

    @model_validator(mode="after")
    def _one(self) -> WaitSpec:
        _exactly_one(self, ("dataset", "job", "invocation"), "a wait")
        return self

    @property
    def kind(self) -> str:
        return next(k for k in ("dataset", "job", "invocation") if k in self.model_fields_set)


class WaitStep(_Strict):
    wait: WaitSpec


VERIFY_KINDS = ("history", "dataset", "dataset_text", "workflow", "invocation", "page", "file")


class VerifySpec(_Strict):
    history: str | None = None
    dataset: str | None = None
    dataset_text: str | None = None
    workflow: str | None = None
    invocation: str | None = None
    page: str | None = None
    file: str | None = None

    @model_validator(mode="after")
    def _one(self) -> VerifySpec:
        _exactly_one(self, VERIFY_KINDS, "a verify")
        return self

    @property
    def kind(self) -> str:
        return next(k for k in VERIFY_KINDS if k in self.model_fields_set)

    @property
    def target(self) -> str:
        return getattr(self, self.kind)  # type: ignore[no-any-return]


class VerifyStep(_Strict):
    verify: VerifySpec
    expect: list[Assertion] = Field(min_length=1)
    divergences: list[str] = Field(default_factory=list)


def _key_tag(*keys: str) -> Callable[[Any], str | None]:
    def pick(v: Any) -> str | None:
        if isinstance(v, BaseModel):
            source: Any = v.model_fields_set
        elif isinstance(v, dict):
            source = v
        else:
            return None
        found = [k for k in keys if k in source]
        return found[0] if len(found) == 1 else None

    return pick


# A tagged union, so a malformed step reports the errors of the step it looks like instead
# of three unrelated ones.
Step = Annotated[
    Annotated[CallStep, Tag("call")]
    | Annotated[WaitStep, Tag("wait")]
    | Annotated[VerifyStep, Tag("verify")],
    Discriminator(_key_tag("call", "wait", "verify")),
]


# --- scenarios and packs -----------------------------------------------------------------


class Requires(_Strict):
    tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)

    @field_validator("capabilities")
    @classmethod
    def _known(cls, caps: list[str]) -> list[str]:
        for cap in caps:
            if cap.lstrip("!") not in CAPABILITIES:
                raise ValueError(f"unknown capability {cap!r}; known: {list(CAPABILITIES)}")
        return caps


def _both_servers() -> list[ServerName]:
    return ["python", "typescript"]


SERVER_ENV_NAME = re.compile(r"^GALAXY_MCP_[A-Z0-9_]+$")


class Scenario(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    note: str = Field(min_length=1)
    ports: list[str] = Field(default_factory=list)
    servers: list[ServerName] = Field(default_factory=_both_servers)
    requires: Requires = Field(default_factory=Requires)
    divergences: list[str] = Field(default_factory=list)
    fixtures: list[Fixture] = Field(default_factory=list)
    # Extra environment for the servers under test, for behaviour that is configured at
    # startup rather than per call. Values may reference fixtures, so a scenario that sets
    # any gets a server process of its own, started after its fixtures exist and stopped
    # when it ends; every other scenario shares the session's one process, as before.
    server_env: dict[str, str] = Field(default_factory=dict)
    steps: list[Step] = Field(min_length=1)
    timeout: float = 600

    @model_validator(mode="after")
    def _unique_names(self) -> Scenario:
        names = [f.as_ for f in self.fixtures]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"{self.id}: fixture names used twice: {sorted(dupes)}")
        if not self.servers:
            raise ValueError(f"{self.id}: servers must not be empty")
        # Only the servers' own settings: the connection and the process environment the
        # harness builds from nothing stay the harness's.
        bad = sorted(k for k in self.server_env if not SERVER_ENV_NAME.match(k))
        if bad:
            raise ValueError(f"{self.id}: server_env may only set GALAXY_MCP_* variables: {bad}")
        return self


class Uncovered(_Strict):
    tool: str
    reason: str = Field(min_length=1)


class Pack(_Strict):
    comment: str | None = Field(default=None, alias="$comment")
    pack: str = Field(pattern=r"^[a-z0-9-]+$")
    tools: list[str]
    uncovered: list[Uncovered] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)


# --- the registry's live section ---------------------------------------------------------


class DropOp(_Strict):
    drop: list[str] = Field(min_length=1)


class RstripSpec(_Strict):
    paths: list[str] = Field(min_length=1)
    chars: str = Field(min_length=1)


class RstripOp(_Strict):
    rstrip: RstripSpec


class SubSpec(_Strict):
    path: str
    pattern: str
    replace: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        re.compile(v)
        return v


class SubOp(_Strict):
    sub: list[SubSpec] = Field(min_length=1)


class CutFromSpec(_Strict):
    path: str
    marker: str = Field(min_length=1)


class CutFromOp(_Strict):
    cut_from: CutFromSpec


class SetSpec(_Strict):
    path: str
    from_: Any = Field(alias="from")
    to: Any


class SetOp(_Strict):
    """Swap one documented value for another; the only op that reaches a non-string.

    Only a value equal to `from` is rewritten, so a server that starts answering something
    else entirely is not papered over: the step sees the new value (and the row goes stale).
    """

    set: list[SetSpec] = Field(min_length=1)


class ExpectFailureOp(_Strict):
    expect: Literal["failure"]
    # What the server answers instead, as "success" or "failure:<kind>". Required on a row
    # at `outcome`: that row excuses this one wrong outcome, not any outcome at all.
    outcome: Literal["success", "failure:tool", "failure:arguments"] | None = None


AdapterOp = Annotated[
    Annotated[DropOp, Tag("drop")]
    | Annotated[RstripOp, Tag("rstrip")]
    | Annotated[SubOp, Tag("sub")]
    | Annotated[CutFromOp, Tag("cut_from")]
    | Annotated[SetOp, Tag("set")]
    | Annotated[ExpectFailureOp, Tag("expect")],
    Discriminator(_key_tag("drop", "rstrip", "sub", "cut_from", "set", "expect")),
]


class LiveRow(_Strict):
    tool: str
    where: str
    kind: Literal[
        "value-mismatch",
        "shape-mismatch",
        "failure-text-mismatch",
        "outcome-mismatch",
        "server-bug",
    ]
    server: ServerName
    observed: str = Field(min_length=1)
    adapter: AdapterOp | list[AdapterOp]
    status: Literal["intentional", "pending-port", "pending-decision", "unreviewed-gap"]
    reason: str = Field(min_length=1)

    @property
    def key(self) -> str:
        # No server in it: a bug both servers share is one key with a row per server, and
        # one citation picks whichever row is about the server under test.
        return f"{self.tool} :: {self.where} :: {self.kind}"

    @property
    def ops(self) -> list[AdapterOp]:
        return list(self.adapter) if isinstance(self.adapter, list) else [self.adapter]

    @property
    def expects_failure(self) -> bool:
        return any(isinstance(op, ExpectFailureOp) for op in self.ops)

    @property
    def documented_outcome(self) -> str | None:
        found = [op.outcome for op in self.ops if isinstance(op, ExpectFailureOp)]
        return found[0] if found else None

    @model_validator(mode="after")
    def _expect_alone(self) -> LiveRow:
        if self.expects_failure and len(self.ops) != 1:
            raise ValueError(f"{self.key}: expect failure is an adapter on its own")
        if self.expects_failure and self.where == "outcome" and not self.documented_outcome:
            raise ValueError(
                f"{self.key}: an expect-failure row at `outcome` names the outcome it "
                'documents ("outcome": "success" | "failure:tool" | "failure:arguments")'
            )
        return self


class NormalizationRule(_Strict):
    enabled: bool
    status: Literal["intentional", "pending-port", "pending-decision", "unreviewed-gap"]
    reason: str = Field(min_length=1)


class LiveNormalization(_Strict):
    argumentValidationText: NormalizationRule  # noqa: N815 -- the registry's own key


class LiveRatchet(_Strict):
    comment: str | None = Field(default=None, alias="$comment")
    unreviewedGaps: int = Field(ge=0)  # noqa: N815 -- matches the existing ratchet's key


class LiveSection(_Strict):
    comment: str | None = Field(default=None, alias="$comment")
    ratchet: LiveRatchet
    normalization: LiveNormalization
    divergences: list[LiveRow] = Field(default_factory=list)

    def unreviewed_count(self) -> int:
        rules = [self.normalization.argumentValidationText]
        return sum(r.status == "unreviewed-gap" for r in self.divergences) + sum(
            r.status == "unreviewed-gap" for r in rules
        )


SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "contract" / "scenarios" / "scenario.schema.json"
)


def pack_schema() -> dict[str, Any]:
    schema = Pack.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$comment"] = (
        "Generated from python/tests/live/model.py by `uv run python -m tests.live.model` "
        "in python/. Do not edit by hand."
    )
    return schema


def render_schema() -> str:
    return json.dumps(pack_schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.write_text(render_schema())
    print(f"wrote {SCHEMA_PATH}")
