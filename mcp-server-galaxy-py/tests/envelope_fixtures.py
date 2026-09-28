"""Generate the golden result envelopes the TypeScript surfaces are checked against.

This server is the contract. Every tool answers with a ``GalaxyResult`` -- ``data``,
``success``, ``message``, ``count``, ``pagination`` -- and FastMCP sends it as
``pydantic_core.to_json(result, fallback=str)``, so what a client actually reads is
decided here and nowhere else. The TypeScript MCP server and CLI in
``galaxy-agent-tools`` have to produce the same thing, and the only way to hold them to
it is to write down what this side emits and replay it there.

Each case stores two files under ``testdata/envelopes/<tool>/``:

* ``<case>.json`` -- the envelope, exactly the JSON FastMCP would send, re-indented for
  review. Whitespace between tokens is the only difference; keys, nulls and values are
  the bytes ``pydantic_core.to_json`` produced.
* ``<case>.galaxy.json`` -- the Galaxy replies this case was run against, as a small
  routing table. The two implementations do not make the same requests (bioblend asks
  for a window and then counts unpaged; the TypeScript op fetches once and slices), so
  the table is matched by method and path, narrowed by query only where one case needs
  two different answers from one path. The most specific matching route wins on both
  sides.

``index.json`` lists every case with the input to call the tool with, spelled in this
server's parameter names -- which is what the MCP wire takes, so the other side can pass
it straight through.

Nothing here talks to a Galaxy. Every request is answered by ``responses``, and the key
below is a placeholder that no server ever sees.

Regenerate with ``uv run python -m tests.envelope_fixtures``;
``tests/test_envelope_fixtures.py`` fails when the checked-in files are stale.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydantic_core
import responses
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp import server
from galaxy_mcp.server import GalaxyResult, galaxy_state
from galaxy_mcp.version import clear_version_cache

from .mcp_session import LiveMCPSession
from .test_helpers import (
    cancel_workflow_invocation_fn,
    create_history_fn,
    create_page_fn,
    create_user_tool_fn,
    delete_user_tool_fn,
    get_collection_details_fn,
    get_dataset_details_fn,
    get_histories_fn,
    get_history_contents_fn,
    get_history_details_fn,
    get_invocations_fn,
    get_iwc_workflow_details_fn,
    get_iwc_workflows_fn,
    get_job_details_fn,
    get_page_fn,
    get_page_revision_fn,
    get_server_info_fn,
    get_tool_citations_fn,
    get_tool_details_fn,
    get_tool_input_template_fn,
    get_tool_panel_fn,
    get_tool_run_examples_fn,
    get_user_fn,
    get_workflow_details_fn,
    get_workflow_input_template_fn,
    import_workflow_from_iwc_fn,
    invoke_workflow_fn,
    list_history_ids_fn,
    list_page_revisions_fn,
    list_pages_fn,
    list_user_tools_fn,
    list_workflows_fn,
    recommend_biocontainer_fn,
    recommend_iwc_workflows_fn,
    revert_page_revision_fn,
    run_tool_fn,
    run_user_tool_fn,
    search_iwc_workflows_fn,
    search_tools_by_keywords_fn,
    search_tools_fn,
    update_history_fn,
    update_page_fn,
)

FIXTURE_ROOT = Path(__file__).parent / "testdata" / "envelopes"
REGENERATE_COMMAND = "uv run python -m tests.envelope_fixtures"

# Where the fake Galaxy lives. The TypeScript replay uses the same base, so a route's
# path reads the same on both sides.
GALAXY_URL = "https://galaxy.example"
# The same address as a connected session holds it. ``connect`` stores whatever it was
# given with a trailing slash added, and two tools read that string rather than a
# client's normalised copy of it: get_server_info answers with it, and get_job_details
# joins "api/jobs/{id}" straight onto it. So a session on this Galaxy is set up here the
# way connect would leave it, and the reply table hands the same spelling to the other
# side -- pointing the two surfaces at one address written two ways would be a
# difference in the fixture rather than in either server.
GALAXY_BASE_URL = f"{GALAXY_URL}/"
# Not a credential: every request in this module is answered by `responses`, and the
# string exists only because bioblend wants one.
PLACEHOLDER_KEY = "fixture-not-a-key"

IWC_MANIFEST_URL = "https://iwc.galaxyproject.org/workflow_manifest.json"


def route(
    path: str,
    body: Any,
    *,
    query: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    status: int = 200,
    method: str = "GET",
    body_text: str | None = None,
) -> dict[str, Any]:
    """One canned Galaxy reply.

    ``query`` is only set where a case needs two answers from one path; a route that
    declares none answers any query on that path.

    ``body_text`` serves those exact bytes instead of a JSON rendering of ``body``, and
    is what every failure route uses. It has to: a failure sentence quotes the reply
    body -- twice, for a bioblend GET, once as a Python ``bytes`` repr and once as text
    -- so the two sides have to be answered with the same bytes and not merely with the
    same value. ``json.dumps`` writes ``{"a": 1}`` and ``JSON.stringify`` writes
    ``{"a":1}``, which is a sentence two bytes apart for a reason that has nothing to do
    with either server.
    """
    return {
        "method": method,
        "path": path,
        "query": query or {},
        "status": status,
        "headers": headers or {},
        # One or the other, never both: a route whose bytes matter says so, and a reader
        # cannot then wonder which of two spellings of the same reply was served.
        **({"body": body} if body_text is None else {"bodyText": body_text}),
    }


# Galaxy 26.1, for the two tools that refuse anything older. Both surfaces ask
# /api/version before they run one.
VERSION_ROUTE = route("/api/version", {"version_major": "26.1", "version_minor": "1"})


@dataclass
class Case:
    """One call, the replies it is answered with, and what it pins."""

    tool: str
    name: str
    note: str
    # The tool's arguments in this server's parameter names, which is what the MCP wire
    # takes -- so the other side calls the tool with this object unchanged.
    input: dict[str, Any]
    # How to make the call, for a case that succeeds. A failure case leaves this out and
    # is driven by name through a real MCP client instead -- see ``run_case``.
    call: Callable[[], GalaxyResult] | None
    routes: list[dict[str, Any]] = field(default_factory=list)
    # Whether this case pins a failure. A success is pinned as the envelope the tool
    # built; a failure has no envelope to pin, so what is written down is the tool result
    # the MCP wire carries.
    failure: bool = False


# How a section below adds one case: the same six arguments ``cases()`` takes, so a
# group can live in a function of its own without changing what a case looks like.
AddCase = Callable[
    [str, str, str, dict[str, Any], Callable[[], GalaxyResult], list[dict[str, Any]]], None
]

# The same, for a failure: no callable, because a failure case is driven by tool name and
# arguments through a real MCP client rather than by calling the function.
AddFailure = Callable[[str, str, str, dict[str, Any], list[dict[str, Any]]], None]


# ---------------------------------------------------------------------------
# Corpora
#
# Small, boring and complete: every record carries every field both surfaces read, so a
# difference in a fixture is a difference in the envelope rather than a field one side
# defaulted and the other omitted.
# ---------------------------------------------------------------------------


def tool_rows(count: int, *, filler: str = "") -> list[dict[str, Any]]:
    return [
        {
            "id": f"tool{i}",
            "name": f"Matchable Tool {i}",
            "description": f"matchable tool number {i}{filler}",
            "versions": ["1.0.0"],
        }
        for i in range(count)
    ]


def history_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"h{i:04d}",
            "name": f"History {i}",
            "state": "ok",
            "update_time": f"2026-01-0{(i % 9) + 1}T00:00:00",
        }
        for i in range(count)
    ]


def content_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"d{i:04d}",
            "hid": i + 1,
            "name": f"dataset {i}",
            "state": "ok",
            "deleted": False,
            "visible": True,
            "history_content_type": "dataset",
            "extension": "txt",
        }
        for i in range(count)
    ]


def workflow_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"w{i:04d}",
            "name": f"Workflow {i}",
            "owner": "tester",
            "published": False,
            "deleted": False,
            "tags": [],
        }
        for i in range(count)
    ]


def user_tool_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"ut{i:04d}",
            "uuid": f"0000-{i:04d}",
            "tool_id": f"user_tool_{i}",
            "active": True,
            "representation": {"name": f"User Tool {i}", "version": "1.0"},
        }
        for i in range(count)
    ]


def panel_sections(sections: int, tools_per_section: int) -> list[dict[str, Any]]:
    """A tool panel: `sections` sections, each holding `tools_per_section` tools."""
    return [
        {
            "id": f"section{s}",
            "name": f"Section {s}",
            "model_class": "ToolSection",
            "elems": [
                {
                    "id": f"sec{s}_tool{t}",
                    "name": f"Section {s} Tool {t}",
                    "description": f"tool {t} of section {s}",
                    "versions": ["1.0.0"],
                    "model_class": "Tool",
                }
                for t in range(tools_per_section)
            ],
        }
        for s in range(sections)
    ]


def iwc_manifest(count: int, *, readme_filler: str = "") -> list[dict[str, Any]]:
    """The IWC manifest shape: a list of repositories, each holding workflows."""
    return [
        {
            "repo": f"repo{i}",
            "workflows": [
                {
                    "trsID": f"#workflow/github.com/iwc-workflows/wf{i}/main",
                    "definition": {
                        "name": f"Workflow {i}",
                        "annotation": f"rnaseq analysis number {i}",
                        "tags": ["transcriptomics", "rnaseq"],
                        "license": "MIT",
                        "creator": [{"class": "Person", "name": "IWC", "identifier": ""}],
                        "steps": {
                            "0": {"type": "tool", "tool_id": "fastqc"},
                            "1": {"type": "tool", "tool_id": "hisat2"},
                        },
                    },
                    "readme": f"Workflow {i} runs rnaseq quality control.{readme_filler}",
                    "categories": ["Transcriptomics"],
                }
            ],
        }
        for i in range(count)
    ]


def invocation_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"inv{i:04d}",
            "state": "scheduled",
            "workflow_id": "w0000",
            "history_id": "h0000",
            "create_time": "2026-01-01T00:00:00",
            "update_time": "2026-01-01T00:10:00",
        }
        for i in range(count)
    ]


def page_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"p{i:04d}",
            "title": f"Page {i}",
            "slug": f"page-{i}",
            "history_id": "h0000",
            "latest_revision_id": f"r{i:04d}",
            "revision_ids": [f"r{i:04d}"],
            "model_class": "Page",
            "username": "tester",
            "email_hash": "0" * 32,
            "author_deleted": False,
            "deleted": False,
            "importable": False,
            "published": False,
            "tags": [],
            "create_time": "2026-01-01T00:00:00",
            "update_time": "2026-01-02T00:00:00",
        }
        for i in range(count)
    ]


def revision_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"r{i:04d}",
            "page_id": "p0000",
            "edit_source": "agent" if i else "user",
            "create_time": "2026-01-01T00:00:00",
            "update_time": "2026-01-01T00:00:00",
        }
        for i in range(count)
    ]


# A row wide enough that a full page of them cannot fit the 50,000 byte budget, so the
# cut cases really are cut. Kept to one field so the fixture stays readable.
CUT_FILLER = " padding" * 60


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def cases() -> list[Case]:  # noqa: PLR0915 -- a flat table reads better than helpers
    out: list[Case] = []

    def add(
        tool: str,
        name: str,
        note: str,
        inp: dict[str, Any],
        call: Callable[[], GalaxyResult],
        routes: list[dict[str, Any]],
    ) -> None:
        out.append(Case(tool=tool, name=name, note=note, input=inp, call=call, routes=routes))

    # -- search_tools_by_name ------------------------------------------------
    tools_40 = tool_rows(40)
    tools_index = [route("/api/tools", tools_40)]
    add(
        "search_tools_by_name",
        "full_page",
        "a full page in the middle of the matches",
        {"query": "matchable", "limit": 10, "offset": 10},
        lambda: search_tools_fn("matchable", limit=10, offset=10),
        tools_index,
    )
    add(
        "search_tools_by_name",
        "last_short_page",
        "the last page, shorter than the limit asked for",
        {"query": "matchable", "limit": 15, "offset": 30},
        lambda: search_tools_fn("matchable", limit=15, offset=30),
        tools_index,
    )
    add(
        "search_tools_by_name",
        "empty",
        "nothing matched at all",
        {"query": "nothing matches this", "limit": 10, "offset": 0},
        lambda: search_tools_fn("nothing matches this", limit=10, offset=0),
        tools_index,
    )
    add(
        "search_tools_by_name",
        "past_the_end",
        "an offset past the last match",
        {"query": "matchable", "limit": 10, "offset": 400},
        lambda: search_tools_fn("matchable", limit=10, offset=400),
        tools_index,
    )
    add(
        "search_tools_by_name",
        "cut_by_the_budget",
        "a page cut short to fit the output budget, which the helper text has to say",
        {"query": "matchable", "limit": 100, "offset": 0},
        lambda: search_tools_fn("matchable", limit=100, offset=0),
        [route("/api/tools", tool_rows(90, filler=CUT_FILLER))],
    )
    # A needle and a haystack are both lowercased before one is looked for in the
    # other, and lowercasing is not only a table: a capital sigma at the end of a
    # word becomes a FINAL sigma, and "the end of a word" is a question about which
    # characters are cased. U+1C8A was given a case after this interpreter's Unicode
    # edition, so here the sigma after it is an ordinary one and a query of an
    # ordinary sigma finds the tool. A runtime on newer tables reads U+1C8A as a
    # cased letter, makes the sigma final, and finds nothing -- with neither
    # character's own lowercase differing at all.
    add(
        "search_tools_by_name",
        "sigma_after_a_letter_assigned_after_unicode_15",
        "a query whose match depends on which characters count as cased",
        {"query": "σ", "limit": 10, "offset": 0},
        lambda: search_tools_fn("σ", limit=10, offset=0),
        [route("/api/tools", [{"id": "t", "name": "ᲊΣ", "description": ""}])],
    )

    # -- search_tools_by_keywords -------------------------------------------
    # Every tool matches on its description, so no detail lookup happens; the
    # keyword op's extension path needs one reply per tool and is not what these
    # cases are about.
    panel_30 = panel_sections(1, 30)
    keyword_routes = [route("/api/tools", panel_30)]
    add(
        "search_tools_by_keywords",
        "full_page",
        "a full page of keyword matches",
        {"keywords": ["section 0 tool"], "limit": 10, "offset": 0},
        lambda: search_tools_by_keywords_fn(["section 0 tool"], limit=10, offset=0),
        keyword_routes,
    )
    add(
        "search_tools_by_keywords",
        "last_short_page",
        "the last page of keyword matches",
        {"keywords": ["section 0 tool"], "limit": 20, "offset": 20},
        lambda: search_tools_by_keywords_fn(["section 0 tool"], limit=20, offset=20),
        keyword_routes,
    )
    add(
        "search_tools_by_keywords",
        "past_the_end",
        "an offset past the last keyword match",
        {"keywords": ["section 0 tool"], "limit": 10, "offset": 300},
        lambda: search_tools_by_keywords_fn(["section 0 tool"], limit=10, offset=300),
        keyword_routes,
    )
    # Two keywords, because the sentence joins them with ", " and one keyword cannot
    # show that. The second matches nothing, and the first still matches every tool,
    # so no detail lookup happens here either.
    add(
        "search_tools_by_keywords",
        "two_keywords",
        "two keywords, which the sentence lists in the order they were given",
        {"keywords": ["section 0 tool", "nothing matches this"], "limit": 10, "offset": 0},
        lambda: search_tools_by_keywords_fn(
            ["section 0 tool", "nothing matches this"], limit=10, offset=0
        ),
        keyword_routes,
    )

    # -- get_tool_panel ------------------------------------------------------
    panel = panel_sections(12, 4)
    panel_routes = [route("/api/tools", panel)]
    add(
        "get_tool_panel",
        "overview_full_page",
        "the sections themselves, one page of them",
        {"limit": 5, "offset": 0},
        lambda: get_tool_panel_fn(limit=5, offset=0),
        panel_routes,
    )
    add(
        "get_tool_panel",
        "overview_last_page",
        "the last page of sections",
        {"limit": 5, "offset": 10},
        lambda: get_tool_panel_fn(limit=5, offset=10),
        panel_routes,
    )
    add(
        "get_tool_panel",
        "overview_past_the_end",
        "an offset past the last section",
        {"limit": 5, "offset": 50},
        lambda: get_tool_panel_fn(limit=5, offset=50),
        panel_routes,
    )
    add(
        "get_tool_panel",
        "section_full_page",
        "one section's tools, which is a different data shape from the overview",
        {"section_id": "section3", "limit": 3, "offset": 0},
        lambda: get_tool_panel_fn(section_id="section3", limit=3, offset=0),
        panel_routes,
    )
    add(
        "get_tool_panel",
        "section_cut_by_the_budget",
        "a section's tools cut short to fit the output budget",
        {"section_id": "section0", "limit": 500, "offset": 0},
        lambda: get_tool_panel_fn(section_id="section0", limit=500, offset=0),
        [
            route(
                "/api/tools",
                [
                    {
                        "id": "section0",
                        "name": "Section 0",
                        "model_class": "ToolSection",
                        "elems": [
                            {
                                "id": f"sec0_tool{t}",
                                "name": f"Section 0 Tool {t}",
                                "description": f"tool {t}{CUT_FILLER}",
                                "versions": ["1.0.0"],
                                "model_class": "Tool",
                            }
                            for t in range(90)
                        ],
                    }
                ],
            )
        ],
    )

    # A panel that is not flat: a section holding a tool, a sub-section and a
    # divider label, and a tool sitting outside any section. A section entry's
    # tool_count counts the tools directly in it -- a sub-section is neither a tool
    # nor counted through, and a label is not a tool -- so a recursive count would
    # read 2 here and a flat one over every node would read 3. Neither number is
    # what either server reports, and the drill-in lists the one tool for the same
    # reason. There is no panel-wide total on this tool at all, on either side.
    nested_panel = [
        {
            "id": "outer",
            "name": "Outer",
            "model_class": "ToolSection",
            "elems": [
                {
                    "id": "outer_tool",
                    "name": "Outer Tool",
                    "description": "directly in the section",
                    "versions": ["1.0.0"],
                    "model_class": "Tool",
                },
                {
                    "id": "inner",
                    "name": "Inner",
                    "model_class": "ToolSection",
                    "elems": [
                        {
                            "id": "inner_tool",
                            "name": "Inner Tool",
                            "description": "one level down",
                            "versions": ["1.0.0"],
                            "model_class": "Tool",
                        }
                    ],
                },
                {"id": "divider", "name": "Divider", "model_class": "ToolSectionLabel"},
            ],
        },
        {
            "id": "loose_tool",
            "name": "Loose Tool",
            "description": "outside any section",
            "versions": ["1.0.0"],
            "model_class": "Tool",
        },
    ]
    nested_panel_routes = [route("/api/tools", nested_panel)]
    add(
        "get_tool_panel",
        "nested_overview",
        "a nested panel, where a section entry counts only the tools directly in it",
        {"limit": 5, "offset": 0},
        lambda: get_tool_panel_fn(limit=5, offset=0),
        nested_panel_routes,
    )
    add(
        "get_tool_panel",
        "nested_section",
        "opening a section that holds a sub-section and a label as well as a tool",
        {"section_id": "outer", "limit": 5, "offset": 0},
        lambda: get_tool_panel_fn(section_id="outer", limit=5, offset=0),
        nested_panel_routes,
    )

    # -- list_history_ids ----------------------------------------------------
    histories_40 = history_rows(40)
    histories_index = [route("/api/histories", histories_40)]
    add(
        "list_history_ids",
        "full_page",
        "a full page of history ids",
        {"limit": 10, "offset": 10},
        lambda: list_history_ids_fn(limit=10, offset=10),
        histories_index,
    )
    add(
        "list_history_ids",
        "last_short_page",
        "the last, short page of history ids",
        {"limit": 15, "offset": 30},
        lambda: list_history_ids_fn(limit=15, offset=30),
        histories_index,
    )
    add(
        "list_history_ids",
        "empty",
        "an account with no histories at all",
        {"limit": 10, "offset": 0},
        lambda: list_history_ids_fn(limit=10, offset=0),
        [route("/api/histories", [])],
    )
    add(
        "list_history_ids",
        "past_the_end",
        "an offset past the last history",
        {"limit": 10, "offset": 400},
        lambda: list_history_ids_fn(limit=10, offset=400),
        histories_index,
    )
    # Right on the boundary. Two names of 24,835 characters put a full page of two
    # within eleven bytes of the 50,000 byte budget, and the budget is measured on
    # the whole envelope -- message included -- so a sentence eleven bytes shorter
    # on one surface keeps a row the other cuts. That is not a hypothetical: it is
    # what the two surfaces did here while their messages were still their own.
    boundary_histories = [{"id": f"h{i}", "name": "n" * 24_835} for i in range(2)]
    # A history with no name at all, one whose name Galaxy sent as null, and one
    # named the empty string. The word "Unnamed" stands in for the missing key and
    # for nothing else, so all three have to be here to say which is which.
    add(
        "list_history_ids",
        "nameless",
        "histories with no name, a null name and an empty name, which are three answers",
        {"limit": 10, "offset": 0},
        lambda: list_history_ids_fn(limit=10, offset=0),
        [
            route(
                "/api/histories",
                [
                    {"id": "hmissing", "state": "ok"},
                    {"id": "hnull", "name": None, "state": "ok"},
                    {"id": "hempty", "name": "", "state": "ok"},
                ],
            )
        ],
    )
    add(
        "list_history_ids",
        "budget_boundary",
        "two histories whose full page lands within a few bytes of the output budget",
        {"limit": 2, "offset": 0},
        lambda: list_history_ids_fn(limit=2, offset=0),
        [route("/api/histories", boundary_histories)],
    )

    # -- get_histories -------------------------------------------------------
    # One unpaged fetch on both sides now: this server filters by name before it
    # windows, because bioblend's name filter runs after Galaxy has cut the page.
    histories_index_unpaged = [route("/api/histories", histories_40)]
    add(
        "get_histories",
        "limited_full_page",
        "a limit and an offset, so a pagination block comes back",
        {"limit": 10, "offset": 10},
        lambda: get_histories_fn(limit=10, offset=10),
        histories_index_unpaged,
    )
    add(
        "get_histories",
        "limited_last_page",
        "the last page under a limit",
        {"limit": 15, "offset": 30},
        lambda: get_histories_fn(limit=15, offset=30),
        histories_index_unpaged,
    )
    add(
        "get_histories",
        "limited_past_the_end",
        "an offset past the last history, under a limit",
        {"limit": 10, "offset": 400},
        lambda: get_histories_fn(limit=10, offset=400),
        histories_index_unpaged,
    )
    add(
        "get_histories",
        "no_limit",
        "no limit asked for, which is the branch with no pagination block at all",
        {},
        lambda: get_histories_fn(),
        [route("/api/histories", history_rows(3))],
    )
    add(
        "get_histories",
        "empty",
        "an account with no histories, under a limit",
        {"limit": 10, "offset": 0},
        lambda: get_histories_fn(limit=10, offset=0),
        [route("/api/histories", [])],
    )
    # Two histories, one of them named B: the window has to be cut from the matches,
    # or page one of name="B" answers with A's window -- empty, and claiming there is
    # more -- and the page after it counts a match that was never there.
    histories_ab = [
        {"id": "hA", "name": "A", "state": "ok", "update_time": "2026-01-01T00:00:00"},
        {"id": "hB", "name": "B", "state": "ok", "update_time": "2026-01-02T00:00:00"},
    ]
    histories_ab_route = [route("/api/histories", histories_ab)]
    add(
        "get_histories",
        "no_limit_with_offset",
        "an offset and no limit, which skips without describing a window",
        {"offset": 1},
        lambda: get_histories_fn(offset=1),
        histories_ab_route,
    )
    add(
        "get_histories",
        "name_filter_first_page",
        "page one of a name filter whose only match is not in the first window",
        {"limit": 1, "offset": 0, "name": "B"},
        lambda: get_histories_fn(limit=1, offset=0, name="B"),
        histories_ab_route,
    )
    add(
        "get_histories",
        "name_filter_past_the_end",
        "the page after the only match, which still totals one",
        {"limit": 1, "offset": 1, "name": "B"},
        lambda: get_histories_fn(limit=1, offset=1, name="B"),
        histories_ab_route,
    )
    add(
        "get_histories",
        "zero_limit",
        "limit=0, which bioblend reads as no window at all: one page holding everything",
        {"limit": 0, "offset": 0},
        lambda: get_histories_fn(limit=0, offset=0),
        [route("/api/histories", history_rows(1))],
    )

    # -- get_history_contents ------------------------------------------------
    contents_25 = content_rows(25)
    contents_route = [route("/api/histories/h0000/contents", contents_25)]
    add(
        "get_history_contents",
        "full_page",
        "a page of contents, wrapped in this tool's own data shape",
        {"history_id": "h0000", "limit": 10, "offset": 10},
        lambda: get_history_contents_fn("h0000", limit=10, offset=10),
        contents_route,
    )
    add(
        "get_history_contents",
        "last_short_page",
        "the last, short page of contents",
        {"history_id": "h0000", "limit": 10, "offset": 20},
        lambda: get_history_contents_fn("h0000", limit=10, offset=20),
        contents_route,
    )
    add(
        "get_history_contents",
        "empty",
        "an empty history",
        {"history_id": "h0000", "limit": 10, "offset": 0},
        lambda: get_history_contents_fn("h0000", limit=10, offset=0),
        [route("/api/histories/h0000/contents", [])],
    )
    add(
        "get_history_contents",
        "past_the_end",
        "an offset past the last item",
        {"history_id": "h0000", "limit": 10, "offset": 400},
        lambda: get_history_contents_fn("h0000", limit=10, offset=400),
        contents_route,
    )

    # -- get_history_details -------------------------------------------------
    # Two requests, in this order: the history itself, then its contents, which are
    # fetched only to be counted. The count is every item the history holds, deleted
    # and hidden included, which is why the metadata's own numbers will not do.
    history_record = {
        "id": "h0000",
        "name": "History 0",
        "state": "ok",
        "deleted": False,
        "purged": False,
        "published": False,
        "annotation": None,
        "tags": [],
        "size": 2048,
        "create_time": "2026-01-01T00:00:00",
        "update_time": "2026-01-02T00:00:00",
        "contents_active": {"active": 3, "deleted": 1, "hidden": 0},
    }
    add(
        "get_history_details",
        "with_contents",
        "a history and the number of items in it, wrapped in this tool's own data shape",
        {"history_id": "h0000"},
        lambda: get_history_details_fn("h0000"),
        [
            route("/api/histories/h0000", history_record),
            route("/api/histories/h0000/contents", content_rows(4)),
        ],
    )
    add(
        "get_history_details",
        "empty",
        "a history with nothing in it, which still carries the same note",
        {"history_id": "h0000"},
        lambda: get_history_details_fn("h0000"),
        [
            route("/api/histories/h0000", history_record),
            route("/api/histories/h0000/contents", []),
        ],
    )
    # The name is read with dict.get and falls back to the id that was asked for, so a
    # record carrying no name key at all is the branch that shows the fallback.
    add(
        "get_history_details",
        "no_name_on_the_record",
        "a history record with no name key, which the sentence names by its id",
        {"history_id": "h0000"},
        lambda: get_history_details_fn("h0000"),
        [
            route(
                "/api/histories/h0000",
                {k: v for k, v in history_record.items() if k != "name"},
            ),
            route("/api/histories/h0000/contents", content_rows(1)),
        ],
    )

    # -- list_workflows ------------------------------------------------------
    workflows_30 = workflow_rows(30)
    workflows_route = [route("/api/workflows", workflows_30)]
    add(
        "list_workflows",
        "full_page",
        "a full page of workflows",
        {"limit": 10, "offset": 10},
        lambda: list_workflows_fn(limit=10, offset=10),
        workflows_route,
    )
    add(
        "list_workflows",
        "last_short_page",
        "the last, short page of workflows",
        {"limit": 20, "offset": 20},
        lambda: list_workflows_fn(limit=20, offset=20),
        workflows_route,
    )
    add(
        "list_workflows",
        "empty",
        "no workflows at all",
        {"limit": 10, "offset": 0},
        lambda: list_workflows_fn(limit=10, offset=0),
        [route("/api/workflows", [])],
    )
    add(
        "list_workflows",
        "past_the_end",
        "an offset past the last workflow",
        {"limit": 10, "offset": 400},
        lambda: list_workflows_fn(limit=10, offset=400),
        workflows_route,
    )

    # -- list_user_tools -----------------------------------------------------
    user_tools_20 = user_tool_rows(20)
    user_tools_route = [route("/api/unprivileged_tools", user_tools_20)]
    add(
        "list_user_tools",
        "full_page",
        "a full page of user-defined tools",
        {"limit": 5, "offset": 5},
        lambda: list_user_tools_fn(limit=5, offset=5),
        user_tools_route,
    )
    add(
        "list_user_tools",
        "last_short_page",
        "the last, short page of user-defined tools",
        {"limit": 15, "offset": 15},
        lambda: list_user_tools_fn(limit=15, offset=15),
        user_tools_route,
    )
    add(
        "list_user_tools",
        "empty",
        "no user-defined tools",
        {"limit": 5, "offset": 0},
        lambda: list_user_tools_fn(limit=5, offset=0),
        [route("/api/unprivileged_tools", [])],
    )
    add(
        "list_user_tools",
        "past_the_end",
        "an offset past the last user-defined tool",
        {"limit": 5, "offset": 400},
        lambda: list_user_tools_fn(limit=5, offset=400),
        user_tools_route,
    )

    # -- get_iwc_workflows / search_iwc_workflows ---------------------------
    manifest_25 = iwc_manifest(25)
    manifest_route = [route(IWC_MANIFEST_URL, manifest_25)]
    add(
        "get_iwc_workflows",
        "full_page",
        "a page of IWC summaries",
        {"limit": 10, "offset": 10},
        lambda: get_iwc_workflows_fn(limit=10, offset=10),
        manifest_route,
    )
    add(
        "get_iwc_workflows",
        "last_short_page",
        "the last, short page of IWC summaries",
        {"limit": 10, "offset": 20},
        lambda: get_iwc_workflows_fn(limit=10, offset=20),
        manifest_route,
    )
    add(
        "get_iwc_workflows",
        "empty",
        "an empty IWC manifest",
        {"limit": 10, "offset": 0},
        lambda: get_iwc_workflows_fn(limit=10, offset=0),
        [route(IWC_MANIFEST_URL, [])],
    )
    add(
        "get_iwc_workflows",
        "past_the_end",
        "an offset past the last IWC workflow",
        {"limit": 10, "offset": 400},
        lambda: get_iwc_workflows_fn(limit=10, offset=400),
        manifest_route,
    )
    # Steps whose keys arrive out of order, with a key that is not an index at all.
    # A JSON object has no order the two servers agree on, so the order tools_used
    # comes out in is the stated one: canonical non-negative integers ascending,
    # then the rest as they arrived. Insertion order alone would answer
    # hisat2, multiqc, fastqc, cutadapt; ascending-everything would put cutadapt
    # somewhere it does not belong.
    out_of_order_steps = {
        "2": {"type": "tool", "tool_id": "toolshed.g2.bx.psu.edu/repos/iuc/hisat2/hisat2/2.2.1"},
        "10": {"type": "tool", "tool_id": "multiqc"},
        "1": {"type": "tool", "tool_id": "fastqc"},
        "x": {"type": "tool", "tool_id": "cutadapt"},
    }
    out_of_order_manifest = [
        {
            "repo": "repo0",
            "workflows": [
                {
                    "trsID": "#workflow/github.com/iwc-workflows/wf0/main",
                    "definition": {
                        "name": "Workflow 0",
                        "annotation": "rnaseq analysis number 0",
                        "tags": ["transcriptomics", "rnaseq"],
                        "license": "MIT",
                        "creator": [{"class": "Person", "name": "IWC", "identifier": ""}],
                        "steps": out_of_order_steps,
                    },
                    "readme": "Workflow 0 runs rnaseq quality control.",
                    "categories": ["Transcriptomics"],
                }
            ],
        }
    ]
    add(
        "get_iwc_workflows",
        "steps_out_of_order",
        "a workflow whose step keys arrive out of order, which decides tools_used",
        {"limit": 10, "offset": 0},
        lambda: get_iwc_workflows_fn(limit=10, offset=0),
        [route(IWC_MANIFEST_URL, out_of_order_manifest)],
    )

    # -- get_iwc_workflow_details -------------------------------------------
    # The same steps read a second way: this tool walks them for inputs and
    # outputs, so its two lists are the same order question again.
    details_steps = {
        "2": {
            "type": "data_input",
            "label": "Reads",
            "annotation": "the fastq",
            "workflow_outputs": [{"label": "copy", "output_name": "output"}],
        },
        "10": {"type": "parameter_input", "label": "Threads", "annotation": ""},
        "1": {
            "type": "tool",
            "tool_id": "fastqc",
            "label": "QC",
            "workflow_outputs": [{"output_name": "html_file"}],
        },
        "x": {"type": "data_collection_input", "label": "Extra", "annotation": ""},
    }
    details_manifest = [
        {
            "repo": "repo0",
            "workflows": [
                {
                    "trsID": "#workflow/github.com/iwc-workflows/wf0/main",
                    "definition": {
                        "name": "Workflow 0",
                        "annotation": "rnaseq analysis number 0",
                        "tags": ["transcriptomics"],
                        "license": "MIT",
                        "creator": [{"class": "Person", "name": "IWC", "identifier": ""}],
                        "steps": details_steps,
                    },
                    "readme": "Workflow 0 runs rnaseq quality control.",
                    "categories": ["Transcriptomics"],
                }
            ],
        }
    ]
    add(
        "get_iwc_workflow_details",
        "steps_out_of_order",
        "one workflow's inputs and outputs, which are the step order read twice",
        {"trs_id": "#workflow/github.com/iwc-workflows/wf0/main"},
        lambda: get_iwc_workflow_details_fn("#workflow/github.com/iwc-workflows/wf0/main"),
        [route(IWC_MANIFEST_URL, details_manifest)],
    )

    # A .ga export states a label it does not have as null, and the other server
    # passes a present null (or an empty string) through: only an absent key gets
    # the "Input N" / "Step N" / output_name fallback.
    null_label_steps = {
        "1": {
            "type": "data_input",
            "label": None,
            "annotation": None,
            "workflow_outputs": [
                {"label": None, "output_name": "output"},
                {"label": "", "output_name": "out2"},
                {"output_name": "out3"},
                {},
            ],
        },
        "2": {"type": "tool", "tool_id": "t", "workflow_outputs": [{"label": "named"}]},
        "3": {"type": "parameter_input"},
    }
    null_label_manifest = [
        {
            "repo": "repo0",
            "workflows": [
                {
                    "trsID": "#workflow/github.com/iwc-workflows/wf1/main",
                    "definition": {
                        "name": "Workflow 1",
                        "annotation": "labels stated as null",
                        "tags": [],
                        "license": "MIT",
                        "creator": [],
                        "steps": null_label_steps,
                    },
                    "readme": "Workflow 1.",
                    "categories": [],
                }
            ],
        }
    ]
    add(
        "get_iwc_workflow_details",
        "labels_null_empty_and_absent",
        "inputs and outputs whose labels are null, empty, or missing",
        {"trs_id": "#workflow/github.com/iwc-workflows/wf1/main"},
        lambda: get_iwc_workflow_details_fn("#workflow/github.com/iwc-workflows/wf1/main"),
        [route(IWC_MANIFEST_URL, null_label_manifest)],
    )

    add(
        "search_iwc_workflows",
        "full_page",
        "a page of IWC search matches",
        {"query": "rnaseq", "limit": 10, "offset": 0},
        lambda: search_iwc_workflows_fn("rnaseq", limit=10, offset=0),
        manifest_route,
    )
    add(
        "search_iwc_workflows",
        "last_short_page",
        "the last, short page of IWC search matches",
        {"query": "rnaseq", "limit": 10, "offset": 20},
        lambda: search_iwc_workflows_fn("rnaseq", limit=10, offset=20),
        manifest_route,
    )
    add(
        "search_iwc_workflows",
        "empty",
        "an IWC query nothing matches",
        {"query": "nothing matches this", "limit": 10, "offset": 0},
        lambda: search_iwc_workflows_fn("nothing matches this", limit=10, offset=0),
        manifest_route,
    )
    add(
        "search_iwc_workflows",
        "past_the_end",
        "an offset past the last IWC match",
        {"query": "rnaseq", "limit": 10, "offset": 400},
        lambda: search_iwc_workflows_fn("rnaseq", limit=10, offset=400),
        manifest_route,
    )

    # -- recommend_iwc_workflows --------------------------------------------
    # A minority of the corpus on topic: BM25 scores a term most documents share at
    # zero or below, so a ranking needs something the rest do not say.
    ranked_manifest = iwc_manifest(20)
    for i, entry in enumerate(ranked_manifest):
        if i % 4:
            entry["workflows"][0]["definition"]["tags"] = ["proteomics", "maxquant"]
            entry["workflows"][0]["definition"]["annotation"] = f"proteomics run {i}"
            entry["workflows"][0]["readme"] = f"Workflow {i} quantifies proteins."
    add(
        "recommend_iwc_workflows",
        "ranked",
        "a ranking, which carries no pagination block at all",
        {"intent": "rnaseq quality control", "limit": 5},
        lambda: recommend_iwc_workflows_fn("rnaseq quality control", limit=5),
        [route(IWC_MANIFEST_URL, ranked_manifest)],
    )
    # 26 fat on-topic entries in a corpus of 60: enough to overflow the budget, and
    # still a minority, which BM25 needs -- a term most of the corpus shares scores
    # nothing at all.
    cut_manifest = iwc_manifest(60)
    for i, entry in enumerate(cut_manifest):
        definition = entry["workflows"][0]["definition"]
        if i < 26:
            definition["annotation"] = f"rnaseq quality control run {i}." + CUT_FILLER * 4
        else:
            definition["tags"] = ["proteomics"]
            definition["annotation"] = f"proteomics run {i}"
            entry["workflows"][0]["readme"] = f"Workflow {i} quantifies proteins."
    add(
        "recommend_iwc_workflows",
        "cut_by_the_budget",
        "a ranking cut to fit the output budget, which this tool explains in the message",
        {"intent": "rnaseq quality control", "limit": 25},
        lambda: recommend_iwc_workflows_fn("rnaseq quality control", limit=25),
        [route(IWC_MANIFEST_URL, cut_manifest)],
    )
    add(
        "recommend_iwc_workflows",
        "no_terms_match",
        "an intent nothing scores against",
        {"intent": "zzzz", "limit": 5},
        lambda: recommend_iwc_workflows_fn("zzzz", limit=5),
        [route(IWC_MANIFEST_URL, ranked_manifest)],
    )
    # The two early returns, which say why there is nothing rather than reporting
    # zero matches. Order matters: the manifest is tested before the query.
    add(
        "recommend_iwc_workflows",
        "empty_manifest",
        "an IWC manifest with nothing in it, which is not the same as nothing matching",
        {"intent": "rnaseq quality control", "limit": 5},
        lambda: recommend_iwc_workflows_fn("rnaseq quality control", limit=5),
        [route(IWC_MANIFEST_URL, [])],
    )
    add(
        "recommend_iwc_workflows",
        "no_searchable_terms",
        "an intent that tokenises to nothing, over a manifest that is not empty",
        {"intent": "", "limit": 5},
        lambda: recommend_iwc_workflows_fn("", limit=5),
        [route(IWC_MANIFEST_URL, ranked_manifest)],
    )
    add(
        "recommend_iwc_workflows",
        "only_stop_words",
        "an intent that is nothing but the words the tokeniser drops",
        {"intent": "the and for with", "limit": 5},
        lambda: recommend_iwc_workflows_fn("the and for with", limit=5),
        [route(IWC_MANIFEST_URL, ranked_manifest)],
    )
    # The tokeniser's word boundaries are Unicode's: the accent continues the word,
    # so there is no run of ASCII letters standing alone and nothing to search for.
    # An engine whose boundaries are ASCII pulls "caf" out of this and reports a
    # ranking that matched nothing, which is a different answer to a different
    # question -- hence a case rather than a unit test.
    add(
        "recommend_iwc_workflows",
        "accented_intent",
        "an intent of one accented word, which tokenises to nothing at all",
        {"intent": "café", "limit": 5},
        lambda: recommend_iwc_workflows_fn("café", limit=5),
        [route(IWC_MANIFEST_URL, ranked_manifest)],
    )
    # The other side of the same rule, and the one that costs something: U+A7CB is a
    # letter Unicode assigned after this interpreter's edition, so here it is a
    # separator, "rnaseq" beside it is a word, and the ranking comes back full. A
    # tokeniser reading a newer edition sees one long word, finds no term and answers
    # "No searchable terms in query" -- five workflows against none, for one character
    # a user could paste in without noticing. Pinned as a case because the answer
    # depends on which Unicode edition is the contract, and this server is.
    add(
        "recommend_iwc_workflows",
        "letter_added_after_unicode_15",
        "an intent ending in a letter this interpreter has not been told about",
        {"intent": "rnaseqꟋ", "limit": 5},
        lambda: recommend_iwc_workflows_fn("rnaseqꟋ", limit=5),
        [route(IWC_MANIFEST_URL, ranked_manifest)],
    )

    # -- get_invocations -----------------------------------------------------
    add(
        "get_invocations",
        "list",
        "a list of invocations: a count, and no pagination block",
        {"limit": 5},
        lambda: get_invocations_fn(limit=5),
        [route("/api/invocations", invocation_rows(3))],
    )
    add(
        "get_invocations",
        "single",
        "one invocation by id: no count and no pagination",
        {"invocation_id": "inv0000"},
        lambda: get_invocations_fn(invocation_id="inv0000"),
        [route("/api/invocations/inv0000", invocation_rows(1)[0])],
    )
    add(
        "get_invocations",
        "empty",
        "no invocations",
        {"limit": 5},
        lambda: get_invocations_fn(limit=5),
        [route("/api/invocations", [])],
    )

    # -- list_pages ----------------------------------------------------------
    add(
        "list_pages",
        "full_page",
        "a page of pages, whose total comes from the total_matches header",
        {"limit": 5, "offset": 0},
        lambda: list_pages_fn(limit=5, offset=0),
        [
            VERSION_ROUTE,
            route("/api/pages", page_rows(5), headers={"total_matches": "12"}),
        ],
    )
    add(
        "list_pages",
        "last_page",
        "the last page of pages",
        {"limit": 5, "offset": 10},
        lambda: list_pages_fn(limit=5, offset=10),
        [
            VERSION_ROUTE,
            route("/api/pages", page_rows(2), headers={"total_matches": "12"}),
        ],
    )
    add(
        "list_pages",
        "empty",
        "no pages, and a total_matches of zero",
        {"limit": 5, "offset": 0},
        lambda: list_pages_fn(limit=5, offset=0),
        [VERSION_ROUTE, route("/api/pages", [], headers={"total_matches": "0"})],
    )
    add(
        "list_pages",
        "past_the_end",
        "an offset past the last page, which the shared helper has a sentence for",
        {"limit": 5, "offset": 40},
        lambda: list_pages_fn(limit=5, offset=40),
        [VERSION_ROUTE, route("/api/pages", [], headers={"total_matches": "12"})],
    )
    # The header is the total, except that rows in hand prove a floor: a server that
    # under-reports cannot make a page it just sent disappear. That is the one line in
    # the shared helper that makes a total-from-a-header expressible through it.
    add(
        "list_pages",
        "header_below_the_rows_in_hand",
        "a total_matches smaller than the page it came with",
        {"limit": 5, "offset": 0},
        lambda: list_pages_fn(limit=5, offset=0),
        [VERSION_ROUTE, route("/api/pages", page_rows(3), headers={"total_matches": "1"})],
    )

    # -- list_page_revisions -------------------------------------------------
    add(
        "list_page_revisions",
        "some",
        "a page's revisions: a count, and no pagination block",
        {"page_id": "p0000"},
        lambda: list_page_revisions_fn("p0000"),
        [VERSION_ROUTE, route("/api/pages/p0000/revisions", revision_rows(3))],
    )
    add(
        "list_page_revisions",
        "empty",
        "a page with no revisions recorded",
        {"page_id": "p0000"},
        lambda: list_page_revisions_fn("p0000"),
        [VERSION_ROUTE, route("/api/pages/p0000/revisions", [])],
    )

    # -- get_tool_run_examples ----------------------------------------------
    test_cases = [
        {"test_index": 0, "inputs": {"input1": "1.bed"}, "outputs": {"out_file1": "2.bed"}},
        {"test_index": 1, "inputs": {"input1": "3.bed"}, "outputs": {"out_file1": "4.bed"}},
    ]
    add(
        "get_tool_run_examples",
        "with_version",
        "test cases counted, with the version the caller asked for echoed back",
        {"tool_id": "cat1", "tool_version": "1.0.0"},
        lambda: get_tool_run_examples_fn("cat1", tool_version="1.0.0"),
        [route("/api/tools/cat1/test_data", test_cases)],
    )
    add(
        "get_tool_run_examples",
        "none",
        "a tool with no test cases",
        {"tool_id": "cat1", "tool_version": "1.0.0"},
        lambda: get_tool_run_examples_fn("cat1", tool_version="1.0.0"),
        [route("/api/tools/cat1/test_data", [])],
    )
    # No version asked for, which is still a stated one: requested_version comes back
    # null rather than missing, because an absent key reads as "no such field".
    add(
        "get_tool_run_examples",
        "no_version_asked_for",
        "no version named, which the answer states as null rather than leaving out",
        {"tool_id": "cat1"},
        lambda: get_tool_run_examples_fn("cat1"),
        [route("/api/tools/cat1/test_data", test_cases)],
    )

    # -- get_collection_details ----------------------------------------------
    # The elements come back normalised rather than as Galaxy sent them: one flat
    # record per element, its index counted from the head of the returned list, and
    # the dataset behind it read one level deep and no further. A nested collection
    # is an element whose object is another collection, and it stays that way -- the
    # fields a dataset would fill are simply empty.
    def collection_element(index: int, *, size: int | None = 1024) -> dict[str, Any]:
        return {
            "id": f"dce{index}",
            "element_index": index,
            "element_identifier": f"sample{index}",
            "element_type": "hda",
            "model_class": "DatasetCollectionElement",
            "object": {
                "id": f"d{index:04d}",
                "name": f"sample{index}.txt",
                "state": "ok",
                "extension": "txt",
                "file_size": size,
                "history_id": "h0000",
            },
        }

    def collection_record(elements: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
        record = {
            "id": "dc000001",
            "name": "Sample list",
            "collection_type": "list",
            "element_count": len(elements),
            "populated": True,
            "state": "ok",
            "history_content_type": "dataset_collection",
            "elements": elements,
        }
        record.update(over)
        return record

    add(
        "get_collection_details",
        "some_elements",
        "a collection whose elements all fit under the cap",
        {"collection_id": "dc000001"},
        lambda: get_collection_details_fn("dc000001"),
        [
            route(
                "/api/dataset_collections/dc000001",
                collection_record([collection_element(i) for i in range(3)]),
            )
        ],
    )
    add(
        "get_collection_details",
        "truncated",
        "more elements than max_elements, which the note and the flag both have to say",
        {"collection_id": "dc000001", "max_elements": 2},
        lambda: get_collection_details_fn("dc000001", max_elements=2),
        [
            route(
                "/api/dataset_collections/dc000001",
                collection_record([collection_element(i) for i in range(5)]),
            )
        ],
    )
    add(
        "get_collection_details",
        "nested_collection",
        "a list of pairs, where an element's object is another collection and is not walked",
        {"collection_id": "dc000002"},
        lambda: get_collection_details_fn("dc000002"),
        [
            route(
                "/api/dataset_collections/dc000002",
                collection_record(
                    [
                        {
                            "id": "dce0",
                            "element_index": 0,
                            "element_identifier": "pair0",
                            "element_type": "dataset_collection",
                            "model_class": "DatasetCollectionElement",
                            "object": {
                                "id": "dc000003",
                                "collection_type": "paired",
                                "element_count": 2,
                                "populated": True,
                                "elements": [collection_element(0), collection_element(1)],
                            },
                        }
                    ],
                    id="dc000002",
                    name="Paired samples",
                    collection_type="list:paired",
                ),
            )
        ],
    )
    # Every field of the normalised collection missing at once: the defaults are not
    # all the same, and only a reply that leaves them out says which is which.
    add(
        "get_collection_details",
        "sparse_record",
        "a collection Galaxy answered with almost nothing, so every default shows",
        {"collection_id": "dc000004"},
        lambda: get_collection_details_fn("dc000004"),
        [
            route(
                "/api/dataset_collections/dc000004",
                {"elements": [{"element_identifier": "lonely", "object": {}}]},
            )
        ],
    )

    # -- get_workflow_input_template -----------------------------------------
    # Three independent reads of one workflow, and the two servers do not spell all
    # three the same: the run model is `/api/workflows/{id}/download?style=run` on
    # both, but bioblend's .ga export asks `/api/workflows/download/{id}` while the
    # other side asks the first path with no style. Both spellings are the same
    # endpoint to Galaxy, and this table answers questions rather than replaying a
    # log, so the .ga body is registered under both.
    def wf_routes(
        workflow_id: str,
        run_model: dict[str, Any],
        definition: dict[str, Any],
        show: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            route(f"/api/workflows/{workflow_id}/download", run_model, query={"style": "run"}),
            route(f"/api/workflows/{workflow_id}/download", definition),
            route(f"/api/workflows/download/{workflow_id}", definition),
            route(f"/api/workflows/{workflow_id}", show),
        ]

    # One tool step carrying an unconnected RuntimeValue, which is the legacy pattern
    # the template warns about; the input steps around it are the slots.
    def ga_definition(input_steps: dict[str, Any]) -> dict[str, Any]:
        steps = dict(input_steps)
        steps["9"] = {
            "type": "tool",
            "tool_id": "fastqc",
            "label": None,
            "tool_state": json.dumps({"adapters": {"__class__": "RuntimeValue"}}),
        }
        return {"a_galaxy_workflow": "true", "name": "Reads QC", "steps": steps}

    wf_show = {
        "id": "wf000001",
        "name": "Reads QC",
        "version": 3,
        "annotation": "quality control over sequencing reads",
        "readme": "# Reads QC\n\nRuns quality control over reads and reports on them.\n",
        "help": "",
        "source_metadata": {
            "trs_tool_id": "#workflow/github.com/iwc-workflows/reads-qc/main",
            "trs_url": "https://dockstore.org/api/ga4gh/trs/v2/tools/reads-qc",
        },
    }

    def run_model(steps: dict[str, Any], **over: Any) -> dict[str, Any]:
        model = {
            "id": "wf000001",
            "name": "Reads QC",
            "has_upgrade_messages": False,
            "step_version_changes": [],
            "steps": steps,
        }
        model.update(over)
        return model

    data_step = {
        "step_type": "data_input",
        "step_index": 0,
        "step_label": "Input FASTQ",
        "uuid": "11111111-1111-1111-1111-111111111111",
        "inputs": [
            {
                "extensions": ["fastqsanger"],
                "acceptable_extensions": ["fastqsanger", "fastqsanger.gz"],
                "optional": False,
            }
        ],
    }
    add(
        "get_workflow_input_template",
        "simple",
        "one data input off the run model, with a guide and a legacy warning",
        {"workflow_id": "wf000001"},
        lambda: get_workflow_input_template_fn("wf000001"),
        wf_routes(
            "wf000001",
            run_model({"0": data_step, "1": {"step_type": "tool", "step_index": 1, "inputs": []}}),
            ga_definition({"0": {"type": "data_input", "label": "Input FASTQ"}}),
            wf_show,
        ),
    )
    add(
        "get_workflow_input_template",
        "collection_input",
        "a collection slot whose type comes from collection_types rather than collection_type",
        {"workflow_id": "wf000001"},
        lambda: get_workflow_input_template_fn("wf000001"),
        wf_routes(
            "wf000001",
            run_model(
                {
                    "0": {
                        "step_type": "data_collection_input",
                        "step_index": 0,
                        "step_label": "Paired reads",
                        "uuid": "22222222-2222-2222-2222-222222222222",
                        "inputs": [
                            {
                                "extensions": [],
                                "collection_types": ["list:paired", "paired"],
                                "optional": True,
                            }
                        ],
                    }
                }
            ),
            ga_definition({"0": {"type": "data_collection_input", "label": "Paired reads"}}),
            wf_show,
        ),
    )
    # A selector and its choices: the run model hands options over as
    # [label, value, selected] triples, and the template keeps label and value.
    add(
        "get_workflow_input_template",
        "parameter_selector",
        "a parameter slot whose options are a selector's choices",
        {"workflow_id": "wf000001"},
        lambda: get_workflow_input_template_fn("wf000001"),
        wf_routes(
            "wf000001",
            run_model(
                {
                    "0": {
                        "step_type": "parameter_input",
                        "step_index": 0,
                        "uuid": "33333333-3333-3333-3333-333333333333",
                        "inputs": [
                            {
                                "label": "Reference genome",
                                "parameter_type": "text",
                                "optional": False,
                                "options": [
                                    ["Human (hg38)", "hg38", True],
                                    ["Mouse (mm10)", "mm10", False],
                                ],
                            }
                        ],
                    }
                }
            ),
            ga_definition({"0": {"type": "parameter_input", "label": "Reference genome"}}),
            wf_show,
        ),
    )
    # More options than the inline cap, so the sample and its sentence show up. The
    # step carries NO uuid key at all, which is the difference between a slot whose
    # step_uuid is null and a slot that has no such field.
    add(
        "get_workflow_input_template",
        "capped_options",
        "a selector with more choices than the template inlines, on a step with no uuid",
        {"workflow_id": "wf000001"},
        lambda: get_workflow_input_template_fn("wf000001"),
        wf_routes(
            "wf000001",
            run_model(
                {
                    "0": {
                        "step_type": "parameter_input",
                        "step_index": 0,
                        "step_label": "Build",
                        "inputs": [
                            {
                                "parameter_type": "text",
                                "options": [[f"Build {i}", f"b{i}", i == 0] for i in range(30)],
                            }
                        ],
                    }
                }
            ),
            ga_definition({"0": {"type": "parameter_input", "label": "Build"}}),
            wf_show,
        ),
    )
    # Everything the run model is allowed to leave out: no step_index (the index comes
    # from order_index), no uuid, no label anywhere (so the slot names itself), and
    # parameter_type on the step rather than on the param.
    add(
        "get_workflow_input_template",
        "sparse_run_step",
        "a run-model step that names almost nothing, so every fallback is exercised",
        {"workflow_id": "wf000001"},
        lambda: get_workflow_input_template_fn("wf000001"),
        wf_routes(
            "wf000001",
            run_model(
                {
                    "0": {
                        "step_type": "parameter_input",
                        "order_index": 4,
                        "id": 7,
                        "parameter_type": "integer",
                        "inputs": [{}],
                    }
                }
            ),
            ga_definition({"0": {"type": "parameter_input"}}),
            wf_show,
        ),
    )
    # The run model answers with no input steps at all, so both servers fall back to
    # the .ga export: options come from `restrictions`, nothing is server-resolved,
    # and the guide says so in a note.
    add(
        "get_workflow_input_template",
        "ga_fallback",
        "the .ga path, which resolves no options and says so in the guide's notes",
        {"workflow_id": "wf000002"},
        lambda: get_workflow_input_template_fn("wf000002"),
        wf_routes(
            "wf000002",
            run_model({"0": {"step_type": "tool", "step_index": 0, "inputs": []}}),
            ga_definition(
                {
                    "0": {
                        "type": "data_input",
                        "label": "",
                        "uuid": "44444444-4444-4444-4444-444444444444",
                        "tool_state": json.dumps({"format": "tabular", "optional": False}),
                    },
                    "1": {
                        "type": "parameter_input",
                        "uuid": None,
                        "tool_state": json.dumps(
                            {
                                "parameter_type": "text",
                                "restrictions": ["alpha", "beta"],
                                "optional": True,
                            }
                        ),
                    },
                }
            ),
            {**wf_show, "readme": "", "help": "", "annotation": "a bare annotation"},
        ),
    )
    # The same order question on the .ga path: the slots come out in step order, and
    # the key that is not an index is not a slot. Insertion order alone would put
    # step 2 first; treating every integer-ish key as an index would make a slot of
    # "007", which is a key a workflow should not have.
    add(
        "get_workflow_input_template",
        "steps_out_of_order",
        "step keys out of order, plus two keys that are not step indexes",
        {"workflow_id": "wf000003"},
        lambda: get_workflow_input_template_fn("wf000003"),
        wf_routes(
            "wf000003",
            run_model({}),
            {
                "a_galaxy_workflow": "true",
                "name": "Out of order",
                "steps": {
                    "2": {"type": "data_input", "label": "Second", "tool_state": "{}"},
                    "10": {"type": "data_input", "label": "Tenth", "tool_state": "{}"},
                    "1": {"type": "data_input", "label": "First", "tool_state": "{}"},
                    "x": {"type": "data_input", "label": "Not a step", "tool_state": "{}"},
                    "007": {"type": "data_input", "label": "Padded", "tool_state": "{}"},
                },
            },
            {**wf_show, "readme": "", "help": "", "annotation": "out of order"},
        ),
    )

    # -- get_tool_citations --------------------------------------------------
    citations = [
        {"type": "bibtex", "content": "@article{one, title={One}}"},
        {"type": "doi", "content": "10.1000/two"},
    ]
    add(
        "get_tool_citations",
        "some",
        "citations counted",
        {"tool_id": "cat1"},
        lambda: get_tool_citations_fn("cat1"),
        [
            route(
                "/api/tools/cat1",
                {"id": "cat1", "name": "Concatenate", "version": "1.0.0", "citations": citations},
            )
        ],
    )
    add(
        "get_tool_citations",
        "none",
        "a tool that cites nothing",
        {"tool_id": "cat1"},
        lambda: get_tool_citations_fn("cat1"),
        [
            route(
                "/api/tools/cat1",
                {"id": "cat1", "name": "Concatenate", "version": "1.0.0", "citations": []},
            )
        ],
    )

    single_record_cases(add)
    mutation_cases(add)
    biocontainer_cases(add)

    def add_failure(
        tool: str,
        name: str,
        note: str,
        inp: dict[str, Any],
        routes: list[dict[str, Any]],
    ) -> None:
        out.append(
            Case(tool=tool, name=name, note=note, input=inp, call=None, routes=routes, failure=True)
        )

    failure_cases(add_failure)

    return out


# ---------------------------------------------------------------------------
# One record in, one record out
#
# Sections of their own rather than more of the table above: these tools answer with
# a record or a small wrapper around one, so what a case pins is which fields a reply
# may leave out rather than where a page was cut.
# ---------------------------------------------------------------------------


def single_record_cases(add: AddCase) -> None:
    server_info_cases(add)
    user_cases(add)
    dataset_details_cases(add)
    tool_details_cases(add)
    tool_input_template_cases(add)
    workflow_details_cases(add)
    job_details_cases(add)


def server_info_cases(add: AddCase) -> None:
    """What the connected Galaxy is, and what it is too old to run.

    Three things this tool answers with are its own rather than Galaxy's: the address
    is the one the session stored, ``config`` is sixteen named fields lifted out of
    Galaxy's configuration with their own defaults (a brand that is missing reads
    "Galaxy", everything else missing reads null), and ``unsupported_tools`` is the
    declared tools this version cannot run, by name. So the cases are a current server,
    one a version too old for the page tools, one that will not say what it is, and a
    configuration bare enough that every default shows.
    """
    config = {
        "brand": "Example Galaxy",
        "logo_url": "/static/images/galaxyIcon_noText.png",
        "welcome_url": "/static/welcome.html",
        "support_url": "https://galaxy.example/support",
        "citation_url": "https://galaxy.example/citing",
        "terms_url": None,
        "allow_user_creation": True,
        "allow_user_deletion": False,
        "enable_quotas": True,
        "ftp_upload_site": None,
        "wiki_url": "https://galaxy.example/wiki",
        "screencasts_url": "https://galaxy.example/screencasts",
        "library_import_dir": None,
        "user_library_import_dir": None,
        "allow_library_path_paste": False,
        "enable_unique_workflow_defaults": False,
        # Fields this tool does not lift out, so a case proves they do not travel.
        "version_major": "26.1",
        "email_from": "noreply@galaxy.example",
        "server_startttime": 1767225845,
    }
    config_route = route("/api/configuration", config)
    add(
        "get_server_info",
        "current_galaxy",
        "a server new enough for everything, so nothing is ruled out",
        {},
        get_server_info_fn,
        [VERSION_ROUTE, config_route],
    )
    add(
        "get_server_info",
        "too_old_for_the_page_tools",
        "one minor version short, so the six declared tools are named",
        {},
        get_server_info_fn,
        [
            route("/api/version", {"version_major": "26.0", "version_minor": "0"}),
            config_route,
        ],
    )
    add(
        "get_server_info",
        "version_unreadable",
        "a version route that says nothing, where nothing is known and nothing is refused",
        {},
        get_server_info_fn,
        [route("/api/version", {}), config_route],
    )
    add(
        "get_server_info",
        "bare_configuration",
        "a configuration carrying almost nothing, so every default shows",
        {},
        get_server_info_fn,
        [VERSION_ROUTE, route("/api/configuration", {"logo_url": None})],
    )


def user_cases(add: AddCase) -> None:
    """The current user, as Galaxy describes them.

    Whatever the record carries is what the tool answers with -- there is no field
    list here -- so the cases are a full DetailedUserModel and one stripped to the
    three fields anything reading it needs.
    """
    add(
        "get_user",
        "detailed",
        "the whole user record, every field Galaxy sent",
        {},
        get_user_fn,
        [
            route(
                "/api/users/current",
                {
                    "id": "u0000001",
                    "username": "curator",
                    "email": "curator@galaxy.example",
                    "model_class": "User",
                    "deleted": False,
                    "purged": False,
                    "is_admin": False,
                    "total_disk_usage": 1048576,
                    "nice_total_disk_usage": "1.0 MB",
                    "quota_percent": 12.5,
                    "quota": "10.0 GB",
                    "quota_bytes": 10737418240,
                    "tags_used": ["rnaseq"],
                    "preferences": {"extra_user_preferences": "{}"},
                },
            )
        ],
    )
    add(
        "get_user",
        "anonymous",
        "the reply an unauthenticated session gets, which this tool answers with as it is",
        {},
        get_user_fn,
        [route("/api/users/current", {"total_disk_usage": 0, "quota_percent": None})],
    )
    add(
        "get_user",
        "bare",
        "a record carrying only the three fields anything reading it needs",
        {},
        get_user_fn,
        [
            route(
                "/api/users/current",
                {"id": "u0000001", "username": "curator", "email": "curator@galaxy.example"},
            )
        ],
    )


def dataset_details_cases(add: AddCase) -> None:
    """A dataset's metadata, wrapped, with Galaxy's text peek sliced beside it.

    The record does not come back on its own: it arrives under ``dataset`` next to the
    id that was asked for, so a caller reading ``data.state`` is reading the wrapper
    and not the dataset. The peek is only taken for a dataset in the ok state and only
    when it was asked for, and it has three shapes -- sliced text, a datatype Galaxy
    has no text for, and a peek that could not be taken -- so the cases are the
    branches rather than one happy path.
    """
    record = {
        "id": "d0000001",
        "name": "reads.txt",
        "state": "ok",
        "extension": "txt",
        "file_size": 1024,
        "history_id": "h0000",
        "deleted": False,
        "visible": True,
        "hid": 1,
        "misc_info": None,
        "model_class": "HistoryDatasetAssociation",
    }
    twelve = "\n".join(f"line{n}" for n in range(1, 13))
    text_route = route(
        "/api/datasets/d0000001/get_content_as_text",
        {"item_data": twelve, "truncated": False, "item_url": "/datasets/d0000001/display"},
    )
    add(
        "get_dataset_details",
        "peek_sliced",
        "twelve lines of peek cut to the ten asked for by default",
        {"dataset_id": "d0000001"},
        lambda: get_dataset_details_fn("d0000001"),
        [route("/api/datasets/d0000001", record), text_route],
    )
    add(
        "get_dataset_details",
        "peek_shorter_than_asked_for",
        "fewer lines than the slice, so nothing is cut and the count is what arrived",
        {"dataset_id": "d0000001", "preview_lines": 20},
        lambda: get_dataset_details_fn("d0000001", preview_lines=20),
        [route("/api/datasets/d0000001", record), text_route],
    )
    add(
        "get_dataset_details",
        "galaxy_cut_the_peek_itself",
        "Galaxy's own truncated flag, which is not the same fact as our line slice",
        {"dataset_id": "d0000001", "preview_lines": 20},
        lambda: get_dataset_details_fn("d0000001", preview_lines=20),
        [
            route("/api/datasets/d0000001", record),
            route(
                "/api/datasets/d0000001/get_content_as_text",
                {"item_data": "line1\nline2\n", "truncated": True},
            ),
        ],
    )
    add(
        "get_dataset_details",
        "no_text_preview",
        "a datatype Galaxy previews nothing for, so item_data is null",
        {"dataset_id": "d0000002"},
        lambda: get_dataset_details_fn("d0000002"),
        [
            route(
                "/api/datasets/d0000002",
                {**record, "id": "d0000002", "name": "alignment.bam", "extension": "bam"},
            ),
            route(
                "/api/datasets/d0000002/get_content_as_text",
                {"item_data": None, "truncated": False},
            ),
        ],
    )
    add(
        "get_dataset_details",
        "preview_not_asked_for",
        "include_preview false, so the text route is never asked and there is no preview key",
        {"dataset_id": "d0000001", "include_preview": False},
        lambda: get_dataset_details_fn("d0000001", include_preview=False),
        [route("/api/datasets/d0000001", record)],
    )
    # The name is read with dict.get and falls back to the id that was asked for.
    add(
        "get_dataset_details",
        "no_name_on_the_record",
        "a dataset record with no name key, which the sentence names by its id",
        {"dataset_id": "d0000004", "include_preview": False},
        lambda: get_dataset_details_fn("d0000004", include_preview=False),
        [
            route(
                "/api/datasets/d0000004",
                {k: v for k, v in record.items() if k != "name"} | {"id": "d0000004"},
            )
        ],
    )
    add(
        "get_dataset_details",
        "state_not_ok",
        "a dataset still running, where no peek is taken whatever was asked for",
        {"dataset_id": "d0000003"},
        lambda: get_dataset_details_fn("d0000003"),
        [
            route(
                "/api/datasets/d0000003",
                {**record, "id": "d0000003", "name": "half.txt", "state": "running"},
            )
        ],
    )


def tool_details_cases(add: AddCase) -> None:
    """Galaxy's tool record, passed through as it arrived.

    ``io_details`` changes the request and not the answer, and the two clients spell a
    boolean query parameter differently -- ``io_details=True`` from requests,
    ``io_details=true`` from this side -- so a route narrowed on it would answer one
    surface and 404 the other. The two cases therefore ask about two tools, each with
    its own reply, which is what the flag is for anyway.
    """
    add(
        "get_tool_details",
        "metadata_only",
        "the default: the tool record without its parameter list",
        {"tool_id": "cat1"},
        lambda: get_tool_details_fn("cat1"),
        [
            route(
                "/api/tools/cat1",
                {
                    "id": "cat1",
                    "name": "Concatenate datasets",
                    "version": "1.0.0",
                    "description": "tail-to-head",
                    "model_class": "Tool",
                    "panel_section_id": "text_manipulation",
                    "panel_section_name": "Text Manipulation",
                },
            )
        ],
    )
    add(
        "get_tool_details",
        "with_io_details",
        "io_details=True, so the parameter and output lists come too",
        {"tool_id": "fastqc", "io_details": True},
        lambda: get_tool_details_fn("fastqc", io_details=True),
        [
            route(
                "/api/tools/fastqc",
                {
                    "id": "fastqc",
                    "name": "FastQC",
                    "version": "0.74+galaxy1",
                    "description": "Read Quality reports",
                    "model_class": "Tool",
                    "inputs": [
                        {
                            "name": "input_file",
                            "label": "Short read data from your current history",
                            "type": "data",
                            "optional": False,
                            "multiple": False,
                            "extensions": ["fastqsanger", "bam"],
                            "value": None,
                        },
                        {
                            "name": "contaminants",
                            "label": "Contaminant list",
                            "type": "data",
                            "optional": True,
                            "multiple": False,
                            "extensions": ["tabular"],
                            "value": None,
                        },
                    ],
                    "outputs": [
                        {
                            "name": "html_file",
                            "format": "html",
                            "label": "${tool.name} on ${on_string}: Webpage",
                        },
                        {
                            "name": "text_file",
                            "format": "txt",
                            "label": "${tool.name} on ${on_string}: RawData",
                        },
                    ],
                },
            )
        ],
    )


def tool_input_template_cases(add: AddCase) -> None:
    """The skeleton and the compact parameter list, built from one io_details schema.

    Every nesting the builder walks is in the first case -- a data param, a repeat, a
    conditional inside it, a section, and a select whose options are capped inline --
    because each of them decides a different flattened key.
    """
    add(
        "get_tool_input_template",
        "every_nesting",
        "a data param, a repeat, a conditional, a section and a select in one schema",
        {"tool_id": "build_list"},
        lambda: get_tool_input_template_fn("build_list"),
        [
            route(
                "/api/tools/build_list",
                {
                    "id": "build_list",
                    "name": "Build list",
                    "version": "1.2.0",
                    "inputs": [
                        {
                            "name": "input_file",
                            "type": "data",
                            "optional": False,
                            "multiple": False,
                            "extensions": ["tabular"],
                        },
                        {
                            "name": "datasets",
                            "type": "repeat",
                            "title": "Dataset",
                            "inputs": [
                                {"name": "input", "type": "data", "optional": False},
                                {
                                    "name": "id_cond",
                                    "type": "conditional",
                                    "test_param": {
                                        "name": "id_select",
                                        "type": "select",
                                        "value": "idx",
                                        "options": [
                                            ["use index", "idx", True],
                                            ["manual", "manual", False],
                                        ],
                                    },
                                    "cases": [
                                        {"value": "idx", "inputs": []},
                                        {
                                            "value": "manual",
                                            "inputs": [
                                                {"name": "identifier", "type": "text", "value": ""}
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "name": "advanced",
                            "type": "section",
                            "title": "Advanced",
                            "inputs": [
                                {
                                    "name": "sort_by",
                                    "type": "select",
                                    "value": "name",
                                    "options": [
                                        ["Name", "name", True],
                                        ["Identifier", "identifier", False],
                                    ],
                                },
                                {"name": "threshold", "type": "integer", "value": 5},
                            ],
                        },
                    ],
                },
            )
        ],
    )
    add(
        "get_tool_input_template",
        "no_inputs",
        "a tool whose definition arrived without a parameter list at all",
        {"tool_id": "upload1"},
        lambda: get_tool_input_template_fn("upload1"),
        [route("/api/tools/upload1", {"id": "upload1", "name": "Upload File", "version": "1.1.7"})],
    )


def workflow_details_cases(add: AddCase) -> None:
    """A stored workflow, passed through.

    ``version`` changes the request and not the answer, so -- as with io_details above
    -- the two cases ask about two workflows rather than narrowing one path on a query
    the two clients spell differently.
    """
    add(
        "get_workflow_details",
        "latest",
        "no version asked for, so Galaxy answers with the newest",
        {"workflow_id": "wf000001"},
        lambda: get_workflow_details_fn("wf000001"),
        [
            route(
                "/api/workflows/wf000001",
                {
                    "id": "wf000001",
                    "name": "Reads QC",
                    "version": 3,
                    "owner": "curator",
                    "annotation": "quality control over sequencing reads",
                    "published": False,
                    "deleted": False,
                    "inputs": {"0": {"label": "Input FASTQ", "value": "", "uuid": None}},
                    "steps": {
                        "0": {
                            "id": 0,
                            "type": "data_input",
                            "annotation": None,
                            "input_steps": {},
                        },
                        "1": {
                            "id": 1,
                            "type": "tool",
                            "tool_id": "fastqc",
                            "annotation": None,
                            "input_steps": {
                                "input_file": {"source_step": 0, "step_output": "output"}
                            },
                        },
                    },
                },
            )
        ],
    )
    # The name is read with dict.get and falls back to the id that was asked for.
    add(
        "get_workflow_details",
        "no_name_on_the_record",
        "a workflow record with no name key, which the sentence names by its id",
        {"workflow_id": "wf000003"},
        lambda: get_workflow_details_fn("wf000003"),
        [
            route(
                "/api/workflows/wf000003",
                {
                    "id": "wf000003",
                    "version": 1,
                    "owner": "curator",
                    "published": False,
                    "deleted": False,
                    "inputs": {},
                    "steps": {},
                },
            )
        ],
    )
    add(
        "get_workflow_details",
        "pinned_version",
        "an older version asked for by number",
        {"workflow_id": "wf000002", "version": 1},
        lambda: get_workflow_details_fn("wf000002", version=1),
        [
            route(
                "/api/workflows/wf000002",
                {
                    "id": "wf000002",
                    "name": "Reads QC",
                    "version": 1,
                    "owner": "curator",
                    "annotation": None,
                    "published": False,
                    "deleted": False,
                    "inputs": {},
                    "steps": {"0": {"id": 0, "type": "data_input", "input_steps": {}}},
                },
            )
        ],
    )


def job_details_cases(add: AddCase) -> None:
    """The job behind a dataset, found two ways.

    With a history the provenance record is asked first and its ``job_id`` is used;
    without one, or when provenance answers without a job id, the dataset's own
    ``creating_job`` is read instead. Both paths end at the same GET /api/jobs/{id},
    which this server reaches with requests rather than through bioblend -- joined onto
    the session's stored address, which is why that address has to be the one connect
    would have stored.
    """
    job = {
        "id": "j0000001",
        "state": "ok",
        "tool_id": "fastqc",
        "exit_code": 0,
        "create_time": "2026-01-02T03:04:05.000000",
        "update_time": "2026-01-02T03:14:05.000000",
        "params": {"input_file": '{"values": [{"id": 12, "src": "hda"}]}'},
        "inputs": {"input_file": {"id": "d0000001", "src": "hda", "uuid": None}},
        "outputs": {"html_file": {"id": "d0000002", "src": "hda", "uuid": None}},
    }
    job_route = route("/api/jobs/j0000001", job)
    dataset_route = route(
        "/api/datasets/d0000002",
        {
            "id": "d0000002",
            "name": "FastQC on data 1: Webpage",
            "state": "ok",
            "extension": "html",
            "creating_job": "j0000001",
            "history_id": "h0000",
        },
    )
    add(
        "get_job_details",
        "from_provenance",
        "a history was named, so the provenance record answers with the job id",
        {"dataset_id": "d0000002", "history_id": "h0000"},
        lambda: get_job_details_fn("d0000002", history_id="h0000"),
        [
            route(
                "/api/histories/h0000/contents/d0000002/provenance",
                {
                    "id": "d0000002",
                    "job_id": "j0000001",
                    "tool_id": "fastqc",
                    "uuid": "22222222-2222-2222-2222-222222222222",
                    "parameters": {},
                },
            ),
            job_route,
        ],
    )
    add(
        "get_job_details",
        "from_creating_job",
        "no history named, so the dataset's own creating_job is read",
        {"dataset_id": "d0000002"},
        lambda: get_job_details_fn("d0000002"),
        [dataset_route, job_route],
    )
    add(
        "get_job_details",
        "provenance_without_a_job_id",
        "provenance answered, but named no job, so the dataset is read anyway",
        {"dataset_id": "d0000002", "history_id": "h0000"},
        lambda: get_job_details_fn("d0000002", history_id="h0000"),
        [
            route(
                "/api/histories/h0000/contents/d0000002/provenance",
                {"id": "d0000002", "tool_id": "fastqc", "parameters": {}},
            ),
            dataset_route,
            job_route,
        ],
    )


# ---------------------------------------------------------------------------
# Mutations
#
# Ordinary cases: the POST, PUT or DELETE reply is canned like any other, and what
# a case pins is the record the tool hands back afterwards. Success paths only --
# the failure envelope is its own piece of work -- and nothing here reaches a
# Galaxy, so "creating" a history costs one canned reply.
# ---------------------------------------------------------------------------


def mutation_cases(add: AddCase) -> None:
    history_mutation_cases(add)
    run_tool_cases(add)
    invoke_workflow_cases(add)
    page_cases(add)
    user_tool_mutation_cases(add)
    invocation_mutation_cases(add)
    revision_cases(add)


def history_mutation_cases(add: AddCase) -> None:
    created = {
        "id": "h0000new",
        "name": "RNA-seq Sample A",
        "state": "new",
        "deleted": False,
        "published": False,
        "annotation": None,
        "tags": [],
        "create_time": "2026-01-02T03:04:05.000000",
        "update_time": "2026-01-02T03:04:05.000000",
        "model_class": "History",
    }
    add(
        "create_history",
        "named",
        "a new history, which is one POST and the record Galaxy answers with",
        {"history_name": "RNA-seq Sample A"},
        lambda: create_history_fn("RNA-seq Sample A"),
        [route("/api/histories", created, method="POST")],
    )
    add(
        "update_history",
        "name_only",
        "one field changed, which the message has to name and the record has to show",
        {"history_id": "h0000new", "name": "RNA-seq Sample A (final)"},
        lambda: update_history_fn("h0000new", name="RNA-seq Sample A (final)"),
        [
            route(
                "/api/histories/h0000new",
                {
                    **created,
                    "name": "RNA-seq Sample A (final)",
                    "update_time": "2026-01-03T00:00:00",
                },
                method="PUT",
            )
        ],
    )
    add(
        "update_history",
        "several_fields",
        "an annotation, tags and published at once",
        {
            "history_id": "h0000new",
            "annotation": "QC'd and ready",
            "tags": ["rnaseq", "final"],
            "published": True,
        },
        lambda: update_history_fn(
            "h0000new", annotation="QC'd and ready", tags=["rnaseq", "final"], published=True
        ),
        [
            route(
                "/api/histories/h0000new",
                {
                    **created,
                    "annotation": "QC'd and ready",
                    "tags": ["rnaseq", "final"],
                    "published": True,
                    "update_time": "2026-01-03T00:00:00",
                },
                method="PUT",
            )
        ],
    )


# A page, as Galaxy answers with one. `content` is the same document with its embeds
# expanded for export and is the large half of the reply; `content_editor` is the
# editable markdown a caller edits and sends back.
PAGE_MARKDOWN = (
    "# Reads QC\n\n```galaxy\nhistory_dataset_display(history_dataset_id=d0000002)\n```\n"
)
PAGE_RENDERED = "# Reads QC\n\n<div class='embedded'>expanded</div>\n"


def page_record(**over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "pg000001",
        "title": "Reads QC report",
        "slug": "reads-qc-report",
        "history_id": "h0000",
        "latest_revision_id": "rev00002",
        "revision_ids": ["rev00001", "rev00002"],
        "source_invocation_id": None,
        "model_class": "Page",
        "username": "curator",
        "email_hash": "d41d8cd98f00b204e9800998ecf8427e",
        "author_deleted": False,
        "deleted": False,
        "importable": False,
        "published": False,
        "tags": [],
        "create_time": "2026-01-02T03:04:05.000000",
        "update_time": "2026-01-02T03:14:05.000000",
        "content_editor": PAGE_MARKDOWN,
        "content": PAGE_RENDERED,
        "content_format": "markdown",
        "annotation": None,
        "edit_source": "agent",
        "generate_time": None,
        "generate_version": None,
    }
    record.update(over)
    return record


def page_cases(add: AddCase) -> None:
    """A page read, created and updated, and what happens to the expanded render.

    The rendered ``content`` is dropped unless it was asked for, and asking is only
    possible on the read -- create and update never return it. What the cases pin is
    that the drop is unconditional: an HTML page arrives with an empty
    ``content_editor``, because Galaxy fills that field on the markdown path only, and
    its body is in ``content`` -- and ``content`` goes anyway. get_page is not refused
    against an older Galaxy, so its cases are answered no version; create_page and
    update_page are, so theirs are.
    """
    add(
        "get_page",
        "markdown",
        "the default read: the editable markdown stays, the expanded render goes",
        {"page_id": "pg000001"},
        lambda: get_page_fn("pg000001"),
        [route("/api/pages/pg000001", page_record())],
    )
    add(
        "get_page",
        "with_rendered",
        "include_rendered, so the expanded render comes too",
        {"page_id": "pg000001", "include_rendered": True},
        lambda: get_page_fn("pg000001", include_rendered=True),
        [route("/api/pages/pg000001", page_record())],
    )
    add(
        "get_page",
        "html_page_without_content_editor",
        "an HTML page, whose only body is the expanded render -- which still goes",
        {"page_id": "pg000002"},
        lambda: get_page_fn("pg000002"),
        [
            route(
                "/api/pages/pg000002",
                page_record(
                    id="pg000002",
                    title="Legacy report",
                    slug="legacy-report",
                    history_id=None,
                    content_format="html",
                    content_editor="",
                    content="<h1>Legacy report</h1>",
                    edit_source="user",
                ),
            )
        ],
    )
    add(
        "get_page",
        "no_rendered_content_at_all",
        "a reply carrying no content key, where there is nothing to drop",
        {"page_id": "pg000003"},
        lambda: get_page_fn("pg000003"),
        [
            route(
                "/api/pages/pg000003",
                {k: v for k, v in page_record(id="pg000003").items() if k != "content"},
            )
        ],
    )
    add(
        "create_page",
        "notebook",
        "a page attached to a history, which is the notebook shape",
        {"history_id": "h0000", "title": "Reads QC report", "content": "# Reads QC\n"},
        lambda: create_page_fn(history_id="h0000", title="Reads QC report", content="# Reads QC\n"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages",
                page_record(
                    content_editor="# Reads QC\n",
                    content="# Reads QC\n",
                    latest_revision_id="rev00001",
                    revision_ids=["rev00001"],
                    edit_source="user",
                ),
                method="POST",
            ),
        ],
    )
    # The id in the sentence is read off the REPLY with dict.get, defaulted to the
    # empty string, so a create Galaxy answers without one still makes a sentence.
    add(
        "create_page",
        "reply_without_an_id",
        "a create whose reply carries no id, which the sentence leaves empty",
        {"title": "Untitled", "slug": "untitled"},
        lambda: create_page_fn(title="Untitled", slug="untitled"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages",
                {
                    k: v
                    for k, v in page_record(
                        title="Untitled",
                        slug="untitled",
                        history_id=None,
                        content_editor="",
                        content="",
                        edit_source="user",
                    ).items()
                    if k != "id"
                },
                method="POST",
            ),
        ],
    )
    add(
        "create_page",
        "standalone_report",
        "no history, so a report -- which needs both a title and a slug",
        {"title": "Cohort summary", "slug": "cohort-summary", "annotation": "for the paper"},
        lambda: create_page_fn(
            title="Cohort summary", slug="cohort-summary", annotation="for the paper"
        ),
        [
            VERSION_ROUTE,
            route(
                "/api/pages",
                page_record(
                    id="pg000004",
                    title="Cohort summary",
                    slug="cohort-summary",
                    history_id=None,
                    annotation="for the paper",
                    content_editor="",
                    content="",
                    latest_revision_id="rev00010",
                    revision_ids=["rev00010"],
                    edit_source="user",
                ),
                method="POST",
            ),
        ],
    )
    add(
        "update_page",
        "content_changed",
        "new markdown, which writes a revision the agent is recorded against",
        {"page_id": "pg000001", "content": "# Reads QC\n\nrewritten\n"},
        lambda: update_page_fn("pg000001", content="# Reads QC\n\nrewritten\n"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages/pg000001",
                page_record(
                    content_editor="# Reads QC\n\nrewritten\n",
                    latest_revision_id="rev00003",
                    revision_ids=["rev00001", "rev00002", "rev00003"],
                ),
                method="PUT",
            ),
        ],
    )
    add(
        "update_page",
        "title_only_on_an_html_page",
        "a title change on a page whose body is the expanded render, which still goes",
        {"page_id": "pg000002", "title": "Legacy report (renamed)"},
        lambda: update_page_fn("pg000002", title="Legacy report (renamed)"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages/pg000002",
                page_record(
                    id="pg000002",
                    title="Legacy report (renamed)",
                    slug="legacy-report",
                    history_id=None,
                    content_format="html",
                    content_editor="",
                    content="<h1>Legacy report</h1>",
                ),
                method="PUT",
            ),
        ],
    )


def user_tool_mutation_cases(add: AddCase) -> None:
    """A user-defined tool created, run and deactivated.

    The representation is the tool's own definition, and run_user_tool is two
    requests: the uuid is read for the tool id and version, then the run is POSTed to
    /api/tools like any other. The definition it reads back is also the schema this
    server checks the supplied inputs against, so a dataset handed to a data parameter
    is checked here without a second lookup.
    """
    representation = {
        "class": "GalaxyUserTool",
        "id": "row_filter",
        "version": "0.1.0",
        "name": "Row filter",
        "description": "Keep rows above a threshold",
        "container": "quay.io/biocontainers/python:3.12",
        "shell_command": "python3 -c 'pass'",
        "inputs": [
            {"name": "table", "type": "data", "format": "tabular"},
            {"name": "threshold", "type": "integer"},
        ],
        "outputs": [
            {"name": "kept", "type": "data", "format": "tabular", "from_work_dir": "out.tsv"}
        ],
    }
    created_tool = {
        "id": "ut000001",
        "uuid": "61d15277-a911-45ef-aa66-5385146578cc",
        "tool_id": "row_filter",
        "active": True,
        "create_time": "2026-01-02T03:04:05.000000",
        "representation": representation,
    }
    add(
        "create_user_tool",
        "created",
        "a representation POSTed, and the record Galaxy answers with",
        {"representation": representation},
        lambda: create_user_tool_fn(representation),
        [route("/api/unprivileged_tools", created_tool, method="POST")],
    )
    # The name is read off the REPRESENTATION with dict.get, falling back to its id.
    # Both fields are required, so the fallback is unreachable -- but a field stated as
    # null is present, and an f-string renders that None on both sides.
    add(
        "create_user_tool",
        "name_stated_as_null",
        "a representation whose name is null, which is present and so is not defaulted",
        {"representation": {**representation, "name": None}},
        lambda: create_user_tool_fn({**representation, "name": None}),
        [
            route(
                "/api/unprivileged_tools",
                {**created_tool, "representation": {**representation, "name": None}},
                method="POST",
            )
        ],
    )
    add(
        "delete_user_tool",
        "deactivated",
        "a soft delete, where the answer is built here rather than read from Galaxy",
        {"uuid": "61d15277-a911-45ef-aa66-5385146578cc"},
        lambda: delete_user_tool_fn("61d15277-a911-45ef-aa66-5385146578cc"),
        [
            route(
                "/api/unprivileged_tools/61d15277-a911-45ef-aa66-5385146578cc",
                {"id": "ut000001", "active": False},
                method="DELETE",
            )
        ],
    )
    submission = {
        "outputs": [
            {
                "id": "d0000010",
                "hid": 4,
                "name": "Row filter on data 1",
                "state": "new",
                "history_id": "h0000",
                "extension": "tabular",
            }
        ],
        "output_collections": [],
        "jobs": [
            {
                "model_class": "Job",
                "id": "j0000010",
                "state": "new",
                "tool_id": "row_filter",
                "tool_version": "0.1.0",
                "create_time": "2026-01-02T03:04:05.000000",
            }
        ],
        "implicit_collections": [],
        "produces_entry_points": False,
    }
    lookup = route("/api/unprivileged_tools/61d15277-a911-45ef-aa66-5385146578cc", created_tool)
    add(
        "run_user_tool",
        "scalar_input",
        "a run whose inputs are all scalars, so the checker has nothing to resolve",
        {
            "history_id": "h0000",
            "tool_uuid": "61d15277-a911-45ef-aa66-5385146578cc",
            "inputs": {"threshold": 5},
        },
        lambda: run_user_tool_fn("h0000", "61d15277-a911-45ef-aa66-5385146578cc", {"threshold": 5}),
        [lookup, route("/api/tools", submission, method="POST")],
    )
    add(
        "run_user_tool",
        "dataset_input",
        "a dataset handed to a data parameter, checked against the tool's own definition",
        {
            "history_id": "h0000",
            "tool_uuid": "61d15277-a911-45ef-aa66-5385146578cc",
            "inputs": {"table": {"src": "hda", "id": "d0000001"}, "threshold": 5},
        },
        lambda: run_user_tool_fn(
            "h0000",
            "61d15277-a911-45ef-aa66-5385146578cc",
            {"table": {"src": "hda", "id": "d0000001"}, "threshold": 5},
        ),
        [lookup, route("/api/tools", submission, method="POST")],
    )


def run_tool_cases(add: AddCase) -> None:
    """A tool submitted, and the submission record Galaxy answers with.

    This tool queues and returns: the jobs come back in the "new" state and nothing
    here waits for them. Two things happen before the POST and neither shows in the
    answer. Supplied inputs are checked against the tool's schema, but only when one
    of them looks like a dataset reference -- a run made entirely of scalars skips the
    lookup -- so the second case registers the schema and the first does not need it.
    And stored credentials for the tool are looked up best-effort; this table answers
    that lookup with nothing, so the run goes out without a credentials context, which
    is what an ordinary tool run does.
    """
    submission = {
        "outputs": [
            {
                "id": "d0000020",
                "hid": 5,
                "name": "FastQC on data 1: Webpage",
                "state": "new",
                "history_id": "h0000",
                "output_name": "html_file",
                "file_ext": "html",
                "model_class": "HistoryDatasetAssociation",
            }
        ],
        "output_collections": [],
        "jobs": [
            {
                "model_class": "Job",
                "id": "j0000020",
                "state": "new",
                "tool_id": "fastqc",
                "tool_version": "0.74+galaxy1",
                "create_time": "2026-01-02T03:04:05.000000",
                "update_time": "2026-01-02T03:04:05.000000",
                "exit_code": None,
            }
        ],
        "implicit_collections": [],
        "produces_entry_points": False,
    }
    tools_post = route("/api/tools", submission, method="POST")
    add(
        "run_tool",
        "scalar_inputs",
        "nothing that looks like a dataset, so the schema is never fetched",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"contaminants": "", "limits": ""},
        },
        lambda: run_tool_fn("h0000", "fastqc", {"contaminants": "", "limits": ""}),
        [tools_post],
    )
    fastqc_schema = {
        "id": "fastqc",
        "name": "FastQC",
        "version": "0.74+galaxy1",
        "inputs": [
            {
                "name": "input_file",
                "type": "data",
                "optional": False,
                "multiple": False,
                "extensions": ["fastqsanger"],
            }
        ],
    }
    add(
        "run_tool",
        "dataset_input_checked_first",
        "a dataset reference, so the tool's schema is read and the inputs are checked",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"input_file": {"src": "hda", "id": "d0000001"}},
        },
        lambda: run_tool_fn("h0000", "fastqc", {"input_file": {"src": "hda", "id": "d0000001"}}),
        [route("/api/tools/fastqc", fastqc_schema), tools_post],
    )
    add(
        "run_tool",
        "schema_unreadable",
        "the schema read refused, so the run says the inputs went unchecked",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"input_file": {"src": "hda", "id": "d0000001"}},
        },
        lambda: run_tool_fn("h0000", "fastqc", {"input_file": {"src": "hda", "id": "d0000001"}}),
        [fail("/api/tools/fastqc", 404, MISSING), tools_post],
    )
    add(
        "run_tool",
        "schema_without_a_parameter_list",
        "a schema with no inputs key is not checkable, which the run says rather than implies",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"input_file": {"src": "hda", "id": "d0000001"}},
        },
        lambda: run_tool_fn("h0000", "fastqc", {"input_file": {"src": "hda", "id": "d0000001"}}),
        [
            route(
                "/api/tools/fastqc", {"id": "fastqc", "name": "FastQC", "version": "0.74+galaxy1"}
            ),
            tools_post,
        ],
    )
    add(
        "run_tool",
        "with_stored_credentials",
        "credentials configured for this tool, which the run carries and says it carried",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"contaminants": ""},
        },
        lambda: run_tool_fn("h0000", "fastqc", {"contaminants": ""}),
        [
            route("/api/users/current", {"id": "u0000001", "username": "curator"}),
            route(
                "/api/users/u0000001/credentials",
                [
                    {
                        "id": "cred0001",
                        "name": "api_token",
                        "version": "1",
                        "current_group_id": "g0000001",
                        "groups": [
                            {"id": "g0000001", "name": "default"},
                            {"id": "g0000002", "name": "staging"},
                        ],
                    }
                ],
            ),
            tools_post,
        ],
    )
    add(
        "run_tool",
        "section_inputs_legacy_keys",
        "a parameter inside a section, spelled the flat legacy way the tool sends it",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"advanced|threshold": 9, "contaminants": ""},
        },
        lambda: run_tool_fn("h0000", "fastqc", {"advanced|threshold": 9, "contaminants": ""}),
        [tools_post],
    )
    add(
        "run_tool",
        "pinned_version",
        "a version asked for by name, which this server posts itself rather than bioblend",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"contaminants": ""},
            "tool_version": "0.74+galaxy1",
        },
        lambda: run_tool_fn("h0000", "fastqc", {"contaminants": ""}, tool_version="0.74+galaxy1"),
        [tools_post],
    )
    # Asking for a version does not make it so: Galaxy's toolbox falls back to an
    # installed one, and only the jobs say which ran. Two more branches of that
    # sentence -- a version that is not the one requested, and a reply that names none.
    add(
        "run_tool",
        "version_galaxy_did_not_honour",
        "a version asked for that the jobs came back naming differently",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"contaminants": ""},
            "tool_version": "0.72",
        },
        lambda: run_tool_fn("h0000", "fastqc", {"contaminants": ""}, tool_version="0.72"),
        [tools_post],
    )
    add(
        "run_tool",
        "version_no_job_reports",
        "a version asked for and a reply whose job names none, so nothing is claimed",
        {
            "history_id": "h0000",
            "tool_id": "fastqc",
            "inputs": {"contaminants": ""},
            "tool_version": "0.74+galaxy1",
        },
        lambda: run_tool_fn("h0000", "fastqc", {"contaminants": ""}, tool_version="0.74+galaxy1"),
        [
            route(
                "/api/tools",
                {
                    **submission,
                    "jobs": [
                        {k: v for k, v in submission["jobs"][0].items() if k != "tool_version"}
                    ],
                },
                method="POST",
            )
        ],
    )


def invoke_workflow_cases(add: AddCase) -> None:
    """A workflow submitted, and what comes back.

    The answer is the invocation record Galaxy sent and nothing else: this is a
    submission, not a wait, and the invocation is "new" when it arrives. Supplying
    inputs turns on a preflight, which reads the run model, the datatype hierarchy and
    each supplied dataset before anything is posted -- all mocked here, so the check
    really runs and really passes rather than being skipped by a failed lookup. A
    batch invocation is the third case: Galaxy answers with a list of invocations and
    that list is the answer.
    """
    invocation = {
        "id": "inv00002",
        "state": "new",
        "model_class": "WorkflowInvocation",
        "workflow_id": "wf000001",
        "history_id": "h0000",
        "uuid": "33333333-3333-3333-3333-333333333333",
        "create_time": "2026-01-02T03:04:05.000000",
        "update_time": "2026-01-02T03:04:05.000000",
        "inputs": {},
        "steps": [
            {
                "id": "st000001",
                "order_index": 0,
                "state": "new",
                "job_id": None,
                "workflow_step_label": "Input FASTQ",
                "model_class": "WorkflowInvocationStep",
            }
        ],
        "outputs": {},
        "output_collections": {},
    }
    invocations_route = route("/api/workflows/wf000001/invocations", invocation, method="POST")
    add(
        "invoke_workflow",
        "no_inputs",
        "nothing supplied, so nothing to check -- one POST and the record back",
        {"workflow_id": "wf000001", "history_id": "h0000"},
        lambda: invoke_workflow_fn("wf000001", history_id="h0000"),
        [invocations_route],
    )
    run_model_for_invoke = {
        "id": "wf000001",
        "name": "Reads QC",
        "has_upgrade_messages": False,
        "step_version_changes": [],
        "steps": {
            "0": {
                "step_type": "data_input",
                "step_index": 0,
                "step_label": "Input FASTQ",
                "uuid": "11111111-1111-1111-1111-111111111111",
                "inputs": [
                    {
                        "extensions": ["fastqsanger"],
                        "acceptable_extensions": ["fastqsanger", "fastqsanger.gz"],
                        "optional": False,
                    }
                ],
            }
        },
    }
    add(
        "invoke_workflow",
        "inputs_checked_before_submitting",
        "a dataset supplied for the one slot, checked against the run model first",
        {
            "workflow_id": "wf000001",
            "history_id": "h0000",
            "inputs": {"0": {"id": "d0000001", "src": "hda"}},
        },
        lambda: invoke_workflow_fn(
            "wf000001", inputs={"0": {"id": "d0000001", "src": "hda"}}, history_id="h0000"
        ),
        [
            route(
                "/api/workflows/wf000001/download",
                run_model_for_invoke,
                query={"style": "run"},
            ),
            route(
                "/api/datatypes/types_and_mapping",
                {
                    "datatypes_mapping": {
                        "ext_to_class_name": {
                            "fastqsanger": "galaxy.datatypes.sequence.FastqSanger"
                        },
                        "class_to_classes": {
                            "galaxy.datatypes.sequence.FastqSanger": {
                                "galaxy.datatypes.sequence.FastqSanger": True
                            }
                        },
                    }
                },
            ),
            route(
                "/api/datasets/d0000001",
                {"id": "d0000001", "name": "reads.fastqsanger", "extension": "fastqsanger"},
            ),
            invocations_route,
        ],
    )
    add(
        "invoke_workflow",
        "batch_answers_with_a_list",
        "Galaxy expanded the run, so the answer is every invocation it made",
        {"workflow_id": "wf000001", "history_id": "h0000"},
        lambda: invoke_workflow_fn("wf000001", history_id="h0000"),
        [
            route(
                "/api/workflows/wf000001/invocations",
                [invocation, {**invocation, "id": "inv00003"}],
                method="POST",
            )
        ],
    )


def invocation_mutation_cases(add: AddCase) -> None:
    add(
        "cancel_workflow_invocation",
        "cancelled",
        "a DELETE, and the invocation Galaxy answers with wrapped beside a flag",
        {"invocation_id": "inv00001"},
        lambda: cancel_workflow_invocation_fn("inv00001"),
        [
            route(
                "/api/invocations/inv00001",
                {
                    "id": "inv00001",
                    "state": "cancelling",
                    "workflow_id": "wf000001",
                    "history_id": "h0000",
                    "create_time": "2026-01-02T03:04:05.000000",
                    "update_time": "2026-01-02T03:05:05.000000",
                    "model_class": "WorkflowInvocation",
                },
                method="DELETE",
            )
        ],
    )
    # bioblend posts an imported definition to /api/workflows/upload where the other
    # side posts it to /api/workflows. Both are the same create to Galaxy, and this
    # table answers questions rather than replaying a log, so the record is registered
    # under both spellings.
    imported = {
        "id": "wf000009",
        "name": "Workflow 0",
        "owner": "curator",
        "number_of_steps": 2,
        "published": False,
        "deleted": False,
        "model_class": "StoredWorkflow",
    }
    add(
        "import_workflow_from_iwc",
        "imported",
        "a definition found in the IWC manifest and POSTed to Galaxy",
        {"trs_id": "#workflow/github.com/iwc-workflows/wf0/main"},
        lambda: import_workflow_from_iwc_fn("#workflow/github.com/iwc-workflows/wf0/main"),
        [
            route(IWC_MANIFEST_URL, iwc_manifest(2)),
            route("/api/workflows/upload", imported, method="POST"),
            route("/api/workflows", imported, method="POST"),
        ],
    )


def revision_cases(add: AddCase) -> None:
    """A page revision read and restored.

    Both tools give a revision one field to edit whatever the server sent and say which
    field that was, so the three cases are the three answers: the revision's own
    content_editor, the expanded content standing in for it, and a revision carrying
    neither. Both are refused against a Galaxy older than 26.1, so every case here is
    answered a version first.
    """
    revision = {
        "id": "rev00002",
        "page_id": "pg000001",
        "edit_source": "agent",
        "title": "Reads QC report",
        "content_format": "markdown",
        "create_time": "2026-01-02T03:04:05.000000",
        "update_time": "2026-01-02T03:04:05.000000",
    }
    add(
        "get_page_revision",
        "content_editor_from_the_server",
        "the revision carries its own editable markdown",
        {"page_id": "pg000001", "revision_id": "rev00002"},
        lambda: get_page_revision_fn("pg000001", "rev00002"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages/pg000001/revisions/rev00002",
                {
                    **revision,
                    "content_editor": (
                        "# Reads QC\n\n```galaxy\n"
                        "history_dataset_display(history_dataset_id=d0000002)\n```\n"
                    ),
                    "content": "# Reads QC\n\n<div class='embedded'>expanded</div>\n",
                },
            ),
        ],
    )
    add(
        "get_page_revision",
        "content_editor_from_content",
        "an older server sent no content_editor, so the expanded content stands in",
        {"page_id": "pg000001", "revision_id": "rev00001"},
        lambda: get_page_revision_fn("pg000001", "rev00001"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages/pg000001/revisions/rev00001",
                {
                    **revision,
                    "id": "rev00001",
                    "edit_source": "user",
                    "content": "# Reads QC\n\n<div class='embedded'>expanded</div>\n",
                },
            ),
        ],
    )
    add(
        "get_page_revision",
        "no_content_at_all",
        "a revision carrying neither field, where content_editor is null",
        {"page_id": "pg000001", "revision_id": "rev00000"},
        lambda: get_page_revision_fn("pg000001", "rev00000"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages/pg000001/revisions/rev00000",
                {**revision, "id": "rev00000", "edit_source": "user", "content_editor": ""},
            ),
        ],
    )
    add(
        "revert_page_revision",
        "restored",
        "the new revision a restore writes, read back the same way",
        {"page_id": "pg000001", "revision_id": "rev00001"},
        lambda: revert_page_revision_fn("pg000001", "rev00001"),
        [
            VERSION_ROUTE,
            route(
                "/api/pages/pg000001/revisions/rev00001/revert",
                {
                    **revision,
                    "id": "rev00003",
                    "edit_source": "restore",
                    "content_editor": "# Reads QC\n\nthe old body\n",
                    "content": "# Reads QC\n\nthe old body\n",
                },
                method="POST",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# The one tool that does not talk to a Galaxy
# ---------------------------------------------------------------------------


def quay_repository(name: str) -> str:
    """Where a biocontainer's tag list lives, on both sides."""
    return f"https://quay.io/api/v1/repository/biocontainers/{name}"


def biocontainer_cases(add: AddCase) -> None:
    """A conda package list resolved to a built quay.io image.

    The only tool here that reaches a host other than the Galaxy: quay.io answers with a
    repository's tag list, and the recommender picks a tag from it and then asks again to
    verify the one it picked -- two requests to one route, which one canned reply answers
    on both sides. A single package is looked up under its own name; several are a
    mulled-v2 repository whose name hashes the sorted package names and whose tag hashes
    their versions. The inputs are plain ASCII with an ordinary JSON reply on purpose: the
    accepted divergences of the port are all about codecs, charsets and timing, and these
    cases are about the algorithm.
    """
    samtools_tags = {
        "tags": {
            "1.17--hd87286a_2": {"name": "1.17--hd87286a_2"},
            "1.17--h00b6d6a_0": {"name": "1.17--h00b6d6a_0"},
            "1.16.1--h6899075_1": {"name": "1.16.1--h6899075_1"},
            "latest": {"name": "latest"},
        }
    }
    samtools = route(quay_repository("samtools"), samtools_tags)
    add(
        "recommend_biocontainer",
        "single_pinned",
        "one package at a version that is built, which is an exact match",
        {"packages": ["samtools=1.17"]},
        lambda: recommend_biocontainer_fn(["samtools=1.17"]),
        [samtools],
    )
    add(
        "recommend_biocontainer",
        "single_unpinned",
        "one package with no version, so the newest built tag and a name-only match",
        {"packages": ["samtools"]},
        lambda: recommend_biocontainer_fn(["samtools"]),
        [samtools],
    )
    add(
        "recommend_biocontainer",
        "version_never_built",
        "a version nobody built, which falls back to the newest and says so in a note",
        {"packages": ["samtools=9.9"]},
        lambda: recommend_biocontainer_fn(["samtools=9.9"]),
        [samtools],
    )
    add(
        "recommend_biocontainer",
        "no_such_repository",
        "a package with no biocontainer at all: no image, and nothing to verify",
        {"packages": ["nosuchpackage"]},
        lambda: recommend_biocontainer_fn(["nosuchpackage"]),
        [route(quay_repository("nosuchpackage"), {"detail": "Not Found"}, status=404)],
    )
    # Two packages are a mulled-v2 image: the repository name is a hash of the sorted
    # package names and the tag is a hash of their versions, both written out here so a
    # reader can see that the two sides hash the same thing.
    mulled_v2 = "mulled-v2-fe8faa35dbf6dc65a0f7f5d4ea12e31a79f73e40"
    mulled_tag = "49f21fe4737f78633ac5b610f847c089b6251c74-0"
    add(
        "recommend_biocontainer",
        "two_packages",
        "a pinned pair, which is a mulled-v2 repository and a hashed tag",
        {"packages": ["bwa=0.7.17", "samtools=1.17"]},
        lambda: recommend_biocontainer_fn(["bwa=0.7.17", "samtools=1.17"]),
        [
            route(
                quay_repository(mulled_v2),
                {"tags": {mulled_tag: {"name": mulled_tag}, "latest": {"name": "latest"}}},
            )
        ],
    )


# ---------------------------------------------------------------------------
# Failures
#
# A tool that fails raises, and what a client reads is not an envelope at all: FastMCP
# turns the exception into a tool result with ``isError`` and one text block carrying
# `Error calling tool '<name>': ` and then the tool's own sentence. Measured, not assumed
# -- see ``failure_json``.
#
# Every route here spells its reply out as bytes rather than as a value. The sentences
# quote what Galaxy said, and a bioblend GET quotes it twice: once as a Python ``bytes``
# repr and once as text. Two JSON writers that agree on the value and disagree on the
# spaces would put the two surfaces two bytes apart for a reason belonging to neither.
# ---------------------------------------------------------------------------

# One error body, spelled the way Galaxy spells one: MessageExceptionModel is
# {err_msg, err_code} and nothing else.
DENIED = '{"err_msg": "History is not accessible by user", "err_code": 403002}'
MISSING = '{"err_msg": "History not found", "err_code": 404001}'
BROKEN = '{"err_msg": "Uncaught exception in exposed API method:", "err_code": 0}'


def fail(
    path: str,
    status: int,
    body: str,
    *,
    method: str = "GET",
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A route that answers with an error, byte for byte."""
    return route(path, None, status=status, method=method, query=query, body_text=body)


def failure_cases(add: AddFailure) -> None:
    http_failure_cases(add)
    more_http_failure_cases(add)
    run_failure_cases(add)
    iwc_failure_cases(add)
    refusal_cases(add)
    argument_refusal_cases(add)
    order_of_refusal_cases(add)


def http_failure_cases(add: AddFailure) -> None:
    """What a tool says when Galaxy refuses the request it made.

    The sentence is ``format_error``'s, and the middle of it is the exception's own text,
    which is the client library's rather than this server's. Three shapes reach it:

    * a bioblend client GET, which reports `GET: error <status>: <bytes>, 0 attempts left:
      <text>` -- the body twice, and the attempt counter of a default max_get_attempts
      of 1;
    * a bioblend write, which reports `Unexpected HTTP status code: <status>: <text>`;
    * a raw ``make_get_request`` followed by ``raise_for_status``, which reports requests'
      own `<status> Client Error: <reason> for url: <url>`.

    A case per shape, and per status class where the hint differs.
    """
    # -- the bioblend GET shape ---------------------------------------------
    add(
        "get_workflow_details",
        "not_found",
        "a 404 from a bioblend GET: the not-found hint, and a context with a None in it",
        {"workflow_id": "w0000404"},
        [fail("/api/workflows/w0000404", 404, MISSING)],
    )
    add(
        "get_workflow_details",
        "server_error",
        "a 500 from the same request, which picks the other hint",
        {"workflow_id": "w0000500"},
        [fail("/api/workflows/w0000500", 500, BROKEN)],
    )
    add(
        "get_tool_details",
        "unauthorized",
        "a 401, whose hint is about the API key",
        {"tool_id": "cat1"},
        [fail("/api/tools/cat1", 401, DENIED)],
    )
    add(
        "get_history_contents",
        "permission_denied",
        "a 403, whose hint is about the account",
        {"history_id": "h0000403"},
        [fail("/api/histories/h0000403/contents", 403, DENIED)],
    )
    add(
        "get_tool_run_examples",
        "refused_without_a_hint",
        "a 400, which no hint speaks to -- so the sentence is the text and the context",
        {"tool_id": "cat1"},
        [fail("/api/tools/cat1/test_data", 400, DENIED)],
    )
    add(
        "get_user",
        "unauthorized",
        "the tool's own sentence, which takes neither a hint nor a context",
        {},
        [fail("/api/users/current", 401, DENIED)],
    )
    add(
        "get_histories",
        "server_error",
        "the tool's own sentence, with its own advice after the exception text",
        {},
        [fail("/api/histories", 500, BROKEN)],
    )
    # -- the bioblend write shape ------------------------------------------
    add(
        "update_history",
        "refused_by_galaxy",
        "a 400 from a PUT: no hint, and this tool passes no context",
        {"history_id": "h0000400", "name": "renamed"},
        [fail("/api/histories/h0000400", 400, DENIED, method="PUT")],
    )
    add(
        "cancel_workflow_invocation",
        "not_found",
        "a 404 from a DELETE, which reports the write shape and takes the hint",
        {"invocation_id": "i0000404"},
        [fail("/api/invocations/i0000404", 404, MISSING, method="DELETE")],
    )
    # -- the raise_for_status shape ----------------------------------------
    add(
        "get_page",
        "not_found",
        "requests' own text, which names the URL it could not read",
        {"page_id": "p0000404"},
        [fail("/api/pages/p0000404", 404, MISSING)],
    )
    # -- a raw request whose status the tool checks itself ------------------
    add(
        "list_user_tools",
        "server_error",
        "the index refused, which reads as requests' text because the tool checks the status",
        {},
        [fail("/api/unprivileged_tools", 500, BROKEN, query={"active": "true"})],
    )
    add(
        "delete_user_tool",
        "not_found",
        "a DELETE that failed, which is a failure and not a deactivation",
        {"uuid": "u0000404"},
        [fail("/api/unprivileged_tools/u0000404", 404, MISSING, method="DELETE")],
    )
    add(
        "run_user_tool",
        "lookup_refused",
        "the uuid lookup refused, before anything is submitted",
        {"history_id": "h0001", "tool_uuid": "u0000403", "inputs": {}},
        [fail("/api/unprivileged_tools/u0000403", 403, DENIED)],
    )
    # -- an error body under a 200 -----------------------------------------
    add(
        "get_invocations",
        "error_body_under_a_200",
        "Galaxy answers 200 with a MessageExceptionModel, which is refused as an error",
        {},
        [
            route(
                "/api/invocations",
                {"err_msg": "History is not accessible by user", "err_code": 403002},
            )
        ],
    )


def more_http_failure_cases(add: AddFailure) -> None:
    """The rest of the tools' refused requests, one shape each."""
    add(
        "search_tools_by_keywords",
        "server_error",
        "its own sentence over the panel fetch",
        {"keywords": ["align"]},
        [fail("/api/tools", 500, BROKEN)],
    )
    add(
        "get_tool_input_template",
        "not_found",
        "the schema read that the template is built from",
        {"tool_id": "nosuchtool"},
        [fail("/api/tools/nosuchtool", 404, MISSING)],
    )
    add(
        "get_tool_citations",
        "not_found",
        "the same record, read by another tool",
        {"tool_id": "nosuchtool"},
        [fail("/api/tools/nosuchtool", 404, MISSING)],
    )
    add(
        "list_history_ids",
        "server_error",
        "its own sentence over the listing get_histories reads",
        {},
        [fail("/api/histories", 500, BROKEN)],
    )
    add(
        "get_server_info",
        "configuration_refused",
        "the configuration is read first, so a server that answers the version still fails",
        {},
        [VERSION_ROUTE, fail("/api/configuration", 500, BROKEN)],
    )
    add(
        "get_collection_details",
        "not_found",
        "the tool's own 404 sentence",
        {"collection_id": "c0000404"},
        [fail("/api/dataset_collections/c0000404", 404, MISSING)],
    )
    add(
        "get_collection_details",
        "permission_denied",
        "and format_error's, for a status it has nothing of its own to say about",
        {"collection_id": "c0000403"},
        [fail("/api/dataset_collections/c0000403", 403, DENIED)],
    )
    add(
        "create_history",
        "refused_by_galaxy",
        "the one tool with no try block, so bioblend's text is the whole sentence",
        {"history_name": "nope"},
        [fail("/api/histories", 400, DENIED, method="POST")],
    )
    add(
        "run_tool",
        "permission_denied",
        "a status Galaxy refuses a run with that is not about the inputs",
        {
            "tool_id": "cat1",
            "history_id": "h0001",
            "inputs": {"input1": {"src": "hda", "id": "d1"}},
        },
        [fail("/api/tools", 403, DENIED, method="POST")],
    )
    add(
        "get_workflow_input_template",
        "server_error",
        "the export the template falls back to, which is the only request that can fail here",
        {"workflow_id": "w0000500"},
        # The same export, spelled the two ways the two clients spell it: bioblend builds
        # /api/workflows/download/<id> and the other side asks /api/workflows/<id>/download.
        [
            fail("/api/workflows/download/w0000500", 500, BROKEN),
            fail("/api/workflows/w0000500/download", 500, BROKEN),
        ],
    )
    add(
        "invoke_workflow",
        "refused_by_galaxy",
        "every argument that decides where the run went is in the context",
        {"workflow_id": "w0000400", "history_id": "h0001"},
        [fail("/api/workflows/w0000400/invocations", 400, DENIED, method="POST")],
    )
    add(
        "create_user_tool",
        "refused_by_galaxy",
        "the context names the id out of the representation, with dict.get",
        {
            "representation": {
                "class": "GalaxyUserTool",
                "id": "utool",
                "version": "0.1.0",
                "name": "A user tool",
                "container": "python:3.12-slim",
                "shell_command": "echo hi",
            }
        },
        [fail("/api/unprivileged_tools", 400, DENIED, method="POST")],
    )
    add(
        "list_pages",
        "permission_denied",
        "a raw GET whose text names the URL, query string and all",
        {},
        [fail("/api/pages", 403, DENIED)],
    )
    add(
        "create_page",
        "refused_by_galaxy",
        "a write, so the sentence is the status and the body",
        {"title": "A report", "slug": "a-report", "content": "# hi"},
        [VERSION_ROUTE, fail("/api/pages", 400, DENIED, method="POST")],
    )
    add(
        "update_page",
        "not_found",
        "a write to a page that is not there",
        {"page_id": "p0000404", "title": "Renamed"},
        [VERSION_ROUTE, fail("/api/pages/p0000404", 404, MISSING, method="PUT")],
    )
    add(
        "list_page_revisions",
        "not_found",
        "the URL in the text carries the sort_desc this tool always sends",
        {"page_id": "p0000404"},
        [VERSION_ROUTE, fail("/api/pages/p0000404/revisions", 404, MISSING)],
    )
    add(
        "get_page_revision",
        "not_found",
        "two ids in the context",
        {"page_id": "p0001", "revision_id": "r0000404"},
        [VERSION_ROUTE, fail("/api/pages/p0001/revisions/r0000404", 404, MISSING)],
    )
    add(
        "revert_page_revision",
        "not_found",
        "a write, with the same two ids",
        {"page_id": "p0001", "revision_id": "r0000404"},
        [
            VERSION_ROUTE,
            fail("/api/pages/p0001/revisions/r0000404/revert", 404, MISSING, method="POST"),
        ],
    )
    add(
        "get_job_details",
        "dataset_not_found",
        "a 404 keeps the tool's own sentence, which does not claim which of the two it was",
        {"dataset_id": "d0000404"},
        [fail("/api/datasets/d0000404", 404, MISSING)],
    )
    add(
        "get_job_details",
        "no_job_made_this_dataset",
        "a dataset that reads fine and names no job -- a refusal, not a failed request",
        {"dataset_id": "d0001"},
        [route("/api/datasets/d0001", {"id": "d0001", "name": "uploaded.txt", "state": "ok"})],
    )
    add(
        "get_job_details",
        "the_job_read_refused",
        "the jobs API is asked through requests itself, so its failure reads differently",
        {"dataset_id": "d0002"},
        [
            route(
                "/api/datasets/d0002",
                {"id": "d0002", "name": "out.txt", "state": "ok", "creating_job": "j0000500"},
            ),
            fail("/api/jobs/j0000500", 500, BROKEN),
        ],
    )


def run_failure_cases(add: AddFailure) -> None:
    """A refused run, which is the failure an agent running a tool is most likely to meet.

    A 400 is the status Galaxy refuses a tool form with, and the sentence for one is not the
    bare failure: the parameter list is read back and offered, a tool test supplies a
    structural example, and the whole thing warns about Galaxy's misleading wording. A failure
    whose text mentions credentials takes a different branch again, and it is checked first.
    """
    fastqc_schema = {
        "id": "fastqc",
        "name": "FastQC",
        "version": "0.74+galaxy1",
        "inputs": [
            {
                "name": "input_file",
                "type": "data",
                "optional": False,
                "multiple": False,
                "extensions": ["fastqsanger"],
            }
        ],
    }
    tool_tests = [
        {
            "name": "Test-1",
            "tool_id": "fastqc",
            "inputs": {"input_file": [{"src": "hda", "id": "0123456789abcdef"}]},
        }
    ]
    # The inputs here are ones the preflight is happy with, and Galaxy refuses anyway --
    # which is an ordinary thing for it to do, since its own validation is the stricter of
    # the two. It also keeps the two clauses that need the input checker out of the sentence,
    # and those are the two the TypeScript side cannot produce; see the release notes.
    add(
        "run_tool",
        "refused_over_the_inputs",
        "a 400, with the parameter list and a tool test's example read back for the caller",
        {
            "history_id": "h0001",
            "tool_id": "fastqc",
            "inputs": {"input_file": {"src": "hda", "id": "d0000001"}},
        },
        [
            route("/api/tools/fastqc", fastqc_schema),
            route("/api/tools/fastqc/test_data", tool_tests),
            fail("/api/tools", 400, DENIED, method="POST"),
        ],
    )
    add(
        "run_tool",
        "refused_with_nothing_readable",
        "the same 400 with neither the schema nor a test readable, so it points at a call",
        {"history_id": "h0001", "tool_id": "fastqc", "inputs": {"input_file": "not-a-dataset"}},
        [fail("/api/tools", 400, DENIED, method="POST")],
    )
    add(
        "run_tool",
        "refused_over_credentials",
        "a refusal whose text mentions credentials, which is checked before the status is",
        {"history_id": "h0001", "tool_id": "fastqc", "inputs": {"contaminants": ""}},
        [
            fail(
                "/api/tools",
                400,
                '{"err_msg": "Tool requires service credentials that are not set", '
                '"err_code": 400008}',
                method="POST",
            )
        ],
    )
    add(
        "run_user_tool",
        "refused_over_the_inputs",
        "the representation it already holds is the schema the sentence offers",
        {
            "history_id": "h0001",
            "tool_uuid": "61d15277-a911-45ef-aa66-5385146578cc",
            "inputs": {"input": {"src": "hda", "id": "d0000001"}},
        },
        [
            route(
                "/api/unprivileged_tools/61d15277-a911-45ef-aa66-5385146578cc",
                {
                    "tool_id": "row_filter",
                    "representation": {
                        "id": "row_filter",
                        "version": "0.1.0",
                        "inputs": [{"name": "input", "type": "data", "optional": False}],
                    },
                },
            ),
            fail("/api/tools", 400, DENIED, method="POST"),
        ],
    )


def iwc_failure_cases(add: AddFailure) -> None:
    """The four manifest tools, which reach past Galaxy to the IWC.

    The manifest is fetched with requests and checked with raise_for_status, so all four
    quote requests' own text -- and each wraps it in a sentence of its own.
    """
    manifest = absolute(IWC_MANIFEST_URL)
    for tool, name, inp in (
        ("get_iwc_workflows", "manifest_refused", {}),
        ("search_iwc_workflows", "manifest_refused", {"query": "rna"}),
        ("recommend_iwc_workflows", "manifest_refused", {"intent": "assemble a genome"}),
        ("get_iwc_workflow_details", "manifest_refused", {"trs_id": "#workflow/x/y/1"}),
        ("import_workflow_from_iwc", "manifest_refused", {"trs_id": "#workflow/x/y/1"}),
    ):
        add(tool, name, "the manifest itself refused", inp, [fail(manifest, 500, BROKEN)])
    add(
        "get_iwc_workflow_details",
        "no_such_trs_id",
        "a refusal raised inside the try, so the tool's own sentence wraps it too",
        {"trs_id": "#workflow/nobody/has/this"},
        [route(IWC_MANIFEST_URL, iwc_manifest(2))],
    )
    add(
        "import_workflow_from_iwc",
        "no_such_trs_id",
        "the same refusal, worded by the tool that was asked",
        {"trs_id": "#workflow/nobody/has/this"},
        [route(IWC_MANIFEST_URL, iwc_manifest(2))],
    )
    add(
        "import_workflow_from_iwc",
        "refused_by_galaxy",
        "the import itself refused, which is a bioblend write",
        {"trs_id": "#workflow/github.com/iwc-workflows/wf0/main"},
        # Registered under both spellings of the same create, as the success cases are:
        # bioblend posts to /api/workflows/upload and the other side to /api/workflows.
        [
            route(IWC_MANIFEST_URL, iwc_manifest(2)),
            fail("/api/workflows/upload", 400, DENIED, method="POST"),
            fail("/api/workflows", 400, DENIED, method="POST"),
        ],
    )


def refusal_cases(add: AddFailure) -> None:
    """What a tool says before it asks Galaxy anything, or instead of answering.

    These sentences are the server's own from end to end -- no exception text, no hint and
    no context -- so they are the ones a port has to get exactly right on its own.
    """
    add(
        "list_workflows",
        "limit_above_the_ceiling",
        "the page ceiling, with the advice to use offset that a pageable tool gets",
        {"limit": 5000},
        [],
    )
    add(
        "get_history_contents",
        "limit_below_one",
        "the floor, on a listing with no ceiling",
        {"history_id": "h0001", "limit": 0},
        [],
    )
    add(
        "search_tools_by_name",
        "negative_offset",
        "the offset floor",
        {"query": "cat", "offset": -1},
        [],
    )
    add(
        "get_history_details",
        "not_found",
        "the tool's own 404 sentence, which says what kind of argument it wanted",
        {"history_id": "h0000404"},
        [fail("/api/histories/h0000404", 404, MISSING)],
    )
    add(
        "get_tool_panel",
        "no_such_section",
        "a refusal after a request that succeeded",
        {"section_id": "not-a-section"},
        [route("/api/tools", panel_sections(2, 2))],
    )
    add(
        "update_history",
        "nothing_to_update",
        "no field to change, refused before anything is sent",
        {"history_id": "h0001"},
        [],
    )
    add(
        "get_dataset_details",
        "is_a_collection",
        "the id turned out to name a collection, which the tool says by name",
        {"dataset_id": "c0001"},
        [
            fail("/api/datasets/c0001", 404, MISSING),
            route(
                "/api/dataset_collections/c0001",
                {"id": "c0001", "name": "My collection", "collection_type": "list"},
            ),
        ],
    )


def order_of_refusal_cases(add: AddFailure) -> None:
    """Where the two surfaces used to refuse in a different order, or not at all."""
    add(
        "list_history_ids",
        "history_without_an_id",
        "the id is read with [] rather than .get, so a history without one raises",
        {},
        [route("/api/histories", [{"name": "Nameless"}])],
    )
    add(
        "update_history",
        "every_field_null",
        "a null is an unset field, so a call that sets all of them to null updates nothing",
        {
            "history_id": "h0001",
            "name": None,
            "annotation": None,
            "tags": None,
            "deleted": None,
            "published": None,
        },
        [],
    )
    add(
        "create_page",
        "report_without_a_title",
        "a report missing both required fields, which Galaxy is left to refuse",
        {"slug": None, "title": None, "content": "# hi"},
        [VERSION_ROUTE, fail("/api/pages", 400, DENIED, method="POST")],
    )
    add(
        "create_page",
        "galaxy_too_old",
        "the version guard, which refuses before anything is sent",
        {"title": "A report", "slug": "a-report", "content": "# hi"},
        [route("/api/version", {"version_major": "24.1", "version_minor": "0"})],
    )


def argument_refusal_cases(add: AddFailure) -> None:
    """Arguments this server refuses on its own terms, before it asks Galaxy anything."""
    add(
        "create_user_tool",
        "representation_missing_a_field",
        "the first of six required fields that is not there",
        {"representation": {"class": "GalaxyUserTool", "id": "utool"}},
        [],
    )
    add(
        "create_user_tool",
        "wrong_class",
        "a representation of something else",
        {
            "representation": {
                "class": "GalaxyTool",
                "id": "utool",
                "version": "0.1.0",
                "name": "A user tool",
                "container": "python:3.12-slim",
                "shell_command": "echo hi",
            }
        },
        [],
    )
    add(
        "create_user_tool",
        "container_is_not_a_string",
        "the type is named the way Python names a type",
        {
            "representation": {
                "class": "GalaxyUserTool",
                "id": "utool",
                "version": "0.1.0",
                "name": "A user tool",
                "container": 3,
                "shell_command": "echo hi",
            }
        },
        [],
    )
    add(
        "recommend_biocontainer",
        "package_entry_with_no_name",
        "the entry is quoted with repr, so the sentence shows what arrived",
        {"packages": ["=1.17"]},
        [],
    )


# ---------------------------------------------------------------------------
# Running and writing
# ---------------------------------------------------------------------------


def absolute(path: str) -> str:
    return path if path.startswith("http") else f"{GALAXY_URL}{path}"


def _reset_caches() -> None:
    server.get_manifest_json.cache_clear()
    server._TOOL_SCHEMA_CACHE.clear()
    server._DATATYPES_MAPPING_CACHE.clear()
    clear_version_cache()
    _clear_recommendation_cache()


def _clear_recommendation_cache() -> None:
    """Forget what quay.io said, which is remembered module-wide for five minutes.

    Three cases ask about the same package and expect three lookups; without this the
    second and third would be answered from the first. Imported inside the function
    because the recommender only exists with the container-recommend extra -- which is
    also what registers the tool -- and an install without it has no memo to clear. The
    cases that need the extra then fail with the tool's own sentence about installing it,
    which says more than an ImportError from here would.
    """
    try:
        from galaxy.tool_util.deps.mulled import recommend as mulled_recommend
    except ImportError:
        return
    with mulled_recommend._cache_lock:
        mulled_recommend._cache.clear()


def run_case(case: Case, session: LiveMCPSession | None = None) -> GalaxyResult | dict[str, Any]:
    """Answer this case's requests from its table and return what the tool answered.

    A success case calls the tool function and the envelope it built is what gets written
    down. A failure case is driven through ``session`` -- a real in-memory MCP client --
    because there is no envelope to write down: the tool raises, FastMCP turns the
    exception into a tool result with ``isError`` and one text block, and that result is
    the only thing a client ever sees. Calling the function directly would pin the
    exception instead, which is not what crosses the wire.
    """
    _reset_caches()
    gi = GalaxyInstance(url=GALAXY_BASE_URL, key=PLACEHOLDER_KEY)
    previous = galaxy_state.copy()
    galaxy_state.update(
        {"url": GALAXY_BASE_URL, "api_key": PLACEHOLDER_KEY, "gi": gi, "connected": True}
    )
    try:
        with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
            # Most specific first: `responses` takes the first registration that matches,
            # and the TypeScript replay picks the same one by the same rule.
            for spec in sorted(case.routes, key=lambda r: -len(r["query"])):
                body = spec.get("bodyText")
                mock.add(
                    method=spec["method"],
                    url=absolute(spec["path"]),
                    status=spec["status"],
                    headers=spec["headers"],
                    match=(
                        [responses.matchers.query_param_matcher(spec["query"], strict_match=False)]
                        if spec["query"]
                        else []
                    ),
                    **(
                        {"body": body, "content_type": "application/json"}
                        if body is not None
                        else {"json": spec["body"]}
                    ),
                )
            if case.failure:
                assert session is not None, "a failure case needs an MCP session"
                return session.wire_result(case.tool, case.input)
            assert case.call is not None, "a success case needs something to call"
            return case.call()
    finally:
        galaxy_state.clear()
        galaxy_state.update(previous)
        _reset_caches()


def envelope_json(result: GalaxyResult) -> str:
    """The envelope as FastMCP sends it, re-indented so a diff is readable."""
    sent = pydantic_core.to_json(result, fallback=str)
    return json.dumps(json.loads(sent), indent=2, ensure_ascii=False) + "\n"


# What FastMCP prefixes a tool's own message with on its way out. Measured rather than
# assumed (fastmcp 3.4.2, `mask_error_details` False by default): a tool that raises
# anything other than a ToolError reaches the client as
# `ToolError(f"Error calling tool {name!r}: {e}")`, and that is the text the content block
# carries. The prefix belongs to the MCP surface and not to the tool, so it is stripped
# off here as well as written out -- a surface with no FastMCP under it says the sentence
# on its own, and the two halves are kept apart rather than left for a reader to guess at.
def tool_error_prefix(tool: str) -> str:
    return f"Error calling tool {tool!r}: "


def failure_json(case: Case, result: dict[str, Any]) -> str:
    """A failed call, as the wire carries it, plus the tool's own sentence."""
    blocks = result.get("content") or []
    assert result.get("isError") is True, f"{case.tool}/{case.name} did not fail: {result}"
    assert len(blocks) == 1, f"{case.tool}/{case.name} answered {len(blocks)} content blocks"
    text = blocks[0]["text"]
    prefix = tool_error_prefix(case.tool)
    assert text.startswith(prefix), f"{case.tool}/{case.name} is not wrapped: {text!r}"
    return (
        json.dumps(
            {
                "$comment": (
                    "A failed tool call from the Python MCP server -- do not edit by hand. "
                    f"Regenerate with `{REGENERATE_COMMAND}`. `result` is the CallToolResult "
                    "the MCP wire carries, dumped the way the SDK dumps it (by alias, JSON "
                    "mode, None-valued fields left out). `sentence` is the same text with "
                    "FastMCP's own wrapper removed, which is what the tool itself said."
                ),
                "outcome": "failure",
                "result": result,
                "sentence": text[len(prefix) :],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def replies_json_for(case: Case) -> str:
    """The canned replies, as a table both sides match the same way.

    Nothing here names the case: one table often answers several windows onto the
    same corpus, and index.json is what says which case reads which file.
    """
    return (
        json.dumps(
            {
                "$comment": (
                    "Canned Galaxy replies -- do not edit by hand. Regenerate with "
                    f"`{REGENERATE_COMMAND}`. Matched by method and path; where a route "
                    "declares a query, every declared parameter must match, and the most "
                    "specific matching route wins."
                ),
                "baseUrl": GALAXY_BASE_URL,
                "routes": case.routes,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def build() -> dict[str, str]:
    """Every file this generator owns, as path -> contents."""
    files: dict[str, str] = {}
    index: list[dict[str, Any]] = []
    # Several cases are the same corpus read through different windows, and a
    # manifest written out once per case is most of what this directory would
    # weigh. The first case to use a table owns the file; the rest of the index
    # points at it.
    by_replies: dict[str, str] = {}
    all_cases = cases()
    # One client for every failure case, because one connection means one session id and
    # the session-scoped connection store then does its job across the calls -- the same
    # reasoning tests/mcp_session.py is built on. Opened only when something needs it.
    with ExitStack() as stack:
        session = (
            stack.enter_context(LiveMCPSession())
            if any(case.failure for case in all_cases)
            else None
        )
        for case in all_cases:
            result = run_case(case, session)
            replies = replies_json_for(case)
            path = by_replies.get(replies)
            if path is None:
                path = f"{case.tool}/{case.name}.galaxy.json"
                by_replies[replies] = path
                files[path] = replies
            if case.failure:
                assert isinstance(result, dict)
                files[f"{case.tool}/{case.name}.json"] = failure_json(case, result)
            else:
                assert isinstance(result, GalaxyResult)
                files[f"{case.tool}/{case.name}.json"] = envelope_json(result)
            index.append(
                {
                    "tool": case.tool,
                    "case": case.name,
                    "note": case.note,
                    "outcome": "failure" if case.failure else "success",
                    "input": case.input,
                    "envelope": f"{case.tool}/{case.name}.json",
                    "replies": path,
                }
            )
    files["index.json"] = (
        json.dumps(
            {
                "$comment": (
                    "Generated golden envelopes from the Python MCP server -- do not edit by "
                    f"hand. Regenerate with `{REGENERATE_COMMAND}`. `input` is spelled in the "
                    "server's own parameter names, which is what the MCP wire takes. A case "
                    'whose `outcome` is "failure" pins the tool result a failed call '
                    "answers with instead of an envelope."
                ),
                "source": "mcp-server-galaxy-py/src/galaxy_mcp/server.py",
                "caseCount": len(index),
                "failureCaseCount": sum(1 for row in index if row["outcome"] == "failure"),
                "cases": index,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return files


def main() -> None:
    files = build()
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, text in files.items():
        path = FIXTURE_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, newline="\n")
        written += 1
    # Anything left from a case that was renamed or dropped.
    known = {FIXTURE_ROOT / name for name in files}
    for stale in sorted(FIXTURE_ROOT.rglob("*.json")):
        if stale not in known:
            stale.unlink()
            print(f"removed stale {stale}")
    print(f"wrote {written} files under {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
