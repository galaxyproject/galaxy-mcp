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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydantic_core
import responses
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp import server
from galaxy_mcp.server import GalaxyResult, galaxy_state
from galaxy_mcp.version import clear_version_cache

from .test_helpers import (
    get_collection_details_fn,
    get_histories_fn,
    get_history_contents_fn,
    get_history_details_fn,
    get_invocations_fn,
    get_iwc_workflow_details_fn,
    get_iwc_workflows_fn,
    get_tool_citations_fn,
    get_tool_panel_fn,
    get_tool_run_examples_fn,
    get_workflow_input_template_fn,
    list_history_ids_fn,
    list_page_revisions_fn,
    list_pages_fn,
    list_user_tools_fn,
    list_workflows_fn,
    recommend_iwc_workflows_fn,
    search_iwc_workflows_fn,
    search_tools_by_keywords_fn,
    search_tools_fn,
)

FIXTURE_ROOT = Path(__file__).parent / "testdata" / "envelopes"
REGENERATE_COMMAND = "uv run python -m tests.envelope_fixtures"

# Where the fake Galaxy lives. The TypeScript replay uses the same base, so a route's
# path reads the same on both sides.
GALAXY_URL = "https://galaxy.example"
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
) -> dict[str, Any]:
    """One canned Galaxy reply.

    ``query`` is only set where a case needs two answers from one path; a route that
    declares none answers any query on that path.
    """
    return {
        "method": method,
        "path": path,
        "query": query or {},
        "status": status,
        "headers": headers or {},
        "body": body,
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
    call: Callable[[], GalaxyResult]
    routes: list[dict[str, Any]] = field(default_factory=list)


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

    return out


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


def run_case(case: Case) -> GalaxyResult:
    """Answer this case's requests from its table and return what the tool built."""
    _reset_caches()
    gi = GalaxyInstance(url=GALAXY_URL, key=PLACEHOLDER_KEY)
    previous = galaxy_state.copy()
    galaxy_state.update(
        {"url": GALAXY_URL, "api_key": PLACEHOLDER_KEY, "gi": gi, "connected": True}
    )
    try:
        with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
            # Most specific first: `responses` takes the first registration that matches,
            # and the TypeScript replay picks the same one by the same rule.
            for spec in sorted(case.routes, key=lambda r: -len(r["query"])):
                mock.add(
                    method=spec["method"],
                    url=absolute(spec["path"]),
                    json=spec["body"],
                    status=spec["status"],
                    headers=spec["headers"],
                    match=(
                        [responses.matchers.query_param_matcher(spec["query"], strict_match=False)]
                        if spec["query"]
                        else []
                    ),
                )
            return case.call()
    finally:
        galaxy_state.clear()
        galaxy_state.update(previous)
        _reset_caches()


def envelope_json(result: GalaxyResult) -> str:
    """The envelope as FastMCP sends it, re-indented so a diff is readable."""
    sent = pydantic_core.to_json(result, fallback=str)
    return json.dumps(json.loads(sent), indent=2, ensure_ascii=False) + "\n"


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
                "baseUrl": GALAXY_URL,
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
    for case in cases():
        result = run_case(case)
        replies = replies_json_for(case)
        path = by_replies.get(replies)
        if path is None:
            path = f"{case.tool}/{case.name}.galaxy.json"
            by_replies[replies] = path
            files[path] = replies
        files[f"{case.tool}/{case.name}.json"] = envelope_json(result)
        index.append(
            {
                "tool": case.tool,
                "case": case.name,
                "note": case.note,
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
                    "server's own parameter names, which is what the MCP wire takes."
                ),
                "source": "mcp-server-galaxy-py/src/galaxy_mcp/server.py",
                "caseCount": len(index),
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
