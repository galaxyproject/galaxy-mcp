"""Tests for Galaxy Pages (notebook/report) operations.

The page tools call Galaxy's /api/pages* REST endpoints through bioblend's
thread-safe make_get_request / make_post_request / make_put_request helpers
(which carry the configured User-Agent and the _gi_lock), so the tests mock
those methods on the injected GalaxyInstance. make_get_request returns a
requests.Response; make_post_request / make_put_request return decoded JSON.
"""

from unittest.mock import Mock

import pytest
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp.server import galaxy_state
from tests.test_helpers import (
    create_page_fn,
    get_page_fn,
    get_page_revision_fn,
    list_page_revisions_fn,
    list_pages_fn,
    revert_page_revision_fn,
    update_page_fn,
)

# bioblend's GalaxyInstance.url is the API root (`<base>/api`), not the server
# base URL -- mirror that here or these URL assertions guard a shape production
# can never produce.
GALAXY_BASE_URL = "http://localhost:8080"
GALAXY_API_URL = f"{GALAXY_BASE_URL}/api"


def test_bioblend_url_is_the_api_root():
    """Pin the contract the mocked fixture below imitates.

    Every URL assertion in this module sets `gi.url` by hand, so it can only be
    as honest as this assumption. GalaxyInstance only touches the network when
    the URL has no scheme, so constructing one here stays offline.
    """
    gi = GalaxyInstance(url=GALAXY_BASE_URL, key="notakey")
    assert gi.url == GALAXY_API_URL


def _get_response(json_data, headers=None):
    """Fake requests.Response as returned by gi.make_get_request."""
    resp = Mock()
    resp.json.return_value = json_data
    resp.headers = headers or {}
    resp.raise_for_status.return_value = None
    return resp


class TestPageOperations:
    def setup_method(self):
        self.gi = Mock()
        self.gi.url = GALAXY_API_URL
        galaxy_state["connected"] = True
        galaxy_state["gi"] = self.gi

    def teardown_method(self):
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    def test_list_pages(self):
        self.gi.make_get_request.return_value = _get_response(
            [
                {"id": "page1", "title": "Notebook 1", "history_id": "hist1"},
                {"id": "page2", "title": "Report 2", "history_id": None},
            ],
            headers={"total_matches": "5"},
        )

        result = list_pages_fn(limit=2, offset=0)

        assert result.success is True
        assert result.count == 2
        assert result.data[0]["id"] == "page1"
        # total_matches header is surfaced through pagination metadata
        assert result.pagination is not None
        assert result.pagination.total_items == 5
        assert result.pagination.has_next is True

        args, kwargs = self.gi.make_get_request.call_args
        assert args[0] == f"{GALAXY_API_URL}/pages"
        # REST index defaults (show_own/show_published) are wrong for an agent;
        # confirm we override them explicitly.
        params = kwargs["params"]
        assert params["show_own"] == "true"
        assert params["show_published"] == "false"
        assert params["show_shared"] == "false"
        assert params["limit"] == 2

    def test_list_pages_history_filter(self):
        self.gi.make_get_request.return_value = _get_response(
            [{"id": "page1", "title": "Notebook 1", "history_id": "hist1"}],
            headers={"total_matches": "1"},
        )

        result = list_pages_fn(history_id="hist1", show_published=True)

        assert result.count == 1
        params = self.gi.make_get_request.call_args.kwargs["params"]
        assert params["history_id"] == "hist1"
        assert params["show_published"] == "true"

    def test_list_pages_total_matches_defaults_when_header_absent(self):
        # Galaxy always sets the header, but the tool must not crash without it.
        self.gi.make_get_request.return_value = _get_response([{"id": "page1"}, {"id": "page2"}])

        result = list_pages_fn(limit=50)

        assert result.count == 2
        assert result.pagination.total_items == 2
        assert result.pagination.has_next is False

    def test_get_page_strips_rendered_by_default(self):
        self.gi.make_get_request.return_value = _get_response(
            {
                "id": "page1",
                "title": "Notebook 1",
                "content": "<rendered html>",
                "content_editor": "raw markdown",
                "edit_source": "agent",
            }
        )

        result = get_page_fn("page1")

        assert result.success is True
        assert result.data["content_editor"] == "raw markdown"
        # The large rendered form is dropped unless explicitly requested.
        assert "content" not in result.data
        assert self.gi.make_get_request.call_args.args[0] == f"{GALAXY_API_URL}/pages/page1"

    def test_get_page_include_rendered(self):
        self.gi.make_get_request.return_value = _get_response(
            {
                "id": "page1",
                "title": "Notebook 1",
                "content": "<rendered html>",
                "content_editor": "raw markdown",
            }
        )

        result = get_page_fn("page1", include_rendered=True)

        assert result.data["content"] == "<rendered html>"
        assert result.data["content_editor"] == "raw markdown"

    def test_create_notebook(self):
        self.gi.make_post_request.return_value = {
            "id": "page1",
            "title": "My History",
            "content": "<rendered>",
            "content_editor": "# notes",
        }

        result = create_page_fn(history_id="hist1", content="# notes")

        assert result.success is True
        assert result.data["id"] == "page1"
        # create returns the editable form, not the rendered one
        assert "content" not in result.data

        args, kwargs = self.gi.make_post_request.call_args
        assert args[0] == f"{GALAXY_API_URL}/pages"
        payload = kwargs["payload"]
        assert payload["content_format"] == "markdown"
        assert payload["history_id"] == "hist1"
        assert payload["content"] == "# notes"

    def test_create_report(self):
        self.gi.make_post_request.return_value = {
            "id": "page9",
            "title": "My Report",
            "content_editor": "",
        }

        result = create_page_fn(title="My Report", slug="my-report")

        assert result.data["id"] == "page9"
        payload = self.gi.make_post_request.call_args.kwargs["payload"]
        assert payload["title"] == "My Report"
        assert payload["slug"] == "my-report"
        assert "history_id" not in payload

    def test_create_report_missing_slug_error(self):
        # Standalone reports require a unique slug; Galaxy rejects the request and
        # bioblend's make_post_request raises on the non-200.
        self.gi.make_post_request.side_effect = Exception("Unexpected HTTP status code: 400")

        with pytest.raises(ValueError, match="Create page failed"):
            create_page_fn(title="My Report")

    def test_update_page_attributes_to_agent(self):
        self.gi.make_put_request.return_value = {
            "id": "page1",
            "title": "Notebook 1",
            "content": "<rendered>",
            "content_editor": "# updated",
        }

        result = update_page_fn("page1", content="# updated")

        assert result.success is True
        assert "content" not in result.data
        args, kwargs = self.gi.make_put_request.call_args
        assert args[0] == f"{GALAXY_API_URL}/pages/page1"
        payload = kwargs["payload"]
        assert payload["edit_source"] == "agent"
        assert payload["content"] == "# updated"

    def test_list_page_revisions(self):
        self.gi.make_get_request.return_value = _get_response(
            [
                {"id": "rev1", "page_id": "page1", "edit_source": "user"},
                {"id": "rev2", "page_id": "page1", "edit_source": "agent"},
            ]
        )

        result = list_page_revisions_fn("page1", sort_desc=True)

        assert result.success is True
        assert result.count == 2
        assert result.data[1]["edit_source"] == "agent"
        args, kwargs = self.gi.make_get_request.call_args
        assert args[0] == f"{GALAXY_API_URL}/pages/page1/revisions"
        assert kwargs["params"]["sort_desc"] == "true"

    def test_get_page_revision_returns_content(self):
        # A revision's editable markdown lives in `content` (no content_editor),
        # and unlike get_page it is NOT stripped.
        self.gi.make_get_request.return_value = _get_response(
            {
                "id": "rev1",
                "page_id": "page1",
                "content": "raw md",
                "edit_source": "user",
            }
        )

        result = get_page_revision_fn("page1", "rev1")

        assert result.success is True
        assert result.data["content"] == "raw md"
        assert result.data["edit_source"] == "user"
        assert (
            self.gi.make_get_request.call_args.args[0]
            == f"{GALAXY_API_URL}/pages/page1/revisions/rev1"
        )

    def test_revert_page_revision(self):
        self.gi.make_post_request.return_value = {
            "id": "rev3",
            "page_id": "page1",
            "content": "restored md",
            "edit_source": "restore",
        }

        result = revert_page_revision_fn("page1", "rev1")

        assert result.success is True
        assert result.data["id"] == "rev3"
        assert result.data["edit_source"] == "restore"
        # restored revision content is returned, not stripped
        assert result.data["content"] == "restored md"
        assert (
            self.gi.make_post_request.call_args.args[0]
            == f"{GALAXY_API_URL}/pages/page1/revisions/rev1/revert"
        )

    def test_list_pages_not_connected(self):
        galaxy_state["connected"] = False
        with pytest.raises(ValueError, match="Not connected to Galaxy"):
            list_pages_fn()
