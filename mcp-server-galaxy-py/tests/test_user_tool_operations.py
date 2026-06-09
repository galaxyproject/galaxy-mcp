"""Tests for user-defined tool (UDT) operations."""

from unittest.mock import patch

import pytest

from .test_helpers import galaxy_state, run_user_tool_fn


class TestRunUserTool:
    """run_user_tool submits UDTs via the portable synchronous tools endpoint."""

    def _gi(self, mock_galaxy_instance):
        gi = mock_galaxy_instance
        gi.url = "http://localhost:8080/api"
        gi.make_get_request.return_value.json.return_value = {
            "tool_id": "my_udt_tool_id",
            "representation": {"version": "1.2.3"},
        }
        gi.make_post_request.return_value = {
            "outputs": [{"id": "out_1", "name": "result"}],
            "jobs": [{"id": "job_1", "state": "new"}],
        }
        return gi

    def test_run_user_tool_posts_to_tools_endpoint(self, mock_galaxy_instance):
        """Submits via POST /api/tools with tool_uuid -- never the 26.0-broken /api/jobs path."""
        gi = self._gi(mock_galaxy_instance)
        uuid = "8a049c53-f4f2-4fdd-a9a6-a2560494e0ec"
        inputs = {"msg": "hello"}

        with patch.dict(galaxy_state, {"connected": True, "gi": gi}):
            result = run_user_tool_fn("hist_1", uuid, inputs)

        assert result.success is True
        gi.make_post_request.assert_called_once()
        posted_url = gi.make_post_request.call_args.args[0]
        posted_payload = gi.make_post_request.call_args.kwargs["payload"]

        assert posted_url == "http://localhost:8080/api/tools"
        assert "/jobs" not in posted_url
        assert posted_payload["tool_uuid"] == uuid
        # tool_id and tool_uuid are mutually exclusive on /api/tools; send only the uuid
        assert "tool_id" not in posted_payload
        assert posted_payload["history_id"] == "hist_1"
        assert posted_payload["inputs"] == inputs
        assert posted_payload["input_format"] == "legacy"
        assert "use_cached_jobs" not in posted_payload

    def test_run_user_tool_resolves_version_from_unprivileged_tools(self, mock_galaxy_instance):
        """tool_version is resolved from the UDT record and forwarded in the submission."""
        gi = self._gi(mock_galaxy_instance)
        uuid = "abc"

        with patch.dict(galaxy_state, {"connected": True, "gi": gi}):
            run_user_tool_fn("hist_1", uuid, {})

        gi.make_get_request.assert_called_once_with(
            "http://localhost:8080/api/unprivileged_tools/abc"
        )
        assert gi.make_post_request.call_args.kwargs["payload"]["tool_version"] == "1.2.3"

    def test_run_user_tool_missing_tool_raises(self, mock_galaxy_instance):
        """A UUID with no resolvable tool yields a clear error, no submission attempted."""
        gi = self._gi(mock_galaxy_instance)
        gi.make_get_request.return_value.json.return_value = {}

        with patch.dict(galaxy_state, {"connected": True, "gi": gi}):
            with pytest.raises(ValueError, match="Run user tool failed"):
                run_user_tool_fn("hist_1", "missing", {})
        gi.make_post_request.assert_not_called()

    def test_run_user_tool_error(self, mock_galaxy_instance):
        """Submission failures are surfaced as a Run user tool error."""
        gi = self._gi(mock_galaxy_instance)
        gi.make_post_request.side_effect = Exception("boom")

        with patch.dict(galaxy_state, {"connected": True, "gi": gi}):
            with pytest.raises(ValueError, match="Run user tool failed"):
                run_user_tool_fn("hist_1", "abc", {})

    def test_run_user_tool_not_connected(self):
        """Fails fast when not connected."""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(Exception):
                run_user_tool_fn("hist_1", "abc", {})
