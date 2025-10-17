"""
Test history-related operations
"""

from unittest.mock import call, patch

import pytest

from .test_helpers import (
    galaxy_state,
    get_histories_fn,
    get_history_details_fn,
)


class TestHistoryOperations:
    """Test history operations"""

    def test_get_histories_fn(self, mcp_server_instance, mock_galaxy_instance):
        """Test get_histories returns paginated results"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_histories_fn()

            assert isinstance(result, dict)
            assert "histories" in result
            assert "pagination" in result
            assert len(result["histories"]) == 2
            assert result["histories"][0]["id"] == "test_history_1"
            assert result["histories"][0]["name"] == "Test History 1"
            assert result["pagination"]["total_items"] == 2
            assert result["pagination"]["paginated"] is False

    def test_get_histories_empty(self, mock_galaxy_instance):
        """Test get_histories with no histories"""
        mock_galaxy_instance.histories.get_histories.side_effect = lambda *args, **kwargs: []

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_histories_fn()

            assert result["histories"] == []
            assert result["pagination"]["total_items"] == 0

    def test_get_histories_ids_only(self, mock_galaxy_instance):
        """Test get_histories ids_only mode returns simplified list"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_histories_fn(ids_only=True)

            assert isinstance(result["histories"], list)
            assert len(result["histories"]) == 2
            assert result["histories"][0] == {"id": "test_history_1", "name": "Test History 1"}
            assert result["histories"][1] == {"id": "test_history_2", "name": "Test History 2"}
            assert result["ids_only"] is True

    def test_get_history_details_fn(self, mock_galaxy_instance):
        """Test get_history_details with valid ID"""
        mock_galaxy_instance.histories.show_history.side_effect = [
            {"id": "test_history_1", "name": "Test History 1", "state": "ok"},  # History info
            ["dataset1", "dataset2"],  # All contents for count
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_history_details_fn("test_history_1")

            assert "history" in result
            assert "contents_summary" in result
            assert result["history"]["id"] == "test_history_1"
            assert result["contents_summary"]["total_items"] == 2
            assert "get_history_contents(" in result["contents_summary"]["note"]
            assert "create_time-dsc" in result["contents_summary"]["note"]

    def test_get_history_details_with_dict_string(self, mock_galaxy_instance):
        """Test get_history_details treats dict string as regular ID (should fail)"""
        mock_galaxy_instance.histories.show_history.side_effect = Exception("404 Not Found")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # User passes string representation of dict - should be treated as invalid ID
            dict_string = "{'id': 'test_history_1', 'name': 'Test History 1'}"

            with pytest.raises(ValueError, match="History ID .* not found"):
                get_history_details_fn(dict_string)

    def test_get_history_details_invalid_id(self, mock_galaxy_instance):
        """Test get_history_details with invalid ID"""
        mock_galaxy_instance.histories.show_history.side_effect = Exception("Invalid history ID")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Check for the exact validation error or API error
            with pytest.raises(ValueError) as exc_info:
                get_history_details_fn("invalid_id")

            # Either validation error or API error is acceptable
            assert "Failed to get history details" in str(exc_info.value)

    def test_not_connected_errors(self):
        """Test operations fail when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(Exception):
                get_histories_fn()

            with pytest.raises(Exception):
                get_histories_fn(ids_only=True)

            with pytest.raises(Exception):
                get_history_details_fn("any_id")

    def test_get_history_contents_paginated(self, mock_galaxy_instance):
        """Test get_history_contents with pagination"""
        # Mock datasets API calls
        mock_galaxy_instance.datasets.get_datasets.side_effect = [
            [
                {"id": "dataset1", "visible": True, "deleted": False},
                {"id": "dataset2", "visible": True, "deleted": False},
            ],  # Paginated contents
            [
                {"id": "dataset1", "visible": True, "deleted": False},
                {"id": "dataset2", "visible": True, "deleted": False},
                {"id": "dataset3", "visible": True, "deleted": False},
                {"id": "dataset4", "visible": True, "deleted": False},
                {"id": "dataset5", "visible": True, "deleted": False},
            ],  # Total contents
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1", limit=2, offset=0)

            assert "contents" in result
            assert "pagination" in result
            assert result["history_id"] == "test_history_1"
            assert len(result["contents"]) == 2
            assert result["pagination"]["total_items"] == 5
            assert result["pagination"]["current_page"] == 1
            assert result["pagination"]["has_next"] is True
            assert result["pagination"]["has_previous"] is False
            assert result["pagination"]["next_offset"] == 2

    def test_get_history_contents_no_pagination(self, mock_galaxy_instance):
        """Test get_history_contents without pagination (default)"""
        mock_galaxy_instance.datasets.get_datasets.side_effect = [
            [
                {"id": "dataset1", "visible": True, "deleted": False},
                {"id": "dataset2", "visible": True, "deleted": False},
                {"id": "dataset3", "visible": True, "deleted": False},
            ],  # All contents
            [
                {"id": "dataset1", "visible": True, "deleted": False},
                {"id": "dataset2", "visible": True, "deleted": False},
                {"id": "dataset3", "visible": True, "deleted": False},
            ],  # Total contents (same)
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1")

            assert len(result["contents"]) == 3
            assert result["pagination"]["total_items"] == 3
            assert result["pagination"]["has_next"] is False
            assert result["pagination"]["has_previous"] is False

    def test_get_history_contents_most_recent_first(self, mock_galaxy_instance):
        """Test get_history_contents with ordering for most recent datasets"""
        mock_galaxy_instance.datasets.get_datasets.side_effect = [
            [
                {
                    "id": "dataset5",
                    "create_time": "2023-12-01T10:00:00",
                    "visible": True,
                    "deleted": False,
                },
                {
                    "id": "dataset4",
                    "create_time": "2023-11-30T09:00:00",
                    "visible": True,
                    "deleted": False,
                },
            ],  # Most recent first
            [
                {
                    "id": "dataset5",
                    "create_time": "2023-12-01T10:00:00",
                    "visible": True,
                    "deleted": False,
                },
                {
                    "id": "dataset4",
                    "create_time": "2023-11-30T09:00:00",
                    "visible": True,
                    "deleted": False,
                },
                {
                    "id": "dataset3",
                    "create_time": "2023-11-29T08:00:00",
                    "visible": True,
                    "deleted": False,
                },
            ],  # All contents
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1", limit=2, order="create_time-dsc")

            assert len(result["contents"]) == 2
            assert result["contents"][0]["id"] == "dataset5"  # Most recent first
            assert result["contents"][1]["id"] == "dataset4"
            assert result["pagination"]["total_items"] == 3

            # Verify datasets API was called with correct order parameter for pagination
            calls = mock_galaxy_instance.datasets.get_datasets.call_args_list
            assert len(calls) == 2
            # First call should be the paginated call
            assert calls[0] == call(
                limit=2, offset=0, history_id="test_history_1", order="create_time-dsc"
            )
            # Second call should be for total count
            assert calls[1] == call(history_id="test_history_1", order="create_time-dsc")
