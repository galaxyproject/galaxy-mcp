"""
Test history-related operations
"""

from unittest.mock import patch

import pytest

from .test_helpers import (
    galaxy_state,
    get_histories_fn,
    get_history_details_fn,
    list_history_ids_fn,
)


class TestHistoryOperations:
    """Test history operations"""

    def test_get_histories_fn(self, mcp_server_instance, mock_galaxy_instance):
        """Test get_histories returns paginated results"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_histories_fn()

            assert result.success is True
            assert result.count == 2
            assert len(result.data) == 2
            assert result.data[0]["id"] == "test_history_1"
            assert result.data[0]["name"] == "Test History 1"
            # No pagination when limit is not specified
            assert result.pagination is None

    def test_get_histories_empty(self, mock_galaxy_instance):
        """Test get_histories with no histories"""
        mock_galaxy_instance.histories.get_histories.return_value = []

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_histories_fn()

            assert result.success is True
            assert result.data == []
            assert result.count == 0

    def test_list_history_ids_fn(self, mock_galaxy_instance):
        """Test list_history_ids returns simplified list"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = list_history_ids_fn()

            assert result.success is True
            assert result.count == 2
            assert len(result.data) == 2
            assert result.data[0] == {"id": "test_history_1", "name": "Test History 1"}
            assert result.data[1] == {"id": "test_history_2", "name": "Test History 2"}

    def test_get_history_details_fn(self, mock_galaxy_instance):
        """Test get_history_details with valid ID"""
        mock_galaxy_instance.histories.show_history.side_effect = [
            {"id": "test_history_1", "name": "Test History 1", "state": "ok"},  # History info
            ["dataset1", "dataset2"],  # All contents for count
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_history_details_fn("test_history_1")

            assert result.success is True
            assert "history" in result.data
            assert "contents_summary" in result.data
            assert result.data["history"]["id"] == "test_history_1"
            assert result.data["contents_summary"]["total_items"] == 2
            assert "get_history_contents(" in result.data["contents_summary"]["note"]
            assert "create_time-dsc" in result.data["contents_summary"]["note"]

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
                list_history_ids_fn()

            with pytest.raises(Exception):
                get_history_details_fn("any_id")

    def test_get_history_contents_paginated(self, mock_galaxy_instance):
        """Test get_history_contents with pagination"""
        # Mock show_history to return all contents
        mock_galaxy_instance.histories.show_history.return_value = [
            {"id": "dataset1", "hid": 1, "visible": True, "deleted": False},
            {"id": "dataset2", "hid": 2, "visible": True, "deleted": False},
            {"id": "dataset3", "hid": 3, "visible": True, "deleted": False},
            {"id": "dataset4", "hid": 4, "visible": True, "deleted": False},
            {"id": "dataset5", "hid": 5, "visible": True, "deleted": False},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1", limit=2, offset=0)

            assert result.success is True
            assert "contents" in result.data
            assert result.data["history_id"] == "test_history_1"
            assert len(result.data["contents"]) == 2
            assert result.pagination is not None
            assert result.pagination.total_items == 5
            assert result.pagination.has_next is True
            assert result.pagination.has_previous is False
            assert result.pagination.next_offset == 2

    def test_get_history_contents_no_pagination(self, mock_galaxy_instance):
        """Test get_history_contents without pagination (default)"""
        mock_galaxy_instance.histories.show_history.return_value = [
            {"id": "dataset1", "hid": 1, "visible": True, "deleted": False},
            {"id": "dataset2", "hid": 2, "visible": True, "deleted": False},
            {"id": "dataset3", "hid": 3, "visible": True, "deleted": False},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1")

            assert result.success is True
            assert len(result.data["contents"]) == 3
            assert result.pagination is not None
            assert result.pagination.total_items == 3
            assert result.pagination.has_next is False
            assert result.pagination.has_previous is False

    def test_get_history_contents_most_recent_first(self, mock_galaxy_instance):
        """Test get_history_contents with ordering for most recent datasets"""
        mock_galaxy_instance.histories.show_history.return_value = [
            {
                "id": "dataset3",
                "hid": 3,
                "create_time": "2023-11-29T08:00:00",
                "visible": True,
                "deleted": False,
            },
            {
                "id": "dataset4",
                "hid": 4,
                "create_time": "2023-11-30T09:00:00",
                "visible": True,
                "deleted": False,
            },
            {
                "id": "dataset5",
                "hid": 5,
                "create_time": "2023-12-01T10:00:00",
                "visible": True,
                "deleted": False,
            },
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1", limit=2, order="create_time-dsc")

            assert result.success is True
            assert len(result.data["contents"]) == 2
            assert result.data["contents"][0]["id"] == "dataset5"  # Most recent first
            assert result.data["contents"][1]["id"] == "dataset4"
            assert result.pagination.total_items == 3

            # Verify show_history was called
            mock_galaxy_instance.histories.show_history.assert_called_once_with(
                "test_history_1", contents=True
            )

    def test_get_history_contents_with_collections(self, mock_galaxy_instance):
        """Test get_history_contents returns both datasets and collections with proper type flags"""
        # Mock show_history to return both datasets and collections
        mock_galaxy_instance.histories.show_history.return_value = [
            {
                "id": "dataset1",
                "name": "My Dataset",
                "hid": 1,
                "visible": True,
                "deleted": False,
                "state": "ok",
            },
            {
                "id": "collection1",
                "name": "My Collection",
                "hid": 2,
                "collection_type": "list",
                "visible": True,
                "deleted": False,
                "state": "ok",
            },
            {
                "id": "dataset2",
                "name": "Another Dataset",
                "hid": 3,
                "visible": True,
                "deleted": False,
                "state": "ok",
            },
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            from tests.test_helpers import get_history_contents_fn

            result = get_history_contents_fn("test_history_1")

            assert result.success is True
            assert len(result.data["contents"]) == 3
            assert result.pagination.total_items == 3

            # Check that datasets are marked correctly
            assert result.data["contents"][0]["history_content_type"] == "dataset"
            assert result.data["contents"][0]["id"] == "dataset1"

            # Check that collections are marked correctly
            assert result.data["contents"][1]["history_content_type"] == "dataset_collection"
            assert result.data["contents"][1]["id"] == "collection1"
            assert result.data["contents"][1]["collection_type"] == "list"

            # Check third item is also a dataset
            assert result.data["contents"][2]["history_content_type"] == "dataset"
            assert result.data["contents"][2]["id"] == "dataset2"

            mock_galaxy_instance.histories.show_history.assert_called_once_with(
                "test_history_1", contents=True
            )
