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
        """Test get_histories returns list of histories"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            histories = get_histories_fn()

            assert isinstance(histories, list)
            assert len(histories) == 2
            assert histories[0]["id"] == "test_history_1"
            assert histories[0]["name"] == "Test History 1"

    def test_get_histories_empty(self, mock_galaxy_instance):
        """Test get_histories with no histories"""
        mock_galaxy_instance.histories.get_histories.return_value = []

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            histories = get_histories_fn()

            assert histories == []

    def test_list_history_ids_fn(self, mock_galaxy_instance):
        """Test list_history_ids returns simplified list"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            history_ids = list_history_ids_fn()

            assert isinstance(history_ids, list)
            assert len(history_ids) == 2
            assert history_ids[0] == {"id": "test_history_1", "name": "Test History 1"}
            assert history_ids[1] == {"id": "test_history_2", "name": "Test History 2"}

    def test_get_history_details_fn(self, mock_galaxy_instance):
        """Test get_history_details with valid ID"""
        mock_galaxy_instance.histories.show_history.side_effect = [
            {"id": "test_history_1", "name": "Test History 1", "state": "ok"},
            ["dataset1", "dataset2"],  # Contents
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_history_details_fn("test_history_1")

            assert "history" in result
            assert "contents" in result
            assert result["history"]["id"] == "test_history_1"
            assert len(result["contents"]) == 2

    def test_get_history_details_with_dict_string(self, mock_galaxy_instance):
        """Test get_history_details handles dict string input"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # User passes string representation of dict
            dict_string = "{'id': 'test_history_1', 'name': 'Test History 1'}"

            msg = "Invalid history_id: expected a history ID string"
            with pytest.raises(ValueError, match=msg):
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
