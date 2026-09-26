"""
Test dataset-related operations
"""

from unittest.mock import Mock, patch

import pytest

from .test_helpers import (
    download_dataset_fn,
    galaxy_state,
    get_collection_details_fn,
    get_dataset_details_fn,
    upload_file_fn,
    upload_file_from_url_fn,
)


def _serves_text(gi, item_data, truncated=False):
    """Galaxy's GET /api/datasets/{id}/get_content_as_text answer."""
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "item_data": item_data,
        "truncated": truncated,
        "item_url": "/datasets/x/display",
    }
    gi.make_get_request.return_value = response
    return response


class TestDatasetOperations:
    """Test dataset operations"""

    def test_upload_file(self, mock_galaxy_instance):
        """Test file upload to history"""
        mock_galaxy_instance.tools.upload_file.return_value = {
            "outputs": [{"id": "new_dataset_1", "name": "test.txt"}]
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with patch("os.path.exists", return_value=True):
                result = upload_file_fn("/path/to/test.txt", "test_history_1")

                assert result.success is True
                assert result.data["outputs"][0]["id"] == "new_dataset_1"
                assert result.data["outputs"][0]["name"] == "test.txt"
                mock_galaxy_instance.tools.upload_file.assert_called_once()

    def test_upload_file_not_found(self, mock_galaxy_instance):
        """Test upload with non-existent file"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(ValueError, match="File not found"):
                    upload_file_fn("/nonexistent/file.txt", "test_history_1")

    def test_get_dataset_details_with_preview(self, mock_galaxy_instance):
        """Test getting dataset details with preview"""
        dataset_id = "dataset123"

        # Mock dataset info
        mock_dataset_info = {
            "id": dataset_id,
            "name": "test_data.txt",
            "state": "ok",
            "extension": "txt",
            "file_size": 1024,
        }

        mock_galaxy_instance.datasets.show_dataset.return_value = mock_dataset_info
        _serves_text(mock_galaxy_instance, "line1\nline2\nline3\nline4\nline5\n")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_dataset_details_fn(dataset_id, include_preview=True, preview_lines=3)

            assert result.success is True
            assert result.data["dataset_id"] == dataset_id
            assert result.data["dataset"]["name"] == "test_data.txt"
            assert result.data["preview"]["lines"] == "line1\nline2\nline3"
            assert result.data["preview"]["preview_lines"] == 3
            assert result.data["preview"]["truncated"] is True
            assert result.data["preview"]["content_truncated_by_galaxy"] is False
            # No total: what came back is a peek, and a line count taken from it
            # reads as a count of the dataset.
            assert set(result.data["preview"]) == {
                "lines",
                "preview_lines",
                "truncated",
                "content_truncated_by_galaxy",
            }

            mock_galaxy_instance.datasets.show_dataset.assert_called_once_with(dataset_id)
            # The whole dataset is never fetched for a preview any more.
            mock_galaxy_instance.datasets.download_dataset.assert_not_called()

    def test_get_dataset_details_preview_asks_the_text_route(self, mock_galaxy_instance):
        """The URL matters: this is the route that stops at 1 MB."""
        mock_galaxy_instance.datasets.show_dataset.return_value = {
            "id": "dataset123",
            "name": "test_data.txt",
            "state": "ok",
        }
        _serves_text(mock_galaxy_instance, "a\nb\n")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            get_dataset_details_fn("dataset123", include_preview=True)

        mock_galaxy_instance.make_get_request.assert_called_once_with(
            "http://localhost:8080/api/datasets/dataset123/get_content_as_text"
        )

    def test_get_dataset_details_passes_galaxys_own_truncation_flag_through(
        self, mock_galaxy_instance
    ):
        """truncated is our line slice; content_truncated_by_galaxy is Galaxy's flag.

        The two are different facts and both are reported. Galaxy's own flag is no
        guarantee either: it comes from the stored byte size while the read counts
        decoded characters, so compressed text can be clipped with it false and
        multibyte text can be complete with it true. That is why the preview reports
        no totals of its own.
        """
        mock_galaxy_instance.datasets.show_dataset.return_value = {
            "id": "dataset123",
            "name": "big.txt",
            "state": "ok",
        }
        _serves_text(mock_galaxy_instance, "a\nb\n", truncated=True)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_dataset_details_fn("dataset123", include_preview=True, preview_lines=10)

        preview = result.data["preview"]
        assert preview["content_truncated_by_galaxy"] is True
        # Our own line slice did not cut anything; Galaxy's byte cap did.
        assert preview["truncated"] is False
        assert preview["lines"] == "a\nb\n"

    def test_get_dataset_details_preview_failure_is_not_fatal(self, mock_galaxy_instance):
        """A preview that cannot be fetched still leaves the metadata usable."""
        mock_galaxy_instance.datasets.show_dataset.return_value = {
            "id": "dataset123",
            "name": "test_data.txt",
            "state": "ok",
        }
        mock_galaxy_instance.make_get_request.side_effect = Exception("boom")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_dataset_details_fn("dataset123", include_preview=True)

        assert result.success is True
        assert result.data["dataset"]["name"] == "test_data.txt"
        assert "Preview unavailable" in result.data["preview"]["error"]
        assert result.data["preview"]["lines"] is None

    def test_get_dataset_details_no_preview(self, mock_galaxy_instance):
        """Test getting dataset details without preview"""
        dataset_id = "dataset123"

        mock_dataset_info = {
            "id": dataset_id,
            "name": "test_data.txt",
            "state": "ok",
            "extension": "txt",
        }

        mock_galaxy_instance.datasets.show_dataset.return_value = mock_dataset_info

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_dataset_details_fn(dataset_id, include_preview=False)

            assert result.success is True
            assert result.data["dataset_id"] == dataset_id
            assert result.data["dataset"]["name"] == "test_data.txt"
            assert "preview" not in result.data

            # Should not call download_dataset for preview
            mock_galaxy_instance.datasets.download_dataset.assert_not_called()

    def test_get_dataset_details_binary_file(self, mock_galaxy_instance):
        """Test getting dataset details with binary content"""
        dataset_id = "dataset123"

        mock_dataset_info = {
            "id": dataset_id,
            "name": "test_data.bin",
            "state": "ok",
            "extension": "bin",
        }

        mock_galaxy_instance.datasets.show_dataset.return_value = mock_dataset_info
        # hda_manager.text_data returns nothing at all for a datatype that is not text,
        # so Galaxy answers with item_data null rather than bytes to guess at.
        _serves_text(mock_galaxy_instance, None)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_dataset_details_fn(dataset_id, include_preview=True)

            assert result.success is True
            assert result.data["dataset_id"] == dataset_id
            assert result.data["preview"]["lines"] is None
            assert "text datatypes only" in result.data["preview"]["error"]

    def test_download_dataset_with_file_path(self, mock_galaxy_instance):
        """Test dataset download to specific file path"""
        dataset_id = "dataset123"
        file_path = "/tmp/test_download.txt"

        mock_dataset_info = {
            "id": dataset_id,
            "name": "test_data.txt",
            "state": "ok",
            "extension": "txt",
            "file_size": 1024,
        }

        mock_galaxy_instance.datasets.show_dataset.return_value = mock_dataset_info
        mock_galaxy_instance.datasets.download_dataset.return_value = file_path

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=1024):
                    result = download_dataset_fn(dataset_id, file_path=file_path)

                    assert result.success is True
                    assert result.data["dataset_id"] == dataset_id
                    assert result.data["file_path"] == file_path
                    assert result.data["file_size"] == 1024
                    assert result.data["dataset_info"]["name"] == "test_data.txt"

                    mock_galaxy_instance.datasets.download_dataset.assert_called_once_with(
                        dataset_id,
                        file_path=file_path,
                        use_default_filename=False,
                        require_ok_state=True,
                    )

    def test_download_dataset_default_filename(self, mock_galaxy_instance):
        """Test dataset download to memory (no file path specified)"""
        dataset_id = "dataset123"

        mock_dataset_info = {
            "id": dataset_id,
            "name": "test_data",
            "state": "ok",
            "extension": "txt",
            "file_size": 1024,
        }

        mock_content = b"test file content"

        mock_galaxy_instance.datasets.show_dataset.return_value = mock_dataset_info
        mock_galaxy_instance.datasets.download_dataset.return_value = mock_content

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = download_dataset_fn(dataset_id)

            assert result.success is True
            assert result.data["dataset_id"] == dataset_id
            assert result.data["file_path"] is None  # No file saved
            assert result.data["suggested_filename"] == "test_data.txt"
            assert result.data["content_available"] is True
            assert result.data["file_size"] == len(mock_content)
            assert result.data["dataset_info"]["name"] == "test_data"
            assert "memory" in result.data["note"]

            # Verify bioblend was called with use_default_filename=False
            mock_galaxy_instance.datasets.download_dataset.assert_called_once_with(
                dataset_id, use_default_filename=False, require_ok_state=True
            )

    def test_download_dataset_not_ok_state(self, mock_galaxy_instance):
        """Test download fails when dataset not in ok state"""
        dataset_id = "dataset123"

        mock_dataset_info = {
            "id": dataset_id,
            "name": "test_data.txt",
            "state": "running",
            "extension": "txt",
        }

        mock_galaxy_instance.datasets.show_dataset.return_value = mock_dataset_info

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Dataset .* is in state 'running', not 'ok'"):
                download_dataset_fn(dataset_id)

    def test_upload_file_from_url(self, mock_galaxy_instance):
        """Test file upload from URL"""
        url = "https://example.com/data.fasta"
        history_id = "test_history_1"

        mock_galaxy_instance.tools.put_url.return_value = {
            "outputs": [{"id": "new_dataset_1", "name": "data.fasta"}]
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = upload_file_from_url_fn(
                url, history_id=history_id, file_type="fasta", dbkey="hg38"
            )

            assert result.success is True
            assert result.data["outputs"][0]["id"] == "new_dataset_1"
            assert result.data["outputs"][0]["name"] == "data.fasta"
            mock_galaxy_instance.tools.put_url.assert_called_once_with(
                url, history_id=history_id, file_type="fasta", dbkey="hg38"
            )

    def test_upload_file_from_url_with_custom_name(self, mock_galaxy_instance):
        """Test file upload from URL with custom filename"""
        url = "https://example.com/data.txt"

        mock_galaxy_instance.tools.put_url.return_value = {
            "outputs": [{"id": "new_dataset_1", "name": "custom_name.txt"}]
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = upload_file_from_url_fn(url, file_name="custom_name.txt", file_type="tabular")

            assert result.success is True
            assert result.data["outputs"][0]["name"] == "custom_name.txt"
            mock_galaxy_instance.tools.put_url.assert_called_once_with(
                url,
                history_id=None,
                file_type="tabular",
                dbkey="?",
                file_name="custom_name.txt",
            )

    def test_upload_file_from_url_error(self, mock_galaxy_instance):
        """Test error handling for URL upload"""
        mock_galaxy_instance.tools.put_url.side_effect = Exception("Upload failed")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Upload file from URL failed"):
                upload_file_from_url_fn("https://example.com/data.txt")

    def test_dataset_operations_not_connected(self):
        """Test dataset operations fail when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                upload_file_fn("/path/to/file.txt", "history_1")

            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                upload_file_from_url_fn("https://example.com/data.txt")

            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                get_dataset_details_fn("dataset123")

            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                download_dataset_fn("dataset123")

    def test_get_collection_details_list_collection(self, mock_galaxy_instance):
        """Test getting details of a list-type dataset collection"""
        mock_galaxy_instance.dataset_collections = Mock()
        collection_id = "collection123"

        # Mock collection info with 3 elements
        mock_collection_info = {
            "id": collection_id,
            "name": "My Sample Collection",
            "collection_type": "list",
            "element_count": 3,
            "populated": True,
            "state": "ok",
            "elements": [
                {
                    "element_identifier": "sample1",
                    "element_type": "hda",
                    "object": {
                        "id": "dataset1",
                        "name": "sample1.fastq",
                        "state": "ok",
                        "extension": "fastqsanger",
                        "file_size": 12345,
                    },
                },
                {
                    "element_identifier": "sample2",
                    "element_type": "hda",
                    "object": {
                        "id": "dataset2",
                        "name": "sample2.fastq",
                        "state": "ok",
                        "extension": "fastqsanger",
                        "file_size": 23456,
                    },
                },
                {
                    "element_identifier": "sample3",
                    "element_type": "hda",
                    "object": {
                        "id": "dataset3",
                        "name": "sample3.fastq",
                        "state": "ok",
                        "extension": "fastqsanger",
                        "file_size": 34567,
                    },
                },
            ],
        }

        mock_galaxy_instance.dataset_collections.show_dataset_collection.return_value = (
            mock_collection_info
        )

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_collection_details_fn(collection_id)

            assert result.success is True
            assert result.data["collection_id"] == collection_id
            assert result.data["history_content_type"] == "dataset_collection"
            assert result.data["collection"]["name"] == "My Sample Collection"
            assert result.data["collection"]["collection_type"] == "list"
            assert result.data["collection"]["element_count"] == 3
            assert result.data["elements_truncated"] is False
            assert len(result.data["elements"]) == 3

            # Check first element structure
            assert result.data["elements"][0]["element_identifier"] == "sample1"
            assert result.data["elements"][0]["object_id"] == "dataset1"
            assert result.data["elements"][0]["name"] == "sample1.fastq"
            assert result.data["elements"][0]["state"] == "ok"

            mock_galaxy_instance.dataset_collections.show_dataset_collection.assert_called_once_with(
                collection_id, instance_type="history"
            )

    def test_get_collection_details_truncation(self, mock_galaxy_instance):
        """Test collection details with truncation when exceeding max_elements"""
        mock_galaxy_instance.dataset_collections = Mock()
        collection_id = "large_collection"

        # Create a collection with 150 elements
        elements = []
        for i in range(150):
            elements.append(
                {
                    "element_identifier": f"sample{i}",
                    "element_type": "hda",
                    "object": {
                        "id": f"dataset{i}",
                        "name": f"sample{i}.fastq",
                        "state": "ok",
                        "extension": "fastqsanger",
                        "file_size": 10000 + i,
                    },
                }
            )

        mock_collection_info = {
            "id": collection_id,
            "name": "Large Collection",
            "collection_type": "list",
            "element_count": 150,
            "populated": True,
            "state": "ok",
            "elements": elements,
        }

        mock_galaxy_instance.dataset_collections.show_dataset_collection.return_value = (
            mock_collection_info
        )

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_collection_details_fn(collection_id, max_elements=50)

            assert result.success is True
            assert result.data["collection"]["element_count"] == 150
            assert result.data["elements_truncated"] is True
            assert len(result.data["elements"]) == 50
            assert result.data["elements"][0]["element_identifier"] == "sample0"
            assert result.data["elements"][49]["element_identifier"] == "sample49"

    def test_get_dataset_details_with_collection_id(self, mock_galaxy_instance):
        """Test that get_dataset_details raises helpful error when given a collection ID"""
        mock_galaxy_instance.dataset_collections = Mock()
        collection_id = "collection123"

        # Mock show_dataset to fail (not a dataset)
        mock_galaxy_instance.datasets.show_dataset.side_effect = Exception("Dataset not found")

        # Mock show_dataset_collection to succeed (it IS a collection)
        mock_galaxy_instance.dataset_collections.show_dataset_collection.return_value = {
            "id": collection_id,
            "name": "My Sample Collection",
            "collection_type": "list",
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(
                ValueError,
                match=(
                    "is a dataset collection, not a dataset.*My Sample Collection.*"
                    "get_collection_details"
                ),
            ):
                get_dataset_details_fn(collection_id)

            # Verify both API calls were made
            mock_galaxy_instance.datasets.show_dataset.assert_called_once_with(collection_id)
            mock_galaxy_instance.dataset_collections.show_dataset_collection.assert_called_once_with(
                collection_id, instance_type="history"
            )
