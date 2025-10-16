"""
Tests for unified search and fetch tools.
"""

from unittest.mock import patch

import pytest

from .test_helpers import fetch_fn, galaxy_state, search_fn


class TestSearchAndFetch:
    """Verify search and fetch integration."""

    def test_search_returns_results_for_all_sources(self, mock_galaxy_instance):
        """Search should return hits for each supported resource type."""
        with patch.dict(
            galaxy_state,
            {
                "connected": True,
                "gi": mock_galaxy_instance,
                "url": "https://galaxy.example",
            },
        ):
            response = search_fn(term="Test")

        assert response["term"] == "Test"
        assert response["per_source_limit"] == 5
        assert response["total"] > 0

        sources_with_hits = {item["source"] for item in response["results"]}
        for expected in {
            "histories",
            "tools",
            "workflows",
            "datasets",
            "dataset_collections",
            "libraries",
            "library_datasets",
            "jobs",
            "invocations",
        }:
            assert expected in sources_with_hits

        tool_results = [item for item in response["results"] if item["source"] == "tools"]
        assert tool_results, "Expected at least one tools search result."
        for item in tool_results:
            metadata = item.get("metadata", {})
            versions = metadata.get("versions", [])
            assert versions, "Grouped tool result should include version metadata."
            for version in versions:
                assert version.get("id"), "Version entries must include a Galaxy tool identifier."
                assert version.get("resource_id"), "Version entries must expose a resource identifier."

        for item in response["results"]:
            assert ":" in item["id"]
            assert item["score"] >= 0
            assert item["name"]

    def test_fetch_dispatches_to_correct_client(self, mock_galaxy_instance):
        """Fetch should delegate to the specific Galaxy client."""
        with patch.dict(
            galaxy_state,
            {
                "connected": True,
                "gi": mock_galaxy_instance,
                "url": "https://galaxy.example",
            },
        ):
            response = search_fn(term="Test")

            # Build a representative item per source
            representative: dict[str, str] = {}
            for item in response["results"]:
                representative.setdefault(item["source"], item["id"])

            # Reset call history to test fetch execution only
            mock_galaxy_instance.histories.show_history.reset_mock()
            mock_galaxy_instance.tools.show_tool.reset_mock()
            mock_galaxy_instance.workflows.show_workflow.reset_mock()
            mock_galaxy_instance.datasets.show_dataset.reset_mock()
            mock_galaxy_instance.dataset_collections.show_dataset_collection.reset_mock()
            mock_galaxy_instance.libraries.show_library.reset_mock()
            mock_galaxy_instance.libraries.show_dataset.reset_mock()
            mock_galaxy_instance.jobs.show_job.reset_mock()
            mock_galaxy_instance.invocations.show_invocation.reset_mock()

            for expected in [
                "histories",
                "tools",
                "workflows",
                "datasets",
                "dataset_collections",
                "libraries",
                "library_datasets",
                "jobs",
                "invocations",
            ]:
                assert expected in representative, f"Missing representative result for {expected}"

            # Histories
            metadata = fetch_fn(resource_id=representative["histories"])
            assert metadata["source"] == "histories"
            mock_galaxy_instance.histories.show_history.assert_called_once()

            # Tools
            tool_fetch = fetch_fn(resource_id=representative["tools"])
            assert tool_fetch["source"] == "tools"
            versions = tool_fetch["metadata"]["versions"]
            assert versions
            assert mock_galaxy_instance.tools.show_tool.call_count >= len(versions)

            mock_galaxy_instance.tools.show_tool.reset_mock()
            first_version_resource = versions[0]["resource_id"]
            version_fetch = fetch_fn(resource_id=first_version_resource)
            assert len(version_fetch["metadata"]["versions"]) == 1
            mock_galaxy_instance.tools.show_tool.assert_called_once()
            mock_galaxy_instance.tools.show_tool.reset_mock()

            # Workflows
            fetch_fn(resource_id=representative["workflows"])
            mock_galaxy_instance.workflows.show_workflow.assert_called_once()

            # Datasets
            fetch_fn(resource_id=representative["datasets"])
            mock_galaxy_instance.datasets.show_dataset.assert_called_once()

            # Dataset collections
            fetch_fn(resource_id=representative["dataset_collections"])
            mock_galaxy_instance.dataset_collections.show_dataset_collection.assert_called_once()

            # Libraries
            fetch_fn(resource_id=representative["libraries"])
            mock_galaxy_instance.libraries.show_library.assert_called_once()

            # Library datasets
            fetch_fn(resource_id=representative["library_datasets"])
            mock_galaxy_instance.libraries.show_dataset.assert_called_once()

            # Jobs
            fetch_fn(resource_id=representative["jobs"])
            mock_galaxy_instance.jobs.show_job.assert_called_once()

            # Invocations
            fetch_fn(resource_id=representative["invocations"])
            mock_galaxy_instance.invocations.show_invocation.assert_called_once()

    def test_fetch_unknown_resource(self, mock_galaxy_instance):
        """Fetch should raise when the identifier is malformed or unsupported."""
        with patch.dict(
            galaxy_state,
            {
                "connected": True,
                "gi": mock_galaxy_instance,
                "url": "https://galaxy.example",
            },
        ):
            with pytest.raises(ValueError):
                fetch_fn(resource_id="invalid")

            with pytest.raises(ValueError):
                fetch_fn(resource_id="unknown:identifier")
