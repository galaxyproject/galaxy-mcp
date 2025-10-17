"""
Tests for the individual Galaxy search tools.
"""

from collections.abc import Callable
from unittest.mock import patch

import pytest

from .test_helpers import (
    galaxy_state,
    search_dataset_collections_fn,
    search_datasets_fn,
    search_histories_fn,
    search_invocations_fn,
    search_jobs_fn,
    search_libraries_fn,
    search_library_datasets_fn,
    search_tools_fn,
    search_workflows_fn,
)

SearchFn = Callable[[str], dict]


@pytest.mark.parametrize(
    ("search_fn", "source"),
    [
        (search_histories_fn, "histories"),
        (search_tools_fn, "tools"),
        (search_workflows_fn, "workflows"),
        (search_datasets_fn, "datasets"),
        (search_dataset_collections_fn, "dataset_collections"),
        (search_libraries_fn, "libraries"),
        (search_library_datasets_fn, "library_datasets"),
        (search_jobs_fn, "jobs"),
        (search_invocations_fn, "invocations"),
    ],
)
def test_search_functions_return_results(
    search_fn: SearchFn,
    source: str,
    mock_galaxy_instance,
) -> None:
    """Each search tool should return structured matches containing details."""
    with patch.dict(
        galaxy_state,
        {
            "connected": True,
            "gi": mock_galaxy_instance,
            "url": "https://galaxy.example",
        },
    ):
        response = search_fn(term="Test")

    assert response["source"] == source
    assert response["limit"] >= 1
    assert response["total"] == len(response["matches"])
    assert response["matches"], f"Expected at least one {source} match"

    for match in response["matches"]:
        assert match["source"] == source
        assert match["score"] >= 0
        assert match["resource_id"].startswith(f"{source}:")
        assert match["details"], "Expected search result to include details payload"


def test_search_tools_includes_versions_and_citations(mock_galaxy_instance) -> None:
    """Tool search should surface version list and deduplicated citations."""
    with patch.dict(
        galaxy_state,
        {
            "connected": True,
            "gi": mock_galaxy_instance,
            "url": "https://galaxy.example",
        },
    ):
        response = search_tools_fn(term="Test Tool")

    assert response["matches"], "Expected at least one tool match"
    first = response["matches"][0]
    details = first["details"]
    versions = details["versions"]
    assert versions, "Tool details should include version metadata"
    citation_titles = [citation["title"] for citation in details["citations"]]
    assert len(citation_titles) == len(set(citation_titles)), "Citations should be deduplicated"
