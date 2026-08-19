"""Tests for user-defined tool (UDT) operations."""

import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from galaxy_mcp.server import _shape_biocontainer_recommendation

from .test_helpers import galaxy_state, recommend_biocontainer_fn, run_user_tool_fn


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


def _fake_recommend_tree(recommend, verify):
    """Build a fake ``galaxy.tool_util.deps.mulled.recommend`` module tree so the
    tool's lazy import resolves without a real galaxy-tool-util install."""

    class FakePackageSpec:
        def __init__(self, name, version=None):
            self.name = name
            self.version = version

    recommend_mod = types.ModuleType("galaxy.tool_util.deps.mulled.recommend")
    recommend_mod.recommend_container = recommend
    recommend_mod.biocontainer_tag_built = verify
    recommend_mod.PackageSpec = FakePackageSpec

    tree = {
        "galaxy": types.ModuleType("galaxy"),
        "galaxy.tool_util": types.ModuleType("galaxy.tool_util"),
        "galaxy.tool_util.deps": types.ModuleType("galaxy.tool_util.deps"),
        "galaxy.tool_util.deps.mulled": types.ModuleType("galaxy.tool_util.deps.mulled"),
        "galaxy.tool_util.deps.mulled.recommend": recommend_mod,
    }
    tree["galaxy"].tool_util = tree["galaxy.tool_util"]
    tree["galaxy.tool_util"].deps = tree["galaxy.tool_util.deps"]
    tree["galaxy.tool_util.deps"].mulled = tree["galaxy.tool_util.deps.mulled"]
    tree["galaxy.tool_util.deps.mulled"].recommend = recommend_mod
    return tree


class TestRecommendBiocontainer:
    """recommend_biocontainer resolves a verified quay.io/biocontainers image."""

    def test_shape_recommendation(self):
        """The pure shaper flattens a ContainerRecommendation into the wire dict."""
        rec = SimpleNamespace(
            image="quay.io/biocontainers/samtools:1.17--h00cdaf9_0",
            found=True,
            match_quality=SimpleNamespace(value="exact_version"),
            source=SimpleNamespace(value="quay_single"),
            notes=["resolved against built tags"],
        )
        assert _shape_biocontainer_recommendation(rec, verified=True) == {
            "image": "quay.io/biocontainers/samtools:1.17--h00cdaf9_0",
            "found": True,
            "match_quality": "exact_version",
            "source": "quay_single",
            "notes": ["resolved against built tags"],
            "verified": True,
        }

    @pytest.mark.parametrize("verified", [False, None])
    def test_shape_passes_through_unverified(self, verified):
        """verified is reported verbatim -- False (absent) and None (uncheckable) differ."""
        rec = SimpleNamespace(
            image="quay.io/biocontainers/seaborn:0.13.2--pyhd8ed1ab_3",
            found=True,
            match_quality=SimpleNamespace(value="name_only"),
            source=SimpleNamespace(value="quay_mulled_v2"),
            notes=[],
        )
        shaped = _shape_biocontainer_recommendation(rec, verified=verified)
        assert shaped["verified"] is verified
        assert shaped["match_quality"] == "name_only"
        assert shaped["source"] == "quay_mulled_v2"

    def test_not_found_recommendation_shapes_cleanly(self):
        """A no-match recommendation still shapes -- image is null, not an exception."""
        rec = SimpleNamespace(
            image=None,
            found=False,
            match_quality=SimpleNamespace(value="not_found"),
            source=SimpleNamespace(value="quay_single"),
            notes=["no built tag for the requested packages"],
        )
        shaped = _shape_biocontainer_recommendation(rec, verified=None)
        assert shaped["image"] is None
        assert shaped["found"] is False
        assert shaped["verified"] is None

    def test_resolves_and_parses_versions(self):
        """'name=version' entries become PackageSpecs; the resolved image is shaped and verified."""
        calls = {}

        def fake_recommend(specs):
            calls["specs"] = [(s.name, s.version) for s in specs]
            return SimpleNamespace(
                image="quay.io/biocontainers/samtools:1.17--h00cdaf9_0",
                found=True,
                match_quality=SimpleNamespace(value="exact_version"),
                source=SimpleNamespace(value="single"),
                notes=[],
            )

        def fake_verify(image):
            calls["verified_image"] = image
            return True

        tree = _fake_recommend_tree(fake_recommend, fake_verify)
        with patch.dict(sys.modules, tree):
            result = recommend_biocontainer_fn(["samtools=1.17", "bwa"])

        assert result.success is True
        assert result.data["image"] == "quay.io/biocontainers/samtools:1.17--h00cdaf9_0"
        assert result.data["match_quality"] == "exact_version"
        assert result.data["verified"] is True
        assert calls["specs"] == [("samtools", "1.17"), ("bwa", None)]
        assert calls["verified_image"] == "quay.io/biocontainers/samtools:1.17--h00cdaf9_0"

    def test_no_image_found(self):
        """A miss reports found=False, a null image, and verified stays None."""

        def fake_recommend(specs):
            return SimpleNamespace(
                image=None,
                found=False,
                match_quality=SimpleNamespace(value="not_found"),
                source=SimpleNamespace(value="none"),
                notes=["no biocontainer for these packages"],
            )

        def fake_verify(image):  # pragma: no cover - not reached when image is None
            raise AssertionError("verify should not run without an image")

        tree = _fake_recommend_tree(fake_recommend, fake_verify)
        with patch.dict(sys.modules, tree):
            result = recommend_biocontainer_fn(["definitely-not-a-package"])

        assert result.success is True
        assert result.data["image"] is None
        assert result.data["found"] is False
        assert result.data["verified"] is None

    def test_empty_packages_rejected_before_dependency(self):
        """An empty list is rejected before the optional dependency is imported, so a
        stock install reports the bad request rather than a missing dependency. (No
        fake module tree here -- that's the point: the guard must fire pre-import.)"""
        with pytest.raises(ValueError, match="at least one conda package"):
            recommend_biocontainer_fn([])

    def test_malformed_entry_rejected(self):
        """Entries with no package name (blank, or '=version') are rejected explicitly."""
        for bad in ["   ", "=1.2"]:
            with pytest.raises(ValueError, match="invalid package entry"):
                recommend_biocontainer_fn([bad])

    def test_graceful_without_extra(self):
        """Without the container-recommend extra, the tool explains how to install it."""
        try:
            has_recommender = (
                importlib.util.find_spec("galaxy.tool_util.deps.mulled.recommend") is not None
            )
        except ModuleNotFoundError:
            has_recommender = False
        if has_recommender:
            pytest.skip("galaxy-tool-util with mulled-recommend is installed")
        with pytest.raises(ValueError, match="galaxy-tool-util>=26.1"):
            recommend_biocontainer_fn(["samtools=1.17"])
