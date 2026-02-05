"""Tests for the Galaxy MCP CLI (gxy)."""

import json
from unittest.mock import Mock, patch

import pytest
from galaxy_mcp.cli.config import _load_planemo_profile, list_profiles, load_profile
from galaxy_mcp.cli.main import app
from galaxy_mcp.server import GalaxyResult
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    """Set up mock environment for CLI tests."""
    monkeypatch.setenv("GALAXY_URL", "https://test.galaxy.com")
    monkeypatch.setenv("GALAXY_API_KEY", "test_api_key")


class TestCliHelp:
    """Test CLI help and basic functionality."""

    def test_help(self):
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Galaxy command-line interface" in result.stdout

    def test_tools_help(self):
        """Test tools subcommand help."""
        result = runner.invoke(app, ["tools", "--help"])
        assert result.exit_code == 0
        assert "search" in result.stdout
        assert "details" in result.stdout

    def test_history_help(self):
        """Test history subcommand help."""
        result = runner.invoke(app, ["history", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "create" in result.stdout

    def test_iwc_help(self):
        """Test iwc subcommand help."""
        result = runner.invoke(app, ["iwc", "--help"])
        assert result.exit_code == 0
        assert "search" in result.stdout
        assert "recommend" in result.stdout


class TestConnectCommand:
    """Test the connect command."""

    def test_connect_success(self):
        """Test successful connection."""
        mock_result = GalaxyResult(
            data={"connected": True, "user": {"username": "testuser"}},
            success=True,
            message="Connected to Galaxy",
        )

        with patch("galaxy_mcp.cli.main.connect_tool") as mock_connect:
            mock_connect.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["connect"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["success"] is True
        assert output["data"]["connected"] is True

    def test_connect_missing_credentials(self, monkeypatch):
        """Test connection without credentials."""
        monkeypatch.delenv("GALAXY_URL", raising=False)
        monkeypatch.delenv("GALAXY_API_KEY", raising=False)

        result = runner.invoke(app, ["connect"])

        assert result.exit_code == 1
        output = json.loads(result.stderr)
        assert "Missing credentials" in output["error"]


class TestToolsCommands:
    """Test tools commands."""

    def test_search(self):
        """Test tools search command."""
        mock_result = GalaxyResult(
            data=[{"id": "fastqc", "name": "FastQC"}],
            success=True,
            message="Found 1 tools",
            count=1,
        )

        with patch("galaxy_mcp.cli.commands.tools.search_tools_by_name") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["tools", "search", "fastqc"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["count"] == 1
        assert output["data"][0]["id"] == "fastqc"

    def test_search_pretty(self):
        """Test tools search with pretty output."""
        mock_result = GalaxyResult(
            data=[{"id": "fastqc", "name": "FastQC"}],
            success=True,
            message="Found 1 tools",
            count=1,
        )

        with patch("galaxy_mcp.cli.commands.tools.search_tools_by_name") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["--pretty", "tools", "search", "fastqc"])

        assert result.exit_code == 0
        # Pretty output should be indented
        assert "  " in result.stdout

    def test_details(self):
        """Test tools details command."""
        mock_result = GalaxyResult(
            data={"id": "fastqc", "name": "FastQC", "inputs": []},
            success=True,
            message="Retrieved details",
        )

        with patch("galaxy_mcp.cli.commands.tools.get_tool_details") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["tools", "details", "fastqc"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["data"]["id"] == "fastqc"


class TestHistoryCommands:
    """Test history commands."""

    def test_list(self):
        """Test history list command."""
        mock_result = GalaxyResult(
            data=[{"id": "hist1", "name": "My History"}],
            success=True,
            message="Retrieved 1 histories",
            count=1,
        )

        with patch("galaxy_mcp.cli.commands.history.get_histories") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["history", "list"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["count"] == 1

    def test_create(self):
        """Test history create command."""
        mock_result = GalaxyResult(
            data={"id": "new_hist", "name": "New History"},
            success=True,
            message="Created history",
        )

        with patch("galaxy_mcp.cli.commands.history.create_history") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["history", "create", "New History"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["data"]["name"] == "New History"


class TestIWCCommands:
    """Test IWC commands."""

    def test_search(self):
        """Test iwc search command."""
        mock_result = GalaxyResult(
            data=[{"trsID": "#workflow/test", "name": "RNA-seq"}],
            success=True,
            message="Found 1 workflows",
            count=1,
        )

        with patch("galaxy_mcp.cli.commands.iwc.search_iwc_workflows") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["iwc", "search", "rna-seq"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["count"] == 1

    def test_recommend(self):
        """Test iwc recommend command."""
        mock_result = GalaxyResult(
            data=[{"trsID": "#workflow/test", "name": "RNA-seq", "match_score": 10.5}],
            success=True,
            message="Found 1 workflows",
            count=1,
        )

        with patch("galaxy_mcp.cli.commands.iwc.recommend_iwc_workflows") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(
                app, ["iwc", "recommend", "differential expression from RNA-seq"]
            )

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["data"][0]["match_score"] == 10.5


class TestDatasetCommands:
    """Test dataset commands."""

    def test_details(self):
        """Test dataset details command."""
        mock_result = GalaxyResult(
            data={"dataset": {"id": "ds1", "name": "data.txt"}},
            success=True,
            message="Retrieved details",
        )

        with patch("galaxy_mcp.cli.commands.dataset.get_dataset_details") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["dataset", "details", "ds1"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["data"]["dataset"]["id"] == "ds1"


class TestWorkflowCommands:
    """Test workflow commands."""

    def test_list(self):
        """Test workflow list command."""
        mock_result = GalaxyResult(
            data=[{"id": "wf1", "name": "My Workflow"}],
            success=True,
            message="Found 1 workflows",
            count=1,
        )

        with patch("galaxy_mcp.cli.commands.workflow.list_workflows") as mock:
            mock.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["workflow", "list"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["count"] == 1


class TestErrorHandling:
    """Test error handling."""

    def test_tool_error(self):
        """Test error handling in tool commands."""
        with patch("galaxy_mcp.cli.commands.tools.search_tools_by_name") as mock:
            mock.fn = Mock(side_effect=ValueError("Connection failed"))
            result = runner.invoke(app, ["tools", "search", "test"])

        assert result.exit_code == 1
        output = json.loads(result.stderr)
        assert "Connection failed" in output["error"]


class TestPlanemoProfiles:
    """Test planemo profile reading."""

    def test_load_planemo_profile(self, tmp_path):
        """Test loading an external_galaxy planemo profile."""
        profile_dir = tmp_path / "myserver"
        profile_dir.mkdir()
        options = {
            "galaxy_url": "https://usegalaxy.eu/",
            "galaxy_user_key": "eu-key-123",
            "galaxy_admin_key": None,
            "engine": "external_galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        with patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path):
            result = _load_planemo_profile("myserver")

        assert result is not None
        assert result.url == "https://usegalaxy.eu/"
        assert result.api_key == "eu-key-123"

    def test_load_planemo_profile_admin_key(self, tmp_path):
        """Test that admin key is used when user key is absent."""
        profile_dir = tmp_path / "admin"
        profile_dir.mkdir()
        options = {
            "galaxy_url": "https://galaxy.example.com/",
            "galaxy_admin_key": "admin-key-456",
            "engine": "external_galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        with patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path):
            result = _load_planemo_profile("admin")

        assert result is not None
        assert result.api_key == "admin-key-456"

    def test_load_planemo_profile_skips_local_engine(self, tmp_path):
        """Test that local Galaxy profiles are ignored."""
        profile_dir = tmp_path / "local"
        profile_dir.mkdir()
        options = {
            "database_type": "sqlite",
            "database_connection": "sqlite:///foo.sqlite",
            "engine": "galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        with patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path):
            result = _load_planemo_profile("local")

        assert result is None

    def test_load_planemo_profile_missing(self, tmp_path):
        """Test that a missing profile returns None."""
        with patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path):
            result = _load_planemo_profile("nonexistent")

        assert result is None

    def test_load_profile_falls_back_to_planemo(self, tmp_path, monkeypatch):
        """Test that load_profile falls back to planemo when gxy config doesn't have the profile."""
        monkeypatch.delenv("GALAXY_URL", raising=False)
        monkeypatch.delenv("GALAXY_API_KEY", raising=False)

        # Create a planemo profile
        profile_dir = tmp_path / "planemo_profiles" / "usegalaxy-eu"
        profile_dir.mkdir(parents=True)
        options = {
            "galaxy_url": "https://usegalaxy.eu/",
            "galaxy_user_key": "eu-key",
            "engine": "external_galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        # Point to a nonexistent gxy config so it gets skipped
        fake_config = tmp_path / "config.toml"

        with (
            patch("galaxy_mcp.cli.config.CONFIG_FILE", fake_config),
            patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path / "planemo_profiles"),
        ):
            result = load_profile("usegalaxy-eu")

        assert result.url == "https://usegalaxy.eu/"
        assert result.api_key == "eu-key"

    def test_gxy_profile_takes_precedence(self, tmp_path, monkeypatch):
        """Test that gxy config profile wins over planemo profile of the same name."""
        monkeypatch.delenv("GALAXY_URL", raising=False)
        monkeypatch.delenv("GALAXY_API_KEY", raising=False)

        # Create gxy config
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[myserver]\nurl = "https://gxy.example.com/"\napi_key = "gxy-key"\n'
        )

        # Create planemo profile with same name
        profile_dir = tmp_path / "planemo_profiles" / "myserver"
        profile_dir.mkdir(parents=True)
        options = {
            "galaxy_url": "https://planemo.example.com/",
            "galaxy_user_key": "planemo-key",
            "engine": "external_galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        with (
            patch("galaxy_mcp.cli.config.CONFIG_FILE", config_file),
            patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path / "planemo_profiles"),
        ):
            result = load_profile("myserver")

        assert result.url == "https://gxy.example.com/"
        assert result.api_key == "gxy-key"

    def test_list_profiles_includes_planemo(self, tmp_path, monkeypatch):
        """Test that list_profiles includes planemo external profiles."""
        # Create gxy config with one profile
        config_file = tmp_path / "config.toml"
        config_file.write_text('[default]\nurl = "https://usegalaxy.org/"\napi_key = "key"\n')

        # Create planemo profile
        profile_dir = tmp_path / "planemo_profiles" / "usegalaxy-eu"
        profile_dir.mkdir(parents=True)
        options = {
            "galaxy_url": "https://usegalaxy.eu/",
            "galaxy_user_key": "eu-key",
            "engine": "external_galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        # Create a local-engine planemo profile (should be excluded)
        local_dir = tmp_path / "planemo_profiles" / "local-dev"
        local_dir.mkdir(parents=True)
        local_options = {"engine": "galaxy", "database_type": "sqlite"}
        (local_dir / "planemo_profile_options.json").write_text(json.dumps(local_options))

        with (
            patch("galaxy_mcp.cli.config.CONFIG_FILE", config_file),
            patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path / "planemo_profiles"),
        ):
            profiles = list_profiles()

        assert "default" in profiles
        assert "usegalaxy-eu" in profiles
        assert "local-dev" not in profiles

    def test_connect_with_planemo_profile(self, monkeypatch, tmp_path):
        """Test that --profile can load a planemo profile for the connect command."""
        monkeypatch.delenv("GALAXY_URL", raising=False)
        monkeypatch.delenv("GALAXY_API_KEY", raising=False)

        profile_dir = tmp_path / "planemo_profiles" / "test-server"
        profile_dir.mkdir(parents=True)
        options = {
            "galaxy_url": "https://test.galaxy.org/",
            "galaxy_user_key": "test-key",
            "engine": "external_galaxy",
        }
        (profile_dir / "planemo_profile_options.json").write_text(json.dumps(options))

        mock_result = GalaxyResult(
            data={"connected": True, "user": {"username": "testuser"}},
            success=True,
            message="Connected",
        )

        fake_config = tmp_path / "config.toml"

        with (
            patch("galaxy_mcp.cli.config.CONFIG_FILE", fake_config),
            patch("galaxy_mcp.cli.config.PLANEMO_PROFILES_DIR", tmp_path / "planemo_profiles"),
            patch("galaxy_mcp.cli.main.connect_tool") as mock_connect,
        ):
            mock_connect.fn = Mock(return_value=mock_result)
            result = runner.invoke(app, ["--profile", "test-server", "connect"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["data"]["connected"] is True
