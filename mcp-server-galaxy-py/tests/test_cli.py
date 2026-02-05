"""Tests for the Galaxy MCP CLI (gxy)."""

import json
from unittest.mock import Mock, patch

import pytest
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
