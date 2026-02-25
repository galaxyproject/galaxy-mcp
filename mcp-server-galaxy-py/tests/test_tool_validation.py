"""Tests for structured tool contracts and validated tool submission."""

from unittest.mock import patch

import pytest

from galaxy_mcp import server

from .test_helpers import (
    galaxy_state,
    inspect_tool_contract_fn,
    run_tool_validated_fn,
    validate_tool_inputs_fn,
)


def _sample_tool_info() -> dict:
    return {
        "id": "example_tool",
        "name": "Example Tool",
        "version": "1.0.0",
        "description": "Example for contract tests",
        "inputs": [
            {
                "name": "input_dataset",
                "type": "data",
                "label": "Input dataset",
                "optional": False,
            },
            {
                "name": "mode",
                "type": "conditional",
                "optional": False,
                "test_param": {
                    "name": "selector",
                    "type": "select",
                    "label": "Mode selector",
                    "options": [
                        ["Alpha", "alpha", True],
                        ["Beta", "beta", False],
                    ],
                },
                "cases": [
                    {
                        "value": "alpha",
                        "inputs": [
                            {
                                "name": "alpha_threshold",
                                "type": "float",
                                "label": "Alpha threshold",
                                "optional": False,
                            }
                        ],
                    },
                    {
                        "value": "beta",
                        "inputs": [
                            {
                                "name": "beta_flag",
                                "type": "boolean",
                                "label": "Beta flag",
                                "optional": False,
                            }
                        ],
                    },
                ],
            },
        ],
    }


def _sample_tool_info_simple() -> dict:
    return {
        "id": "simple_tool",
        "name": "Simple Tool",
        "version": "1.0.0",
        "description": "Simple schema",
        "inputs": [
            {
                "name": "input_dataset",
                "type": "data",
                "label": "Input dataset",
                "optional": False,
            }
        ],
    }


def _sample_tool_info_batch_size() -> dict:
    return {
        "id": "batch_tool",
        "name": "Batch Tool",
        "version": "1.0.0",
        "description": "Batch schema",
        "inputs": [
            {
                "name": "batch_size",
                "type": "integer",
                "label": "Batch size",
                "optional": True,
            }
        ],
    }


def _sample_tool_info_select() -> dict:
    return {
        "id": "select_tool",
        "name": "Select Tool",
        "version": "1.0.0",
        "description": "Select schema",
        "inputs": [
            {
                "name": "strategy",
                "type": "select",
                "label": "Strategy",
                "optional": False,
                "options": [
                    ["Option A", "a", True],
                    ["Option B", "b", False],
                ],
            }
        ],
    }


def _sample_tool_info_repeat() -> dict:
    return {
        "id": "repeat_tool",
        "name": "Repeat Tool",
        "version": "1.0.0",
        "description": "Repeat schema",
        "inputs": [
            {
                "name": "images_zip_repeat",
                "type": "repeat",
                "optional": False,
                "inputs": [
                    {
                        "name": "images_zip",
                        "type": "data",
                        "label": "Images archive",
                        "optional": False,
                    }
                ],
            }
        ],
    }


def _sample_tool_info_with_extension() -> dict:
    return {
        "id": "ext_tool",
        "name": "Extension Tool",
        "version": "1.0.0",
        "description": "Extension-aware schema",
        "inputs": [
            {
                "name": "input_dataset",
                "type": "data",
                "label": "Input dataset",
                "optional": False,
                "extensions": ["csv"],
            }
        ],
    }


def _sample_tool_info_collection() -> dict:
    return {
        "id": "collection_tool",
        "name": "Collection Tool",
        "version": "1.0.0",
        "description": "Collection schema",
        "inputs": [
            {
                "name": "input_collection",
                "type": "data_collection",
                "label": "Input collection",
                "optional": False,
                "collection_types": ["list"],
            }
        ],
    }


def _sample_tool_info_conditional_dataset() -> dict:
    return {
        "id": "conditional_dataset_tool",
        "name": "Conditional Dataset Tool",
        "version": "1.0.0",
        "description": "Conditional dataset schema",
        "inputs": [
            {
                "name": "mode",
                "type": "conditional",
                "optional": False,
                "test_param": {
                    "name": "selector",
                    "type": "select",
                    "label": "Selector",
                    "options": [["Alpha", "alpha", True], ["Beta", "beta", False]],
                },
                "cases": [
                    {
                        "value": "alpha",
                        "inputs": [
                            {"name": "alpha_input", "type": "data", "optional": True},
                        ],
                    },
                    {
                        "value": "beta",
                        "inputs": [
                            {"name": "beta_input", "type": "data", "optional": True},
                        ],
                    },
                ],
            }
        ],
    }


def _sample_tool_info_column_and_text() -> dict:
    return {
        "id": "column_text_tool",
        "name": "Column/Text Tool",
        "version": "1.0.0",
        "description": "Column and text schema",
        "inputs": [
            {
                "name": "table_input",
                "type": "data",
                "label": "Input table",
                "optional": False,
            },
            {
                "name": "target_column",
                "type": "data_column",
                "label": "Target column",
                "optional": False,
            },
            {
                "name": "hyperparameters",
                "type": "text",
                "label": "Hyperparameters JSON",
                "optional": True,
            },
        ],
    }


class TestToolValidation:
    def test_inspect_tool_contract_returns_structured_output(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"mode": {"selector": "alpha"}},
            "errors": {},
            "tool_errors": None,
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = inspect_tool_contract_fn("example_tool", history_id="hist1")

        assert result.success is True
        contract = result.data["contract"]
        assert "fields" in contract
        assert "conditional_groups" in contract
        assert "input_dataset" in contract["required_keys"]
        assert any(g["selector_key"] == "mode|selector" for g in contract["conditional_groups"])
        assert "agent_hints" in result.data
        assert result.data["agent_identity"] == server.USER_AGENT

    def test_inspect_tool_contract_includes_agent_hints_for_special_field_types(
        self, mock_galaxy_instance
    ):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_column_and_text()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = inspect_tool_contract_fn("column_text_tool", history_id="hist1")

        assert result.success is True
        codes = {hint["code"] for hint in result.data["agent_hints"]}
        assert "data_column_requires_index" in codes
        assert "text_json_escape_risk" in codes

    def test_validate_tool_inputs_detects_missing_conditional_selector(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"mode": {"selector": "alpha"}},
            "errors": {},
            "tool_errors": None,
        }

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "mode|alpha_threshold": 0.1,
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "example_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is False
        assert any(
            err["code"] == "conditional_selector_missing"
            for err in result.data["validation"]["errors"]
        )
        assert result.data["agent_identity"] == server.USER_AGENT
        assert mock_galaxy_instance.tools.build.call_count >= 1
        first_call = mock_galaxy_instance.tools.build.call_args_list[0]
        assert first_call.args[0] == "example_tool"
        assert first_call.kwargs["history_id"] == "hist1"

    def test_validate_tool_inputs_accepts_valid_payload(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {
                "input_dataset": {"src": "hda", "id": "dataset1"},
                "mode": {"selector": "alpha", "alpha_threshold": 0.1},
            },
            "errors": {},
            "tool_errors": None,
        }

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "mode|selector": "alpha",
            "mode|alpha_threshold": 0.1,
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "example_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is True
        assert result.data["validation"]["errors"] == []
        assert mock_galaxy_instance.tools.build.call_count >= 1

    def test_run_tool_validated_rejects_invalid_payload(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"mode": {"selector": "alpha"}},
            "errors": {},
            "tool_errors": None,
        }

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "mode|alpha_threshold": 0.1,
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Run tool validated failed"):
                run_tool_validated_fn("hist1", "example_tool", inputs)

        assert mock_galaxy_instance.tools.build.call_count >= 1
        mock_galaxy_instance.tools.run_tool.assert_not_called()

    def test_run_tool_validated_submits_when_valid(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {
                "input_dataset": {"src": "hda", "id": "dataset1"},
                "mode": {"selector": "alpha", "alpha_threshold": 0.1},
            },
            "errors": {},
            "tool_errors": None,
        }
        mock_galaxy_instance.tools.run_tool.return_value = {
            "jobs": [{"id": "job1", "state": "queued"}],
            "outputs": [{"id": "out1"}],
        }

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "mode|selector": "alpha",
            "mode|alpha_threshold": 0.1,
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_validated_fn("hist1", "example_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is True
        assert "submission" in result.data
        assert "build_rounds" in result.data
        assert "build_converged" in result.data
        assert "dataset_preflight" in result.data
        assert result.data["agent_identity"] == server.USER_AGENT
        mock_galaxy_instance.tools.run_tool.assert_called_once_with(
            "hist1",
            "example_tool",
            inputs,
        )

    def test_unknown_keys_are_errors_by_default(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_simple()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "data_train": {"src": "hda", "id": "dataset2"},
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "simple_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is False
        assert any(err["code"] == "unknown_input_keys" for err in result.data["validation"]["errors"])

    def test_unknown_keys_can_be_warning_in_non_strict_mode(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_simple()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [{"id": "job1"}], "outputs": []}

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "data_train": {"src": "hda", "id": "dataset2"},
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            validation = validate_tool_inputs_fn(
                "hist1", "simple_tool", inputs, strict_unknown_keys=False
            )

        assert validation.success is True
        assert validation.data["validation"]["valid"] is True
        assert any(
            warning["code"] == "unknown_input_keys"
            for warning in validation.data["validation"]["warnings"]
        )

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            submission = run_tool_validated_fn(
                "hist1",
                "simple_tool",
                inputs,
                strict_unknown_keys=False,
            )

        assert submission.success is True
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_repeat_indexed_keys_are_accepted(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_repeat()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}

        inputs = {"images_zip_repeat_0|images_zip": {"src": "hda", "id": "dataset_zip"}}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "repeat_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is True
        assert result.data["validation"]["unknown_keys"] == []

    def test_non_repeat_suffix_key_is_not_normalized(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_batch_size()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}

        inputs = {"batch_size_2": 32}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "batch_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is False
        assert any(
            err["code"] == "unknown_input_keys" for err in result.data["validation"]["errors"]
        )

    def test_data_column_requires_1_indexed_integer(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_column_and_text()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}

        invalid_inputs = {
            "table_input": {"src": "hda", "id": "dataset_table"},
            "target_column": "patient_id",
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            invalid_result = validate_tool_inputs_fn("hist1", "column_text_tool", invalid_inputs)

        assert invalid_result.success is True
        assert invalid_result.data["validation"]["valid"] is False
        assert any(
            err["code"] == "invalid_data_column_value"
            for err in invalid_result.data["validation"]["errors"]
        )

        valid_inputs = {"table_input": {"src": "hda", "id": "dataset_table"}, "target_column": 2}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            valid_result = validate_tool_inputs_fn("hist1", "column_text_tool", valid_inputs)

        assert valid_result.success is True
        assert valid_result.data["validation"]["valid"] is True

    def test_json_blob_in_text_field_warns(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_column_and_text()
        mock_galaxy_instance.tools.build.return_value = {"state_inputs": {}, "errors": {}, "tool_errors": None}

        inputs = {
            "table_input": {"src": "hda", "id": "dataset_table"},
            "target_column": 1,
            "hyperparameters": "{\"learning_rate\":0.01}",
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "column_text_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is True
        assert any(
            warning["code"] == "json_blob_in_text_field"
            for warning in result.data["validation"]["warnings"]
        )

    def test_invalid_select_value_is_rejected(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_select()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"strategy": "invalid_choice"},
            "errors": {},
            "tool_errors": None,
        }

        inputs = {"strategy": "invalid_choice"}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "select_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is False
        assert any(
            err["code"] == "invalid_select_value"
            for err in result.data["validation"]["errors"]
        )

    def test_iterative_build_preserves_agent_values(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info()
        mock_galaxy_instance.tools.build.side_effect = [
            {
                "state_inputs": {
                    "input_dataset": {"src": "hda", "id": "dataset1"},
                    "mode": {"selector": "alpha", "alpha_threshold": 0.5},
                },
                "errors": {},
                "tool_errors": None,
            },
            {
                "state_inputs": {
                    "input_dataset": {"src": "hda", "id": "dataset1"},
                    "mode": {"selector": "alpha", "alpha_threshold": 0.5},
                },
                "errors": {},
                "tool_errors": None,
            },
        ]

        inputs = {
            "input_dataset": {"src": "hda", "id": "dataset1"},
            "mode|selector": "alpha",
            "mode|alpha_threshold": 0.1,
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "example_tool", inputs)

        assert result.success is True
        assert result.data["resolved_inputs"]["mode"]["alpha_threshold"] == 0.5
        assert mock_galaxy_instance.tools.build.call_count >= 2
        second_call = mock_galaxy_instance.tools.build.call_args_list[1]
        assert second_call.kwargs["inputs"]["mode"]["alpha_threshold"] == 0.1
        assert result.data["build_rounds"] >= 2
        assert result.data["build_converged"] is True

    def test_iterative_build_warns_when_not_converged(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_select()
        mock_galaxy_instance.tools.build.side_effect = [
            {"state_inputs": {"strategy": "a"}, "errors": {}, "tool_errors": None},
            {"state_inputs": {"strategy": "b"}, "errors": {}, "tool_errors": None},
            {"state_inputs": {"strategy": "a"}, "errors": {}, "tool_errors": None},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "select_tool", {})

        assert result.success is True
        warning_codes = {warning["code"] for warning in result.data["validation"]["warnings"]}
        assert "build_not_converged" in warning_codes or "build_oscillation_detected" in warning_codes
        assert result.data["build_rounds"] == 3

    def test_dataset_preflight_dataset_not_found(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_simple()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"input_dataset": {"src": "hda", "id": "dataset_missing"}},
            "errors": {},
            "tool_errors": None,
        }
        mock_galaxy_instance.datasets.show_dataset.side_effect = Exception("404 not found")

        inputs = {"input_dataset": {"src": "hda", "id": "dataset_missing"}}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "simple_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is False
        assert any(
            err["code"] == "dataset_not_found"
            for err in result.data["validation"]["errors"]
        )

    def test_dataset_preflight_history_deleted(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_simple()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"input_dataset": {"src": "hda", "id": "dataset1"}},
            "errors": {},
            "tool_errors": None,
        }
        mock_galaxy_instance.histories.show_history.return_value = {"id": "hist1", "deleted": True}

        inputs = {"input_dataset": {"src": "hda", "id": "dataset1"}}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "simple_tool", inputs)

        assert result.success is True
        assert any(
            err["code"] == "history_deleted"
            for err in result.data["validation"]["errors"]
        )

    def test_dataset_preflight_extension_mismatch_warns(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_with_extension()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"input_dataset": {"src": "hda", "id": "dataset1"}},
            "errors": {},
            "tool_errors": None,
        }
        mock_galaxy_instance.datasets.show_dataset.return_value = {
            "id": "dataset1",
            "state": "ok",
            "history_id": "hist1",
            "extension": "txt",
        }

        inputs = {"input_dataset": {"src": "hda", "id": "dataset1"}}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "ext_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is True
        assert any(
            warning["code"] == "dataset_extension_mismatch"
            for warning in result.data["validation"]["warnings"]
        )

    def test_dataset_preflight_collection_type_mismatch(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_collection()
        mock_galaxy_instance.tools.build.return_value = {
            "state_inputs": {"input_collection": {"src": "hdca", "id": "collection1"}},
            "errors": {},
            "tool_errors": None,
        }
        mock_collection_client = type("MockCollectionClient", (), {})()
        mock_collection_client.show_dataset_collection = (
            lambda _collection_id: {
                "id": "collection1",
                "collection_type": "paired",
                "history_id": "hist1",
            }
        )
        mock_galaxy_instance.dataset_collections = mock_collection_client

        inputs = {"input_collection": {"src": "hdca", "id": "collection1"}}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "collection_tool", inputs)

        assert result.success is True
        assert result.data["validation"]["valid"] is False
        assert any(
            err["code"] == "collection_type_mismatch"
            for err in result.data["validation"]["errors"]
        )

    def test_dataset_reference_removed_after_resolution_warns(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = _sample_tool_info_conditional_dataset()
        mock_galaxy_instance.tools.build.side_effect = [
            {
                "state_inputs": {"mode": {"selector": "alpha"}},
                "errors": {},
                "tool_errors": None,
            },
            {
                "state_inputs": {"mode": {"selector": "alpha"}},
                "errors": {},
                "tool_errors": None,
            },
        ]

        inputs = {
            "mode|selector": "alpha",
            "mode|beta_input": {"src": "hda", "id": "dataset_beta"},
        }
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = validate_tool_inputs_fn("hist1", "conditional_dataset_tool", inputs)

        assert result.success is True
        assert any(
            warning["code"] == "dataset_reference_removed_by_resolution"
            for warning in result.data["validation"]["warnings"]
        )