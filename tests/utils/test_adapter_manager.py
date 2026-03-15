# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.adapter_manager import AdapterManager


@patch("vllm.utils.adapter_manager.time.sleep")
@patch("vllm.utils.adapter_manager._load_adapter")
@patch("vllm.utils.adapter_manager.Path")
def test_get_modal_adapter_retries_after_missing_path(
    mock_path_cls, mock_load_adapter, mock_sleep
):
    manager = AdapterManager(cache_dir="/tmp/vllm-adapter-manager-test")
    source_path = MagicMock()
    source_path.exists.side_effect = [False, True]
    mock_path_cls.return_value = source_path
    mock_load_adapter.return_value = {"tensor": "ok"}

    result = manager.get_adapter("modal://volume/run/adapter\n")

    assert result == {"tensor": "ok"}
    mock_sleep.assert_called_once_with(5.0)
    mock_load_adapter.assert_called_once_with("modal://volume/run/adapter")


@patch("vllm.utils.adapter_manager.time.sleep")
@patch("vllm.utils.adapter_manager._load_adapter")
@patch("vllm.utils.adapter_manager.Path")
def test_get_modal_adapter_retries_after_file_not_found(
    mock_path_cls, mock_load_adapter, mock_sleep
):
    manager = AdapterManager(cache_dir="/tmp/vllm-adapter-manager-test")
    source_path = MagicMock()
    source_path.exists.return_value = True
    mock_path_cls.return_value = source_path
    mock_load_adapter.side_effect = [FileNotFoundError("not ready"), {"tensor": "ok"}]

    result = manager.get_adapter("modal://volume/run/adapter")

    assert result == {"tensor": "ok"}
    mock_sleep.assert_called_once_with(5.0)
    assert mock_load_adapter.call_count == 2


@patch("vllm.utils.adapter_manager.time.sleep")
@patch("vllm.utils.adapter_manager.Path")
def test_get_modal_adapter_raises_after_retry_if_still_missing(
    mock_path_cls, mock_sleep
):
    manager = AdapterManager(cache_dir="/tmp/vllm-adapter-manager-test")
    source_path = MagicMock()
    source_path.exists.side_effect = [False, False]
    mock_path_cls.return_value = source_path

    with pytest.raises(FileNotFoundError, match="Modal volume adapter not found"):
        manager.get_adapter("modal://volume/run/adapter")

    mock_sleep.assert_called_once_with(5.0)
