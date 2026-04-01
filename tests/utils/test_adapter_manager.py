# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.adapter_manager import AdapterManager, FORCE_REDOWNLOAD_ENV_VAR


@patch("vllm.utils.adapter_manager.time.sleep")
@patch("vllm.utils.adapter_manager._load_adapter")
@patch("vllm.utils.adapter_manager.Path")
def test_get_modal_adapter_retries_after_missing_path(
    mock_path_cls, mock_load_adapter, mock_sleep
):
    manager = AdapterManager(cache_dir="/tmp/vllm-adapter-manager-test")
    source_path = MagicMock()
    mock_path_cls.return_value = source_path
    mock_load_adapter.side_effect = [FileNotFoundError("not ready"), {"tensor": "ok"}]

    result = manager.get_adapter("modal://volume/run/adapter\n")

    assert result == {"tensor": "ok"}
    mock_sleep.assert_called_once_with(5.0)
    assert mock_load_adapter.call_count == 2


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
    mock_path_cls.return_value = source_path

    with patch(
        "vllm.utils.adapter_manager._load_adapter",
        side_effect=FileNotFoundError("not ready"),
    ), pytest.raises(FileNotFoundError, match="not ready"):
        manager.get_adapter("modal://volume/run/adapter")

    mock_sleep.assert_called_once_with(5.0)


@patch.dict(f"os.environ", {FORCE_REDOWNLOAD_ENV_VAR: "1"})
@patch("vllm.utils.adapter_manager._load_adapter")
@patch("vllm.utils.adapter_manager.Path")
def test_get_s3_adapter_force_redownload_env_ignores_existing_cache(
    mock_path_cls, mock_load_adapter
):
    manager = AdapterManager(cache_dir="/tmp/vllm-adapter-manager-test")
    source_path = MagicMock()
    source_path.open.return_value.__enter__.return_value.read.side_effect = [b"", b""]
    mock_path_cls.return_value = source_path
    mock_load_adapter.return_value = {"tensor": "ok"}

    with patch("vllm.utils.adapter_manager.filelock.FileLock"), patch(
        "pathlib.Path.exists", return_value=True
    ), patch("builtins.open", MagicMock()), patch(
        "pathlib.Path.mkdir"
    ), patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)):
        result = manager.get_adapter("s3://bucket/adapter.pt")

    assert result == {"tensor": "ok"}
    source_path.open.assert_called_once_with("rb")
