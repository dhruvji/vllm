# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLLM shutdown cleanup."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.engine.async_llm import AsyncLLM


def test_async_llm_shutdown_tolerates_partial_init(monkeypatch) -> None:
    monkeypatch.setattr("vllm.v1.engine.async_llm.shutdown_prometheus", lambda: None)

    engine = AsyncLLM.__new__(AsyncLLM)
    engine.engine_core = MagicMock()
    engine.input_processor = SimpleNamespace()
    engine.output_processor = SimpleNamespace(close=MagicMock())
    engine.io_processor = SimpleNamespace(shutdown=MagicMock())
    engine.output_handler = None

    engine.shutdown()

    engine.engine_core.shutdown.assert_called_once()
    engine.output_processor.close.assert_called_once()
    engine.io_processor.shutdown.assert_called_once()
