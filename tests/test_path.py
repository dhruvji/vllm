# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from vllm.path import S3ReadFile

pytestmark = pytest.mark.skip_global_cleanup


def test_s3_read_retries_signature_mismatch():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 4}
    signature_error = ClientError(
        {"Error": {"Code": "SignatureDoesNotMatch"}}, "GetObject"
    )
    body = MagicMock()
    body.read.return_value = b"data"
    client.get_object.side_effect = [signature_error, {"Body": body}]

    with patch("vllm.path.time.sleep") as sleep:
        source = S3ReadFile("bucket", "adapter.safetensors", client, "rb")
        assert source.read(4) == b"data"

    expected_call = call(Bucket="bucket", Key="adapter.safetensors", Range="bytes=0-3")
    assert client.get_object.call_args_list == [expected_call, expected_call]
    sleep.assert_called_once()
    assert source.tell() == 4


def test_s3_read_does_not_retry_other_client_errors():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 4}
    access_error = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    client.get_object.side_effect = access_error

    with patch("vllm.path.time.sleep") as sleep:
        source = S3ReadFile("bucket", "adapter.safetensors", client, "rb")
        with pytest.raises(ClientError) as exc_info:
            source.read(4)

    assert exc_info.value is access_error
    client.get_object.assert_called_once_with(
        Bucket="bucket", Key="adapter.safetensors", Range="bytes=0-3"
    )
    sleep.assert_not_called()
    assert source.tell() == 0
