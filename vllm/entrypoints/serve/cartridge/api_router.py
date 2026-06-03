# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.cartridge_loader import (
    cleanup_cartridge_shm,
    load_cartridge,
    save_cartridge_to_shm,
)

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/v1/cartridges/update")
async def update_cartridge(raw_request: Request):
    """Update a cartridge's KV cache mid-generation."""
    body = await raw_request.json()
    cartridge_id = body.get("cartridge_id")
    cartridge_path = body.get("cartridge_path")
    if not cartridge_id or not cartridge_path:
        return JSONResponse(
            status_code=400,
            content={"error": "cartridge_id and cartridge_path are required"},
        )

    cartridge_data = load_cartridge(cartridge_path, force_redownload=True)
    stacked_kv = cartridge_data.get_stacked_kv()
    if stacked_kv is None:
        return JSONResponse(
            status_code=400,
            content={"error": "cartridge does not contain stacked KV tensors"},
        )
    # Remove old shm file so save_cartridge_to_shm overwrites
    cleanup_cartridge_shm(cartridge_id)
    shm_path = save_cartridge_to_shm(cartridge_id, stacked_kv)
    result = await engine_client(raw_request).update_cartridge(cartridge_id, shm_path)
    return JSONResponse(content={"status": result, "cartridge_id": cartridge_id})


def attach_router(app: FastAPI):
    app.include_router(router)
