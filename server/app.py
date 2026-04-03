"""
server/app.py — FastAPI application entry point for ProteinEnv.

Implements all OpenEnv HTTP validation endpoints plus the full
episode loop endpoints used by inference agents:
  GET  /health    — liveness check
  GET  /metadata  — environment name and description
  GET  /schema    — action, observation, state JSON schemas
  POST /mcp       — JSON-RPC 2.0 MCP compatibility shim
  POST /reset     — start a new episode
  POST /step      — execute one action
  GET  /state     — current episode state snapshot
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from ..models import ProteinAction, ProteinObservation, ProteinState
    from ..server.protein_environment import ProteinEnvironment
except ImportError:
    from models import ProteinAction, ProteinObservation, ProteinState
    from server.protein_environment import ProteinEnvironment

logger = logging.getLogger(__name__)

try:
    from openenv_core.server import create_app
    app = create_app(
        ProteinEnvironment,
        ProteinAction,
        ProteinObservation,
        env_name="protein-env",
    )
except ImportError:
    # openenv_core not on PyPI yet — use a plain FastAPI app.
    app = FastAPI(title="protein-env", version="0.1.0")

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def read_root():
        return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health() -> dict:
    """Liveness check. OpenEnv validate expects status == 'healthy'.

    Args:
        None.

    Returns:
        dict with status, env name, and version string.

    Raises:
        Nothing.
    """
    return {"status": "healthy", "env": "protein-env", "version": "0.1.0"}


@app.get("/metadata")
async def metadata() -> dict:
    """Environment metadata required by openenv validate.

    Args:
        None.

    Returns:
        dict with name and description fields.

    Raises:
        Nothing.
    """
    return {
        "name": "protein-env",
        "description": (
            "OpenEnv environment for protein function prediction using ESM2. "
            "Supports three task tiers: family classification (easy), "
            "GO term prediction (medium), and disease variant assessment (hard)."
        ),
        "version": "0.1.0",
        "author": "Meta AI — ESM2 team",
        "task_types": ["easy", "medium", "hard"],
    }


@app.get("/schema")
async def schema() -> dict:
    """JSON schemas for action, observation, and state models.

    Args:
        None.

    Returns:
        dict with action, observation, and state JSON Schema objects.

    Raises:
        Nothing.
    """
    return {
        "action": ProteinAction.model_json_schema(),
        "observation": ProteinObservation.model_json_schema(),
        "state": ProteinState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request) -> JSONResponse:
    """JSON-RPC 2.0 MCP compatibility shim required by openenv validate.

    Handles the standard JSON-RPC ping/initialize handshake so that
    MCP-aware orchestrators can discover this environment.

    Args:
        request: Incoming FastAPI Request with a JSON-RPC 2.0 body.

    Returns:
        JSONResponse with jsonrpc=2.0 and method-appropriate result.

    Raises:
        Nothing (errors are returned as JSON-RPC error objects).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    rpc_id = body.get("id", 1)
    method = body.get("method", "")

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "protein-env", "version": "0.1.0"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "get_esm2_embedding",
                    "description": "Compute a 320-dim ESM2 embedding for a protein sequence.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sequence": {
                                "type": "string",
                                "description": "Amino acid sequence (single-letter codes).",
                            }
                        },
                        "required": ["sequence"],
                    },
                }
            ]
        }
    else:
        result = {"message": f"method '{method}' acknowledged"}

    return JSONResponse(
        content={"jsonrpc": "2.0", "id": rpc_id, "result": result}
    )


# ── Episode loop endpoints ─────────────────────────────────────────────────────


@app.post("/reset")
async def reset(req: ResetRequest) -> ProteinObservation:
    """Start a new episode and return the initial observation.

    Args:
        req: ResetRequest with task_type, optional seed and episode_id.

    Returns:
        ProteinObservation for the first step.

    Raises:
        Nothing (errors propagate as 500).
    """
    logger.info("POST /reset task_type=%s seed=%s", req.task_type, req.seed)
    return _env.reset(
        task_type=req.task_type,
        seed=req.seed,
        episode_id=req.episode_id,
    )


@app.post("/step")
async def step(action: ProteinAction) -> dict:
    """Execute one agent action and return the step result.

    Args:
        action: ProteinAction from the agent.

    Returns:
        StepResult serialised as a dict.

    Raises:
        Nothing (errors propagate as 500).
    """
    logger.info("POST /step action_type=%s", action.action_type)
    result = _env.step(action)
    return result.model_dump()


@app.get("/state")
async def state() -> ProteinState:
    """Return the current full episode state snapshot.

    Args:
        None.

    Returns:
        ProteinState with all episode tracking fields.

    Raises:
        Nothing (errors propagate as 500).
    """
    return _env.state()
