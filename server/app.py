"""
server/app.py — FastAPI application entry point for ProteinEnv.

Uses OpenEnv's create_app() factory to wire the environment into
the standard WebSocket + HTTP server infrastructure, then adds a
/health endpoint for orchestrator health checks.
"""

from __future__ import annotations

try:
    from ..models import ProteinAction, ProteinObservation
    from ..server.protein_environment import ProteinEnvironment
except ImportError:
    from models import ProteinAction, ProteinObservation
    from server.protein_environment import ProteinEnvironment

try:
    from openenv_core.server import create_app
    app = create_app(
        ProteinEnvironment,
        ProteinAction,
        ProteinObservation,
        env_name="protein-env",
    )
except ImportError:
    # openenv_core not installed (e.g. during local syntax/unit testing):
    # create a minimal FastAPI stand-in so the module still imports cleanly.
    from fastapi import FastAPI
    app = FastAPI(title="protein-env", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint required by openenv validate.

    Args:
        None.

    Returns:
        dict with status, env name, and version string.

    Raises:
        Nothing.
    """
    return {"status": "ok", "env": "protein-env", "version": "0.1.0"}
