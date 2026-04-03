"""
server — FastAPI application package for the ProteinEnv OpenEnv environment.

Exposes the ASGI app entrypoint (app.py) and all HTTP/WebSocket route handlers.
Uses the dual-import pattern throughout: try relative imports first, fall back
to bare (top-level) imports so the package works both as an installed module and
when run directly with uvicorn from the project root.
"""
