"""
FastAPI entry point for the Sycophancy RL Environment.

This module is the single process-level wiring layer for the environment
server. It is intentionally thin: no business logic lives here. Its three
responsibilities are:

1. **CORS configuration** — install a permissive ``CORSMiddleware`` so the
   Gradio demo Space and any external RL client can call the API from a
   different origin. ``allow_origins=["*"]`` is acceptable here because the
   environment exposes no credentials or user-specific data; the server is
   designed for local / single-tenant use.

2. **SessionManager injection** — instantiate a single global
   :class:`SessionManager` and attach it to ``app.state.session_manager``
   so the route handlers can pull it out via ``request.app.state`` without
   importing the manager directly. This keeps the routes decoupled from
   the concrete storage backend (the in-memory dict can be swapped for
   Redis later without touching the route code).

3. **Router registration** — ``include_router`` the four endpoint groups
   (health, reset, step, grader). Each module under ``server/routes``
   exposes an ``APIRouter`` instance named ``router``.

Run locally with::

    python -m server.main

or, equivalently, the ``if __name__ == "__main__"`` block at the bottom of
this file.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.environment.session_manager import SessionManager

from .routes import health, reset, step, grader

app = FastAPI(title="Sycophancy RL Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global registry shared across every request. Routes reach it via
# ``request.app.state.session_manager`` so they stay decoupled from this
# module and the concrete storage backend.
app.state.session_manager = SessionManager()

app.include_router(health.router)
app.include_router(reset.router)
app.include_router(step.router)
app.include_router(grader.router)

if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
