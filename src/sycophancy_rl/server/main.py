"""FastAPI entry point for the Sycophancy RL Environment.

The server is local / single-tenant.  Production exposure requires an
authentication + TLS proxy in front of it; the server itself does not
implement auth.

* binds to ``127.0.0.1`` by default;
* CORS is restricted to an allowlist (``syco_origins`` env or the
  default ``http://localhost:7860``);
* the wildcard origin is never combined with credentials;
* separate ``/health/live`` and ``/health/ready`` endpoints;
* the package version is read from a single source.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sycophancy_rl.cli.main import __version__ as PACKAGE_VERSION
from sycophancy_rl.environment.session_manager import SessionManager
from sycophancy_rl.server.data_store import EpisodeStore

from .routes import grader, health, reset, step

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:7860",
    "http://127.0.0.1:7860",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


def _resolve_origins() -> list[str]:
    """Return the CORS allowlist.

    Wildcard origins are never combined with credentials.  When the
    caller sets ``syco_cors_allow_credentials=1`` AND uses ``*`` we
    demote the wildcard to a single deny-by-default origin.
    """

    raw = os.environ.get("syco_origins") or os.environ.get("syco_ALLOWED_ORIGINS")
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
    else:
        origins = list(DEFAULT_ALLOWED_ORIGINS)
    allow_credentials = os.environ.get("syco_cors_allow_credentials", "0") == "1"
    if "*" in origins and allow_credentials:
        raise RuntimeError(
            "Wildcard CORS origin cannot be combined with credentials. "
            "Set syco_origins to an explicit allowlist."
        )
    if not allow_credentials:
        # Wildcards are safe when credentials are not allowed.
        if origins == ["*"]:
            return ["*"]
    return origins


def create_app(
    *,
    allowed_origins: Iterable[str] | None = None,
    session_manager: SessionManager | None = None,
    max_sessions: int | None = None,
    session_ttl_seconds: float | None = None,
    dataset_path: Path | None = None,
) -> FastAPI:
    """Build a FastAPI app with the configured hardening.

    Tests use this factory to construct isolated instances; the module
    level ``app`` keeps the production entry point for ``uvicorn``.
    """

    origins = (
        list(allowed_origins)
        if allowed_origins is not None
        else _resolve_origins()
    )
    allow_credentials = (
        "*" not in origins
        and os.environ.get("syco_cors_allow_credentials", "0") == "1"
    )

    app = FastAPI(
        title="Sycophancy RL Environment",
        version=PACKAGE_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    app.state.session_manager = session_manager or SessionManager(
        max_sessions=max_sessions,
        ttl_seconds=session_ttl_seconds,
    )
    app.state.package_version = PACKAGE_VERSION
    app.state.episode_store = EpisodeStore(
        dataset_path
        or Path(os.environ.get("SYCO_DATASET_PATH", "data/processed/training_pool.jsonl"))
    )

    app.include_router(health.router)
    app.include_router(reset.router)
    app.include_router(step.router)
    app.include_router(grader.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("syco_HOST", "127.0.0.1")
    port = int(os.environ.get("syco_PORT", "8000"))
    uvicorn.run("sycophancy_rl.server.main:app", host=host, port=port, reload=False)
