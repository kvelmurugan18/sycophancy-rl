"""
Health check endpoint for the Sycophancy RL Environment server.

This module exposes a single ``GET /health`` route that serves as a basic
**liveness probe** for the running server. It allows external monitors —
container orchestrators (Docker, Kubernetes), HuggingFace Spaces' built-in
health check, uptime monitors, or even a quick ``curl`` during local
debugging — to verify the FastAPI process is up and responsive.

Beyond the trivial "are you alive?" signal, the endpoint also peeks into
the in-memory :class:`SessionManager` registry and reports the number of
active RL episodes currently loaded. That count is a cheap, useful
operational signal: a sudden drop can indicate crashed sessions being
cleaned up, a sudden spike can flag a runaway client opening episodes
without closing them, and a stuck-at-zero count can confirm the server
has just been (re)started and is waiting for its first ``/reset`` call.

No mutation, no I/O, no side effects — safe to call as often as the
monitor likes.
"""

from fastapi import APIRouter, Request

from server.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    """Liveness probe + active-episode count.

    Pulls the global :class:`SessionManager` off ``app.state`` (set in
    :mod:`server.main`) and returns a :class:`HealthResponse` payload with
    a static ``status`` / ``version`` and a live ``episodes_loaded`` count
    from the registry.

    Args:
        request: The incoming FastAPI request. Used only as a handle to
            reach ``request.app.state.session_manager``; the request body
            and headers are not consulted.

    Returns:
        dict: A mapping FastAPI will validate against
            :class:`HealthResponse`, with keys ``status`` (``"ok"``),
            ``version`` (``"1.0.0"``), and ``episodes_loaded`` (the current
            length of ``SessionManager.sessions``).
    """
    sm = request.app.state.session_manager
    return {
        "status": "ok",
        "version": "1.0.0",
        "episodes_loaded": len(sm.sessions),
    }
