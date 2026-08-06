"""Health endpoints.

* ``GET /health/live``  — liveness probe; never touches the database
  or the SessionManager.  Always returns 200 when the process is up.
* ``GET /health/ready`` — readiness probe; verifies the SessionManager
  can be reached and reports the active-session count.  Returns 503
  when the manager is unavailable so an orchestrator can take the
  pod out of rotation.
"""

from fastapi import APIRouter, Request, Response

from sycophancy_rl.server.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "version": getattr(request.app.state, "package_version", "0.0.0"),
        "episodes_loaded": 0,
    }


@router.get("/health/ready", response_model=HealthResponse)
def readiness(request: Request, response: Response) -> dict[str, object]:
    try:
        request.app.state.session_manager.count()
        episodes = request.app.state.episode_store.count()
    except Exception:
        response.status_code = 503
        return {"status": "degraded", "version": "0.0.0", "episodes_loaded": 0}
    return {
        "status": "ok",
        "version": getattr(request.app.state, "package_version", "0.0.0"),
        "episodes_loaded": episodes,
    }


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
def legacy_health(request: Request) -> dict[str, object]:
    """Backwards-compatible health alias — same payload as readiness."""

    return readiness(request, Response())
