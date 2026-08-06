# Environment server image

This image serves the local diagnostic episode API (`/health`, `/reset`,
`/step`, and `/grader`). It contains the committed smoke fixture only and is
separate from the GPU trainer image.

From the repository root:

```powershell
docker compose up --build env-server
```

Compose publishes port 8000 on host loopback only. The service has no
authentication and must not be exposed directly to the internet. It runs as a
non-root user with a read-only root filesystem, no Linux capabilities, and no
host data or Docker socket mounts.
