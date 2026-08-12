# Threat model

The trainer container and the local API are local / single-tenant
tools.  The threat model is calibrated to that scope.

## What we defend against

* **Accidental data loss.**  The setup helpers never overwrite an
  existing pool or split.  Error messages recommend moving the
  artifacts aside rather than deleting them.
* **Test/real contamination.**  A real runner refuses to start when
  the training pool contains fixture rows.  A smoke runner refuses
  to start when the validation split contains benchmark rows.
* **Run-directory reuse.**  The trainer never silently overwrites an
  existing run directory.  Resume is allowed only when the manifest
  hashes match.
* **Pre-flight OOM.**  The preflight checks the VRAM, the disk free
  space, and the model/profile compatibility before doing any work.
* **Untrusted callers on the API.**  The API binds to `127.0.0.1`.  A
  non-default CORS allowlist (`SYCO_ORIGINS`) is required.  The
  wildcard origin is never combined with credentials.

## What we do NOT defend against

* **Hostile code.**  The trainer container is not a sandbox.  For
  untrusted workloads, run the trainer inside a VM or microVM
  (Firecracker, gVisor) and treat the local runner as a convenience
  layer for trusted users.
* **Public multi-tenant exposure.**  The API does not implement
  authentication or rate limiting.  External exposure requires an
  authenticating TLS proxy in front of the API.
* **Inference-time prompt injection.**  The model still produces
  arbitrary text.  Wrap the model call site in your own prompt
  sandbox if you expose it to untrusted user input.
* **Supply-chain compromise of upstream packages.**  The
  `requirements.txt` pins specific versions; CI runs `pip-audit`
  where available.  We do not run a fully isolated SBOM workflow,
  so a brand-new CVE in any pinned package may take a few days to
  reach the registry.
