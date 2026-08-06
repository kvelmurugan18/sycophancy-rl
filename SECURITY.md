# Security policy

## Reporting a vulnerability

Use the repository's private GitHub Security Advisory reporting flow. Do not
open a public issue containing an exploitable secret, private dataset, or
working vulnerability. If private reporting is not enabled, contact the
repository owner through the verified contact on the GitHub profile and share
only enough public detail to establish a secure channel.

The current release line is the only supported line until a formal release
policy is published.

## Scope

The API and trainer are single-tenant local research tools. The API has no
built-in authentication and must remain bound to loopback unless an
authenticating TLS proxy is added. The trainer isolates trusted project code; it
is not a hostile-code sandbox. See `docs/threat_model.md`.

## Secrets

- `.env` is ignored by Git and excluded from Docker build contexts.
- Never place tokens in `.env.example`, source code, CLI arguments, notebooks,
  or issue logs.
- Kaggle tokens belong in Kaggle Secrets. The runner checks token presence but
  never prints token values.
- Treat a token displayed in a terminal, screenshot, or chat as compromised;
  revoke it and create a replacement.

## Supply chain

Runtime requirements are version-pinned per environment, CI performs dependency
and secret checks, containers run as non-root, and trainer capabilities are
dropped. These controls reduce risk but do not replace reviewing upstream model
repositories, package advisories, container images, and host GPU runtimes before
a public release.
