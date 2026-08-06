# Release blocker: choose the repository license

No `LICENSE` file is committed and `pyproject.toml` currently declares the code
proprietary. That is intentional until the owner makes the legal choice. A
public Git repository without an open-source license can be viewed and cloned,
but it does not grant others general permission to use, modify, or redistribute
the code.

## Common choices

| License | Main effect |
|---|---|
| Apache-2.0 | permissive use plus an explicit patent grant and notice obligations |
| MIT | short permissive license without the Apache patent language |
| GPL-3.0 | copyleft: distributed derivatives must remain under GPL terms |

Apache-2.0 is a practical default for a permissive ML engineering project, but
the repository owner must choose. This document is not legal advice.

## Owner action

1. Select the license and verify ownership of all contributed code.
2. Add the complete license text as root `LICENSE`.
3. Replace `license = { text = "Proprietary" }` in `pyproject.toml` with the
   selected SPDX-compatible expression and add the matching classifier if
   desired.
4. Review training-data, benchmark, and base-model licenses independently; the
   repository license does not override their terms.
5. Update `industrial_readiness.md` and the release/LinkedIn text only after the
   change is committed.

Until then, describe the repository as a release candidate shared for review,
not as an open-source release.
