"""Project-wide pytest configuration: keeps the suite offline by default.

This conftest enforces a hard "no external network" policy for the
entire test run.  Any test that accidentally attempts to open a real
socket or talk to Hugging Face / Kaggle fails loudly instead of
silently reaching the network.

The guard is installed at *import time* (i.e. as soon as pytest loads
this file) so it also protects test collection itself, not just the
execution of test bodies.

It blocks four real-network boundaries:

* :func:`socket.create_connection`  - the most common path;
* :meth:`socket.socket.connect`     - direct low-level connect;
* :meth:`socket.socket.connect_ex`  - non-raising variant;
* :func:`socket.getaddrinfo`        - DNS resolution.

All four are guarded without any third-party dependency.  Non-network
operations on sockets (creation, configuration, ``close``, ``bind`` on
an unbound socket, etc.) continue to work, so the FastAPI
``TestClient`` - which uses an in-process ASGI transport and never
opens a real socket - is unaffected.

The conftest also unconditionally sets the three Hugging Face
offline-mode environment variables (``HF_HUB_OFFLINE``,
``TRANSFORMERS_OFFLINE``, ``HF_DATASETS_OFFLINE``) to ``"1"`` so the HF
stack short-circuits before any HTTP call is attempted.

There is no opt-out.  Tests that need real network access do not
belong in this suite.

Why ``connect``/``connect_ex`` are still allowed for socketpair:

The Python standard library's :func:`socket.socketpair` (used by
:mod:`asyncio` on Windows and by some test utilities) creates two
local AF_INET/AF_INET6 loopback sockets and connects them with a
real ``connect`` call.  This is purely in-kernel loopback IPC, not a
network operation, so the guard permits calls made by a
:func:`socket.socketpair` invocation but blocks every other ``connect``
attempt.  The decision is local to each thread (via
:class:`threading.local`) and is only flipped on for the duration of
a :func:`socket.socketpair` call, scoped to loopback addresses
(``127.0.0.1`` / ``::1``).
"""

from __future__ import annotations

import os
import socket
import threading

# ---------------------------------------------------------------------------
# Environment: tell the HF stack we are offline *before* it is imported.
#
# Direct assignment (not ``setdefault``) so the values are guaranteed
# even if the host shell pre-set them to "0".
# ---------------------------------------------------------------------------

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Socket guard: block all real-network operations at the stdlib boundary.
# ---------------------------------------------------------------------------

_GUARD_MESSAGE = (
    "Test attempted to open a real network connection.  "
    "The test suite is offline by default; mock the boundary."
)


def _guard_error(*_args: object, **_kwargs: object) -> None:
    """Helper that always raises the network-guard error."""

    raise RuntimeError(_GUARD_MESSAGE)


# --- preserve the originals so we never accidentally recurse --------------

_original_create_connection = socket.create_connection
_original_getaddrinfo = socket.getaddrinfo
_original_socket = socket.socket
_original_socketpair = socket.socketpair


# --- per-thread loopback allowance -----------------------------------------

# During a :func:`socket.socketpair` call the calling thread flips
# ``_loopback_allowance.active`` to ``True`` so that the very next
# ``connect`` / ``connect_ex`` performed by the socketpair handshake
# (which only ever targets ``127.0.0.1`` / ``::1``) is allowed through
# the guard.  Everything else is blocked.
_loopback_allowance = threading.local()


def _is_loopback_address(address: object) -> bool:
    """Return ``True`` iff ``address`` resolves to a loopback host."""

    host = None
    if isinstance(address, (tuple, list)):
        if address:
            host = address[0]
    else:
        host = address
    if not isinstance(host, str):
        return False
    return host in ("127.0.0.1", "::1", "localhost")


# --- guarded socket class --------------------------------------------------


class _GuardedSocket(_original_socket):  # type: ignore[misc, valid-type]
    """A ``socket.socket`` subclass that blocks ``connect`` and ``connect_ex``.

    Every other method (``bind``, ``listen``, ``setsockopt``, ``close``,
    etc.) is inherited unchanged.

    ``connect`` / ``connect_ex`` are admitted only when:

    * the calling thread has ``_loopback_allowance.active`` set (the
      socketpair handshake flips it for the duration of one call), and
    * the target address is a loopback host (``127.0.0.1`` / ``::1``).

    Any other attempt raises :class:`RuntimeError` via the suite-wide
    network guard.
    """

    def connect(self, address: object) -> None:
        allowed = getattr(_loopback_allowance, "active", False) and _is_loopback_address(address)
        if not allowed:
            _guard_error(method="connect", address=address)
        return super().connect(address)  # type: ignore[arg-type]

    def connect_ex(self, address: object) -> int:
        allowed = getattr(_loopback_allowance, "active", False) and _is_loopback_address(address)
        if not allowed:
            _guard_error(method="connect_ex", address=address)
            return 0  # pragma: no cover
        return super().connect_ex(address)  # type: ignore[arg-type]


# --- guarded module-level factory -----------------------------------------


def _guarded_create_connection(
    address: object, *args: object, **kwargs: object
) -> socket.socket:
    """Replacement for :func:`socket.create_connection` used in tests.

    Raises :class:`RuntimeError` so any code path that tries to open
    a real TCP connection during a test fails immediately and loudly,
    instead of hanging or pulling weights from the internet.
    """

    _guard_error(method="create_connection", address=address, args=args, kwargs=kwargs)
    # Unreachable: _guard_error always raises.
    return _original_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]  # pragma: no cover


def _guarded_getaddrinfo(  # type: ignore[no-untyped-def]
    host, *args, **kwargs
):
    _guard_error(host=host, args=args, kwargs=kwargs)
    # Unreachable: _guard_error always raises.
    return _original_getaddrinfo(host, *args, **kwargs)  # pragma: no cover


# --- socketpair wrapper around the original -------------------------------


def _guarded_socketpair(*args: object, **kwargs: object) -> tuple:
    """Replacement for :func:`socket.socketpair`.

    The stdlib's :func:`socket.socketpair` uses two ``socket`` instances
    internally and stitches them together with a ``connect`` over the
    loopback interface.  When ``socket.socket`` is patched to
    :class:`_GuardedSocket` (as it is in this suite), the loopback
    ``connect`` calls inside the handshake would be blocked unless we
    flip the per-thread loopback allowance on for the duration of the
    call.  Anything else happening on this thread (e.g. real test code
    that inadvertently reaches for a network call) is unaffected.
    """

    previous = getattr(_loopback_allowance, "active", False)
    _loopback_allowance.active = True
    try:
        return _original_socketpair(*args, **kwargs)
    finally:
        _loopback_allowance.active = previous


# --- install the guards ----------------------------------------------------

socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
socket.socketpair = _guarded_socketpair  # type: ignore[assignment]
socket.socket = _GuardedSocket  # type: ignore[assignment]
