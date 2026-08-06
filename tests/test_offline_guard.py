"""Tests for the suite-wide offline network guard.

These tests verify that ``tests/conftest.py`` actually enforces the
"no external network" policy the rest of the suite relies on:

* the three HF offline env vars are set to ``"1"``;
* the real-network socket boundaries all raise :class:`RuntimeError`;
* loopback IPC (``socket.socketpair``) still works;
* the FastAPI ``TestClient`` continues to function under the guard.
"""

from __future__ import annotations

import os
import socket

import pytest

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_hf_offline_env_vars_are_set_to_one() -> None:
    """All three HF offline env vars must be ``"1"`` after conftest import."""

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


# ---------------------------------------------------------------------------
# Real-network boundaries are blocked
# ---------------------------------------------------------------------------


def test_create_connection_is_blocked() -> None:
    """``socket.create_connection`` must raise :class:`RuntimeError`."""

    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80))


def test_connect_is_blocked() -> None:
    """``socket.socket.connect`` must raise :class:`RuntimeError`."""

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError):
            s.connect(("example.com", 80))
    finally:
        s.close()


def test_connect_ex_is_blocked() -> None:
    """``socket.socket.connect_ex`` must raise :class:`RuntimeError`."""

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError):
            s.connect_ex(("example.com", 80))
    finally:
        s.close()


def test_getaddrinfo_is_blocked() -> None:
    """``socket.getaddrinfo`` must raise :class:`RuntimeError`."""

    with pytest.raises(RuntimeError):
        socket.getaddrinfo("example.com", 80)


# ---------------------------------------------------------------------------
# Loopback IPC still works
# ---------------------------------------------------------------------------


def test_socketpair_send_receive_works() -> None:
    """``socket.socketpair`` must still produce a working send/recv pair."""

    a, b = socket.socketpair()
    try:
        a.sendall(b"hello")
        data = b.recv(5)
        assert data == b"hello"
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# FastAPI TestClient works under the guard
# ---------------------------------------------------------------------------


def test_fastapi_testclient_works_under_offline_guard() -> None:
    """The FastAPI in-process TestClient must exercise routes without
    opening real sockets.

    Importing the app here (rather than via package import) so the
    test does not depend on the test collection ordering injecting
    any other module first.
    """

    from fastapi.testclient import TestClient

    from sycophancy_rl.server.main import create_app

    app = create_app()
    client = TestClient(app)

    # ``/health/live`` is part of the router set; hitting it exercises
    # the in-process ASGI transport without any real socket activity.
    response = client.get("/health/live")
    assert response.status_code == 200
