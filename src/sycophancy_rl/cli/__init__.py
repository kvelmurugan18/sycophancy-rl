"""``syco`` command-line interface.

This subpackage exposes the dependency-light command surface for data
preparation, planning, training, benchmarking, comparison, and runner
staging. Network or GPU work occurs only through explicit commands and
the underlying modules retain their own safety guards.

The CLI is exposed as a console-script entry point in ``pyproject.toml``
so ``pip install -e .`` produces a ``syco`` command.
"""

from __future__ import annotations


def main(argv=None):
    """Import the CLI lazily so ``python -m sycophancy_rl.cli.main`` stays warning-free."""

    from sycophancy_rl.cli.main import main as _main

    return _main(argv)

__all__ = ["main"]
