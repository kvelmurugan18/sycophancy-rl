"""Sycophancy-RL training, evaluation, and experiment orchestration package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sycophancy-rl")
except PackageNotFoundError:  # Source checkout without an editable install.
    __version__ = "unknown"

__all__ = ["__version__"]
