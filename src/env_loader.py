"""Centralized configuration loader for the Sycophancy RL project.

Loads environment variables from a .env file (if present) and exposes them
both as a generic getter pair (`get_env` / `require_env`) and as a set of
typed convenience constants used across the codebase.

Typical usage:

    from src.env_loader import HF_TOKEN, MODEL_NAME, require_env

    token = require_env("HF_TOKEN")          # raises if missing
    base_model = get_env("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file in the current working directory (or any
# ancestor) at import time. Existing process environment variables take
# precedence and are NOT overwritten.
load_dotenv()


def get_env(key: str, default: str = None) -> str:
    """Return the value of the environment variable `key`, or `default`.

    Thin wrapper around `os.getenv` kept for symmetry with `require_env` and
    to make call sites read uniformly across the codebase.
    """
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """Return the value of `key` or raise `ValueError` if it is unset.

    Use this for variables the application cannot function without (e.g.
    API tokens, required repo IDs).
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


# --- Convenience constants ----------------------------------------------------
# These are evaluated once at import time. The defaults mirror .env.example so
# that local development works even without a populated .env file.

# HuggingFace write token (required for pushing trained weights).
HF_TOKEN = get_env("HF_TOKEN")

# URL of the deployed environment server, consumed by the demo Space.
HF_SPACE_URL = get_env("HF_SPACE_URL", "http://localhost:7860")

# HuggingFace repo id (e.g. "username/sycophancy-resistant-model") used as
# the push target after training.
HF_REPO_ID = get_env("HF_REPO_ID")

# Base model identifier passed to the trainer.
MODEL_NAME = get_env("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# Port the FastAPI environment server binds to. Coerced to int so callers
# can hand it directly to uvicorn.
ENVIRONMENT_PORT = int(get_env("ENVIRONMENT_PORT", "7860"))

# Application log verbosity. One of: DEBUG, INFO, WARNING, ERROR.
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")


# --- Backwards-compatible helper ---------------------------------------------
def get_hf_token():
    """Return the HuggingFace token from env, .env, or Kaggle secrets.

    Preserved for older callers. Prefer `HF_TOKEN` (or `require_env("HF_TOKEN")`)
    in new code. Looks for `HUGGINGFACE_TOKEN` first, then falls back to
    `HF_TOKEN`.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None

    if dotenv_values is not None:
        try:
            values = dotenv_values()
        except Exception:
            values = None
        if values:
            token = values.get("HUGGINGFACE_TOKEN") or values.get("HF_TOKEN")
            if token:
                return token

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        UserSecretsClient = None

    if UserSecretsClient is not None:
        try:
            secret = UserSecretsClient().get_secret("HUGGINGFACE_TOKEN")
        except Exception:
            secret = None
        if secret:
            return secret

    raise RuntimeError(
        "Hugging Face token not found. Set HUGGINGFACE_TOKEN (or HF_TOKEN) in a "
        ".env file accessible to python-dotenv, or add it as a Kaggle secret "
        "named HUGGINGFACE_TOKEN."
    )
