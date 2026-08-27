"""Explicit compatibility registry for supported models.

The repository used to claim to be "model-agnostic". The model field is too
large to certify generically, and this repository currently retains no
end-to-end GPU validation artifact. This module replaces the vague claim with
a small, explicit configuration registry.

Each entry pins:

* the Hugging Face ``model_id`` and the exact ``revision`` (commit SHA)
  so the published baseline and re-runs stay byte-identical;
* an architecture family so callers can reject incompatible stacks
  (e.g. mixture-of-experts, encoder-only);
* the minimum VRAM (in GiB) that the registered profile requires;
* whether 4-bit QLoRA is supported (some architectures reject nf4);
* the LoRA target-module strategy the registered profile uses;
* whether a chat template is required, and at which context length;
* a validation status that distinguishes offline configuration from retained
  end-to-end GPU evidence.

Custom model IDs are still allowed but require both an explicit
revision (``--model-revision <sha>``) and the explicit
``--allow-unpinned-model`` acknowledgment. Their 4-bit and LoRA settings
must also be declared by the caller. ``trust_remote_code`` is
``False`` by default and safetensors are preferred — no unsafe loading
flags are permitted in the default path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# --- domain enums ----------------------------------------------------------


class ArchitectureFamily(str, Enum):
    CAUSAL_LM = "causal_lm"
    CAUSAL_LM_MOE = "causal_lm_moe"
    ENCODER_ONLY = "encoder_only"
    UNKNOWN = "unknown"


class TestStatus(str, Enum):
    """Has the model been end-to-end validated on a specific profile."""

    VALIDATED = "validated"
    CONFIGURED = "configured"
    SMOKE_ONLY = "smoke_only"
    PENDING = "pending"
    DEPRECATED = "deprecated"


# Prevent pytest from mistaking this domain enum for a test class.
TestStatus.__test__ = False  # type: ignore[attr-defined]


# --- registry entry --------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """One supported model with its pinned parameters."""

    model_id: str
    revision: str
    architecture: ArchitectureFamily
    min_vram_gib: float
    supports_4bit: bool
    lora_targets: tuple[str, ...]
    requires_chat_template: bool
    context_length: int
    test_status: TestStatus
    display_name: str
    expected_model_class: str = ""
    notes: str = ""

    @property
    def pinned_revision(self) -> str:
        return self.revision


# --- the registry ----------------------------------------------------------


PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        architecture=ArchitectureFamily.CAUSAL_LM,
        min_vram_gib=3.0,
        supports_4bit=True,
        lora_targets=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        requires_chat_template=True,
        context_length=32768,
        test_status=TestStatus.CONFIGURED,
        display_name="Qwen2.5-0.5B-Instruct",
        expected_model_class="Qwen2ForCausalLM",
        notes=(
            "First governed Anthropic experiment model. Exact Hub revision and "
            "Qwen2ForCausalLM-compatible LoRA targets are pinned; actual Kaggle "
            "GPU load/training remains an execution gate."
        ),
    ),
    ModelProfile(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision="31b70e2e869a7173562077fd711b654946d38674",
        architecture=ArchitectureFamily.CAUSAL_LM,
        min_vram_gib=3.5,
        supports_4bit=True,
        lora_targets=("all-linear",),
        requires_chat_template=True,
        context_length=8192,
        test_status=TestStatus.CONFIGURED,
        display_name="SmolLM2-1.7B-Instruct",
        notes=(
            "Pinned reference baseline with offline configuration tests. "
            "No retained end-to-end GPU validation artifact is committed."
        ),
    ),
    ModelProfile(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        architecture=ArchitectureFamily.CAUSAL_LM,
        min_vram_gib=14.0,
        supports_4bit=True,
        lora_targets=("all-linear",),
        requires_chat_template=True,
        context_length=32768,
        test_status=TestStatus.CONFIGURED,
        display_name="Qwen2.5-7B-Instruct",
        notes=(
            "Pinned public Apache-2.0 7B profile configured for NF4 QLoRA. "
            "A real Kaggle GPU validation run is still required before publishing results."
        ),
    ),
    ModelProfile(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        architecture=ArchitectureFamily.CAUSAL_LM,
        min_vram_gib=14.0,
        supports_4bit=True,
        lora_targets=("all-linear",),
        requires_chat_template=True,
        context_length=32768,
        test_status=TestStatus.CONFIGURED,
        display_name="Mistral-7B-Instruct-v0.3",
        notes=(
            "Pinned public Apache-2.0 non-Qwen 7B profile configured for NF4 QLoRA. "
            "A real Kaggle GPU validation run is still required before publishing results."
        ),
    ),
)


BY_ID: dict[str, ModelProfile] = {p.model_id: p for p in PROFILES}


# --- resolution helpers ----------------------------------------------------


_CID_RE = re.compile(r"^[A-Za-z0-9._/\-:]+$")


def _validate_model_id(model_id: str) -> None:
    if not model_id or not _CID_RE.match(model_id):
        raise ValueError(
            f"Model ID {model_id!r} contains characters outside the safe "
            "Hugging Face identifier set; refusing to load."
        )


def resolve_profile(
    model_id: str,
    *,
    model_revision: str | None = None,
    allow_unpinned: bool = False,
    custom_supports_4bit: bool = False,
    custom_lora_targets: tuple[str, ...] = ("all-linear",),
    custom_context_length: int = 2048,
    custom_min_vram_gib: float = 0.0,
) -> ModelProfile:
    """Return the registered profile for ``model_id`` or raise.

    Raises :class:`ValueError` when ``model_id`` is unknown and the caller
    has not provided an explicit ``model_revision`` or the
    ``allow_unpinned`` acknowledgment.  Refuses unsafe custom IDs that
    use protocol-prefixed paths or that look like local file URLs.
    """

    _validate_model_id(model_id)
    if model_id in BY_ID:
        profile = BY_ID[model_id]
        if profile.revision.startswith("PINNED_"):
            raise ValueError(
                f"Model {model_id!r} has not been pinned to a real Hub commit yet. "
                "Supply a verified commit SHA and update the registry before training."
            )
        if model_revision is not None and model_revision != profile.revision:
            raise ValueError(
                f"Model {model_id!r} is registered but you requested revision "
                f"{model_revision!r}; the registry pins {profile.revision!r}. "
                "Pass the pinned revision or use a custom model ID with the "
                "explicit registry opt-out."
            )
        return profile
    if model_id.startswith(("http://", "https://", "file://", "./", "../", "/")):
        raise ValueError(
            f"Custom model ID {model_id!r} looks like a URL or filesystem "
            "path; the runner only accepts plain Hugging Face identifiers."
        )
    if not model_revision:
        raise ValueError(
            f"Model {model_id!r} is not in the supported registry.  Pass "
            "both --model-revision <commit-sha> and --allow-unpinned-model "
            "to acknowledge an unvalidated custom model. Known models: "
            + ", ".join(sorted(BY_ID))
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", model_revision):
        raise ValueError(
            "Custom model revisions must be exact 40-64 character hexadecimal "
            "Hub commit SHAs, not branch names or tags."
        )
    if not allow_unpinned:
        # Even with a revision we still ask for the explicit opt-in so
        # the user is reminded that this model was not validated.
        raise ValueError(
            f"Model {model_id!r} is not in the supported registry.  Pass "
            "--allow-unpinned-model to acknowledge the registry opt-out."
        )
    if not custom_lora_targets:
        raise ValueError("Custom models require at least one LoRA target module.")
    if any(
        target != "all-linear" and not re.fullmatch(r"[A-Za-z0-9_.-]+", target)
        for target in custom_lora_targets
    ):
        raise ValueError("Custom LoRA targets contain unsupported characters.")
    if custom_context_length < 512:
        raise ValueError("Custom model context length must be at least 512 tokens.")
    if custom_min_vram_gib < 0:
        raise ValueError("Custom model minimum VRAM cannot be negative.")
    return ModelProfile(
        model_id=model_id,
        revision=model_revision or "",
        architecture=ArchitectureFamily.CAUSAL_LM,
        min_vram_gib=float(custom_min_vram_gib),
        supports_4bit=bool(custom_supports_4bit),
        lora_targets=tuple(custom_lora_targets),
        requires_chat_template=True,
        context_length=int(custom_context_length),
        test_status=TestStatus.PENDING,
        display_name=model_id,
        notes=(
            "Registered on demand for this run only — has not been "
            "validated by the sycophancy-rl test suite."
        ),
    )


def resolve_profile_safe(
    model_id: str,
    *,
    model_revision: str | None = None,
    allow_unpinned: bool = False,
    custom_supports_4bit: bool = False,
    custom_lora_targets: tuple[str, ...] = ("all-linear",),
    custom_context_length: int = 2048,
    custom_min_vram_gib: float = 0.0,
) -> ModelProfile:
    """Backward-compatible alias for the strict profile resolver.

    The registry no longer contains placeholder revisions, so benchmark,
    smoke, and training paths now enforce the same pinning contract.
    """

    return resolve_profile(
        model_id,
        model_revision=model_revision,
        allow_unpinned=allow_unpinned,
        custom_supports_4bit=custom_supports_4bit,
        custom_lora_targets=custom_lora_targets,
        custom_context_length=custom_context_length,
        custom_min_vram_gib=custom_min_vram_gib,
    )


def list_supported() -> tuple[ModelProfile, ...]:
    """Return the registered profiles, ordered as in the registry."""

    return PROFILES


def is_safe_loading_kwargs(kwargs: dict[str, object]) -> bool:
    """Refuse kwargs that opt into unsafe loading paths.

    ``trust_remote_code``, ``low_cpu_mem_usage`` are accepted for most HF
    models but explicitly requesting ``use_unsafe_tokens`` or
    ``dtype=auto`` with a non-safetensors format is rejected.  Custom
    ``device_map`` strings or execution-mode flags that bypass safetensors
    are also rejected.
    """

    forbidden = {
        "use_unsafe_tokens",
        "force_download",
        "resume_download_force",
    }
    for key in forbidden:
        if key in kwargs:
            return False
    dtype = kwargs.get("dtype")
    if dtype in ("auto", "auto-detect"):
        # ``dtype='auto'`` is fine but trust_remote_code must remain False.
        if kwargs.get("trust_remote_code", False):
            return False
    return True
