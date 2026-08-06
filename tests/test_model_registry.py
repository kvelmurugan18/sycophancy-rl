"""Unit tests for the supported-model contract.

These tests do not download model weights.  They exercise the registry's
resolution rules, refusal of unsafe loading paths, and pinning guarantees.
"""

from __future__ import annotations

import pytest

from sycophancy_rl.training.model_registry import (
    ArchitectureFamily,
    TestStatus,
    is_safe_loading_kwargs,
    list_supported,
    resolve_profile,
    resolve_profile_safe,
)


def test_registry_has_at_least_two_models() -> None:
    profiles = list_supported()
    assert len(profiles) >= 3
    ids = {p.model_id for p in profiles}
    # The reference SmolLM2 model must stay in the registry.
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in ids
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    assert "mistralai/Mistral-7B-Instruct-v0.3" in ids


def test_no_two_models_share_a_revision() -> None:
    seen = {}
    for profile in list_supported():
        if profile.revision in seen:
            assert seen[profile.revision] == profile.model_id, (
                f"Revision {profile.revision} is shared between "
                f"{seen[profile.revision]} and {profile.model_id}"
            )
        seen[profile.revision] = profile.model_id


def test_reference_model_is_causal_lm() -> None:
    profile = resolve_profile("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert profile.architecture is ArchitectureFamily.CAUSAL_LM
    assert profile.requires_chat_template is True
    assert profile.supports_4bit is True
    assert profile.test_status is TestStatus.VALIDATED


def test_registered_revision_is_pinned() -> None:
    profile = resolve_profile("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert len(profile.revision) >= 7
    assert ":" not in profile.revision


def test_resolve_rejects_unknown_without_opt_in() -> None:
    with pytest.raises(ValueError, match="not in the supported registry"):
        resolve_profile("meta-llama/Llama-3-8B")


def test_unknown_with_explicit_revision_still_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="--allow-unpinned-model"):
        resolve_profile(
            "meta-llama/Llama-3-8B", model_revision="0" * 40
        )


def test_unknown_with_explicit_opt_in_is_registered() -> None:
    profile = resolve_profile_safe(
        "meta-llama/Llama-3-8B",
        model_revision="0" * 40,
        allow_unpinned=True,
    )
    assert profile.test_status is TestStatus.PENDING
    assert profile.model_id == "meta-llama/Llama-3-8B"


def test_custom_model_can_declare_qlora_contract() -> None:
    profile = resolve_profile(
        "org/custom-7b-instruct",
        model_revision="1" * 40,
        allow_unpinned=True,
        custom_supports_4bit=True,
        custom_lora_targets=("q_proj", "v_proj"),
        custom_context_length=8192,
        custom_min_vram_gib=14.0,
    )
    assert profile.supports_4bit is True
    assert profile.lora_targets == ("q_proj", "v_proj")
    assert profile.context_length == 8192
    assert profile.min_vram_gib == 14.0


def test_registered_7b_profiles_are_configured_for_qlora() -> None:
    for model_id in (
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ):
        profile = resolve_profile(model_id)
        assert profile.supports_4bit is True
        assert profile.min_vram_gib >= 14.0
        assert profile.test_status is TestStatus.CONFIGURED


def test_url_like_model_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_profile("https://example.com/model", allow_unpinned=True)
    with pytest.raises(ValueError):
        resolve_profile("file:///etc/passwd", allow_unpinned=True)
    with pytest.raises(ValueError):
        resolve_profile("./local-model", allow_unpinned=True)


def test_pinned_revision_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="registry pins"):
        resolve_profile(
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            model_revision="0" * 40,
        )


def test_pinned_revision_match_is_accepted() -> None:
    profile = resolve_profile_safe(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    )
    assert resolve_profile_safe(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision=profile.revision,
    ).model_id == profile.model_id


def test_unsafe_loading_kwargs_are_rejected() -> None:
    assert not is_safe_loading_kwargs({"use_unsafe_tokens": True})
    assert not is_safe_loading_kwargs({"force_download": True})
    assert not is_safe_loading_kwargs(
        {"dtype": "auto", "trust_remote_code": True}
    )
    assert is_safe_loading_kwargs({"dtype": "auto"})
    assert is_safe_loading_kwargs({})


def test_all_registered_revisions_are_real_commit_shas() -> None:
    for profile in list_supported():
        assert len(profile.revision) in range(40, 65)
        assert all(character in "0123456789abcdefABCDEF" for character in profile.revision)
