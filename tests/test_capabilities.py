from pathlib import Path

import pytest

from ai_diffusion.backend import capabilities, model_metadata
from ai_diffusion.backend.capabilities import resolve_capabilities
from ai_diffusion.backend.model_metadata import ModelMetadata, ModelMetadataStore
from ai_diffusion.backend.resources import Arch, ResourceKind


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    s = ModelMetadataStore(tmp_path / "model_metadata.json")
    monkeypatch.setattr(model_metadata, "metadata_store", s)
    return s


def test_defaults_follow_architecture(store: ModelMetadataStore):
    caps = resolve_capabilities("moss_image_sft_bf16.safetensors", Arch.moss)
    assert "edit" not in caps
    assert "cfg" in caps
    edit_caps = resolve_capabilities("moss_image_edit.safetensors", Arch.moss_e)
    assert "edit" in edit_caps
    assert "edit_reference" in edit_caps


def test_metadata_replaces_feature_capabilities(store: ModelMetadataStore):
    store.set(
        ResourceKind.checkpoint,
        "qwen_image_edit.safetensors",
        ModelMetadata(capabilities=["edit_reference"]),
    )
    caps = resolve_capabilities("qwen_image_edit.safetensors", Arch.qwen_e)
    assert "edit" not in caps  # user turned the edit model flag off
    assert "edit_reference" in caps
    assert "cfg" in caps  # sampling capabilities still come from the architecture


def test_metadata_can_enable_edit_on_plain_model(store: ModelMetadataStore):
    store.set(
        ResourceKind.checkpoint, "sdxl_turbo.safetensors", ModelMetadata(capabilities=["edit"])
    )
    caps = resolve_capabilities("sdxl_turbo.safetensors", Arch.sdxl)
    assert "edit" in caps
    assert "clip_skip" in caps


def test_metadata_without_capabilities_disables_all_features(store: ModelMetadataStore):
    store.set(ResourceKind.checkpoint, "plain.safetensors", ModelMetadata(name="Plain"))
    caps = resolve_capabilities("plain.safetensors", Arch.qwen_e)
    assert "edit" not in caps
    assert "edit_reference" not in caps


def test_unknown_checkpoint_uses_defaults(store: ModelMetadataStore):
    store.set(ResourceKind.lora, "other.safetensors", ModelMetadata(capabilities=["edit"]))
    caps = resolve_capabilities("qwen_image_edit.safetensors", Arch.qwen_e)
    assert "edit" in caps


def test_empty_checkpoint_uses_defaults(store: ModelMetadataStore):
    assert resolve_capabilities("", Arch.sd15) == capabilities.default_capabilities(Arch.sd15)


def test_describe_labels():
    text = capabilities.describe(["edit", "regions"])
    assert text == "Edit model, Regional prompts"


def test_feature_capabilities_are_subset_of_all():
    assert set(capabilities.feature_capabilities) <= set(capabilities.all_capabilities)
    assert set(capabilities.capability_labels) == set(capabilities.all_capabilities)
