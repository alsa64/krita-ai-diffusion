from __future__ import annotations

from pathlib import Path

from ai_diffusion.backend import resources
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.model_metadata import ModelMetadata, ModelMetadataStore
from ai_diffusion.backend.resources import Arch, ResourceKind
from ai_diffusion.files import File, FileCollection, FileFilter, FileFormat
from ai_diffusion.ui.model_manager import collect_rows, server_models


def test_metadata_roundtrip():
    metadata = ModelMetadata(
        name="Test",
        arch="sdxl",
        type="lora",
        description="desc",
        triggers=["abc", "def"],
        capabilities=["edit"],
        base_model="SDXL",
    )
    restored = ModelMetadata.from_dict(metadata.to_dict())
    assert restored.name == "Test"
    assert restored.arch == "sdxl"
    assert restored.architecture is Arch.sdxl
    assert restored.triggers == ["abc", "def"]
    assert restored.capabilities == ["edit"]


def test_unknown_arch_is_none():
    assert ModelMetadata.from_dict({"arch": "not_an_arch"}).architecture is None
    assert ModelMetadata().architecture is None


def test_store_roundtrip(tmp_path: Path):
    path = tmp_path / "metadata.json"
    store = ModelMetadataStore(path)
    store.set(ResourceKind.lora, "style.safetensors", ModelMetadata(name="Style", arch="sd15"))
    assert store.has(ResourceKind.lora, "style.safetensors")
    assert store.get(ResourceKind.lora, "style.safetensors").name == "Style"
    assert store.all(ResourceKind.lora)["style.safetensors"].arch == "sd15"
    assert store.arch_of(ResourceKind.lora, "style.safetensors") is Arch.sd15

    reloaded = ModelMetadataStore(path)
    assert reloaded.get(ResourceKind.lora, "style.safetensors").name == "Style"
    assert not reloaded.has(ResourceKind.lora, "other.safetensors")


def test_downloads_are_filtered_by_kind_and_arch():
    checkpoints = resources.downloads(ResourceKind.checkpoint)
    assert checkpoints
    assert all(resource.kind is ResourceKind.checkpoint for resource, _ in checkpoints)
    assert all(file.url for _, file in checkpoints)

    sdxl = resources.downloads(ResourceKind.checkpoint, Arch.sdxl)
    assert sdxl
    assert all(resource.arch in (Arch.sdxl, Arch.all) for resource, _ in sdxl)


def test_server_models_and_rows():
    models = ClientModels()
    models.checkpoints["sd_xl_base_1.0.safetensors"] = CheckpointInfo(
        "sd_xl_base_1.0.safetensors", Arch.sdxl, FileFormat.checkpoint
    )
    models.loras = ["detail.safetensors"]

    assert server_models(models, ResourceKind.checkpoint) == ["sd_xl_base_1.0.safetensors"]
    assert server_models(models, ResourceKind.lora) == ["detail.safetensors"]

    rows = collect_rows(models, ResourceKind.checkpoint)
    installed = {row.filename: row for row in rows}
    assert installed["sd_xl_base_1.0.safetensors"].installed
    assert installed["sd_xl_base_1.0.safetensors"].arch == "sdxl"
    # registry entries which are not on the server become downloadable rows
    assert any(not row.installed and row.file is not None for row in rows)


def test_file_filter_row_filter():
    collection = FileCollection()
    collection.extend([File.remote("a.safetensors", FileFormat.lora)])
    proxy = FileFilter(collection)
    proxy.row_filter = lambda name: name.startswith("a")
    assert proxy.rowCount() == 1
    proxy.row_filter = lambda name: name.startswith("b")
    assert proxy.rowCount() == 0
    proxy.row_filter = None
    assert proxy.rowCount() == 1
