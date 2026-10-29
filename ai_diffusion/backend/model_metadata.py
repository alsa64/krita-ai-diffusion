from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import eventloop, util
from ..util import client_logger as log
from .client import Client
from .resources import Arch, ResourceId, ResourceKind, resource_folders

folder_names = resource_folders
_folder_kinds = {folder: kind for kind, folder in resource_folders.items()}


def _kind_from_folder(folder: str) -> ResourceKind | None:
    return _folder_kinds.get(folder)

# Types a model can be used for. Capabilities are not limited by architecture: the
# plugin asks the model for its type instead of using built-in per-architecture tables.
model_types = [
    "checkpoint",
    "edit",
    "inpaint",
    "outpaint",
    "controlnet",
    "style",
    "upscale",
    "text_encoder",
    "vae",
]


@dataclass
class ModelMetadata:
    """User and plugin maintained information about a model file.

    Stored in a sidecar file next to the model on the server, and cached locally so it is
    available when the server is offline.
    """

    name: str = ""
    arch: str = ""
    type: str = ""
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    role: str = ""
    base_model: str = ""
    updated: str = ""

    @property
    def architecture(self) -> Arch | None:
        return Arch.__members__.get(self.arch) if self.arch else None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.name:
            data["name"] = self.name
        if self.arch:
            data["arch"] = self.arch
        if self.type:
            data["type"] = self.type
        if self.description:
            data["description"] = self.description
        if self.triggers:
            data["triggers"] = list(self.triggers)
        if self.capabilities:
            data["capabilities"] = list(self.capabilities)
        if self.role:
            data["role"] = self.role
        if self.base_model:
            data["base_model"] = self.base_model
        if self.updated:
            data["updated"] = self.updated
        return data

    def as_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        if self.updated:
            data["updated"] = self.updated
        return data

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> ModelMetadata:
        data = data or {}
        return ModelMetadata(
            name=str(data.get("name", "")),
            arch=str(data.get("arch", "")),
            type=str(data.get("type", "")),
            description=str(data.get("description", "")),
            triggers=[str(t) for t in data.get("triggers", [])],
            capabilities=[str(c) for c in data.get("capabilities", [])],
            role=str(data.get("role", "")),
            base_model=str(data.get("base_model", "")),
            updated=str(data.get("updated", "")),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _first(header: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = header.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _arch_from_header(header: dict[str, str]) -> str:
    for value in (
        _first(header, "modelspec.architecture"),
        _first(header, "ss_base_model_version"),
        _first(header, "ss_sd_model_name"),
    ):
        name = value.lower().replace("-", "_").replace(" ", "_")
        if not name:
            continue
        arch = Arch.from_string(name)
        if arch is not None:
            return arch.name
        for keyword, keyword_arch in header_arch_keywords:
            if keyword in name:
                return keyword_arch.name
    return ""


header_arch_keywords = [
    ("illustrious", Arch.illu),
    ("noobai", Arch.illu),
    ("pony", Arch.sdxl),
    ("sdxl", Arch.sdxl),
    ("sd3", Arch.sd3),
    ("sd_v3", Arch.sd3),
    ("flux_2", Arch.flux2_4b),
    ("flux2", Arch.flux2_4b),
    ("flux", Arch.flux),
    ("qwen_image_edit", Arch.qwen_e),
    ("qwen_image", Arch.qwen),
    ("qwen", Arch.qwen),
    ("moss", Arch.moss),
    ("z_image", Arch.zimage),
    ("zimage", Arch.zimage),
    ("anima", Arch.anima),
    ("chroma", Arch.chroma),
    ("sd_v1", Arch.sd15),
    ("sd_1", Arch.sd15),
    ("sd15", Arch.sd15),
]


def _triggers_from_header(header: dict[str, str]) -> list[str]:
    value = _first(header, "modelspec.trigger_phrase", "ss_trigger_words")
    return [t.strip() for t in value.split(",") if t.strip()]


class ModelMetadataStore:
    def __init__(self, path: Path | None = None):
        self._cache: dict[str, dict[str, Any]] = {}
        self._headers: dict[str, dict[str, str]] = {}
        self._client: Client | None = None
        self._path = path or util.user_data_dir / "model_metadata.json"
        self._header_path = self._path.with_name("model_headers.json")
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                self._cache = json.loads(self._path.read_text(encoding="utf-8"))
            if self._header_path.exists():
                self._headers = json.loads(self._header_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Failed to read model metadata cache: {e!s}")
            self._cache = {}
            self._headers = {}

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
            self._header_path.write_text(json.dumps(self._headers, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to write model metadata cache: {e!s}")

    @staticmethod
    def _key(kind: ResourceKind | str, filename: str) -> str:
        folder = kind if isinstance(kind, str) else folder_names.get(kind, kind.name)
        return f"{folder}/{filename}"

    def get(self, kind: ResourceKind | str, filename: str) -> ModelMetadata:
        key = self._key(kind, filename)
        return self._with_header(ModelMetadata.from_dict(self._cache.get(key)), key)

    def _with_header(self, metadata: ModelMetadata, key: str) -> ModelMetadata:
        header = self._headers.get(key)
        if not header:
            return metadata
        if not metadata.arch:
            metadata.arch = _arch_from_header(header)
        if not metadata.name:
            metadata.name = _first(header, "modelspec.title", "ss_output_name")
        if not metadata.base_model:
            metadata.base_model = _first(header, "ss_base_model_version", "modelspec.architecture")
        if not metadata.description:
            metadata.description = _first(header, "modelspec.description")
        if not metadata.triggers:
            metadata.triggers = _triggers_from_header(header)
        return metadata

    def header(self, kind: ResourceKind | str, filename: str) -> dict[str, str]:
        return self._headers.get(self._key(kind, filename), {})

    def has(self, kind: ResourceKind | str, filename: str) -> bool:
        return self._key(kind, filename) in self._cache

    def all(self, kind: ResourceKind | str) -> dict[str, ModelMetadata]:
        folder = kind if isinstance(kind, str) else folder_names.get(kind, kind.name)
        prefix = f"{folder}/"
        return {
            key[len(prefix) :]: ModelMetadata.from_dict(value)
            for key, value in self._cache.items()
            if key.startswith(prefix)
        }

    def set(self, kind: ResourceKind | str, filename: str, metadata: ModelMetadata):
        metadata.updated = _now()
        folder = kind if isinstance(kind, str) else folder_names.get(kind, kind.name)
        self._cache[self._key(folder, filename)] = metadata.as_dict()
        self._save()
        if self._client is not None:
            eventloop.run(self._push(self._client, folder, filename, metadata))

    async def _push(self, client: Client, folder: str, filename: str, metadata: ModelMetadata):
        push = getattr(client, "set_model_metadata", None)
        if push is None:
            return
        try:
            await push(folder, filename, metadata.as_dict())
        except Exception as e:
            log.warning(f"Failed to write metadata for {folder}/{filename}: {e!s}")

    def attach(self, client: Client | None):
        self._client = client

    async def sync(self, client: Client):
        """Merge metadata stored on the server into the local cache."""
        self.attach(client)
        read = getattr(client, "get_folder_metadata", None)
        if read is None:
            return
        changed = False
        for folder in folder_names.values():
            try:
                data = await read(folder)
            except Exception as e:
                log.debug(f"No metadata for {folder}: {e!s}")
                continue
            if not data:
                continue
            for filename, value in data.items():
                key = f"{folder}/{filename}"
                if self._cache.get(key) != value:
                    self._cache[key] = value
                    changed = True
        if changed:
            self._save()

    async def sync_headers(self, client: Client | None = None, force=False):
        """Read metadata embedded in model files which is not in the cache yet."""
        client = client or self._client
        if client is None:
            return
        read = getattr(client, "get_model_header", None)
        if read is None:
            return
        changed = False
        for kind, folder in folder_names.items():
            for filename in self.model_files(client, kind):
                key = f"{folder}/{filename}"
                if key in self._headers and not force:
                    continue
                try:
                    header = await read(folder, filename)
                except Exception as e:
                    log.debug(f"No header for {folder}/{filename}: {e!s}")
                    continue
                if header:
                    self._headers[key] = {
                        str(k): str(v) for k, v in header.items() if isinstance(v, str)
                    }
                    changed = True
        if changed:
            self._save()

    @staticmethod
    def model_files(client: Client, kind: ResourceKind) -> list[str]:
        models = getattr(client, "models", None)
        if models is None:
            return []
        if kind is ResourceKind.checkpoint:
            return list(getattr(models, "checkpoints", {}))
        if kind is ResourceKind.lora:
            return list(getattr(models, "loras", []))
        if kind is ResourceKind.vae:
            return list(getattr(models, "vae", []))
        return []

    def capabilities(self, kind: ResourceKind | str, filename: str) -> list[str]:
        metadata = self.get(kind, filename)
        return metadata.capabilities or ([metadata.type] if metadata.type else [])

    def arch_of(self, kind: ResourceKind | str, filename: str) -> Arch | None:
        return self.get(kind, filename).architecture

    def role_of(self, kind: ResourceKind | str, filename: str) -> str:
        return self.get(kind, filename).role

    def apply_roles(self, models):
        """Register models which declare a role (e.g. a LoRA used for inpainting).

        Roles take the place of hardcoded file name patterns: a model with
        `role="inpaint"` becomes the inpaint model of its architecture.
        """
        added = 0
        for key, data in self._cache.items():
            meta = self._with_header(ModelMetadata.from_dict(data), key)
            arch = meta.architecture
            if arch is None or not meta.role:
                continue
            kind = _kind_from_folder(key.split("/", 1)[0])
            if kind is None:
                continue
            filename = key.split("/", 1)[1]
            if kind is ResourceKind.checkpoint:
                continue
            resource = ResourceId(kind, arch, meta.role)
            if models.resources.get(resource.string) is None:
                models.resources[resource.string] = filename
                added += 1
        return added


metadata_store = ModelMetadataStore()
