from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import eventloop, util
from ..util import client_logger as log
from .client import Client
from .resources import Arch, ResourceKind, resource_folders

folder_names = resource_folders

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
            base_model=str(data.get("base_model", "")),
            updated=str(data.get("updated", "")),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ModelMetadataStore:
    def __init__(self, path: Path | None = None):
        self._cache: dict[str, dict[str, Any]] = {}
        self._client: Client | None = None
        self._path = path or util.user_data_dir / "model_metadata.json"
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Failed to read model metadata cache: {e!s}")
            self._cache = {}

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to write model metadata cache: {e!s}")

    @staticmethod
    def _key(kind: ResourceKind | str, filename: str) -> str:
        folder = kind if isinstance(kind, str) else folder_names.get(kind, kind.name)
        return f"{folder}/{filename}"

    def get(self, kind: ResourceKind | str, filename: str) -> ModelMetadata:
        return ModelMetadata.from_dict(self._cache.get(self._key(kind, filename)))

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

    def capabilities(self, kind: ResourceKind | str, filename: str) -> list[str]:
        metadata = self.get(kind, filename)
        return metadata.capabilities or ([metadata.type] if metadata.type else [])

    def arch_of(self, kind: ResourceKind | str, filename: str) -> Arch | None:
        return self.get(kind, filename).architecture


metadata_store = ModelMetadataStore()
