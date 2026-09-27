"""Self-healing of model references stored in styles.

Styles reference checkpoints, LoRAs and VAEs by server-side file name. When a model
is renamed or moved into a subfolder on the server, those references break. This
module resolves references against the models a server currently provides and
records identity hashes, so references survive renames:

- Name-based matching (exact, basename/subfolder, close-match) resolves moves.
- Identity hashes (see ``backend.client.CheckpointInfo`` / ``ClientModels``)
  resolve arbitrary renames: the hash of the referenced model is recorded when a
  reference resolves, and a later lookup prefers a hash match over name matching.

Hashes are persisted per style in ``model-identity.json`` inside the user data
folder. Built-in styles are healed the same way; they are only modified in memory.
"""

from __future__ import annotations

import json

from .backend.api import CheckpointInput, LoraInput
from .backend.client import ClientModels
from .model_resolution import resolve_model_name
from .style import Style, Styles, StyleSettings
from .util import client_logger as log
from .util import user_data_dir

_kinds = ("checkpoints", "loras", "vae")
not_found = "not-found"


class ModelIdentityCache:
    """Maps style file names to the identity hashes of the models they reference."""

    def __init__(self, path=None):
        self._path = user_data_dir / "model-identity.json" if path is None else path
        self._entries: dict[str, dict[str, dict[str, str]]] = {}
        self._dirty = False
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                self._entries = {
                    style: {
                        k: dict(v) for k, v in kinds.items() if k in _kinds and isinstance(v, dict)
                    }
                    for style, kinds in data.items()
                    if isinstance(kinds, dict)
                }
        except Exception as e:
            log.warning(f"Failed to load model identity cache {self._path}: {e!s}")

    def save(self):
        if not self._dirty:
            return
        self._entries = {
            style: kinds for style, kinds in self._entries.items() if any(kinds.values())
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2))
        self._dirty = False

    def hash_of(self, style_filename: str, kind: str, name: str) -> str | None:
        return self._entries.get(style_filename, {}).get(kind, {}).get(name)

    def record(self, style_filename: str, kind: str, name: str, model_hash: str | None):
        if not model_hash or self.hash_of(style_filename, kind, name) == model_hash:
            return
        kinds = self._entries.setdefault(style_filename, {k: {} for k in _kinds})
        kinds.setdefault(kind, {})[name] = model_hash
        self._dirty = True

    def prune(self, existing: set[str]):
        remaining = {style: kinds for style, kinds in self._entries.items() if style in existing}
        if len(remaining) != len(self._entries):
            self._entries = remaining
            self._dirty = True


_instance: ModelIdentityCache | None = None


def identity_cache() -> ModelIdentityCache:
    global _instance
    if _instance is None:
        _instance = ModelIdentityCache()
    return _instance


def resolve_checkpoint(style: Style, name: str, models: ClientModels) -> str:
    return (
        resolve_model_name(
            name,
            list(models.checkpoints.keys()),
            reference_hash=identity_cache().hash_of(style.filename, "checkpoints", name),
            available_hashes={n: cp.hash for n, cp in models.checkpoints.items() if cp.hash},
        )
        or name
    )


def resolve_lora(style: Style, name: str, models: ClientModels) -> str:
    return (
        resolve_model_name(
            name,
            models.loras,
            reference_hash=identity_cache().hash_of(style.filename, "loras", name),
            available_hashes=models.lora_hashes,
            allow_fuzzy=False,
        )
        or name
    )


def resolve_vae(style: Style, name: str, models: ClientModels) -> str:
    if name in ("", StyleSettings.vae.default):
        return name
    return (
        resolve_model_name(
            name,
            models.vae,
            reference_hash=identity_cache().hash_of(style.filename, "vae", name),
            available_hashes=models.vae_hashes,
        )
        or name
    )


def checkpoint_input(style: Style, models: ClientModels) -> CheckpointInput:
    """Like ``Style.get_models``, but resolves references against current server models."""

    result = style.get_models(models.checkpoints)
    cache = identity_cache()

    checkpoint = next(
        (
            name
            for name in (resolve_checkpoint(style, cp, models) for cp in style.checkpoints)
            if name in models.checkpoints
        ),
        result.checkpoint,
    )
    if checkpoint != not_found:
        result.checkpoint = checkpoint
        cache.record(style.filename, "checkpoints", checkpoint, models.checkpoints[checkpoint].hash)

    loras: list[LoraInput] = []
    for lora in result.loras:
        name = resolve_lora(style, lora.name, models)
        if name in models.loras:
            loras.append(LoraInput(name, lora.strength, lora.storage_id))
            cache.record(style.filename, "loras", name, models.lora_hashes.get(name))
        else:
            loras.append(lora)  # may be a local file which gets uploaded instead
    result.loras = loras

    vae = resolve_vae(style, result.vae, models)
    if vae != result.vae and (vae in models.vae or vae == StyleSettings.vae.default):
        result.vae = vae
    if vae != StyleSettings.vae.default:
        cache.record(style.filename, "vae", vae, models.vae_hashes.get(vae))

    cache.save()
    return result


def heal_styles(models: ClientModels) -> bool:
    """Resolve references of all styles against current server models.

    Updates style names in memory (built-in styles included) and records identity
    hashes. Returns whether any style was modified.
    """

    cache = identity_cache()
    changed = False
    styles = Styles.list()
    for style in styles:
        checkpoints = [resolve_checkpoint(style, cp, models) for cp in style.checkpoints]
        if checkpoints != style.checkpoints:
            style.checkpoints = checkpoints
            changed = True
        for cp in checkpoints:
            if info := models.checkpoints.get(cp):
                cache.record(style.filename, "checkpoints", cp, info.hash)

        for lora in style.loras:
            name = lora.get("name", "")
            if not isinstance(name, str):
                continue
            resolved = resolve_lora(style, name, models)
            if resolved != name:
                lora["name"] = resolved
                changed = True
            if model_hash := models.lora_hashes.get(resolved):
                cache.record(style.filename, "loras", resolved, model_hash)

        vae = resolve_vae(style, style.vae, models)
        if vae != style.vae:
            style.vae = vae
            changed = True
        if model_hash := models.vae_hashes.get(vae):
            cache.record(style.filename, "vae", vae, model_hash)

    cache.prune({s.filename for s in styles})
    cache.save()
    if changed:
        log.info("Updated model references in styles after server models changed")
        styles.changed.emit()
    return changed
