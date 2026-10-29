from __future__ import annotations

from collections.abc import Iterable

from .resources import Arch, ResourceKind, default_capabilities

# Capabilities describe what a diffusion model can do and which sampling options apply.
# They come from the metadata sidecar of a checkpoint and fall back to built-in
# defaults for the architecture, so users can change behavior without code changes.

edit = "edit"  # model makes changes to an input image instead of generating from noise
edit_reference = "edit_reference"  # model can also take reference images while generating
controlnet_inpaint = "controlnet_inpaint"  # inpainting with a controlnet model
regions = "regions"  # regional prompting
alpha = "alpha"  # preserve alpha channel of input images
cfg = "cfg"  # classifier free guidance scale applies
clip_skip = "clip_skip"
lcm = "lcm"
attention_guidance = "attention_guidance"

# Capabilities which determine editing features. These are the ones users can override
# in the model metadata. The remaining capabilities describe sampling parameters and
# always follow the architecture.
feature_capabilities = [edit, edit_reference, controlnet_inpaint, regions]

all_capabilities = [
    edit,
    edit_reference,
    controlnet_inpaint,
    regions,
    alpha,
    cfg,
    clip_skip,
    lcm,
    attention_guidance,
]

capability_labels = {
    edit: "Edit model",
    edit_reference: "Reference images",
    controlnet_inpaint: "ControlNet inpaint",
    regions: "Regional prompts",
    alpha: "Alpha channel",
    cfg: "Guidance scale",
    clip_skip: "Clip skip",
    lcm: "LCM LoRA",
    attention_guidance: "Self-attention guidance",
}


def _metadata_capabilities(checkpoint: str) -> set[str] | None:
    if not checkpoint:
        return None
    from .model_metadata import metadata_store

    if not metadata_store.has(ResourceKind.checkpoint, checkpoint):
        return None
    return set(metadata_store.get(ResourceKind.checkpoint, checkpoint).capabilities)


def resolve_capabilities(checkpoint: str, arch: Arch) -> frozenset[str]:
    """Capabilities of the model with the given file name and architecture.

    User edited metadata replaces the feature capabilities. Sampling capabilities
    always come from the architecture because they describe its inference setup.
    """
    defaults = default_capabilities(arch)
    metadata = _metadata_capabilities(checkpoint)
    if metadata is None:
        return defaults
    sampling = {c for c in defaults if c not in feature_capabilities}
    return frozenset(metadata | sampling)


def has(capabilities: Iterable[str], capability: str) -> bool:
    return capability in capabilities


def describe(capabilities: Iterable[str]) -> str:
    return ", ".join(capability_labels.get(c, c) for c in sorted(capabilities))
