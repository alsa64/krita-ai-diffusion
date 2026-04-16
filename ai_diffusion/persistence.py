from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any

from PyQt5.QtCore import QByteArray, QObject
from PyQt5.QtGui import QImageReader
from PyQt5.QtWidgets import QMessageBox

from . import eventloop
from .backend.api import FillMode, InpaintMode
from .defaults import defaults
from .image import ImageCollection
from .localization import translate as _
from .model.control import ControlLayer, ControlLayerList
from .model.custom_workflow import CustomGenerationMode, CustomWorkspace
from .model.jobs import Job, JobKind, JobParams, JobQueue
from .model.model import (
    AnimationTargetLayerDefault,
    DocumentModel,
    InpaintContext,
    QueueMode,
    SamplingQuality,
    TileOverlapMode,
    Workspace,
)
from .model.properties import deserialize, serialize
from .model.region import Region, RootRegion
from .settings import ImageFileFormat, Setting, settings
from .style import Style, Styles
from .util import client_logger as log
from .util import encode_json

# Version of the persistence format, increment when there are breaking changes
version = 1

document_defaults_schema = {
    "workspace": Setting(_("Open Workspace"), Workspace.generation),
}

generation_defaults_schema = {
    "style": Setting(_("Style Preset"), "", _("Style selected for new documents.")),
    "strength": Setting(_("Strength"), 1.0),
    "region_only": Setting(_("Region Only"), False),
    "edit_mode": Setting(_("Edit Mode"), False),
    "batch_count": Setting(_("Batch Count"), 1),
    "fixed_seed": Setting(_("Fixed Seed"), False),
    "resolution_multiplier": Setting(_("Resolution"), 1.5),
    "use_smart_resolution": Setting(_("Smart Resolution"), True),
    "smart_rotate": Setting(_("Smart Rotate"), False),
    "queue_mode": Setting(_("Queue Mode"), QueueMode.back),
    "translation_enabled": Setting(_("Prompt Translation"), True),
    "layer_count": Setting(_("Layer Count"), 4),
    "inpaint_mode": Setting(_("Inpaint Mode"), InpaintMode.automatic),
    "inpaint_fill": Setting(_("Inpaint Fill"), FillMode.neutral),
    "inpaint_use_model": Setting(_("Use Inpaint Model"), True),
    "inpaint_use_prompt_focus": Setting(_("Use Prompt Focus"), False),
    "inpaint_context": Setting(_("Inpaint Context"), InpaintContext.automatic),
}

upscaling_defaults_schema = {
    "upscale_model": Setting(_("Upscale Model"), ""),
    "factor": Setting(_("Scale"), 2.0),
    "use_diffusion": Setting(_("Use Diffusion"), True),
    "strength": Setting(_("Strength"), 0.3),
    "unblur_strength": Setting(_("Image Guidance"), 0.5),
    "tile_overlap_mode": Setting(_("Tile Overlap Mode"), TileOverlapMode.auto),
    "tile_overlap": Setting(_("Tile Overlap"), 128),
    "use_prompt": Setting(_("Use Prompt"), False),
}

live_defaults_schema = {
    "strength": Setting(_("Strength"), 0.3),
    "recording_format": Setting(_("Recording Format"), ImageFileFormat.webp),
    "recording_folder_name_format": Setting(
        _("Recording Folder Name Template"),
        "{document_name}.live-frames",
        "Template for naming the folder used for recorded live frames. Available keys: {document_name}, {document_file}.",
    ),
    "recording_frame_name_format": Setting(
        _("Recording Frame Name Template"),
        "frame-{index}.{extension}",
        "Template for naming recorded live frames. Available keys: {index}, {extension}.",
    ),
}

animation_defaults_schema = {
    "sampling_quality": Setting(_("Sampling Quality"), SamplingQuality.fast),
    "target_layer_default": Setting(_("Target Layer"), AnimationTargetLayerDefault.active),
    "batch_mode": Setting(_("Batch Mode"), True),
    "batch_folder_name_format": Setting(
        _("Batch Output Folder Template"),
        "{document_name}.animation",
        "Template for naming the animation batch output folder. Available keys: {document_name}, {document_file}.",
    ),
    "batch_frame_name_format": Setting(
        _("Batch Frame Name Template"),
        "frame-{frame}.{extension}",
        "Template for naming generated animation batch frames. Available keys: {frame}, {extension}.",
    ),
}

custom_defaults_schema = {
    "workflow_id": Setting(
        _("Workflow"), "", _("Workflow selected for new custom-workspace documents.")
    ),
    "mode": Setting(_("Mode"), CustomGenerationMode.regular),
    "params_ui_height": Setting(
        _("Parameters Height"),
        100,
        _("Initial height of the custom workflow parameters area in new documents."),
    ),
}

workspace_defaults_schema = {
    Workspace.generation: generation_defaults_schema,
    Workspace.upscaling: upscaling_defaults_schema,
    Workspace.live: live_defaults_schema,
    Workspace.animation: animation_defaults_schema,
    Workspace.custom: custom_defaults_schema,
}


def load_document_defaults():
    return defaults.read_section("document", document_defaults_schema)


def save_document_defaults(values: dict[str, Any]):
    defaults.write_section("document", values, document_defaults_schema)


def load_workspace_defaults(workspace: Workspace):
    return defaults.read_section(
        ("workspaces", workspace.name), workspace_defaults_schema[workspace]
    )


def save_workspace_defaults(workspace: Workspace, values: dict[str, Any]):
    defaults.write_section(
        ("workspaces", workspace.name), values, workspace_defaults_schema[workspace]
    )


class RecentlyUsedSync:
    """Stores the most recently used parameters for various settings across all models.
    This is used to initialize new models with the last used parameters if they are
    created from scratch (not opening an existing .kra with stored settings).
    """

    @staticmethod
    def from_settings():
        recent = RecentlyUsedSync()
        recent._migrate_legacy_settings()
        return recent

    def track(self, model: DocumentModel):
        try:
            if _find_annotation(model.document, "ui.json") is None:
                document = load_document_defaults()
                generation = load_workspace_defaults(Workspace.generation)
                upscaling = load_workspace_defaults(Workspace.upscaling)
                live = load_workspace_defaults(Workspace.live)
                animation = load_workspace_defaults(Workspace.animation)
                custom = load_workspace_defaults(Workspace.custom)

                model.workspace = document["workspace"]
                model.style = Styles.list().find(generation["style"]) or Styles.list().default
                model.strength = generation["strength"]
                model.region_only = generation["region_only"]
                model.edit_mode = generation["edit_mode"]
                model.batch_count = generation["batch_count"]
                model.fixed_seed = generation["fixed_seed"]
                model.resolution_multiplier = generation["resolution_multiplier"]
                model.use_smart_resolution = generation["use_smart_resolution"]
                model.smart_rotate = generation["smart_rotate"]
                model.queue_mode = generation["queue_mode"]
                model.translation_enabled = generation["translation_enabled"]
                model.layer_count = generation["layer_count"]
                model.inpaint.mode = generation["inpaint_mode"]
                model.inpaint.fill = generation["inpaint_fill"]
                model.inpaint.use_inpaint = generation["inpaint_use_model"]
                model.inpaint.use_prompt_focus = generation["inpaint_use_prompt_focus"]
                if generation["inpaint_context"] != InpaintContext.layer_bounds:
                    model.inpaint.context = generation["inpaint_context"]

                model.upscale.upscaler = upscaling["upscale_model"]
                model.upscale.factor = upscaling["factor"]
                model.upscale.use_diffusion = upscaling["use_diffusion"]
                model.upscale.strength = upscaling["strength"]
                model.upscale.unblur_strength = upscaling["unblur_strength"]
                model.upscale.tile_overlap_mode = upscaling["tile_overlap_mode"]
                model.upscale.tile_overlap = upscaling["tile_overlap"]
                model.upscale.use_prompt = upscaling["use_prompt"]

                model.live.strength = live["strength"]
                model.live.recording_format = live["recording_format"]
                model.live.recording_folder_name_format = live["recording_folder_name_format"]
                model.live.recording_frame_name_format = live["recording_frame_name_format"]

                model.animation.sampling_quality = animation["sampling_quality"]
                model.animation.target_layer_default = animation["target_layer_default"]
                model.animation.batch_mode = animation["batch_mode"]
                model.animation.batch_folder_name_format = animation["batch_folder_name_format"]
                model.animation.batch_frame_name_format = animation["batch_frame_name_format"]

                model.custom.mode = custom["mode"]
                model.custom.params_ui_height = custom["params_ui_height"]
                if workflow_id := custom["workflow_id"]:
                    model.custom.workflow_id = workflow_id
        except Exception as e:
            log.warning(f"Failed to apply default settings to new document: {type(e)} {e}")

        model.workspace_changed.connect(self._set_document_default("workspace"))
        model.style_changed.connect(self._set(Workspace.generation, "style"))
        model.strength_changed.connect(self._set(Workspace.generation, "strength"))
        model.region_only_changed.connect(self._set(Workspace.generation, "region_only"))
        model.edit_mode_changed.connect(self._set(Workspace.generation, "edit_mode"))
        model.batch_count_changed.connect(self._set(Workspace.generation, "batch_count"))
        model.fixed_seed_changed.connect(self._set(Workspace.generation, "fixed_seed"))
        model.resolution_multiplier_changed.connect(
            self._set(Workspace.generation, "resolution_multiplier")
        )
        model.use_smart_resolution_changed.connect(
            self._set(Workspace.generation, "use_smart_resolution")
        )
        model.smart_rotate_changed.connect(self._set(Workspace.generation, "smart_rotate"))
        model.queue_mode_changed.connect(self._set(Workspace.generation, "queue_mode"))
        model.translation_enabled_changed.connect(
            self._set(Workspace.generation, "translation_enabled")
        )
        model.layer_count_changed.connect(self._set(Workspace.generation, "layer_count"))
        model.inpaint.mode_changed.connect(self._set(Workspace.generation, "inpaint_mode"))
        model.inpaint.fill_changed.connect(self._set(Workspace.generation, "inpaint_fill"))
        model.inpaint.use_inpaint_changed.connect(
            self._set(Workspace.generation, "inpaint_use_model")
        )
        model.inpaint.use_prompt_focus_changed.connect(
            self._set(Workspace.generation, "inpaint_use_prompt_focus")
        )
        model.inpaint.context_changed.connect(self._set(Workspace.generation, "inpaint_context"))

        model.upscale.upscaler_changed.connect(self._set(Workspace.upscaling, "upscale_model"))
        model.upscale.factor_changed.connect(self._set(Workspace.upscaling, "factor"))
        model.upscale.use_diffusion_changed.connect(self._set(Workspace.upscaling, "use_diffusion"))
        model.upscale.strength_changed.connect(self._set(Workspace.upscaling, "strength"))
        model.upscale.unblur_strength_changed.connect(
            self._set(Workspace.upscaling, "unblur_strength")
        )
        model.upscale.tile_overlap_mode_changed.connect(
            self._set(Workspace.upscaling, "tile_overlap_mode")
        )
        model.upscale.tile_overlap_changed.connect(self._set(Workspace.upscaling, "tile_overlap"))
        model.upscale.use_prompt_changed.connect(self._set(Workspace.upscaling, "use_prompt"))

        model.live.strength_changed.connect(self._set(Workspace.live, "strength"))
        model.live.recording_format_changed.connect(self._set(Workspace.live, "recording_format"))
        model.live.recording_folder_name_format_changed.connect(
            self._set(Workspace.live, "recording_folder_name_format")
        )
        model.live.recording_frame_name_format_changed.connect(
            self._set(Workspace.live, "recording_frame_name_format")
        )

        model.animation.sampling_quality_changed.connect(
            self._set(Workspace.animation, "sampling_quality")
        )
        model.animation.target_layer_default_changed.connect(
            self._set(Workspace.animation, "target_layer_default")
        )
        model.animation.batch_mode_changed.connect(self._set(Workspace.animation, "batch_mode"))
        model.animation.batch_folder_name_format_changed.connect(
            self._set(Workspace.animation, "batch_folder_name_format")
        )
        model.animation.batch_frame_name_format_changed.connect(
            self._set(Workspace.animation, "batch_frame_name_format")
        )

        model.custom.workflow_id_changed.connect(self._set(Workspace.custom, "workflow_id"))
        model.custom.mode_changed.connect(self._set(Workspace.custom, "mode"))
        model.custom.params_ui_height_changed.connect(
            self._set(Workspace.custom, "params_ui_height")
        )

    def _set(self, workspace: Workspace, key: str):
        def setter(value):
            if isinstance(value, Style):
                value = value.filename
            if isinstance(value, Enum):
                value = value.name
            values = load_workspace_defaults(workspace)
            values[key] = value
            save_workspace_defaults(workspace, values)

        return setter

    def _set_document_default(self, key: str):
        def setter(value):
            if isinstance(value, Enum):
                value = value.name
            values = load_document_defaults()
            values[key] = value
            save_document_defaults(values)

        return setter

    def _migrate_legacy_settings(self):
        has_workspace_defaults = any(
            defaults.read_section(("workspaces", workspace.name), schema)
            != {key: setting.default for key, setting in schema.items()}
            for workspace, schema in workspace_defaults_schema.items()
        )
        has_document_defaults = load_document_defaults() != {
            key: setting.default for key, setting in document_defaults_schema.items()
        }
        if (has_workspace_defaults or has_document_defaults) or not settings.document_defaults:
            return

        legacy = settings.document_defaults
        save_document_defaults({
            "workspace": legacy.get("workspace", Workspace.generation.name),
        })
        save_workspace_defaults(
            Workspace.generation,
            {
                "style": legacy.get("style", ""),
                "strength": legacy.get("strength", 1.0),
                "region_only": legacy.get("region_only", False),
                "edit_mode": legacy.get("edit_mode", False),
                "batch_count": legacy.get("batch_count", 1),
                "fixed_seed": legacy.get("fixed_seed", False),
                "resolution_multiplier": legacy.get("resolution_multiplier", 1.5),
                "use_smart_resolution": legacy.get("use_smart_resolution", True),
                "smart_rotate": legacy.get("smart_rotate", False),
                "queue_mode": legacy.get("queue_mode", QueueMode.back.name),
                "translation_enabled": legacy.get("translation_enabled", True),
                "layer_count": legacy.get("layer_count", 4),
                "inpaint_mode": legacy.get("inpaint_mode", InpaintMode.automatic.name),
                "inpaint_fill": legacy.get("inpaint_fill", FillMode.neutral.name),
                "inpaint_use_model": legacy.get("inpaint_use_model", True),
                "inpaint_use_prompt_focus": legacy.get("inpaint_use_prompt_focus", False),
                "inpaint_context": legacy.get("inpaint_context", InpaintContext.automatic.name),
            },
        )
        save_workspace_defaults(
            Workspace.upscaling,
            {
                "upscale_model": legacy.get("upscale_model", ""),
                "tile_overlap_mode": legacy.get("tile_overlap_mode", TileOverlapMode.auto.name),
                "tile_overlap": legacy.get("tile_overlap", 128),
            },
        )


@dataclass
class _HistoryResult:
    id: str
    slot: int  # annotation slot where images are stored
    offsets: list[int]  # offsets in bytes for result images
    params: JobParams
    kind: JobKind = JobKind.diffusion
    in_use: dict[int, bool] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["params"] = JobParams.from_dict(data["params"])
        data["kind"] = JobKind[data.get("kind", "diffusion")]
        data["in_use"] = {int(k): v for k, v in data.get("in_use", {}).items()}
        return _HistoryResult(**data)


class ModelSync:
    """Synchronizes the model with the document's annotations."""

    def __init__(self, model: DocumentModel):
        self._model = model
        self._history: list[_HistoryResult] = []
        self._memory_used: dict[int, int] = {}  # slot -> memory used for images in bytes
        self._slot_index = 0
        self._last_change = 0
        self._save_task: asyncio.Task | None = None
        self._image_task: asyncio.Future | None = None
        if state_bytes := _find_annotation(model.document, "ui.json"):
            try:
                self._load(model, state_bytes.data())
            except Exception as e:
                msg = _("Failed to load state from") + f" {model.document.filename}: {e}"
                log.exception(msg)
                QMessageBox.warning(None, "AI Diffusion Plugin", msg)
        self._track(model)

    def __del__(self):
        try:
            if self._image_task is not None and not self._image_task.done():
                eventloop.run_until_complete(self._image_task)
        except Exception as e:
            log.warning(f"Persistence: failed to wait for image task completion: {e}")

    def _save(self):
        model = self._model
        state = _serialize(model)
        state["version"] = version
        state["preview_layer"] = model.preview_layer_id
        state["inpaint"] = _serialize(model.inpaint)
        state["upscale"] = _serialize(model.upscale)
        state["live"] = _serialize(model.live)
        state["animation"] = _serialize(model.animation)
        state["custom"] = _serialize_custom(model.custom)
        state["history"] = [asdict(h) for h in self._history]
        state["root"] = _serialize(model.regions)
        state["edit"] = _serialize(model.edit_regions)
        state["control"] = [_serialize(c) for c in model.regions.control]
        state["regions"] = []
        for region in model.regions:
            state["regions"].append(_serialize(region))
            state["regions"][-1]["control"] = [_serialize(c) for c in region.control]
        state_str = json.dumps(state, indent=2, default=encode_json)
        state_bytes = QByteArray(state_str.encode("utf-8"))
        model.document.annotate("ui.json", state_bytes)

    def _load(self, model: DocumentModel, state_bytes: bytes):
        state = json.loads(state_bytes.decode("utf-8"))
        model.try_set_preview_layer(state.get("preview_layer", ""))
        _deserialize(model, state)
        _deserialize(model.inpaint, state.get("inpaint", {}))
        _deserialize(model.upscale, state.get("upscale", {}))
        _deserialize(model.live, state.get("live", {}))
        _deserialize(model.animation, state.get("animation", {}))
        _deserialize_custom(model.custom, state.get("custom", {}), model.name)
        _deserialize(model.regions, state.get("root", {}))
        _deserialize(model.edit_regions, state.get("edit", {}))
        for control_state in state.get("control", []):
            _deserialize(model.regions.control.emplace(), control_state)
        for region_state in state.get("regions", []):
            region = model.regions.emplace()
            _deserialize(region, region_state)
            for control_state in region_state.get("control", []):
                _deserialize(region.control.emplace(), control_state)

        for result in state.get("history", []):
            item = _HistoryResult.from_dict(result)
            if images_bytes := _find_annotation(model.document, f"result{item.slot}.webp"):
                job = model.jobs.add_job(Job(item.id, item.kind, item.params))
                job.in_use = item.in_use
                results = ImageCollection.from_bytes(images_bytes, item.offsets)
                model.jobs.set_results(job, results)
                model.jobs.notify_finished(job)
                self._history.append(item)
                self._memory_used[item.slot] = images_bytes.size()
                self._slot_index = max(self._slot_index, item.slot + 1)

    def _track(self, model: DocumentModel):
        model.modified.connect(self._save_later)
        model.inpaint.modified.connect(self._save_later)
        model.upscale.modified.connect(self._save_later)
        model.live.modified.connect(self._save_later)
        model.animation.modified.connect(self._save_later)
        model.custom.modified.connect(self._save_later)
        model.jobs.job_finished.connect(self._save_results)
        model.jobs.job_discarded.connect(self._remove_results)
        model.jobs.result_discarded.connect(self._remove_image)
        model.jobs.result_used.connect(self._save_later)
        model.jobs.selection_changed.connect(self._save_later)
        self._track_regions(model.regions)
        self._track_regions(model.edit_regions)

    def _track_control(self, control: ControlLayer):
        self._save()
        control.modified.connect(self._save_later)

    def _track_control_layers(self, control_layers: ControlLayerList):
        control_layers.added.connect(self._track_control)
        control_layers.removed.connect(self._save_later)
        for control in control_layers:
            self._track_control(control)

    def _track_region(self, region: Region):
        region.modified.connect(self._save_later)
        self._track_control_layers(region.control)

    def _track_regions(self, root_region: RootRegion):
        root_region.added.connect(self._track_region)
        root_region.removed.connect(self._save_later)
        root_region.modified.connect(self._save_later)
        self._track_control_layers(root_region.control)
        for region in root_region:
            self._track_region(region)

    def _save_results(self, job: Job):
        if job.kind in [JobKind.diffusion, JobKind.animation] and len(job.results) > 0:
            slot = self._slot_index
            self._slot_index += 1
            self._image_task = eventloop.run(self._save_result_images(job, slot, self._image_task))

    async def _save_result_images(
        self, job: Job, slot: int, prev_task: asyncio.Future | None = None
    ):
        if prev_task is not None:
            await prev_task
        if settings.multi_threading:
            loop = asyncio.get_running_loop()
            image_data, image_offsets = await loop.run_in_executor(
                None, job.results.to_bytes, settings.history_format
            )
        else:
            image_data, image_offsets = job.results.to_bytes(settings.history_format)

        self._model.document.annotate(f"result{slot}.webp", image_data)
        self._history.append(
            _HistoryResult(job.id or "", slot, image_offsets, job.params, job.kind, job.in_use)
        )
        self._memory_used[slot] = image_data.size()
        self._prune()
        self._save()

    def _remove_results(self, job: Job):
        index = next((i for i, h in enumerate(self._history) if h.id == job.id), None)
        if index is not None:
            item = self._history.pop(index)
            self._model.document.remove_annotation(f"result{item.slot}.webp")
            self._memory_used.pop(item.slot, None)
        self._save()

    def _remove_image(self, item: JobQueue.Item):
        if history := next((h for h in self._history if h.id == item.job), None):
            if job := self._model.jobs.find(item.job):
                image_data, history.offsets = job.results.to_bytes()
                self._model.document.annotate(f"result{history.slot}.webp", image_data)
                self._memory_used[history.slot] = image_data.size()
                self._save()

    def _save_later(self):
        self._last_change = time()
        if self._save_task is None or self._save_task.done():
            self._save_task = eventloop.run(self._delayed_save())

    async def _delayed_save(self):
        while time() - self._last_change < 1.0:
            await asyncio.sleep(1.0)
            self._save()

    @property
    def memory_used(self):
        return sum(self._memory_used.values())

    def _prune(self):
        limit = settings.history_storage * 1024 * 1024
        used = self.memory_used
        while used > limit and len(self._history) > 0:
            slot = self._history.pop(0).slot
            self._model.document.remove_annotation(f"result{slot}.webp")
            used -= self._memory_used.pop(slot, 0)


def _serialize(obj: QObject):
    def converter(obj):
        if isinstance(obj, Style):
            return obj.filename
        return obj

    return serialize(obj, converter)


def _deserialize(obj: QObject, data: dict[str, Any]):
    def converter(type, value):
        if type is Style:
            style = Styles.list().find(value)
            return style or Styles.list().default
        return value

    if "unblur_strength" in data and not isinstance(data["unblur_strength"], float):
        data["unblur_strength"] = 0.5

    return deserialize(obj, data, converter)


def _serialize_custom(custom: CustomWorkspace):
    result = _serialize(custom)
    result["workflow_id"] = custom.workflow_id
    result["graph"] = custom.graph.root if custom.graph else None
    return result


def _deserialize_custom(custom: CustomWorkspace, data: dict[str, Any], document_name: str):
    _deserialize(custom, data)
    workflow_id = data.get("workflow_id", "")
    graph = data.get("graph", None)
    if workflow_id and graph:
        custom.set_graph(workflow_id, graph, document_name)
        if params := data.get("params"):  # old documents, replaced by workflow_params
            custom.workflow_params[workflow_id] = params


def _find_annotation(document, name: str):
    if result := document.find_annotation(name):
        return result
    without_ext = name.rsplit(".", 1)[0]
    if result := document.find_annotation(without_ext):
        return result
    return None


def import_prompt_from_file(model: DocumentModel):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    filename = model.document.filename
    if model.regions.positive == "" and model.regions.negative == "" and filename.endswith(exts):
        try:
            reader = QImageReader(filename)
            # A1111
            if text := reader.text("parameters"):
                if "Negative prompt:" in text:
                    positive, negative = text.split("Negative prompt:", 1)
                    model.regions.positive = positive.strip()
                    model.regions.negative = negative.split("Steps:", 1)[0].strip()
            # ComfyUI
            elif text := reader.text("prompt"):
                prompt: dict[str, dict] = json.loads(text)
                for node in prompt.values():
                    if node["class_type"] in _comfy_sampler_types:
                        inputs = node["inputs"]
                        model.regions.positive = _find_text_prompt(prompt, inputs["positive"][0])
                        model.regions.negative = _find_text_prompt(prompt, inputs["negative"][0])

        except Exception as e:
            log.warning(f"Failed to read PNG metadata from {filename}: {e}")


_comfy_sampler_types = ["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"]


def _find_text_prompt(workflow: dict[str, dict], node_key: str):
    if node := workflow.get(node_key):
        if node["class_type"] == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            return text if isinstance(text, str) else ""
        for input in node.get("inputs", {}).values():
            if isinstance(input, list):
                return _find_text_prompt(workflow, input[0])
    return ""
