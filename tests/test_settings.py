import json
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from PyQt5.QtCore import QObject, QUuid, pyqtSignal

krita = types.ModuleType("krita")


class _Krita:
    @staticmethod
    def instance():
        return _Krita()

    def activeDocument(self):
        return None


krita.__dict__["Krita"] = _Krita
sys.modules.setdefault("krita", krita)

from ai_diffusion.backend.api import InpaintMode
from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.defaults import defaults
from ai_diffusion.document import Document
from ai_diffusion.layer import Layer, LayerType
from ai_diffusion.model.connection import Connection
from ai_diffusion.model.custom_workflow import CustomGenerationMode, WorkflowCollection
from ai_diffusion.model.model import (
    AnimationTargetLayerDefault,
    LiveScheduler,
    QueueMode,
    SamplingQuality,
    TileOverlapMode,
    Workspace,
    animation_batch_frame_path,
    animation_batch_output_folder,
    animation_import_layer_name,
    animation_layer_name,
    apply_layer_name,
    generated_layer_prefix,
    layered_batch_layer_prefix,
    live_recording_folder,
    live_recording_frame_path,
    live_recording_import_layer_name,
    preview_layer_name,
    select_default_animation_target_layer_id,
)
from ai_diffusion.model.model import (
    DocumentModel as Model,
)
from ai_diffusion.model.region import RootRegion
from ai_diffusion.persistence import (
    RecentlyUsedSync,
    load_document_defaults,
    load_workspace_defaults,
    save_document_defaults,
    save_workspace_defaults,
)
from ai_diffusion.settings import (
    ImageFileFormat,
    PerformancePreset,
    ServerMode,
    Setting,
    Settings,
    settings,
)
from ai_diffusion.style import (
    SamplerPreset,
    SamplerPresets,
    Style,
    Styles,
    StyleSettings,
    sort_recent_styles,
    style_defaults_schema,
)
from ai_diffusion.style import legacy_map as style_legacy_map


def test_get_set():
    s = Settings()
    assert (
        s.history_size == Settings._history_size.default
        and s.server_mode == Settings._server_mode.default
        and s.negative_prompt_line_count == Settings._negative_prompt_line_count.default
        and s.color_match_generation == Settings._color_match_generation.default
        and s.color_match_edit == Settings._color_match_edit.default
    )
    s.history_size = 5
    s.server_mode = ServerMode.external
    s.negative_prompt_line_count = 4
    s.color_match_generation = False
    s.color_match_edit = True
    assert (
        s.history_size == 5
        and s.server_mode == ServerMode.external
        and s.negative_prompt_line_count == 4
        and not s.color_match_generation
        and s.color_match_edit
    )


def test_restore():
    s = Settings()
    assert s.server_mode == Settings._server_mode.default

    s.history_size = 5
    s.server_mode = ServerMode.external
    s.restore()
    assert s.history_size == Settings._history_size.default and s.server_mode is ServerMode.managed


def test_save():
    original = Settings()
    original.history_size = 5
    original.server_mode = ServerMode.external
    original.performance_preset = PerformancePreset.low
    original.negative_prompt_line_count = 4
    original.color_match_generation = False
    original.color_match_edit = True
    original.upscale_model = "custom-default.pth"
    original.upscale_model_small = "custom-small.pth"
    original.upscale_highres_refine_strength = 0.55
    original.upscale_tile_overlap_auto_base = 24
    original.upscale_tile_overlap_auto_denoise = 80
    original.upscale_model_tile_size = 1536
    original.upscale_model_tile_overlap = 192
    original.live_poll_rate = 0.2
    original.live_default_grace_period = 0.35
    original.live_max_wait_time = 4.5
    original.live_delay_threshold = 2.5
    original.save_image_quality_png = 82
    original.save_image_quality_png_small = 48
    original.save_image_quality_webp = 77
    original.save_image_quality_webp_lossless = 100
    original.save_image_quality_jpeg = 88
    original.control_layer_mode = ControlMode.depth
    original.control_layer_preset_value = 4
    original.control_layer_use_custom_strength = True
    original.control_layer_strength = 1.2
    original.control_layer_start = 0.2
    original.control_layer_end = 0.8
    original.server_connect_retry_attempts = 7
    original.server_connect_retry_delay = 9
    original.server_authorization = "Bearer test-token"
    original.check_server_resources = False
    original.download_retry_attempts = 4
    original.download_retry_delay = 3
    original.download_inactivity_timeout = 45
    original.comfy_get_timeout = 75
    original.comfy_result_image_timeout = 420
    original.comfy_model_inspection_timeout = 654
    original.websocket_ping_timeout = 91
    original.cloud_sign_in_timeout = 360
    original.cloud_auth_poll_interval = 3.5
    original.cloud_job_poll_interval = 0.8
    original.cloud_api_url = "https://api.example.test"
    original.cloud_web_url = "https://app.example.test"
    original.auto_update_check_timeout = 12
    original.flux_inpaint_cfg_scale = 27.5
    original.preview_layer_name_format = "Preview::{prompt}"
    original.apply_layer_name_format = "{prefix}{seed}:{prompt}"
    original.generated_layer_name_prefix = "Gen::"
    original.layered_batch_prefix_format = "L{layer_index}::"
    original.animation_layer_name_format = "Anim::{prompt}"
    original.animation_import_layer_name_format = "Batch::{start}-{end}:{prompt}"
    original.live_recording_layer_name_format = "Rec::{start}-{end}:{prompt}"
    original.new_region_name = "Area {index}"
    original.new_region_layer_name = "Base paint"
    original.new_style_name = "Preset"
    original.new_style_copy_name = "{name} clone"
    result = Settings()
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        filepath = Path(dir) / "test_settings.json"
        original.save(filepath)
        result.load(filepath)
    assert (
        result.history_size == 5
        and result.server_mode == ServerMode.external
        and result.performance_preset == PerformancePreset.low
        and result.negative_prompt_line_count == 4
        and not result.color_match_generation
        and result.color_match_edit
        and result.upscale_model == "custom-default.pth"
        and result.upscale_model_small == "custom-small.pth"
        and result.upscale_highres_refine_strength == 0.55
        and result.upscale_tile_overlap_auto_base == 24
        and result.upscale_tile_overlap_auto_denoise == 80
        and result.upscale_model_tile_size == 1536
        and result.upscale_model_tile_overlap == 192
        and result.live_poll_rate == 0.2
        and result.live_default_grace_period == 0.35
        and result.live_max_wait_time == 4.5
        and result.live_delay_threshold == 2.5
        and result.save_image_quality_png == 82
        and result.save_image_quality_png_small == 48
        and result.save_image_quality_webp == 77
        and result.save_image_quality_webp_lossless == 100
        and result.save_image_quality_jpeg == 88
        and result.control_layer_mode is ControlMode.depth
        and result.control_layer_preset_value == 4
        and result.control_layer_use_custom_strength
        and result.control_layer_strength == 1.2
        and result.control_layer_start == 0.2
        and result.control_layer_end == 0.8
        and result.server_connect_retry_attempts == 7
        and result.server_connect_retry_delay == 9
        and result.server_authorization == "Bearer test-token"
        and result.check_server_resources is False
        and result.download_retry_attempts == 4
        and result.download_retry_delay == 3
        and result.download_inactivity_timeout == 45
        and result.comfy_get_timeout == 75
        and result.comfy_result_image_timeout == 420
        and result.comfy_model_inspection_timeout == 654
        and result.websocket_ping_timeout == 91
        and result.cloud_sign_in_timeout == 360
        and result.cloud_auth_poll_interval == 3.5
        and result.cloud_job_poll_interval == 0.8
        and result.cloud_api_url == "https://api.example.test"
        and result.cloud_web_url == "https://app.example.test"
        and result.auto_update_check_timeout == 12
        and result.flux_inpaint_cfg_scale == 27.5
        and result.preview_layer_name_format == "Preview::{prompt}"
        and result.apply_layer_name_format == "{prefix}{seed}:{prompt}"
        and result.generated_layer_name_prefix == "Gen::"
        and result.layered_batch_prefix_format == "L{layer_index}::"
        and result.animation_layer_name_format == "Anim::{prompt}"
        and result.animation_import_layer_name_format == "Batch::{start}-{end}:{prompt}"
        and result.live_recording_layer_name_format == "Rec::{start}-{end}:{prompt}"
        and result.new_region_name == "Area {index}"
        and result.new_region_layer_name == "Base paint"
        and result.new_style_name == "Preset"
        and result.new_style_copy_name == "{name} clone"
    )


def test_image_format_quality():
    s = Settings()
    s.save_image_quality_png = 81
    s.save_image_quality_png_small = 52
    s.save_image_quality_webp = 79
    s.save_image_quality_webp_lossless = 100
    s.save_image_quality_jpeg = 87

    assert s.image_format_quality(ImageFileFormat.png) == 81
    assert s.image_format_quality(ImageFileFormat.png_small) == 52
    assert s.image_format_quality(ImageFileFormat.webp) == 79
    assert s.image_format_quality(ImageFileFormat.webp_lossless) == 100
    assert s.image_format_quality(ImageFileFormat.jpeg) == 87


def test_performance_preset():
    s = Settings()
    s.performance_preset = PerformancePreset.low
    assert s.batch_size == 2 and s.max_pixel_count == 2 and s.resolution_multiplier == 1.5


def style_is_default(style):
    return all(
        getattr(style, name) == s.default
        for name, s in StyleSettings.__dict__.items()
        if isinstance(s, Setting) and name != "name"
    )


def test_styles(tmp_path_factory):
    builtin_dir = tmp_path_factory.mktemp("builtin")
    user_dir = tmp_path_factory.mktemp("user")

    style = Style(user_dir / "test_style.json")
    style.name = "Test Style"
    style.save()

    styles = Styles(builtin_dir, user_dir)
    assert len(styles) == 1
    loaded_style = styles[0]
    assert loaded_style.filename == style.filename
    assert loaded_style.name == "Test Style"
    assert styles.find(style.filename) == loaded_style
    assert styles.find("nonexistent.json") is None
    assert style_is_default(loaded_style)


def test_style_folders(tmp_path_factory):
    builtin_dir = tmp_path_factory.mktemp("builtin")
    user_dir = tmp_path_factory.mktemp("user")

    builtin = Style(builtin_dir / "test_style.json")
    builtin.name = "Built-in Style"
    builtin.save()

    user = Style(user_dir / "test_style.json")
    user.name = "User Style"
    user.save()

    styles = Styles(builtin_dir, user_dir)
    assert len(styles) == 2
    for style in styles:
        if style.filepath == builtin.filepath:
            assert style.name == "Built-in Style"
        elif style.filepath == user.filepath:
            assert style.name == "User Style"
        else:
            assert False

    only_user = styles.filtered(show_builtin=False)
    assert len(only_user) == 1
    assert only_user[0].name == "User Style"


def test_bad_style_file(tmp_path_factory):
    builtin_dir = tmp_path_factory.mktemp("builtin")
    user_dir = tmp_path_factory.mktemp("user")

    path = user_dir / "test_style.json"
    path.write_text("bad json")
    styles = Styles(builtin_dir, user_dir)
    assert len(styles) == 1  # no error, default style inserted
    assert style_is_default(styles[0])


def test_bad_style_type():
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        path = Path(dir) / "test_style.json"
        path.write_text(json.dumps({"cfg_scale": "bad", "sampler": "bad", "style_prompt": -1}))
        style = Style.load(path)
        assert (
            style is not None
            and style.cfg_scale == StyleSettings.cfg_scale.default
            and style.sampler == StyleSettings.sampler.default
            and style.style_prompt == StyleSettings.style_prompt.default
        )


def test_preferred_style():
    checkpoints = ["cats", "birds", "snakes"]
    style = Style(Path("test_style.json"))
    assert style.preferred_checkpoint(checkpoints) == "not-found"
    style.checkpoints = ["birds"]
    assert style.preferred_checkpoint(checkpoints) == "birds"
    style.checkpoints = ["dogs", "cats"]
    assert style.preferred_checkpoint(checkpoints) == "cats"


def test_default_style(tmp_path_factory):
    styles = Styles(tmp_path_factory.mktemp("builtin"), tmp_path_factory.mktemp("user"))
    style = styles.default
    assert style_is_default(style)


def test_duplicate_style(tmp_path_factory):
    styles = Styles(tmp_path_factory.mktemp("builtin"), tmp_path_factory.mktemp("user"))
    original = styles.create("original.json")
    original.loras.append({"name": "lora", "strength": 1.0})
    original.name = "Original"
    original.live_sampler_steps = 42

    copy = styles.create(original.filename, copy_from=original)
    assert copy.filename == "original-1.json"
    assert copy.name == "Original (Copy)"
    assert copy.loras == original.loras
    assert copy.live_sampler_steps == original.live_sampler_steps

    copy.loras[0] = {"name": "lora2", "strength": 2.0}
    assert copy.loras != original.loras


def test_style_create_uses_configured_names(tmp_path_factory):
    values = {
        "new_style_name": settings.new_style_name,
        "new_style_copy_name": settings.new_style_copy_name,
    }
    try:
        settings.new_style_name = "Template"
        settings.new_style_copy_name = "Copy of {name}"

        styles = Styles(tmp_path_factory.mktemp("builtin"), tmp_path_factory.mktemp("user"))
        original = styles.create("original.json")
        original.name = "Original"

        created = styles.create("new.json")
        copied = styles.create(original.filename, copy_from=original)

        assert created.name == "Template"
        assert copied.name == "Copy of Original"
    finally:
        for name, value in values.items():
            setattr(settings, name, value)


def test_style_create_applies_style_defaults(tmp_path_factory):
    original_path = defaults.path
    defaults._path = tmp_path_factory.mktemp("defaults") / "defaults.json"
    try:
        defaults.write_section(
            "style",
            {"cfg_scale": 3.5, "style_prompt": "cinematic {prompt}"},
            style_defaults_schema,
        )

        styles = Styles(tmp_path_factory.mktemp("builtin"), tmp_path_factory.mktemp("user"))
        style = styles.create("custom.json")

        assert style.cfg_scale == 3.5
        assert style.style_prompt == "cinematic {prompt}"
    finally:
        defaults._path = original_path


def test_workspace_defaults_roundtrip(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_workspace_defaults(
            Workspace.generation,
            {
                "strength": 0.6,
                "edit_mode": True,
                "batch_count": 3,
                "fixed_seed": True,
                "resolution_multiplier": 2.0,
                "use_smart_resolution": False,
                "smart_rotate": True,
                "queue_mode": QueueMode.front.name,
                "layer_count": 6,
                "inpaint_mode": InpaintMode.custom.name,
            },
        )
        save_workspace_defaults(
            Workspace.upscaling,
            {
                "tile_overlap_mode": TileOverlapMode.custom.name,
                "tile_overlap": 160,
            },
        )

        values = load_workspace_defaults(Workspace.generation)
        upscale_values = load_workspace_defaults(Workspace.upscaling)

        assert values["strength"] == 0.6
        assert values["edit_mode"] is True
        assert values["batch_count"] == 3
        assert values["fixed_seed"] is True
        assert values["resolution_multiplier"] == 2.0
        assert values["use_smart_resolution"] is False
        assert values["smart_rotate"] is True
        assert values["queue_mode"] is QueueMode.front
        assert values["layer_count"] == 6
        assert values["inpaint_mode"] is InpaintMode.custom
        assert values["translation_enabled"] is True
        assert upscale_values["tile_overlap_mode"] is TileOverlapMode.custom
        assert upscale_values["tile_overlap"] == 160
        assert upscale_values["use_diffusion"] is True
    finally:
        defaults._path = original_path


def test_document_defaults_roundtrip(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_document_defaults({"workspace": Workspace.upscaling.name})

        values = load_document_defaults()

        assert values["workspace"] is Workspace.upscaling
    finally:
        defaults._path = original_path


def test_live_workspace_defaults_roundtrip(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_workspace_defaults(
            Workspace.live,
            {
                "strength": 0.6,
                "recording_format": ImageFileFormat.jpeg.name,
                "recording_folder_name_format": "{document_name}-takes",
                "recording_frame_name_format": "take-{index:03}.{extension}",
            },
        )

        values = load_workspace_defaults(Workspace.live)

        assert values["strength"] == 0.6
        assert values["recording_format"] is ImageFileFormat.jpeg
        assert values["recording_folder_name_format"] == "{document_name}-takes"
        assert values["recording_frame_name_format"] == "take-{index:03}.{extension}"
    finally:
        defaults._path = original_path


def test_animation_workspace_defaults_roundtrip(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_workspace_defaults(
            Workspace.animation,
            {
                "sampling_quality": "quality",
                "target_layer_default": AnimationTargetLayerDefault.first.name,
                "batch_mode": False,
                "batch_folder_name_format": "{document_name}-shots",
                "batch_frame_name_format": "shot-{frame:04}.{extension}",
            },
        )

        values = load_workspace_defaults(Workspace.animation)

        sampling_quality = values["sampling_quality"]
        assert isinstance(sampling_quality, SamplingQuality)
        assert sampling_quality.name == "quality"
        assert values["target_layer_default"] is AnimationTargetLayerDefault.first
        assert values["batch_mode"] is False
        assert values["batch_folder_name_format"] == "{document_name}-shots"
        assert values["batch_frame_name_format"] == "shot-{frame:04}.{extension}"
    finally:
        defaults._path = original_path


def test_custom_workspace_defaults_roundtrip(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_workspace_defaults(
            Workspace.custom,
            {
                "workflow_id": "graph/default",
                "mode": CustomGenerationMode.live.name,
                "params_ui_height": 320,
            },
        )

        values = load_workspace_defaults(Workspace.custom)

        assert values["workflow_id"] == "graph/default"
        assert values["mode"] is CustomGenerationMode.live
        assert values["params_ui_height"] == 320
    finally:
        defaults._path = original_path


def test_live_recording_paths_use_settings():
    document_path = Path("/tmp/demo.kra")

    folder = live_recording_folder(document_path, "{document_name}-captures")
    frame = live_recording_frame_path(
        folder, 7, ImageFileFormat.png_small, "take-{index:03}.{extension}"
    )

    assert folder == Path("/tmp/demo-captures")
    assert frame == Path("/tmp/demo-captures/take-007.png")


def test_live_recording_folder_falls_back_for_invalid_template():
    document_path = Path("/tmp/demo.kra")

    folder = live_recording_folder(document_path, "{missing}")

    assert folder == Path("/tmp/demo.live-frames")


def test_animation_batch_paths_use_templates():
    document_path = Path("/tmp/demo.kra")

    folder = animation_batch_output_folder(document_path, "{document_name}-batch")
    frame = animation_batch_frame_path(folder, 12, "shot-{frame:04}.{extension}")

    assert folder == Path("/tmp/demo-batch")
    assert frame == Path("/tmp/demo-batch/shot-0012.png")


def test_animation_batch_paths_fall_back_for_invalid_templates():
    document_path = Path("/tmp/demo.kra")
    folder = animation_batch_output_folder(document_path, "{missing}")
    frame = animation_batch_frame_path(folder, 5, "")

    assert folder == Path("/tmp/demo.animation")
    assert frame == Path("/tmp/demo.animation/frame-5.png")


def test_layer_name_templates_use_settings():
    values = {
        "preview_layer_name_format": settings.preview_layer_name_format,
        "apply_layer_name_format": settings.apply_layer_name_format,
        "generated_layer_name_prefix": settings.generated_layer_name_prefix,
        "layered_batch_prefix_format": settings.layered_batch_prefix_format,
        "animation_layer_name_format": settings.animation_layer_name_format,
        "animation_import_layer_name_format": settings.animation_import_layer_name_format,
        "live_recording_layer_name_format": settings.live_recording_layer_name_format,
    }
    try:
        settings.preview_layer_name_format = "Preview::{prompt}"
        settings.apply_layer_name_format = "{prefix}{seed}:{prompt}"
        settings.generated_layer_name_prefix = "Gen::"
        settings.layered_batch_prefix_format = "L{layer_index}::"
        settings.animation_layer_name_format = "Anim::{prompt}"
        settings.animation_import_layer_name_format = "Batch::{start}-{end}:{prompt}"
        settings.live_recording_layer_name_format = "Rec::{start}-{end}:{prompt}"

        assert preview_layer_name("demo") == "Preview::demo"
        assert apply_layer_name("prompt", 42, "Gen::") == "Gen::42:prompt"
        assert generated_layer_prefix() == "Gen::"
        assert layered_batch_layer_prefix(3) == "L3::"
        assert animation_layer_name("move") == "Anim::move"
        assert animation_import_layer_name("move", 1, 8) == "Batch::1-8:move"
        assert live_recording_import_layer_name("move", 4, 7) == "Rec::4-7:move"
    finally:
        for name, value in values.items():
            setattr(settings, name, value)


def test_live_scheduler_uses_settings():
    values = {
        "live_poll_rate": settings.live_poll_rate,
        "live_default_grace_period": settings.live_default_grace_period,
        "live_max_wait_time": settings.live_max_wait_time,
        "live_delay_threshold": settings.live_delay_threshold,
    }
    try:
        settings.live_poll_rate = 0.2
        settings.live_default_grace_period = 0.4
        settings.live_max_wait_time = 6.0
        settings.live_delay_threshold = 2.0

        scheduler = LiveScheduler()
        scheduler._generation_times.extend([2.5, 3.5])

        assert scheduler.poll_rate == 0.2
        assert scheduler.default_grace_period == 0.4
        assert scheduler.max_wait_time == 6.0
        assert scheduler.delay_threshold == 2.0
        assert scheduler.grace_period == 0.4
    finally:
        for name, value in values.items():
            setattr(settings, name, value)


def test_workspace_defaults_migrate_legacy_document_defaults(tmp_path):
    original_path = defaults.path
    previous_document_defaults = settings.document_defaults
    defaults._path = tmp_path / "defaults.json"
    try:
        settings.document_defaults = {
            "workspace": Workspace.live.name,
            "style": "preset.json",
            "strength": 0.4,
            "region_only": True,
            "edit_mode": True,
            "batch_count": 5,
            "fixed_seed": True,
            "resolution_multiplier": 2.3,
            "use_smart_resolution": False,
            "smart_rotate": True,
            "queue_mode": QueueMode.front.name,
            "layer_count": 8,
            "inpaint_mode": InpaintMode.custom.name,
            "tile_overlap_mode": TileOverlapMode.custom.name,
            "tile_overlap": 192,
        }

        RecentlyUsedSync.from_settings()
        document = load_document_defaults()
        values = load_workspace_defaults(Workspace.generation)
        upscale_values = load_workspace_defaults(Workspace.upscaling)

        assert document["workspace"] is Workspace.live
        assert values["style"] == "preset.json"
        assert values["strength"] == 0.4
        assert "region_only" not in values
        assert values["edit_mode"] is True
        assert values["batch_count"] == 5
        assert values["fixed_seed"] is True
        assert values["resolution_multiplier"] == 2.3
        assert values["use_smart_resolution"] is False
        assert values["smart_rotate"] is True
        assert values["queue_mode"] is QueueMode.front
        assert values["layer_count"] == 8
        assert values["inpaint_mode"] is InpaintMode.custom
        assert upscale_values["tile_overlap_mode"] is TileOverlapMode.custom
        assert upscale_values["tile_overlap"] == 192
    finally:
        settings.document_defaults = previous_document_defaults
        defaults._path = original_path


def _create_model():
    connection = Connection()
    workflows = WorkflowCollection(connection)
    return Model(Document(), connection, workflows)


class _FakeLayer:
    def __init__(self, name="Layer", layer_type=LayerType.paint, parent=None, is_root=False):
        self.id = QUuid.createUuid()
        self.name = name
        self.type = layer_type
        self.parent_layer = parent
        self.is_root = is_root
        self.child_layers = []

        if parent is not None:
            parent.child_layers.append(self)

    @property
    def siblings(self):
        if self.parent_layer is None:
            return [], []
        siblings = self.parent_layer.child_layers
        index = siblings.index(self)
        return siblings[:index], siblings[index + 1 :]


class _FakeLayerStore(QObject):
    removed = pyqtSignal(object)
    active_changed = pyqtSignal()
    parent_changed = pyqtSignal(object)

    def __init__(self, layer: _FakeLayer):
        super().__init__()
        self.root = _FakeLayer("Root", LayerType.group, is_root=True)
        layer.parent_layer = self.root
        self.root.child_layers.append(layer)
        self.active = layer
        self.images = [self.root, layer]

    def updated(self):
        return self

    def find(self, layer_id: QUuid):
        return next((layer for layer in self.images if layer.id == layer_id), None)

    def create(self, name: str, parent=None):
        layer = _FakeLayer(name, LayerType.paint, parent or self.root)
        self.images.append(layer)
        self.active = layer
        return layer

    def create_group(self, name: str):
        layer = _FakeLayer(name, LayerType.group, self.root)
        self.images.append(layer)
        self.active = layer
        return layer


class _FakeJobs(QObject):
    job_finished = pyqtSignal()


class _FakeModel(QObject):
    style_changed = pyqtSignal(object)
    edit_mode_changed = pyqtSignal(bool)

    def __init__(self, arch=Arch.sd15):
        super().__init__()
        layer = _FakeLayer()
        self.arch = arch
        self.layers = _FakeLayerStore(layer)
        self.jobs = _FakeJobs()
        self.style = Style(Path("style.json"))


def test_root_region_uses_configured_names_for_new_regions():
    values = {
        "new_region_name": settings.new_region_name,
        "new_region_layer_name": settings.new_region_layer_name,
    }
    try:
        settings.new_region_name = "Area {index}"
        settings.new_region_layer_name = "Base paint"

        root_region = RootRegion(cast(Model, _FakeModel()))
        root_region.emplace().link(root_region.layers.active)
        group_region = root_region.create_region(group=True)
        layer_region = root_region.create_region(group=False)

        group_layer = group_region.first_layer
        layer_region_layer = layer_region.first_layer
        assert group_layer is not None
        assert layer_region_layer is not None
        assert group_layer.name == "Area 1"
        assert group_layer.child_layers[0].name == "Base paint"
        assert layer_region_layer.name == "Area 2"
    finally:
        for name, value in values.items():
            setattr(settings, name, value)


def _patch_root_connection(monkeypatch):
    from ai_diffusion.model import root as root_module

    monkeypatch.setattr(root_module.root, "_connection", Connection(), raising=False)


def test_control_layer_uses_default_preset_settings(monkeypatch):
    from ai_diffusion.model.control import ControlLayer, ControlPresets

    _patch_root_connection(monkeypatch)
    values = {
        "control_layer_preset_value": settings.control_layer_preset_value,
        "control_layer_use_custom_strength": settings.control_layer_use_custom_strength,
        "control_layer_strength": settings.control_layer_strength,
        "control_layer_start": settings.control_layer_start,
        "control_layer_end": settings.control_layer_end,
    }
    try:
        settings.control_layer_preset_value = 4
        settings.control_layer_use_custom_strength = False
        settings.control_layer_strength = 0.3
        settings.control_layer_start = 0.1
        settings.control_layer_end = 0.4

        model = _FakeModel()
        control = ControlLayer(cast(Model, model), ControlMode.depth, model.layers.active.id, 0)
        expected = ControlPresets.instance().interpolate(
            ControlMode.depth,
            Arch.sd15,
            settings.control_layer_preset_value / ControlLayer.max_preset_value,
        )

        assert control.preset_value == 4
        assert control.use_custom_strength is False
        assert control.strength == int(expected.strength * ControlLayer.strength_multiplier)
        assert control.start == expected.range[0]
        assert control.end == expected.range[1]
    finally:
        for name, value in values.items():
            setattr(settings, name, value)


def test_control_layer_uses_default_custom_settings(monkeypatch):
    from ai_diffusion.model.control import ControlLayer

    _patch_root_connection(monkeypatch)
    values = {
        "control_layer_preset_value": settings.control_layer_preset_value,
        "control_layer_use_custom_strength": settings.control_layer_use_custom_strength,
        "control_layer_strength": settings.control_layer_strength,
        "control_layer_start": settings.control_layer_start,
        "control_layer_end": settings.control_layer_end,
    }
    try:
        settings.control_layer_preset_value = 1
        settings.control_layer_use_custom_strength = True
        settings.control_layer_strength = 1.2
        settings.control_layer_start = 0.2
        settings.control_layer_end = 0.8

        model = _FakeModel()
        control = ControlLayer(cast(Model, model), ControlMode.line_art, model.layers.active.id, 0)

        assert control.preset_value == 1
        assert control.use_custom_strength is True
        assert control.strength == 60
        assert control.start == 0.2
        assert control.end == 0.8
    finally:
        for name, value in values.items():
            setattr(settings, name, value)


def test_control_layer_list_uses_and_updates_default_mode(monkeypatch):
    from ai_diffusion.model.control import ControlLayerList

    _patch_root_connection(monkeypatch)
    original_mode = settings.control_layer_mode
    try:
        settings.control_layer_mode = ControlMode.pose
        saved = []
        monkeypatch.setattr(settings, "save", lambda path=None: saved.append(path))

        control_list = ControlLayerList(cast(Model, _FakeModel()))
        control_list.add()

        first_control = next(iter(control_list))
        assert first_control.mode is ControlMode.pose

        first_control.mode = ControlMode.depth

        assert settings.control_layer_mode is ControlMode.depth
        assert saved == [None]

        control_list.add()
        assert list(control_list)[1].mode is ControlMode.depth
    finally:
        settings.control_layer_mode = original_mode


def test_recently_used_sync_applies_new_document_generation_defaults(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_document_defaults({"workspace": Workspace.animation.name})
        save_workspace_defaults(
            Workspace.generation,
            {
                "strength": 0.55,
                "edit_mode": True,
                "resolution_multiplier": 2.4,
                "use_smart_resolution": False,
                "smart_rotate": True,
                "fixed_seed": True,
                "queue_mode": QueueMode.front.name,
                "layer_count": 7,
            },
        )

        recent = RecentlyUsedSync.from_settings()
        model = _create_model()
        recent.track(model)

        assert model.workspace is Workspace.animation
        assert model.strength == 0.55
        assert model.region_only is False
        assert model.edit_mode is True
        assert model.resolution_multiplier == 2.4
        assert model.use_smart_resolution is False
        assert model.smart_rotate is True
        assert model.fixed_seed is True
        assert model.queue_mode is QueueMode.front
        assert model.layer_count == 7
    finally:
        defaults._path = original_path


def test_recently_used_sync_tracks_workspace_and_generation_defaults(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        recent = RecentlyUsedSync.from_settings()
        model = _create_model()
        recent.track(model)

        model.workspace = Workspace.live
        model.strength = 0.65
        model.region_only = True
        model.edit_mode = True
        model.resolution_multiplier = 1.8
        model.use_smart_resolution = False
        model.smart_rotate = True
        model.fixed_seed = True
        model.queue_mode = QueueMode.front
        model.layer_count = 5

        assert load_document_defaults()["workspace"] is Workspace.live

        values = load_workspace_defaults(Workspace.generation)
        assert values["strength"] == 0.65
        assert "region_only" not in values
        assert values["edit_mode"] is True
        assert values["resolution_multiplier"] == 1.8
        assert values["use_smart_resolution"] is False
        assert values["smart_rotate"] is True
        assert values["fixed_seed"] is True
        assert values["queue_mode"] is QueueMode.front
        assert values["layer_count"] == 5
    finally:
        defaults._path = original_path


def test_recently_used_sync_applies_animation_target_layer_default(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_workspace_defaults(
            Workspace.animation,
            {"target_layer_default": AnimationTargetLayerDefault.first.name},
        )

        recent = RecentlyUsedSync.from_settings()
        model = _create_model()
        recent.track(model)

        assert model.animation.target_layer_default is AnimationTargetLayerDefault.first
    finally:
        defaults._path = original_path


def test_recently_used_sync_tracks_animation_target_layer_default(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        recent = RecentlyUsedSync.from_settings()
        model = _create_model()
        recent.track(model)

        model.animation.target_layer_default = AnimationTargetLayerDefault.first

        values = load_workspace_defaults(Workspace.animation)
        assert values["target_layer_default"] is AnimationTargetLayerDefault.first
    finally:
        defaults._path = original_path


def test_select_default_animation_target_layer_id_prefers_active_layer():
    first = cast(
        Layer,
        types.SimpleNamespace(
            id="first", type=types.SimpleNamespace(is_mask=False), parent_layer=None
        ),
    )
    active = cast(
        Layer,
        types.SimpleNamespace(
            id="active", type=types.SimpleNamespace(is_mask=False), parent_layer=None
        ),
    )

    target = select_default_animation_target_layer_id(
        AnimationTargetLayerDefault.active, active, [first, active]
    )

    assert target == "active"


def test_select_default_animation_target_layer_id_falls_back_to_first_image_layer():
    first = cast(
        Layer,
        types.SimpleNamespace(
            id="first", type=types.SimpleNamespace(is_mask=False), parent_layer=None
        ),
    )
    missing = cast(
        Layer,
        types.SimpleNamespace(
            id="missing", type=types.SimpleNamespace(is_mask=False), parent_layer=None
        ),
    )

    target = select_default_animation_target_layer_id(
        AnimationTargetLayerDefault.active, missing, [first]
    )

    assert target == "first"


def test_select_default_animation_target_layer_id_uses_mask_parent_for_active_default():
    parent = cast(
        Layer,
        types.SimpleNamespace(
            id="parent", type=types.SimpleNamespace(is_mask=False), parent_layer=None
        ),
    )
    mask = cast(
        Layer,
        types.SimpleNamespace(
            id="mask", type=types.SimpleNamespace(is_mask=True), parent_layer=parent
        ),
    )

    target = select_default_animation_target_layer_id(
        AnimationTargetLayerDefault.active, mask, [parent]
    )

    assert target == "parent"


def test_recently_used_sync_applies_new_document_custom_defaults(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        save_workspace_defaults(
            Workspace.custom,
            {
                "workflow_id": "graph/default",
                "mode": CustomGenerationMode.animation.name,
                "params_ui_height": 280,
            },
        )

        recent = RecentlyUsedSync.from_settings()
        model = _create_model()
        recent.track(model)

        assert model.custom.workflow_id == "graph/default"
        assert model.custom.mode is CustomGenerationMode.animation
        assert model.custom.params_ui_height == 280
    finally:
        defaults._path = original_path


def test_recently_used_sync_tracks_custom_workspace_defaults(tmp_path):
    original_path = defaults.path
    defaults._path = tmp_path / "defaults.json"
    try:
        recent = RecentlyUsedSync.from_settings()
        model = _create_model()
        recent.track(model)

        model.custom.workflow_id = "graph/default"
        model.custom.mode = CustomGenerationMode.live
        model.custom.params_ui_height = 360

        values = load_workspace_defaults(Workspace.custom)
        assert values["workflow_id"] == "graph/default"
        assert values["mode"] is CustomGenerationMode.live
        assert values["params_ui_height"] == 360
    finally:
        defaults._path = original_path


def test_sampler_presets(tmp_path_factory):
    dir = tmp_path_factory.mktemp("presets")

    builtin_file = dir / "builtin.json"
    builtin_file.write_text(
        json.dumps({
            "Builtin": {"sampler": "dpmpp_2m", "scheduler": "normal", "steps": 42, "cfg": 7.0},
        })
    )

    user_file = dir / "user.json"
    user_file.write_text(
        json.dumps({
            "User": {"sampler": "user_sampler", "scheduler": "normal", "steps": 13, "cfg": 1.0},
        })
    )

    presets = SamplerPresets(builtin_file, user_file)
    assert len(presets) == 2

    builtin = presets["Builtin"]
    assert builtin == SamplerPreset("dpmpp_2m", "normal", 42, 7.0)

    user = presets["User"]
    assert user == SamplerPreset("user_sampler", "normal", 13, 1.0)

    presets.add_missing("DDIM", 99, 2.3)
    assert len(presets) == 3
    assert presets["DDIM"] == SamplerPreset("ddim", "ddim_uniform", 99, 2.3)


def test_sampler_preset_conversion():
    presets = SamplerPresets()
    for old, new in style_legacy_map.items():
        assert presets[old] == presets[new]


def _make_style(tmp_path, filename: str, name: str) -> Style:
    style = Style(tmp_path / filename)
    style.name = name
    return style


def test_sort_recent_styles(tmp_path):
    a = _make_style(tmp_path, "alpha.json", "Alpha")
    b = _make_style(tmp_path, "beta.json", "Beta")
    c = _make_style(tmp_path, "gamma.json", "Gamma")
    d = _make_style(tmp_path, "delta.json", "Delta")
    all_styles = [a, b, c, d]  # alphabetical order

    # Normal case: 2 recent styles at the top
    recent, remaining = sort_recent_styles(all_styles, ["gamma.json", "alpha.json"], 2)
    assert recent == [c, a]
    assert remaining == [b, d]

    # count=0: feature disabled, all styles go to remaining
    recent, remaining = sort_recent_styles(all_styles, ["gamma.json", "alpha.json"], 0)
    assert recent == []
    assert remaining == all_styles

    # Empty recent list
    recent, remaining = sort_recent_styles(all_styles, [], 3)
    assert recent == []
    assert remaining == all_styles

    # count larger than number of recent matches
    recent, remaining = sort_recent_styles(all_styles, ["beta.json"], 5)
    assert recent == [b]
    assert remaining == [a, c, d]

    # Recent list contains a filename not in styles (silently ignored)
    recent, remaining = sort_recent_styles(all_styles, ["missing.json", "delta.json"], 3)
    assert recent == [d]
    assert remaining == [a, b, c]

    # All styles in recent list
    filenames = [s.filename for s in all_styles]
    recent, remaining = sort_recent_styles(all_styles, filenames, 10)
    assert recent == all_styles
    assert remaining == []

    # Recency order is preserved (most recent first)
    recent, remaining = sort_recent_styles(all_styles, ["delta.json", "beta.json"], 2)
    assert recent == [d, b]
    assert remaining == [a, c]


def test_recent_styles_default_is_empty():
    assert Settings._recent_styles.default == []
    assert Settings().recent_styles == []
