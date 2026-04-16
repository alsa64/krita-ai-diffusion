from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from PyQt5.QtCore import QObject, pyqtSignal

from .localization import translate as _
from .platform_tools import is_macos, is_windows
from .util import client_logger as log
from .util import encode_json, read_json_with_comments, user_data_dir


class ServerMode(Enum):
    undefined = -1
    managed = 0
    external = 1
    cloud = 2


class ServerBackend(Enum):
    cpu = (_("Run on CPU"), True)
    cuda = (_("Use CUDA (NVIDIA GPU)"), not is_macos)
    mps = (_("Use MPS (Metal Performance Shader)"), is_macos)
    directml = (_("Use DirectML (GPU)"), is_windows)
    xpu = (_("Use XPU (Intel GPU)"), not is_macos)
    rocm = (_("Use ROCm (AMD GPU)"), not is_macos)

    @staticmethod
    def supported():
        return [b for b in ServerBackend if b.value[1]]

    @staticmethod
    def default():
        if is_macos:
            return ServerBackend.mps
        else:
            return ServerBackend.cuda


class GenerationFinishedAction(Enum):
    none = _("Do Nothing")
    preview = _("Preview")
    apply = _("Apply")


class ApplyBehavior(Enum):
    replace = _("Modify active layer")
    layer = _("New layer on top")
    layer_active = _("New layer above active")


class ApplyRegionBehavior(Enum):
    none = _("Do not update regions")
    replace = _("Modify region layers")
    layer_group = _("Layer group")
    transparency_mask = _("Layer group + mask")
    no_hide = _("Layer group (don't hide)")


class PerformancePreset(Enum):
    auto = _("Automatic")
    cpu = _("CPU")
    low = _("GPU low (up to 6GB)")
    medium = _("GPU medium (6GB to 12GB)")
    high = _("GPU high (more than 12GB)")
    cloud = _("Cloud")
    custom = _("Custom")


class ImageFileFormat(Enum):
    png = "PNG (fast)"  # fast, large files
    png_small = "PNG"  # slow, smaller files
    webp = "WebP"
    webp_lossless = "WebP (lossless)"
    jpeg = "JPEG"

    @staticmethod
    def from_extension(filepath: str | Path):
        extension = Path(filepath).suffix.lower()
        if extension == ".png":
            return ImageFileFormat.png_small
        if extension == ".webp":
            return ImageFileFormat.webp
        if extension in {".jpg", ".jpeg"}:
            return ImageFileFormat.jpeg
        raise ValueError(f"Unsupported image extension: {extension}")

    @property
    def extension(self):
        if self in [ImageFileFormat.png, ImageFileFormat.png_small]:
            return "png"
        elif self in [ImageFileFormat.webp, ImageFileFormat.webp_lossless]:
            return "webp"
        else:
            return "jpg"

    @property
    def quality(self):
        if self in [ImageFileFormat.png]:
            return 85
        elif self in [ImageFileFormat.png_small]:
            return 50
        elif self in [ImageFileFormat.webp]:
            return 80
        elif self in [ImageFileFormat.webp_lossless]:
            return 100
        elif self in [ImageFileFormat.jpeg]:
            return 85
        else:
            return 85

    @property
    def no_webp_fallback(self):
        if self is ImageFileFormat.webp_lossless:
            return ImageFileFormat.png
        if self is ImageFileFormat.webp:
            return ImageFileFormat.jpeg
        return self


class PerformancePresetSettings(NamedTuple):
    batch_size: int = 4
    resolution_multiplier: float = 1.5
    max_pixel_count: int = 6
    tiled_vae: bool = False


@dataclass
class PerformanceSettings:
    batch_size: int = 4
    resolution_multiplier: float = 1.5
    max_pixel_count: int = 6
    dynamic_caching: bool = False
    tiled_vae: bool = False


class Setting:
    def __init__(self, name: str, default, desc="", help="", items=None):
        self.name = name
        self.desc = desc
        self.default = default
        self.help = help
        self.items = items

    def str_to_enum(self, s: str):
        assert isinstance(self.default, Enum)
        EnumType = type(self.default)
        try:
            return EnumType[s]
        except KeyError:
            log.warning(
                f"Invalid value '{s}' for setting '{self.name}', using default '{self.default.name}'"
            )
            log.info(f"Available options are: {', '.join(EnumType.__members__.keys())}")
            return self.default


class Settings(QObject):
    default_path = user_data_dir / "settings.json"

    language: str
    _language = Setting(
        _("Language"),
        "en",
        _("Interface language used by the plugin - requires restart!"),
    )

    auto_update: bool
    _auto_update = Setting(
        _("Enable Automatic Updates"), True, _("Check for new versions of the plugin on startup")
    )

    server_mode: ServerMode
    _server_mode = Setting(
        _("Server Management"),
        ServerMode.undefined,
        _("To generate images, the plugin connects to a ComfyUI server"),
    )

    access_token: str
    _access_token = Setting(_("Cloud Access Token"), "")

    server_path: str
    _server_path = Setting(
        _("Server Path"),
        str(user_data_dir / "server"),
        _(
            "Directory where ComfyUI will be installed. At least {size} GB of free disk space is required for a minimal installation."
        ).format(size=16),
    )

    server_url: str
    _server_url = Setting(
        _("Server URL"),
        "127.0.0.1:8188",
        _("URL used to connect to a running ComfyUI server. Default is 127.0.0.1:8188 (local)."),
    )

    server_backend: ServerBackend
    _server_backend = Setting(_("Server Backend"), ServerBackend.default())

    server_arguments: str
    _server_arguments = Setting(
        _("Server Arguments"), "", _("Additional command line arguments passed to the server")
    )

    server_authorization: str
    _server_authorization = Setting("ComfyUI Authorization Token", "")

    check_server_resources: bool
    _check_server_resources = Setting("Refuse connection if nodes or models are missing", True)

    server_connect_retry_attempts: int
    _server_connect_retry_attempts = Setting(
        _("Server Connection Retry Attempts"),
        5,
        _("How many times to retry connecting to an external ComfyUI server on startup."),
    )

    server_connect_retry_delay: int
    _server_connect_retry_delay = Setting(
        _("Server Connection Retry Delay"),
        5,
        _("Base delay in seconds between external server connection retries on startup."),
    )

    selection_feather: int
    _selection_feather = Setting(
        _("Selection Feather"),
        10,
        _("The border is expanded and blurred by a fraction of selection size"),
    )

    selection_min_transition: int
    _selection_min_transition = Setting(
        "Selection minimum feather", 32, "Minimum smooth grow (feathering) in pixels for denoising"
    )

    selection_grow_offset: int
    _selection_grow_offset = Setting(
        "Selection Grow Offset",
        4,
        "Apply binary grow/dilation in pixels to denoise mask before smooth grow (feathering)",
    )

    selection_blend: int
    _selection_blend = Setting(
        _("Selection Blend"), 25, _("Transition area for alpha blending the result image")
    )

    selection_padding: int
    _selection_padding = Setting(
        _("Selection Padding"), 6, _("Minimum additional padding around the selection area")
    )

    color_match_generation: bool
    _color_match_generation = Setting(
        _("Color Match (Generation Models)"),
        True,
        _(
            "Match peripheral colors and brightness with existing content for generation models. Requires a selection."
        ),
    )

    color_match_edit: bool
    _color_match_edit = Setting(
        _("Color Match (Edit Models)"),
        True,
        _(
            "Match peripheral colors and brightness with existing content for edit models. Requires a selection."
        ),
    )

    nsfw_filter: float
    _nsfw_filter = Setting(
        _("NSFW Filter"), 0.0, _("Attempt to filter out images with explicit content")
    )

    new_seed_after_apply: bool
    _new_seed_after_apply = Setting(
        _("Live: New Seed after Apply"),
        False,
        _("Pick a new seed after copying the result to the canvas in Live mode"),
    )

    live_poll_rate: float
    _live_poll_rate = Setting(
        _("Live Poll Rate"),
        0.1,
        _("How often Live mode checks for document changes while waiting to generate."),
    )

    live_default_grace_period: float
    _live_default_grace_period = Setting(
        _("Live Grace Period"),
        0.25,
        _(
            "Delay after the most recent edit before Live mode starts a new generation when generation is slow."
        ),
    )

    live_max_wait_time: float
    _live_max_wait_time = Setting(
        _("Live Max Wait Time"),
        3.0,
        _("Maximum delay before Live mode generates again while edits continue."),
    )

    live_delay_threshold: float
    _live_delay_threshold = Setting(
        _("Live Delay Threshold"),
        1.5,
        _("Only apply the Live grace period when average generation time exceeds this threshold."),
    )

    prompt_translation: str
    _prompt_translation = Setting(
        _("Prompt Translation"),
        "",
        _("Translate text prompts from the selected language to English"),
    )

    save_image_metadata: bool
    _save_image_metadata = Setting(
        _("Save Image Metadata"),
        False,
        _("When saving generated images from thumbnails, include metadata in the PNG"),
    )

    upscale_model: str
    _upscale_model = Setting(
        _("Default Upscale Model"),
        "4x_NMKD-Superscale-SP_178000_G.pth",
        _("Default model for tiled upscaling and quality refinement passes."),
    )

    upscale_model_small: str
    _upscale_model_small = Setting(
        _("Default Upscale Model (Small)"),
        "OmniSR_X2_DIV2K.safetensors",
        _("Default model for small automatic upscaling refinement passes."),
    )

    upscale_highres_refine_strength: float
    _upscale_highres_refine_strength = Setting(
        _("High-Res Refine Strength"),
        0.4,
        _("Denoise strength used for automatic high-resolution refinement after upscaling."),
    )

    upscale_tile_overlap_auto_base: int
    _upscale_tile_overlap_auto_base = Setting(
        _("Automatic Tile Overlap Base"),
        16,
        _("Base overlap in pixels used when tiled upscale tile overlap is set to Automatic."),
    )

    upscale_tile_overlap_auto_denoise: int
    _upscale_tile_overlap_auto_denoise = Setting(
        _("Automatic Tile Overlap Denoise Scale"),
        64,
        _(
            "Additional overlap in pixels scaled by denoise strength when tiled upscale tile overlap is set to Automatic."
        ),
    )

    save_image_format: ImageFileFormat
    _save_image_format = Setting(
        _("Save Image Format"),
        ImageFileFormat.png_small,
        _("File format for saved images from thumbnails."),
    )

    save_image_quality_png: int
    _save_image_quality_png = Setting(
        _("Image Quality (PNG Fast)"),
        85,
        _("Encoding quality used when writing PNG (fast) images."),
    )

    save_image_quality_png_small: int
    _save_image_quality_png_small = Setting(
        _("Image Quality (PNG)"),
        50,
        _("Encoding quality used when writing PNG images."),
    )

    save_image_quality_webp: int
    _save_image_quality_webp = Setting(
        _("Image Quality (WebP)"),
        80,
        _("Encoding quality used when writing WebP images."),
    )

    save_image_quality_webp_lossless: int
    _save_image_quality_webp_lossless = Setting(
        _("Image Quality (WebP Lossless)"),
        100,
        _("Encoding quality used when writing WebP lossless images."),
    )

    save_image_quality_jpeg: int
    _save_image_quality_jpeg = Setting(
        _("Image Quality (JPEG)"),
        85,
        _("Encoding quality used when writing JPEG images."),
    )

    save_image_file_name_format: str
    _save_image_file_name_format = Setting(
        _("Save Image File Name Template"),
        "{document_name}-generated-{job_timestamp}-{job_index}-{prompt}",
        "Template for naming saved images (without extension). Available keys: {keys}.".format(
            keys="{document_name}, {job_timestamp}, {current_timestamp}, {job_index}, {prompt}"
        ),
    )

    confirm_discard_image: bool
    _confirm_discard_image = Setting("Ask for confirmation when discarding images", True)

    prompt_line_count: int
    _prompt_line_count = Setting(
        _("Prompt Line Count"), 2, _("Size of the text editor for image descriptions")
    )

    negative_prompt_line_count: int
    _negative_prompt_line_count = Setting(
        _("Negative Prompt Line Count"), 3, _("Size of the text editor for negative prompts")
    )

    prompt_line_count_live: int
    _prompt_line_count_live = Setting("Prompt Line Count (Live)", 2)

    show_negative_prompt: bool
    _show_negative_prompt = Setting(
        _("Negative Prompt"), False, _("Show text editor to describe things to avoid")
    )

    generation_finished_action: GenerationFinishedAction
    _generation_finished_action = Setting(
        _("Finished Generation"),
        GenerationFinishedAction.preview,
        _("Action to take when an image generation job finishes"),
    )

    show_steps: bool
    _show_steps = Setting(
        _("Show Steps"), False, _("Display the number of steps to be evaluated in the weights box.")
    )

    tag_files: list[str]
    _tag_files = Setting(
        _("Tag Auto-Completion"),
        [],
        _("Enable text completion for tags from the selected files"),
    )

    apply_behavior: ApplyBehavior
    _apply_behavior = Setting(
        _("Apply Behavior"),
        ApplyBehavior.layer,
        _("Choose how result images are applied to the canvas (generation workspaces)"),
    )

    apply_region_behavior: ApplyRegionBehavior
    _apply_region_behavior = Setting("Apply Region Behavior", ApplyRegionBehavior.layer_group)

    apply_behavior_live: ApplyBehavior
    _apply_behavior_live = Setting(
        _("Apply Behavior (Live)"),
        ApplyBehavior.replace,
        _("Choose how result images are applied to the canvas in Live mode"),
    )

    apply_region_behavior_live: ApplyRegionBehavior
    _apply_region_behavior_live = Setting(
        "Apply Region Behavior (Live)", ApplyRegionBehavior.replace
    )

    show_builtin_styles: bool
    _show_builtin_styles = Setting(_("Show pre-installed styles"), True)

    recent_styles_count: int
    _recent_styles_count = Setting(
        _("Recent Styles"),
        4,
        _("Number of most recently used styles to show at the top of the style list"),
    )

    recent_styles: list[str]
    _recent_styles = Setting(
        "Recent Styles",
        [
            "built-in/edit-flux2.json",
            "built-in/anime-illustrious.json",
            "built-in/cinematic-photo-zimage.json",
            "built-in/digital-artwork-xl.json",
        ],
    )

    history_size: int
    _history_size = Setting(
        _("Active History Size"),
        1000,
        _("Main memory (RAM) used for the history of generated images"),
    )

    history_storage: int
    _history_storage = Setting(
        _("Stored History Size"),
        20,
        _("Memory used to store generated images in .kra files on disk"),
    )

    history_format: ImageFileFormat
    _history_format = Setting(
        _("History Format"),
        ImageFileFormat.webp,
        _("File format for saving generated images in history"),
    )

    multi_threading: bool
    _multi_threading = Setting(
        _("Multi-Threading"),
        True,
        _("Perform certain plugin operations in background threads"),
    )

    performance_preset: PerformancePreset
    _performance_preset = Setting(
        _("Performance Preset"),
        PerformancePreset.auto,
        _("Configures performance settings to match available hardware."),
    )

    batch_size: int
    _batch_size = Setting(
        _("Maximum Batch Size"),
        4,
        _("Increase efficiency by generating multiple images at once"),
    )

    resolution_multiplier: float
    _resolution_multiplier = Setting(
        _("Resolution Multiplier"),
        1.5,
        _(
            "Scaling factor for generation. With Smart Resolution enabled, this is applied to the style or model preferred resolution. Values below 1.0 improve performance for high resolution canvas."
        ),
    )

    max_pixel_count: int
    _max_pixel_count = Setting(
        _("Maximum Pixel Count"),
        6,
        _("Maximum resolution to generate images at, in megapixels (FullHD ~ 2MP, 4k ~ 8MP)."),
    )

    dynamic_caching: bool
    _dynamic_caching = Setting(
        _("Dynamic Caching"),
        False,
        _("Re-use outputs of previous steps (First Block Cache) to speed up generation."),
    )

    tiled_vae: bool
    _tiled_vae = Setting(
        _("Tiled VAE"),
        False,
        _("Conserve memory by processing output images in smaller tiles."),
    )

    download_retry_attempts: int
    _download_retry_attempts = Setting(
        _("Download Retry Attempts"),
        3,
        _("How many times interrupted downloads should be retried before failing."),
    )

    download_retry_delay: int
    _download_retry_delay = Setting(
        _("Download Retry Delay"),
        1,
        _("Delay in seconds before retrying an interrupted download."),
    )

    download_inactivity_timeout: int
    _download_inactivity_timeout = Setting(
        _("Download Inactivity Timeout"),
        30,
        _("Abort a download if no progress is received for this many seconds."),
    )

    comfy_get_timeout: int
    _comfy_get_timeout = Setting(
        _("Comfy GET Timeout"),
        60,
        _("Timeout in seconds for ComfyUI GET requests."),
    )

    comfy_result_image_timeout: int
    _comfy_result_image_timeout = Setting(
        _("Comfy Result Image Timeout"),
        300,
        _("Timeout in seconds for downloading generated result images from ComfyUI."),
    )

    comfy_model_inspection_timeout: int
    _comfy_model_inspection_timeout = Setting(
        _("Comfy Model Inspection Timeout"),
        600,
        _("Maximum time in seconds to spend inspecting ComfyUI model lists during discovery."),
    )

    websocket_ping_timeout: int
    _websocket_ping_timeout = Setting(
        _("Websocket Ping Timeout"),
        60,
        _("Timeout in seconds before considering the ComfyUI websocket connection unresponsive."),
    )

    cloud_sign_in_timeout: int
    _cloud_sign_in_timeout = Setting(
        _("Cloud Sign-In Timeout"),
        300,
        _("How long to wait for cloud sign-in confirmation before failing."),
    )

    cloud_auth_poll_interval: float
    _cloud_auth_poll_interval = Setting(
        _("Cloud Auth Poll Interval"),
        2.0,
        _("Delay in seconds between cloud sign-in confirmation requests."),
    )

    cloud_job_poll_interval: float
    _cloud_job_poll_interval = Setting(
        _("Cloud Job Poll Interval"),
        0.5,
        _("Delay in seconds between cloud job status checks."),
    )

    auto_update_check_timeout: int
    _auto_update_check_timeout = Setting(
        _("Auto-Update Check Timeout"),
        10,
        _("Timeout in seconds for checking whether a plugin update is available."),
    )

    flux_inpaint_cfg_scale: float
    _flux_inpaint_cfg_scale = Setting(
        _("Flux Inpaint CFG Override"),
        30.0,
        _("Guidance strength applied to Flux fill-model inpaint jobs."),
    )

    _performance_presets: ClassVar[dict[PerformancePreset, PerformancePresetSettings]] = {
        PerformancePreset.cpu: PerformancePresetSettings(
            batch_size=1,
            resolution_multiplier=1.5,
            max_pixel_count=2,
        ),
        PerformancePreset.low: PerformancePresetSettings(
            batch_size=2,
            resolution_multiplier=1.5,
            max_pixel_count=2,
            tiled_vae=True,
        ),
        PerformancePreset.medium: PerformancePresetSettings(
            batch_size=4,
            resolution_multiplier=1.5,
            max_pixel_count=6,
        ),
        PerformancePreset.high: PerformancePresetSettings(
            batch_size=6,
            resolution_multiplier=1.5,
            max_pixel_count=8,
        ),
        PerformancePreset.cloud: PerformancePresetSettings(
            batch_size=8,
            resolution_multiplier=1.5,
            max_pixel_count=6,
        ),
    }

    debug_dump_workflow: bool
    _debug_dump_workflow = Setting(
        _("Dump Workflow"),
        False,
        _("Write latest ComfyUI prompt to the log folder for test & debug"),
    )

    document_defaults: dict[str, Any]
    _document_defaults = Setting(_("Document Defaults"), {}, _("Recently used document settings"))

    last_news: str
    _last_news = Setting("Last seen news digest", "")

    # Folder where intermediate images are stored for debug purposes (default: None)
    debug_image_folder = os.environ.get("KRITA_AI_DIFFUSION_DEBUG_IMAGE")

    changed = pyqtSignal(str, object)

    _values: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.restore(init=True)

    def __getattr__(self, name: str):
        if name in self._values:
            return self._values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in self._values:
            if self._values[name] != value:
                self._values[name] = value
                if name != "document_defaults":
                    self.changed.emit(name, value)
                if name == "performance_preset":
                    self.apply_performance_preset(value)
        else:
            object.__setattr__(self, name, value)

    def restore(self, init=False):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }
        if not init:
            self.server_mode = ServerMode.managed

    def save(self, path: Path | None = None):
        path = self.default_path or path
        with open(path, "w") as file:
            file.write(json.dumps(self._values, default=encode_json, indent=4))

    def load(self, path: Path | None = None):
        path = self.default_path or path
        self._migrate_legacy_settings(path)
        if not path.exists():
            self.save()  # create new file with defaults
            return

        log.info(f"Loading settings from {path}")
        try:
            contents = read_json_with_comments(path)
            for k, v in contents.items():
                setting: Setting | None = getattr(Settings, f"_{k}", None)
                if setting is not None:
                    if isinstance(setting.default, Enum):
                        self._values[k] = setting.str_to_enum(v)
                    elif isinstance(setting.default, type(v)):
                        self._values[k] = v
                    else:
                        log.error(f"{path}: {v} is not a valid value for '{k}'")
                        self._values[k] = setting.default
        except Exception as e:
            log.error(f"Failed to load settings: {e}")

    def apply_performance_preset(self, preset: PerformancePreset):
        if preset not in [PerformancePreset.custom, PerformancePreset.auto]:
            for k, v in self._performance_presets[preset]._asdict().items():
                self._values[k] = v

    def image_format_quality(self, format: ImageFileFormat):
        if format is ImageFileFormat.png:
            return self.save_image_quality_png
        if format is ImageFileFormat.png_small:
            return self.save_image_quality_png_small
        if format is ImageFileFormat.webp:
            return self.save_image_quality_webp
        if format is ImageFileFormat.webp_lossless:
            return self.save_image_quality_webp_lossless
        if format is ImageFileFormat.jpeg:
            return self.save_image_quality_jpeg
        return 85

    def __iter__(self):
        return iter(self._values.items())

    def _migrate_legacy_settings(self, path: Path):
        if path == self.default_path:
            legacy_path = Path(__file__).parent / "settings.json"
            if legacy_path.exists() and not path.exists():
                try:
                    legacy_path.rename(path)
                    log.info(f"Migrated settings from {legacy_path} to {path}")
                except Exception as e:
                    log.warning(f"Failed to migrate settings from {legacy_path} to {path}: {e}")


settings = Settings()
