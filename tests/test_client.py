import asyncio
from pathlib import Path

import pytest

from ai_diffusion import eventloop
from ai_diffusion.backend import comfy_client as comfy_client_module
from ai_diffusion.backend import resources
from ai_diffusion.backend.api import (
    CheckpointInput,
    ConditioningInput,
    ImageInput,
    LoraInput,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.backend.client import ClientEvent, ClientModels, resolve_arch
from ai_diffusion.backend.cloud_client import CloudClient
from ai_diffusion.backend.comfy_client import ComfyClient, parse_url, websocket_args, websocket_url
from ai_diffusion.backend.network import NetworkError
from ai_diffusion.backend.resources import ControlMode, ResourceKind, UpscalerName, resource_id
from ai_diffusion.backend.server import Server, ServerBackend, ServerState
from ai_diffusion.files import File, FileFormat, FileLibrary
from ai_diffusion.image import Extent
from ai_diffusion.platform_tools import get_cuda_devices
from ai_diffusion.settings import settings
from ai_diffusion.style import Arch, Style
from ai_diffusion.util import ensure

from .config import default_checkpoint, server_dir
from .conftest import qtapp


@pytest.fixture(scope="session")
def comfy_server(qtapp):
    backend = ServerBackend.cpu
    if len(get_cuda_devices()) > 0:
        backend = ServerBackend.cuda

    server = Server(str(server_dir), backend)
    assert server.state is ServerState.stopped, (
        f"Expected server installation at {server_dir}. To create the default installation run"
        " `pytest tests/test_server.py --test-install`"
    )
    qtapp.run(server.start(port=8189))
    yield server
    qtapp.run(server.stop())


def make_default_work(size=512, steps=20):
    return WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(default_checkpoint[Arch.sd15]),
        images=ImageInput.from_extent(Extent(size, size)),
        conditioning=ConditioningInput("a photo of a cat", "a photo of a dog"),
        sampling=SamplingInput("euler", "normal", cfg_scale=7.0, total_steps=steps),
    )


@qtapp
async def test_connect_bad_url(comfy_server):
    client = ComfyClient("bad_url")
    with pytest.raises(NetworkError):
        await client.connect()


@pytest.mark.parametrize("cancel_point", ["after_enqueue", "after_start", "after_sampling"])
@qtapp
async def test_cancel(comfy_server: Server, cancel_point):
    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    async for _ in client.discover_models(refresh=False):
        pass
    job_id = None
    interrupted = False
    stage = 0

    async for msg in client.listen():
        if msg.event is ClientEvent.error:
            assert False, msg.error

        elif stage == 0:
            assert msg.event is not ClientEvent.finished
            assert msg.job_id in (job_id, "")
            if not job_id:
                job_id = await client.enqueue(make_default_work(steps=1000))
                assert client.queued_count == 1
            if not interrupted:
                if cancel_point == "after_enqueue":
                    await client.cancel([job_id])
                    interrupted = True
                if cancel_point == "after_start" and msg.event is ClientEvent.progress:
                    await client.interrupt()
                    interrupted = True
                if cancel_point == "after_sampling" and msg.progress > 0.1:
                    await client.interrupt()
                    interrupted = True
            if msg.event is ClientEvent.interrupted:
                assert msg.job_id == job_id
                assert not client.is_executing and client.queued_count == 0

                job_id = await client.enqueue(make_default_work(size=320, steps=1))
                stage = 1
                assert client.queued_count == 1
            elif msg.event is ClientEvent.progress:
                assert stage == 0

        elif stage == 1:
            assert msg.event is not ClientEvent.interrupted
            assert msg.job_id in (job_id, "")
            if msg.event is ClientEvent.finished:
                assert msg.images is not None and len(msg.images) > 0
                assert msg.images[0].extent == Extent(320, 320)
                break

    assert not client.is_executing and client.queued_count == 0


@qtapp
async def test_disconnect(comfy_server: Server):
    async def listen(client: ComfyClient):
        async for msg in client.listen():
            assert msg.event is ClientEvent.connected

    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    task = eventloop._loop.create_task(listen(client))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client.is_executing and client.queued_count == 0


@pytest.mark.parametrize(
    "url,expected_http,expected_ws",
    [
        ("http://localhost:8000", "http://localhost:8000", "ws://localhost:8000"),
        ("http://localhost:8000/", "http://localhost:8000", "ws://localhost:8000"),
        ("http://localhost:8000/foo", "http://localhost:8000/foo", "ws://localhost:8000/foo"),
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234", "ws://127.0.0.1:1234"),
        ("localhost:8000", "http://localhost:8000", "ws://localhost:8000"),
        ("https://localhost:8000", "https://localhost:8000", "wss://localhost:8000"),
    ],
)
def test_parse_url(url, expected_http, expected_ws):
    parsed = parse_url(url)
    assert parsed == expected_http and websocket_url(parsed) == expected_ws


def check_client_info(client: ComfyClient):
    assert client.device_info.type in ["cpu", "cuda"]
    assert client.device_info.name != ""
    assert client.device_info.vram > 0

    assert len(client.models.checkpoints) > 0
    for filename, cp in client.models.checkpoints.items():
        assert cp.filename == filename
        assert cp.filename.startswith(cp.name)
        assert cp.format is FileFormat.checkpoint

    assert len(client.models.resources) >= len(resources.required_resource_ids)
    inpaint = client.models.for_arch(Arch.sd15).control[ControlMode.inpaint]
    assert inpaint and "inpaint" in inpaint


def test_configurable_default_upscalers(monkeypatch):
    models = ClientModels()
    models.upscalers = ["custom-default.pth", "custom-small.pth"]
    models.resources = {
        resource_id(
            ResourceKind.upscaler, Arch.all, UpscalerName.default
        ): UpscalerName.default.value,
        resource_id(
            ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x
        ): UpscalerName.fast_2x.value,
    }

    monkeypatch.setattr(settings, "upscale_model", "custom-default.pth")
    monkeypatch.setattr(settings, "upscale_model_small", "custom-small.pth")
    assert models.default_upscaler == "custom-default.pth"
    assert models.default_upscaler_small == "custom-small.pth"


def test_configurable_default_upscalers_fall_back(monkeypatch):
    models = ClientModels()
    models.upscalers = ["available-only.pth"]
    models.resources = {
        resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): "fallback-default.pth",
        resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x): "fallback-small.pth",
    }

    monkeypatch.setattr(settings, "upscale_model", "missing-default.pth")
    monkeypatch.setattr(settings, "upscale_model_small", "missing-small.pth")
    assert models.default_upscaler == "fallback-default.pth"
    assert models.default_upscaler_small == "fallback-small.pth"


@pytest.mark.asyncio
async def test_comfy_client_uses_configured_timeouts(monkeypatch):
    class StubRequests:
        def __init__(self):
            self.calls = []

        async def get(self, url, timeout=None, bearer=None):
            self.calls.append(("get", url, timeout, bearer))
            if "model_info" in url:
                return {"item": {}, "_meta": {"total": 100}}
            return {}

        async def download(self, url, timeout=None):
            self.calls.append(("download", url, timeout))
            raise RuntimeError("download failed")

    client = ComfyClient("http://example.com")
    client._requests = StubRequests()
    monkeypatch.setattr(settings, "comfy_get_timeout", 77)
    monkeypatch.setattr(settings, "comfy_result_image_timeout", 321)
    monkeypatch.setattr(settings, "comfy_model_inspection_timeout", 5)
    monkeypatch.setattr(comfy_client_module, "time", iter([0, 0, 10]).__next__)

    await client._get("system_stats")
    assert client._requests.calls[0] == ("get", "http://example.com/system_stats", 77, None)

    updates = [status async for status in client.try_inspect("checkpoints")]
    assert len(updates) == 1
    assert client._requests.calls[1] == (
        "get",
        "http://example.com/api/etn/model_info/checkpoints?offset=0&limit=8",
        77,
        None,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await client._transfer_result_image("abc")
    assert client._requests.calls[2] == (
        "download",
        "http://example.com/api/etn/image/abc",
        321,
    )


def test_websocket_args_uses_configured_ping_timeout(monkeypatch):
    monkeypatch.setattr(settings, "websocket_ping_timeout", 91)
    assert websocket_args("token") == {
        "max_size": 2**30,
        "ping_timeout": 91,
        "additional_headers": {"Authorization": "Bearer token"},
    }


@pytest.mark.asyncio
async def test_cloud_sign_in_uses_configured_timeout_and_poll_interval(monkeypatch):
    client = CloudClient("https://example.com")
    calls = []
    statuses = iter([
        {"url": "/auth"},
        {"status": "not-found"},
        {"status": "authorized", "token": "token-123"},
    ])

    async def fake_post(op, data):
        calls.append((op, data))
        return next(statuses)

    sleeps = []

    async def fake_sleep(interval):
        sleeps.append(interval)

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(settings, "cloud_sign_in_timeout", 123)
    monkeypatch.setattr(settings, "cloud_auth_poll_interval", 4.5)

    values = []
    async for value in client.sign_in():
        values.append(value)

    assert values == [f"{client.default_web_url}/auth", "token-123"]
    assert sleeps == [4.5]
    assert [op for op, _ in calls] == ["auth/initiate", "auth/confirm", "auth/confirm"]


@pytest.mark.asyncio
async def test_cloud_generate_uses_configured_job_poll_interval(monkeypatch):
    client = CloudClient("https://example.com")
    job = type("Job", (), {"input": {}, "local_id": "local", "state": None})()
    client._user = type("User", (), {"credits": 10})()

    responses = iter([
        {"id": "remote", "worker_id": "worker", "status": "IN_QUEUE", "user": None},
        {"status": "IN_PROGRESS", "output": {"progress": 0.5}},
        {"status": "COMPLETED", "output": {"images": {"offsets": [0, 1], "base64": "AA=="}}},
    ])

    async def fake_post(op, data):
        return next(responses)

    sleeps = []

    async def fake_sleep(interval):
        sleeps.append(interval)

    async def fake_report(*args, **kwargs):
        return None

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_report", fake_report)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(settings, "cloud_job_poll_interval", 1.75)

    await client._generate(job)

    assert job.remote_id == "remote"
    assert job.worker_id == "worker"
    assert job.output == {"images": {"offsets": [0, 1], "base64": "AA=="}}
    assert sleeps == [1.75, 1.75]


def check_resolve_sd_version(client: ComfyClient, arch: Arch):
    checkpoint = next(cp for cp in client.models.checkpoints.values() if cp.arch == arch)
    style = Style(Path("dummy"))
    style.architecture = Arch.auto
    style.checkpoints = [checkpoint.filename]
    assert resolve_arch(style, client) == arch
    assert resolve_arch(style, None) == arch


def check_nunchaku(server: Server, client: ComfyClient):
    if server.backend is ServerBackend.cuda:
        assert "NunchakuFluxDiTLoader" in client.models.node_inputs.nodes


@qtapp
async def test_info(pytestconfig, comfy_server: Server):
    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    async for _ in client.discover_models(refresh=False):
        pass
    check_client_info(client)
    await client.refresh()
    check_client_info(client)
    check_resolve_sd_version(client, Arch.sd15)
    # check_resolve_sd_version(client, Arch.sdxl) # no SDXL checkpoint in default installation
    check_nunchaku(comfy_server, client)


@qtapp
async def test_upload_lora(comfy_server: Server, tmp_path: Path):
    lora_path = tmp_path / "test-lora.safetensors"
    lora_path.write_bytes(b"testdata" * 1024 * 1024)

    files = FileLibrary.instance()
    file = files.loras.add(File.local(lora_path, compute_hash=True))

    assert comfy_server.url is not None
    client = ComfyClient(comfy_server.url)
    await client.connect()
    if file.id in client.models.loras:
        client.models.loras.remove(file.id)

    input = make_default_work()
    assert input.models is not None
    input.models.loras = [LoraInput(file.id, 1.0, storage_id=ensure(file.hash))]

    task = asyncio.get_running_loop().create_task(client.upload_loras(input, "JOB-ID"))
    upload_progress = 0
    async for msg in client.listen():
        if msg.event is ClientEvent.upload:
            assert msg.job_id == "JOB-ID"
            assert msg.progress >= upload_progress
            upload_progress = msg.progress
            if upload_progress == 1.0:
                break

    await task
    assert file.id in client.models.loras
