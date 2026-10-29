from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from typing import Any

import pytest

from ai_diffusion.backend.api import CheckpointInput, LoraInput, WorkflowInput
from ai_diffusion.backend.client import Client, ClientFeatures, ClientModels
from ai_diffusion.backend.client_pool import ClientPool
from ai_diffusion.backend.resources import Arch


class FakeClient(Client):
    def __init__(self, url: str, checkpoints: list[str], loras: list[str] | None = None):
        self.url = url
        self.models = ClientModels()
        for name in checkpoints:
            self.models.checkpoints[name] = _checkpoint(name)
        self.models.loras = list(loras or [])
        self.device_info = Any  # type: ignore[assignment]
        self.queued_count = 0
        self.enqueued: list[str] = []

    async def connect(self):
        pass

    async def discover_models(self, refresh: bool) -> AsyncGenerator[Client.DiscoverStatus, Any]:
        yield Client.DiscoverStatus("checkpoints", 1, 1)

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        job_id = f"{self.url}-{len(self.enqueued)}"
        self.enqueued.append(job_id)
        return job_id

    async def listen(self) -> AsyncGenerator[Any, Any]:
        for event in ():
            yield event

    async def interrupt(self):
        pass

    async def cancel(self, job_ids: Iterable[str]):
        pass

    @property
    def features(self) -> ClientFeatures:
        raise NotImplementedError

    @property
    def performance_settings(self):
        raise NotImplementedError


def _checkpoint(name: str, arch: Arch = Arch.sdxl):
    from ai_diffusion.backend.client import CheckpointInfo
    from ai_diffusion.files import FileFormat

    return CheckpointInfo(name, arch, FileFormat.checkpoint)


def _work(checkpoint: str, arch: Arch = Arch.sdxl, loras: list[str] | None = None):
    return WorkflowInput(
        kind=Any,  # type: ignore[arg-type]
        images=None,
        models=CheckpointInput(
            checkpoint=checkpoint,
            version=arch,
            loras=[LoraInput(name, 1.0) for name in (loras or [])],
        ),
        sampling=Any,  # type: ignore[arg-type]
        conditioning=Any,  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_routes_to_server_with_checkpoint():
    a = FakeClient("a", ["sd_xl.safetensors"])
    b = FakeClient("b", ["sd_xl.safetensors"])
    pool = ClientPool([a, b])
    await pool.enqueue(_work("sd_xl.safetensors"))
    assert a.enqueued == ["a-0"]
    assert b.enqueued == []


@pytest.mark.anyio
async def test_requires_checkpoint_and_loras():
    a = FakeClient("a", ["sd_xl.safetensors"], ["style.safetensors"])
    b = FakeClient("b", ["sd_xl.safetensors"])
    pool = ClientPool([a, b])
    await pool.enqueue(_work("sd_xl.safetensors", loras=["style.safetensors"]))
    assert a.enqueued and not b.enqueued


@pytest.mark.anyio
async def test_prefers_least_busy_and_sticks():
    a = FakeClient("a", ["sd_xl.safetensors"])
    b = FakeClient("b", ["sd_xl.safetensors"])
    pool = ClientPool([a, b])
    await pool.enqueue(_work("sd_xl.safetensors"))
    assert a.enqueued  # tie: first server wins and stays sticky
    await pool.enqueue(_work("sd_xl.safetensors"))
    assert len(a.enqueued) == 2 and not b.enqueued

    a.queued_count = 3
    await pool.enqueue(_work("sd_xl.safetensors"))
    assert len(b.enqueued) == 1


@pytest.mark.anyio
async def test_falls_back_to_primary():
    a = FakeClient("a", ["sd_xl.safetensors"])
    b = FakeClient("b", ["flux.safetensors"])
    pool = ClientPool([a, b])
    await pool.enqueue(_work("unknown.safetensors"))
    assert a.enqueued == ["a-0"]


@pytest.mark.anyio
async def test_cancel_routes_to_owning_server():
    a = FakeClient("a", ["sd_xl.safetensors"])
    b = FakeClient("b", ["sd_xl.safetensors"])
    cancelled: list[tuple[str, list[str]]] = []

    async def cancel(self, job_ids):
        cancelled.append((self.url, list(job_ids)))

    a.cancel = cancel.__get__(a)  # type: ignore[method-assign]
    b.cancel = cancel.__get__(b)  # type: ignore[method-assign]
    pool = ClientPool([a, b])
    a.queued_count = 5
    job_id = await pool.enqueue(_work("sd_xl.safetensors"))
    await pool.cancel([job_id])
    assert cancelled == [("b", [job_id])]
