from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any

from ..util import client_logger as log
from .api import WorkflowInput
from .client import Client, ClientMessage, PerformanceSettings
from .resources import Arch


class ClientPool(Client):
    """Combines several servers into one client.

    Reads (models, features, url) come from the primary server, which is the one shown in
    the UI. Jobs are sent to a server that has the required models and the fewest queued
    jobs, preferring the server which served the same architecture last.
    """

    def __init__(self, clients: Sequence[Client]):
        assert clients, "ClientPool needs at least one client"
        self._clients = list(clients)
        self._job_targets: dict[str, Client] = {}
        self._last_client: dict[Arch, Client] = {}
        self.url = self.primary.url
        self.models = self.primary.models
        self.device_info = self.primary.device_info

    @property
    def clients(self) -> list[Client]:
        return self._clients

    @property
    def primary(self) -> Client:
        return self._clients[0]

    @property
    def features(self):
        return self.primary.features

    @property
    def performance_settings(self) -> PerformanceSettings:
        return self.primary.performance_settings

    @property
    def user(self):
        return self.primary.user

    @property
    def news(self):
        return self.primary.news

    @property
    def missing_resources(self):
        return self.primary.missing_resources

    def supports_arch(self, arch: Arch) -> bool:
        return any(c.supports_arch(arch) for c in self._clients)

    async def connect(self) -> None:
        errors = []
        connected = []
        for client in self._clients:
            try:
                await client.connect()
                connected.append(client)
            except Exception as e:
                if client is self.primary:
                    raise
                log.warning(f"Failed to connect to {client.url}: {e!s}")
                errors.append(client)
        for client in errors:
            self._clients.remove(client)
        assert self._clients, "No server could be connected"

    async def disconnect(self):
        await asyncio.gather(*(c.disconnect() for c in self._clients), return_exceptions=True)

    async def discover_models(self, refresh: bool) -> AsyncGenerator[Client.DiscoverStatus, Any]:
        totals = [0] * len(self._clients)
        for i, client in enumerate(self._clients):
            async for status in client.discover_models(refresh):
                totals[i] = status.total
                yield Client.DiscoverStatus(
                    status.folder, sum(totals[:i]) + status.current, sum(totals)
                )

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        client = self._select(work)
        job_id = await client.enqueue(work, front)
        self._job_targets[job_id] = client
        return job_id

    async def interrupt(self):
        await asyncio.gather(*(c.interrupt() for c in self._clients), return_exceptions=True)

    async def cancel(self, job_ids: Iterable[str]):
        job_ids = list(job_ids)
        by_client: dict[Client, list[str]] = {}
        for job_id in job_ids:
            client = self._job_targets.pop(job_id, self.primary)
            by_client.setdefault(client, []).append(job_id)
        if by_client:
            await asyncio.gather(*(c.cancel(ids) for c, ids in by_client.items()))

    async def get_folder_metadata(self, folder: str) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for client in self._clients:
            read = getattr(client, "get_folder_metadata", None)
            if read is None:
                continue
            data = await read(folder)
            if data:
                result.update(data)
        return result

    async def set_model_metadata(self, folder: str, filename: str, metadata: dict) -> bool:
        write = getattr(self.primary, "set_model_metadata", None)
        if write is None:
            return False
        return bool(await write(folder, filename, metadata))

    async def refresh(self):
        await asyncio.gather(*(c.refresh() for c in self._clients), return_exceptions=True)

    async def translate(self, text: str, lang: str) -> str:
        return await self.primary.translate(text, lang)

    async def listen(self) -> AsyncGenerator[ClientMessage, Any]:
        queue: asyncio.Queue[ClientMessage | None] = asyncio.Queue()

        async def pump(client: Client):
            try:
                async for message in client.listen():
                    await queue.put(message)
            except Exception as e:
                log.warning(f"Server {client.url} stopped listening: {e!s}")
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(pump(c)) for c in self._clients]
        remaining = len(tasks)
        try:
            while remaining > 0:
                message = await queue.get()
                if message is None:
                    remaining -= 1
                else:
                    yield message
        finally:
            for task in tasks:
                task.cancel()

    def _select(self, work: WorkflowInput) -> Client:
        input = work.models
        if input is None:
            return self.primary
        arch = input.version
        loras = [lora.name for lora in input.loras]
        candidates = [
            client for client in self._clients if self._has_models(client, input.checkpoint, loras)
        ]
        if not candidates:
            log.warning(f"No server has '{input.checkpoint}', using {self.primary.url}")
            return self.primary
        last = self._last_client.get(arch)
        chosen = min(
            candidates,
            key=lambda c: (self._queued(c), c is not last),
        )
        self._last_client[arch] = chosen
        return chosen

    def _has_models(self, client: Client, checkpoint: str, loras: list[str]) -> bool:
        models = client.models
        if checkpoint not in models.checkpoints:
            return False
        return not any(lora not in models.loras for lora in loras)

    def _queued(self, client: Client) -> int:
        return int(getattr(client, "queued_count", 0))
