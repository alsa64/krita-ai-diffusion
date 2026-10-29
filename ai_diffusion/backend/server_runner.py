from __future__ import annotations

import asyncio
import time
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .. import platform_tools
from ..localization import translate as _
from ..settings import ServerMode, settings
from ..util import client_logger as log
from .comfy_client import parse_url
from .network import RequestManager
from .server import Server, ServerState
from .server_profile import ServerKind, ServerProfile


class RunnerState(Enum):
    stopped = 0
    starting = 1
    running = 2
    error = 3


_probe_network: RequestManager | None = None


def _network() -> RequestManager:
    global _probe_network
    if _probe_network is None:
        _probe_network = RequestManager()
    return _probe_network


async def probe_server(url: str, timeout: float = 2.0) -> bool:
    try:
        await _network().get(f"{parse_url(url)}/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


class ServerRunner(QObject):
    """Starts, stops and watches a single server profile."""

    state_changed = pyqtSignal()

    def __init__(self, profile: ServerProfile, parent: QObject | None = None):
        super().__init__(parent)
        self.profile = profile
        self.url = profile.url
        self._state = RunnerState.stopped
        self._error = ""
        self._server: Server | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._log_task: asyncio.Task | None = None

    @property
    def state(self) -> RunnerState:
        if self.profile.is_managed and self._server is not None:
            match self._server.state:
                case ServerState.running:
                    return RunnerState.running
                case ServerState.starting:
                    return RunnerState.starting
        return self._state

    @state.setter
    def state(self, state: RunnerState):
        if self._state is not state:
            self._state = state
            self.state_changed.emit()

    @property
    def error(self):
        return self._error

    @property
    def managed_server(self) -> Server | None:
        return self._server

    def _managed(self) -> Server:
        if self._server is None or str(self._server.path) != self.profile.directory:
            self._server = Server(self.profile.directory or settings.server_path)
        return self._server

    async def ensure_running(self) -> str:
        """Return the server URL, starting the server if it is not reachable."""
        if self.profile.is_managed:
            server = self._managed()
            if server.state is ServerState.running:
                self.url = server.url or self.url
                return self.url
            if await probe_server(self.url):
                self.state = RunnerState.running
                return self.url
            if self.profile.auto_start or server.state is ServerState.stopped:
                self.state = RunnerState.starting
                self.url = await server.start()
                self.state = RunnerState.running
                return self.url
            raise RuntimeError(_("Server is not running"))

        if await probe_server(self.url):
            self.state = RunnerState.running
            return self.url

        if not self.profile.auto_start or not self.profile.has_start_command:
            raise RuntimeError(_("Server is not running"))

        await self.start()
        return self.url

    async def start(self):
        if self.profile.is_managed:
            self.state = RunnerState.starting
            try:
                self.url = await self._managed().start()
                self.state = RunnerState.running
            except Exception as e:
                self._error = str(e)
                self.state = RunnerState.error
                raise
            return self.url

        if await probe_server(self.url):
            self.state = RunnerState.running
            return self.url

        self.state = RunnerState.starting
        self._error = ""
        try:
            if self.profile.kind is ServerKind.ssh:
                if not self.profile.ssh_host:
                    raise RuntimeError(_("SSH host is not configured"))
                command = self.profile.ssh_command_line()
            else:
                command = self.profile.command_line()
            log.info(f"Starting server '{self.profile.display_name}': {' '.join(command)}")
            self._process = await platform_tools.create_process(
                command[0],
                *command[1:],
                cwd=Path(self.profile.directory) if self.profile.directory else None,
                additional_env=self.profile.environment,
                is_job=True,
            )
            self._log_task = asyncio.create_task(self._forward_output(self._process))
            await self._wait_until_ready()
            self.state = RunnerState.running
        except Exception as e:
            self._error = str(e)
            self.state = RunnerState.error
            await self._terminate_process()
            raise
        return self.url

    async def _wait_until_ready(self):
        deadline = time.monotonic() + max(1, self.profile.start_timeout)
        while time.monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    _("Server process exited during startup with code {code}").format(
                        code=self._process.returncode
                    )
                )
            if await probe_server(self.url, timeout=2.0):
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(
            _("Server did not become ready within {seconds} seconds").format(
                seconds=self.profile.start_timeout
            )
        )

    async def _forward_output(self, process: asyncio.subprocess.Process):
        assert process.stdout is not None
        while line := await process.stdout.readline():
            log.info(f"[{self.profile.display_name}] {line.decode(errors='replace').rstrip()}")

    async def stop(self):
        if self.profile.is_managed and self._server is not None:
            if self._server.state is ServerState.running:
                await self._server.stop()
            self.state = RunnerState.stopped
            return
        await self._terminate_process()
        self.state = RunnerState.stopped

    async def _terminate_process(self):
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._process.kill()
            self._process = None
        if self._log_task is not None:
            self._log_task.cancel()
            self._log_task = None

    def terminate(self):
        """Synchronous shutdown, called when Krita exits."""
        if self._server is not None:
            self._server.terminate()
        if self._process is not None and self._process.returncode is None:
            # Note: for SSH profiles this only closes the connection, not the remote server.
            self._process.terminate()


class ServerManager(QObject):
    profiles_changed = pyqtSignal()
    state_changed = pyqtSignal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._runners: dict[str, ServerRunner] = {}
        self._profiles: list[ServerProfile] = [
            ServerProfile.from_dict(p) for p in settings.server_profiles
        ]
        self._migrate_settings()
        settings.changed.connect(self._handle_settings_changed)

    def _migrate_settings(self):
        """Convert the legacy single-server modes into a profile, once."""
        mode = settings.server_mode
        if not self._profiles:
            profile = ServerProfile.preset(ServerKind.local, settings.server_path)
            if mode is ServerMode.external:
                profile.name = _("Custom Server")
                profile.url = settings.server_url
                profile.command = ""
            elif mode is ServerMode.cloud:
                profile.name = _("Online Service")
                profile.url = settings.cloud_api_url
                profile.command = ""
            elif mode is ServerMode.undefined:
                self._profiles = [profile]
                return
            self._profiles = [profile]
            settings.active_server = profile.id
            settings.server_mode = ServerMode.servers
            self.save()
            log.info(f"Migrated legacy server mode {mode.name} to server profile")
            return
        if mode is not ServerMode.servers and mode is not ServerMode.undefined:
            settings.server_mode = ServerMode.servers

    @property
    def profiles(self) -> list[ServerProfile]:
        return self._profiles

    @property
    def active(self) -> ServerProfile | None:
        if not self._profiles:
            return None
        return (
            next((p for p in self._profiles if p.id == settings.active_server), None)
            or (self._profiles[0])
        )

    def find(self, profile_id: str) -> ServerProfile | None:
        return next((p for p in self._profiles if p.id == profile_id), None)

    def runner(self, profile: ServerProfile) -> ServerRunner:
        runner = self._runners.get(profile.id)
        if runner is None:
            runner = ServerRunner(profile, self)
            runner.state_changed.connect(self.state_changed)
            self._runners[profile.id] = runner
        else:
            runner.profile = profile
        return runner

    def active_runner(self) -> ServerRunner | None:
        profile = self.active
        return self.runner(profile) if profile is not None else None

    def save(self):
        settings.server_profiles = [p.to_dict() for p in self._profiles]
        if self.active is not None and settings.active_server != self.active.id:
            settings.active_server = self.active.id
        settings.save()
        self.profiles_changed.emit()

    def add(self, profile: ServerProfile):
        self._profiles.append(profile)
        if not settings.active_server:
            settings.active_server = profile.id
        self.save()

    def update(self, profile: ServerProfile, save=True):
        for i, existing in enumerate(self._profiles):
            if existing.id == profile.id:
                self._profiles[i] = profile
                break
        if save:
            self.save()

    def remove(self, profile_id: str):
        self._profiles = [p for p in self._profiles if p.id != profile_id]
        if runner := self._runners.pop(profile_id, None):
            runner.terminate()
            runner.deleteLater()
        if settings.active_server == profile_id:
            settings.active_server = self._profiles[0].id if self._profiles else ""
        self.save()

    def set_active(self, profile_id: str):
        settings.active_server = profile_id
        settings.save()
        self.profiles_changed.emit()

    def ensure_active(self) -> ServerProfile:
        """Return the active profile, creating a default one if the list is empty."""
        if not self._profiles:
            self.add(ServerProfile.preset(ServerKind.local, settings.server_path))
        profile = self.active
        assert profile is not None
        return profile

    async def ensure_running(self, profile: ServerProfile) -> str:
        return await self.runner(profile).ensure_running()

    def _handle_settings_changed(self, key: str, value: object):
        if key == "server_profiles":
            self._profiles = [ServerProfile.from_dict(p) for p in settings.server_profiles]
            self.profiles_changed.emit()

    def terminate_all(self):
        for runner in self._runners.values():
            runner.terminate()
