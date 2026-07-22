import os
from asyncio import Future, get_running_loop
from pathlib import Path

import pytest
from aiohttp import ClientSession
from PyQt5.QtCore import pyqtBoundSignal

from ai_diffusion.backend.network import RequestManager
from ai_diffusion.model.updates import AutoUpdate, UpdateState
from ai_diffusion.platform_tools import ZipFile
from ai_diffusion.settings import settings

from .conftest import CloudService, qtapp


class SignalObserver:
    def __init__(self, signal: pyqtBoundSignal):
        self.events = []
        signal.connect(self.on_changed)

    def on_changed(self, value):
        self.events.append(value)

    def reset(self):
        self.events = []


class StubRequestManager(RequestManager):
    def __init__(self, calls: list[tuple[str, float | None, str | None]]):
        self._calls = calls

    def get(
        self, url: str, timeout: float | None = None, bearer: str | None = None
    ) -> Future[dict[str, str]]:
        self._calls.append((url, timeout, bearer))
        future: Future[dict[str, str]] = get_running_loop().create_future()
        future.set_result({"version": "1.50.6"})
        return future


def http_session(service_url: str):
    service_token = os.environ["INTERSTICE_INFRA_TOKEN"]
    headers = {"Authorization": f"Bearer {service_token}"}
    return ClientSession(service_url, headers=headers)


@qtapp
async def test_auto_update(cloud_service: CloudService, tmp_path: Path):
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")
    await run_auto_update_test(cloud_service, tmp_path)


async def run_auto_update_test(service: CloudService, tmp_path: Path):
    async with http_session(service.url) as session:
        last_version = new_version = "666.6.6"

        # Get the latest plugin version (set from previous test)
        async with session.get(f"/plugin/latest?version={new_version}") as response:
            assert response.status == 200
            result = await response.json()
            last_version = result["version"]
            a, b, c = last_version.split(".")
            new_version = f"{a}.{b}.{int(c) + 1}"

        # Create an existing installation
        install_dir = tmp_path / "install"
        install_plugin_dir = install_dir / "test_plugin"
        install_plugin_dir.mkdir(parents=True)
        install_test_file = install_plugin_dir / "test_file.txt"
        install_test_file.write_text("local produce is the best")

        updater = AutoUpdate(
            current_version=last_version,
            plugin_dir=install_dir,
            api_url=service.url,
        )
        assert updater.state is UpdateState.unknown

        state_changes = SignalObserver(updater.state_changed)
        await updater.check()
        assert state_changes.events == [UpdateState.checking, UpdateState.latest]
        assert updater.state is UpdateState.latest

        # Create a new plugin version
        build_dir = tmp_path / "build"
        build_plugin_dir = build_dir / "test_plugin"
        build_plugin_dir.mkdir(parents=True)
        build_test_file = build_plugin_dir / "test_file.txt"
        build_test_file.write_text("if you're feeling orange, try flying a kite")
        build_archive = build_dir / f"test_plugin-{new_version}.zip"

        # Build the plugin archive
        with ZipFile(build_archive, "w") as zip_file:
            for file in build_plugin_dir.iterdir():
                zip_file.write(file, f"test_plugin/{file.name}")

        # Upload the plugin as new version
        archive_data = build_archive.read_bytes()
        async with session.put(f"/plugin/upload/{new_version}", data=archive_data) as response:
            assert response.status == 200
            uploaded = await response.json()
            assert uploaded["status"] == "uploaded" and uploaded["version"] == new_version

        # Check for new version
        state_changes.reset()
        await updater.check()
        assert state_changes.events == [UpdateState.checking, UpdateState.available]
        assert updater.state is UpdateState.available
        assert updater.latest_version == new_version

        # Run the update
        state_changes.reset()
        await updater.run()
        assert state_changes.events == [
            UpdateState.downloading,
            UpdateState.installing,
            UpdateState.restart_required,
        ]
        assert updater.state is UpdateState.restart_required
        assert updater.latest_version == new_version
        assert install_test_file.read_text() == "if you're feeling orange, try flying a kite"


async def test_authorization(cloud_service: CloudService):
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")
    async with ClientSession(cloud_service.url) as session:
        # Version check is public
        async with session.get("/plugin/latest?version=1.2.3") as response:
            assert response.status == 200

        # Upload requires authorization
        async with session.put("/plugin/upload/1.2.3") as response:
            assert response.status == 401


@qtapp
async def test_auto_update_uses_configured_check_timeout(monkeypatch):
    calls = []

    updater = AutoUpdate(current_version="1.50.6", api_url="https://example.com")
    updater._request_manager = StubRequestManager(calls)
    monkeypatch.setattr(settings, "auto_update_check_timeout", 42)

    await updater.check()

    assert calls == [("https://example.com/plugin/latest?version=1.50.6", 42, None)]
    assert updater.state is UpdateState.latest


@qtapp
async def test_auto_update_defaults_to_configured_cloud_api_url(monkeypatch):
    calls = []

    monkeypatch.setattr(settings, "cloud_api_url", "https://api.example.test")
    monkeypatch.setattr(settings, "auto_update_check_timeout", 42)

    updater = AutoUpdate(current_version="1.50.6")
    updater._request_manager = StubRequestManager(calls)

    await updater.check()

    assert calls == [("https://api.example.test/plugin/latest?version=1.50.6", 42, None)]
    assert updater.state is UpdateState.latest
