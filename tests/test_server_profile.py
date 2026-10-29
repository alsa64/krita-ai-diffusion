from ai_diffusion.backend.server_profile import (
    ServerKind,
    ServerProfile,
    format_environment,
    parse_environment,
)
from ai_diffusion.settings import Settings


def test_local_preset():
    profile = ServerProfile.preset(ServerKind.local, "/data/server")
    assert profile.kind is ServerKind.local
    assert profile.directory == "/data/server"
    assert profile.is_managed
    assert not profile.has_start_command


def test_ssh_preset_has_start_command():
    profile = ServerProfile.preset(ServerKind.ssh)
    assert profile.kind is ServerKind.ssh
    assert profile.has_start_command
    assert not profile.is_managed
    assert profile.auto_start


def test_roundtrip():
    profile = ServerProfile.preset(ServerKind.ssh)
    profile.name = "Remote"
    profile.ssh_host = "gpu-box"
    profile.ssh_user = "vyls"
    profile.ssh_port = 2222
    profile.environment = {"CUDA_VISIBLE_DEVICES": "0"}
    restored = ServerProfile.from_dict(profile.to_dict())
    assert restored == profile
    assert restored.id == profile.id


def test_from_dict_ignores_unknown_and_missing_fields():
    profile = ServerProfile.from_dict({"id": "abc", "name": "x", "unknown": 1})
    assert profile.id == "abc"
    assert profile.name == "x"
    assert profile.kind is ServerKind.local


def test_ssh_command_line():
    profile = ServerProfile.preset(ServerKind.ssh)
    profile.ssh_host = "gpu-box"
    profile.ssh_user = "vyls"
    profile.ssh_port = 2222
    profile.directory = "/srv/comfy"
    profile.command = "python3"
    profile.arguments = "main.py --port 8188"
    assert profile.ssh_target() == "vyls@gpu-box"
    assert profile.ssh_command_line() == [
        "ssh",
        "-p",
        "2222",
        "-o",
        "BatchMode=yes",
        "vyls@gpu-box",
        "cd /srv/comfy && python3 main.py --port 8188",
    ]


def test_command_line():
    profile = ServerProfile.preset(ServerKind.local)
    profile.command = "systemctl --user start comfy"
    assert profile.command_line() == ["systemctl", "--user", "start", "comfy"]


def test_environment_roundtrip():
    env = {"A": "1", "B": "two words"}
    text = format_environment(env)
    assert parse_environment(text) == env
    assert parse_environment("") == {}
    assert parse_environment("# comment\nA=1\ninvalid\n B = 2 ") == {"A": "1", "B": "2"}


def test_migrate_legacy_external_to_profile(monkeypatch):
    from ai_diffusion.backend import server_runner
    from ai_diffusion.settings import ServerMode

    s = Settings()
    s.server_mode = ServerMode.external
    s.server_url = "10.0.0.5:8188"
    monkeypatch.setattr(s, "save", lambda: None)  # type: ignore[method-assign]
    monkeypatch.setattr(server_runner, "settings", s)

    manager = server_runner.ServerManager()

    assert s.server_mode is ServerMode.servers
    profile = manager.active
    assert profile is not None
    assert profile.url == "10.0.0.5:8188"
    assert not profile.has_start_command
    assert s.active_server == profile.id


def test_migrate_keeps_existing_profiles(monkeypatch):
    from ai_diffusion.backend import server_runner
    from ai_diffusion.settings import ServerMode

    existing = ServerProfile.preset(ServerKind.local, "/srv/comfy")
    existing.name = "Keep"
    s = Settings()
    s.server_profiles = [existing.to_dict()]
    s.active_server = existing.id
    s.server_mode = ServerMode.servers
    monkeypatch.setattr(s, "save", lambda: None)  # type: ignore[method-assign]
    monkeypatch.setattr(server_runner, "settings", s)

    manager = server_runner.ServerManager()

    assert [p.name for p in manager.profiles] == ["Keep"]
    assert manager.active is not None
    assert manager.active.id == existing.id
    assert s.server_mode is ServerMode.servers
