from __future__ import annotations

import shlex
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..localization import translate as _


class ServerKind(Enum):
    local = "Local"
    ssh = "SSH"

    @property
    def label(self):
        return _(self.value)


DEFAULT_URL = "127.0.0.1:8188"


@dataclass
class ServerProfile:
    """A ComfyUI server the plugin can connect to.

    Empty `command` means the server is not started by the plugin. For local profiles it
    also means the server is a plugin managed installation at `directory`.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    kind: ServerKind = ServerKind.local
    url: str = DEFAULT_URL
    directory: str = ""
    command: str = ""
    arguments: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    models_path: str = ""
    auto_start: bool = True
    start_timeout: int = 300

    @property
    def display_name(self):
        return self.name or self.url

    @property
    def has_start_command(self):
        return bool(self.command.strip())

    @property
    def is_managed(self):
        return self.kind is ServerKind.local and not self.has_start_command

    def command_line(self) -> list[str]:
        return [*shlex.split(self.command), *shlex.split(self.arguments)]

    def ssh_target(self):
        return f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host

    def ssh_command_line(self) -> list[str]:
        command = " ".join(shlex.quote(a) for a in self.command_line())
        if self.directory:
            command = f"cd {shlex.quote(self.directory)} && {command}"
        return [
            "ssh",
            "-p",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            self.ssh_target(),
            command,
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.name
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ServerProfile:
        data = dict(data)
        kind = data.get("kind", ServerKind.local.name)
        data["kind"] = ServerKind[kind] if kind in ServerKind.__members__ else ServerKind.local
        known = {f.name for f in ServerProfile.__dataclass_fields__.values()}
        return ServerProfile(**{k: v for k, v in data.items() if k in known})

    @staticmethod
    def preset(kind: ServerKind, directory: str = "") -> ServerProfile:
        profile = ServerProfile(kind=kind)
        if kind is ServerKind.local:
            profile.name = _("Local Server")
            profile.directory = directory
        elif kind is ServerKind.ssh:
            profile.name = _("SSH Server")
            profile.command = "python3"
            profile.arguments = "main.py --listen 127.0.0.1 --port 8188"
        return profile


def parse_environment(text: str) -> dict[str, str]:
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def format_environment(env: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in env.items())
