from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .settings import Setting
from .util import client_logger as log
from .util import encode_json, read_json_with_comments, user_data_dir

SectionPath = str | tuple[str, ...]


class DefaultsStore:
    default_path = user_data_dir / "defaults.json"

    def __init__(self, path: Path | None = None):
        self._path = path or self.default_path

    @property
    def path(self):
        return self._path

    def read_section(self, path: SectionPath, schema: dict[str, Setting]):
        data = self._load_all()
        section = self._resolve_section(data, path)
        return {
            name: self._validate_value(setting, section.get(name, setting.default), path, name)
            for name, setting in schema.items()
        }

    def write_section(self, path: SectionPath, values: dict[str, Any], schema: dict[str, Setting]):
        data = self._load_all()
        section = self._resolve_section(data, path, create=True)
        for name, setting in schema.items():
            section[name] = self._validate_value(
                setting, values.get(name, setting.default), path, name
            )
        self._save_all(data)

    def clear_section(self, path: SectionPath):
        data = self._load_all()
        parent_path, leaf = self._split_path(path)
        section = self._resolve_section(data, parent_path)
        if isinstance(section, dict):
            section.pop(leaf, None)
            self._save_all(data)

    def _load_all(self):
        if not self._path.exists():
            return {}
        try:
            data = read_json_with_comments(self._path)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            log.warning(f"Failed to load defaults from {self._path}: {e}")
            return {}

    def _save_all(self, data: dict[str, Any]):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as file:
            file.write(json.dumps(data, default=encode_json, indent=4))

    def _resolve_section(self, data: dict[str, Any], path: SectionPath, create=False):
        node: dict[str, Any] = data
        for part in self._path_parts(path):
            child = node.get(part)
            if not isinstance(child, dict):
                if not create:
                    return {}
                child = {}
                node[part] = child
            node = child
        return node

    def _split_path(self, path: SectionPath):
        parts = self._path_parts(path)
        if len(parts) == 1:
            return (), parts[0]
        return tuple(parts[:-1]), parts[-1]

    def _path_parts(self, path: SectionPath):
        if isinstance(path, str):
            return (path,)
        return path

    def _validate_value(self, setting: Setting, value: Any, path: SectionPath, name: str):
        numtype = (int, float)
        if isinstance(setting.default, Enum):
            if isinstance(value, str):
                try:
                    return type(setting.default)[value]
                except KeyError:
                    pass
            if isinstance(value, type(setting.default)):
                return value
        elif (setting.items is not None and value in setting.items) or (
            (isinstance(setting.default, str) and isinstance(value, str))
            or (isinstance(setting.default, bool) and isinstance(value, bool))
            or (isinstance(setting.default, list) and isinstance(value, list))
            or (isinstance(setting.default, dict) and isinstance(value, dict))
            or (isinstance(setting.default, numtype) and isinstance(value, numtype))
        ):
            return value

        log.warning(f"Invalid default value for {self._path_parts(path)}.{name}: {value}")
        return setting.default


defaults = DefaultsStore()
