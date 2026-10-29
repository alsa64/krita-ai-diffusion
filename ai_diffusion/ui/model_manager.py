from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtGui import QColor
from PyQt6.QtNetwork import QNetworkAccessManager
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop, util
from ..backend import network, resources
from ..backend.capabilities import capability_labels, feature_capabilities
from ..backend.client import ClientModels
from ..backend.model_metadata import ModelMetadata, metadata_store, model_types
from ..backend.resources import Arch, ControlMode, ResourceId, ResourceKind
from ..localization import translate as _
from ..settings import ServerMode, settings
from ..util import client_logger as log

kinds = [
    ResourceKind.checkpoint,
    ResourceKind.lora,
    ResourceKind.vae,
    ResourceKind.text_encoder,
    ResourceKind.controlnet,
    ResourceKind.inpaint,
    ResourceKind.upscaler,
    ResourceKind.model_patch,
    ResourceKind.ip_adapter,
    ResourceKind.clip_vision,
]


def _parse_id(id_string: str) -> ResourceId | None:
    try:
        return ResourceId.parse(id_string)
    except Exception:
        log.debug(f"Invalid resource id: {id_string}")
        return None


def server_models(models: ClientModels, kind: ResourceKind) -> list[str]:
    if kind is ResourceKind.checkpoint:
        return list(models.checkpoints.keys())
    if kind is ResourceKind.lora:
        return list(models.loras)
    if kind is ResourceKind.vae:
        return list(models.vae)
    if kind is ResourceKind.upscaler:
        return list(models.upscalers) + list(models.seedvr2_dit)
    names = set()
    for id_string, filename in models.resources.items():
        if not filename:
            continue
        id = _parse_id(id_string)
        if id is not None and id.kind is kind:
            names.add(filename)
    return sorted(names)


@dataclass
class Row:
    kind: ResourceKind
    filename: str
    name: str
    arch: str = ""
    installed: bool = False
    file: resources.ModelFile | None = None
    metadata: ModelMetadata | None = None


def collect_rows(models: ClientModels | None, kind: ResourceKind) -> list[Row]:
    installed = server_models(models, kind) if models is not None else []
    rows: dict[str, Row] = {}
    for filename in installed:
        metadata = metadata_store.get(kind, filename)
        arch = model_arch(models, kind, filename) if models is not None else None
        rows[filename] = Row(
            kind,
            filename,
            metadata.name or filename,
            metadata.arch or (arch.name if arch else ""),
            True,
            None,
            metadata,
        )
    for resource, file in resources.downloads(kind):
        filename = file.name
        if filename in rows:
            rows[filename].file = file
            continue
        metadata = metadata_store.get(kind, filename)
        rows[filename] = Row(
            kind,
            filename,
            metadata.name or resource.name,
            metadata.arch or resource.arch.name,
            False,
            file,
            metadata,
        )
    return sorted(rows.values(), key=lambda r: (not r.installed, r.name.lower()))


def model_arch(models: ClientModels, kind: ResourceKind, filename: str) -> Arch | None:
    if kind is ResourceKind.checkpoint and filename in models.checkpoints:
        return models.checkpoints[filename].arch
    for id_string, name in models.resources.items():
        if name != filename:
            continue
        id = _parse_id(id_string)
        if id is not None and id.kind is kind and id.arch is not Arch.all:
            return id.arch
    return None


class MetadataDialog(QDialog):
    def __init__(self, row: Row, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(_("Model Metadata"))
        self.setMinimumWidth(420)
        metadata = row.metadata or ModelMetadata()

        layout = QFormLayout(self)
        self._name = QLineEdit(metadata.name or row.name)
        layout.addRow(_("Name"), self._name)

        self._type = QComboBox()
        self._type.addItem(_("(not set)"), "")
        for value in model_types:
            self._type.addItem(value, value)
        self._type.setCurrentIndex(max(0, self._type.findData(metadata.type)))
        layout.addRow(_("Type"), self._type)

        self._arch = QComboBox()
        self._arch.addItem(_("(not set)"), "")
        for arch in Arch.list():
            self._arch.addItem(arch.value, arch.name)
        current = metadata.arch or row.arch
        self._arch.setCurrentIndex(max(0, self._arch.findData(current)))
        layout.addRow(_("Architecture"), self._arch)

        self._role = QComboBox()
        self._role.addItem(_("(not set)"), "")
        for mode in ControlMode:
            self._role.addItem(mode.text, mode.name)
        self._role.setCurrentIndex(max(0, self._role.findData(metadata.role)))
        layout.addRow(_("Role"), self._role)

        self._base = QLineEdit(metadata.base_model)
        layout.addRow(_("Base model"), self._base)

        self._triggers = QLineEdit(", ".join(metadata.triggers))
        self._triggers.setPlaceholderText(_("comma separated trigger words"))
        layout.addRow(_("Triggers"), self._triggers)

        self._description = QPlainTextEdit(metadata.description)
        self._description.setFixedHeight(80)
        layout.addRow(_("Description"), self._description)

        self._capabilities = list(metadata.capabilities)
        for capability in feature_capabilities:
            check = QCheckBox(capability_labels.get(capability, capability))
            check.setChecked(capability in self._capabilities)
            check.toggled.connect(self._toggle_capability(capability))
            layout.addRow("", check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _toggle_capability(self, capability: str):
        def toggle(checked: bool):
            if checked and capability not in self._capabilities:
                self._capabilities.append(capability)
            elif not checked and capability in self._capabilities:
                self._capabilities.remove(capability)

        return toggle

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self._name.text().strip(),
            arch=self._arch.currentData() or "",
            type=self._type.currentData() or "",
            description=self._description.toPlainText().strip(),
            triggers=[t.strip() for t in self._triggers.text().split(",") if t.strip()],
            capabilities=list(self._capabilities),
            role=self._role.currentData() or "",
            base_model=self._base.text().strip(),
        )


class ModelKindTab(QWidget):
    def __init__(self, kind: ResourceKind, manager: ModelManagerWidget):
        super().__init__()
        self._kind = kind
        self._manager = manager
        self._rows: list[Row] = []
        self._network = QNetworkAccessManager(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)

        self._filter = QComboBox()
        self._filter.addItem(_("All architectures"), "")
        for arch in Arch.list():
            self._filter.addItem(arch.value, arch.name)
        self._filter.currentIndexChanged.connect(self.refresh)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            _("Name"),
            _("Architecture"),
            _("Status"),
            _("Type"),
            _("Triggers"),
        ])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        assert header is not None
        header.setStretchLastSection(True)
        vheader = self._table.verticalHeader()
        assert vheader is not None
        vheader.setVisible(False)
        self._table.itemSelectionChanged.connect(self._update_buttons)
        self._table.doubleClicked.connect(self._edit_metadata)

        buttons = QHBoxLayout()
        self._download = QPushButton(_("Download"))
        self._download.clicked.connect(self._download_selected)
        self._download_all = QPushButton(_("Download All Missing"))
        self._download_all.clicked.connect(self._download_missing)
        self._edit = QPushButton(_("Edit Metadata"))
        self._edit.clicked.connect(self._edit_metadata)

        buttons.addWidget(self._filter)
        buttons.addStretch()
        buttons.addWidget(self._download)
        buttons.addWidget(self._download_all)
        buttons.addWidget(self._edit)

        layout.addLayout(buttons)
        layout.addWidget(self._table)
        self._status = QLabel()
        layout.addWidget(self._status)

    def refresh(self):
        models = self._manager.models
        arch_filter = self._filter.currentData()
        self._rows = [
            r for r in collect_rows(models, self._kind) if not arch_filter or r.arch == arch_filter
        ]
        self._table.setRowCount(len(self._rows))
        dim = QColor(140, 140, 140)
        for i, row in enumerate(self._rows):
            name = QTableWidgetItem(row.name)
            arch = QTableWidgetItem(row.arch or "-")
            if row.installed:
                status_text = _("Downloaded")
            elif row.file is not None:
                status_text = _("Not downloaded")
            else:
                status_text = _("Not found")
            status = QTableWidgetItem(status_text)
            type_item = QTableWidgetItem(row.metadata.type if row.metadata else "")
            triggers = QTableWidgetItem(", ".join(row.metadata.triggers) if row.metadata else "")
            for column, item in enumerate((name, arch, status, type_item, triggers)):
                if not row.installed:
                    item.setForeground(dim)
                self._table.setItem(i, column, item)
        self._update_buttons()

    def _selected_row(self) -> Row | None:
        index = self._table.currentRow()
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def _update_buttons(self):
        row = self._selected_row()
        self._edit.setEnabled(row is not None)
        self._download.setEnabled(row is not None and not row.installed and row.file is not None)

    def _edit_metadata(self):
        row = self._selected_row()
        if row is None:
            return
        dialog = MetadataDialog(row, self)
        if dialog.exec():
            metadata_store.set(self._kind, row.filename, dialog.metadata())
            self._manager.metadata_changed()
            self.refresh()

    def _download_selected(self):
        row = self._selected_row()
        if row is None or row.file is None:
            return
        self._download_files([row.file])

    def _download_missing(self):
        files = [row.file for row in self._rows if not row.installed and row.file is not None]
        if files:
            self._download_files(files)

    def _download_files(self, files: list[resources.ModelFile]):
        base_dir = self._manager.download_dir(self._kind)
        self._status.setText(_("Downloading to {}").format(base_dir))
        self._download.setEnabled(False)
        self._download_all.setEnabled(False)

        async def download():
            try:
                for file in files:
                    target = base_dir / Path(file.path).name
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self._status.setText(_("Downloading {}").format(target.name))
                    async for progress in network.download(self._network, file.url, target):
                        self._status.setText(
                            _("Downloading {} {:.0f}%").format(target.name, progress.value * 100)
                        )
                self._status.setText(_("Download finished"))
            except Exception as e:
                log.error(f"Model download failed: {e!s}")
                self._status.setText(str(e))
            finally:
                self._download.setEnabled(True)
                self._download_all.setEnabled(True)
                self._manager.refresh()

        eventloop.run(download())


class ModelManagerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._pages: list[ModelKindTab] = []
        for kind in kinds:
            page = ModelKindTab(kind, self)
            self._pages.append(page)
            self._tabs.addTab(page, _(kind.value))
        layout.addWidget(self._tabs)

        buttons = QHBoxLayout()
        self._path_label = QLabel()
        self._refresh = QPushButton(_("Refresh"))
        self._refresh.clicked.connect(self.refresh)
        buttons.addWidget(self._path_label)
        buttons.addStretch()
        buttons.addWidget(self._refresh)
        layout.addLayout(buttons)

    @property
    def models(self) -> ClientModels | None:
        from ..model.root import root

        if client := root.connection.client_if_connected:
            return client.models
        return None

    def metadata_changed(self):
        from ..model.root import root

        root.sync_model_metadata()
        for page in self._pages:
            page.refresh()

    def download_dir(self, kind: ResourceKind) -> Path:
        from ..model.root import root

        folder = resources.server_folder(kind)
        profile = None
        if settings.server_mode is ServerMode.servers and root.servers is not None:
            profile = root.servers.active
        if profile is not None:
            if profile.models_path:
                return Path(profile.models_path) / folder
            if profile.is_managed:
                return Path(profile.directory or settings.server_path) / "models" / folder
        if settings.server_mode in (ServerMode.managed, ServerMode.external):
            return Path(settings.server_path) / "models" / folder
        return util.user_data_dir / "downloads" / folder

    def refresh(self):
        for page in self._pages:
            page.refresh()
        self._path_label.setText(
            _("Downloads go to {}").format(self.download_dir(ResourceKind.checkpoint).parent)
        )

    def showEvent(self, a0):
        super().showEvent(a0)
        self.refresh()
