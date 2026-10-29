from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend.comfy_client import ComfyClient
from ..backend.server_profile import (
    ServerKind,
    ServerProfile,
    format_environment,
    parse_environment,
)
from ..backend.server_runner import RunnerState, ServerManager, ServerRunner
from ..localization import translate as _
from ..settings import settings
from ..util import client_logger as log

_status_text = {
    RunnerState.stopped: _("Stopped"),
    RunnerState.starting: _("Starting"),
    RunnerState.running: _("Running"),
    RunnerState.error: _("Error"),
}


class ServersWidget(QWidget):
    """List and edit server profiles. The active profile is used for generation."""

    state_changed = pyqtSignal()

    def __init__(self, servers: ServerManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._servers = servers
        self._current: ServerProfile | None = None
        self._loading = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        left = QVBoxLayout()
        self._list = QListWidget(self)
        self._list.currentRowChanged.connect(self._select)
        left.addWidget(self._list, 1)

        buttons = QHBoxLayout()
        self._add_button = QPushButton(_("Add"))
        self._add_button.setMenu(self._create_add_menu())
        self._remove_button = QPushButton(_("Remove"))
        self._remove_button.clicked.connect(self._remove)
        buttons.addWidget(self._add_button)
        buttons.addWidget(self._remove_button)
        buttons.addStretch()
        left.addLayout(buttons)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        form = QFormLayout()
        self._name = QLineEdit(self)
        self._kind = QComboBox(self)
        for kind in ServerKind:
            self._kind.addItem(kind.label, kind)
        self._kind.currentIndexChanged.connect(self._update_kind_fields)
        self._url = QLineEdit(self)
        self._directory = QLineEdit(self)
        self._directory.setPlaceholderText(_("Working directory for the start command"))
        self._command = QLineEdit(self)
        self._command.setPlaceholderText(_("Leave empty to only connect, never start"))
        self._arguments = QLineEdit(self)
        self._ssh_host = QLineEdit(self)
        self._ssh_user = QLineEdit(self)
        self._ssh_port = QSpinBox(self)
        self._ssh_port.setRange(1, 65535)
        self._models_path = QLineEdit(self)
        self._models_path.setPlaceholderText(_("Server-side models directory"))
        self._timeout = QSpinBox(self)
        self._timeout.setRange(5, 3600)
        self._timeout.setSuffix(" s")
        self._environment = QPlainTextEdit(self)
        self._environment.setPlaceholderText("KEY=value")
        self._environment.setFixedHeight(60)

        form.addRow(_("Name"), self._name)
        form.addRow(_("Type"), self._kind)
        form.addRow(_("URL"), self._url)
        form.addRow(_("Directory"), self._directory)
        form.addRow(_("Command"), self._command)
        form.addRow(_("Arguments"), self._arguments)
        form.addRow(_("SSH Host"), self._ssh_host)
        form.addRow(_("SSH User"), self._ssh_user)
        form.addRow(_("SSH Port"), self._ssh_port)
        form.addRow(_("Models"), self._models_path)
        form.addRow(_("Start Timeout"), self._timeout)
        form.addRow(_("Environment"), self._environment)
        right.addLayout(form)

        actions = QHBoxLayout()
        self._connect_button = QPushButton(_("Connect"))
        self._connect_button.clicked.connect(self._connect)
        self._activate_button = QPushButton(_("Make Active"))
        self._activate_button.clicked.connect(self._activate)
        self._start_button = QPushButton(_("Start"))
        self._start_button.clicked.connect(self._start)
        self._stop_button = QPushButton(_("Stop"))
        self._stop_button.clicked.connect(self._stop)
        actions.addWidget(self._connect_button)
        actions.addWidget(self._activate_button)
        actions.addWidget(self._start_button)
        actions.addWidget(self._stop_button)
        actions.addStretch()
        right.addLayout(actions)

        self._status = QLabel(self)
        right.addWidget(self._status)
        right.addStretch()
        layout.addLayout(right, 2)

        for line_edit in (
            self._name,
            self._url,
            self._directory,
            self._command,
            self._arguments,
            self._ssh_host,
            self._ssh_user,
            self._models_path,
        ):
            line_edit.editingFinished.connect(self._apply)
        self._kind.currentIndexChanged.connect(self._apply)
        self._environment.textChanged.connect(self._apply)
        self._timeout.valueChanged.connect(self._apply)

        self._servers.profiles_changed.connect(self.reload)
        self._servers.state_changed.connect(self._update_status)
        self.reload()

    def _create_add_menu(self) -> QMenu:
        menu = QMenu(self)
        for kind in ServerKind:
            action = menu.addAction(kind.label + " " + _("Server"))
            if action is not None:
                action.triggered.connect(lambda _, k=kind: self._add(k))
        return menu

    def reload(self):
        self._loading = True
        active = self._servers.active
        self._list.clear()
        for profile in self._servers.profiles:
            name = profile.display_name
            if active is not None and profile.id == active.id:
                name = "● " + name
            self._list.addItem(name)
        index = next(
            (i for i, p in enumerate(self._servers.profiles) if p is active),
            -1,
        )
        self._list.setCurrentRow(index)
        self._loading = False
        self._select(index)

    def _select(self, row: int):
        if self._loading:
            return
        profiles = self._servers.profiles
        profile = profiles[row] if 0 <= row < len(profiles) else None
        self._current = profile
        enabled = profile is not None
        for widget in (
            self._name,
            self._kind,
            self._url,
            self._directory,
            self._command,
            self._arguments,
            self._ssh_host,
            self._ssh_user,
            self._ssh_port,
            self._models_path,
            self._timeout,
            self._environment,
            self._connect_button,
            self._activate_button,
            self._start_button,
            self._stop_button,
        ):
            widget.setEnabled(enabled)
        self._load(profile)
        self._update_status()

    def _load(self, profile: ServerProfile | None):
        self._loading = True
        if profile is not None:
            self._name.setText(profile.name)
            self._kind.setCurrentIndex(list(ServerKind).index(profile.kind))
            self._url.setText(profile.url)
            self._directory.setText(profile.directory)
            self._command.setText(profile.command)
            self._arguments.setText(profile.arguments)
            self._ssh_host.setText(profile.ssh_host)
            self._ssh_user.setText(profile.ssh_user)
            self._ssh_port.setValue(profile.ssh_port)
            self._models_path.setText(profile.models_path)
            self._timeout.setValue(profile.start_timeout)
            self._environment.setPlainText(format_environment(profile.environment))
        self._update_kind_fields()
        self._loading = False

    def _update_kind_fields(self):
        if self._current is None:
            return
        kind = self._kind.currentData()
        is_ssh = kind is ServerKind.ssh
        self._ssh_host.setEnabled(is_ssh)
        self._ssh_user.setEnabled(is_ssh)
        self._ssh_port.setEnabled(is_ssh)

    def _apply(self):
        if self._loading or self._current is None:
            return
        profile = self._current
        profile.name = self._name.text()
        profile.kind = self._kind.currentData()
        profile.url = self._url.text()
        profile.directory = self._directory.text()
        profile.command = self._command.text()
        profile.arguments = self._arguments.text()
        profile.ssh_host = self._ssh_host.text()
        profile.ssh_user = self._ssh_user.text()
        profile.ssh_port = self._ssh_port.value()
        profile.models_path = self._models_path.text()
        profile.start_timeout = self._timeout.value()
        profile.environment = parse_environment(self._environment.toPlainText())
        self._servers.update(profile, save=False)
        row = self._list.currentRow()
        item = self._list.item(row)
        if item is not None:
            item.setText(profile.display_name)

    def _add(self, kind: ServerKind):
        profile = ServerProfile.preset(kind, settings.server_path)
        if kind is ServerKind.local:
            profile.directory = profile.directory or settings.server_path
        self._servers.add(profile)
        self._servers.set_active(profile.id)
        self.reload()
        self._list.setCurrentRow(len(self._servers.profiles) - 1)

    def _remove(self):
        if self._current is not None:
            self._servers.remove(self._current.id)
            self.reload()

    def _activate(self):
        if self._current is not None:
            self._servers.set_active(self._current.id)
            self.reload()
            self.state_changed.emit()

    def _runner(self) -> ServerRunner | None:
        return self._servers.runner(self._current) if self._current is not None else None

    def _connect(self):
        from ..model.root import root

        runner = self._runner()
        if runner is None:
            return
        self._apply()
        self._servers.save()

        async def connect():
            try:
                url = await runner.ensure_running()
            except Exception as e:
                self._status.setText(str(e))
                return
            root.connection.connect(ComfyClient(url))
            self._update_status()

        eventloop.run(connect())

    def _start(self):
        runner = self._runner()
        if runner is None:
            return
        self._apply()
        self._servers.save()

        async def start():
            try:
                await runner.start()
            except Exception as e:
                log.error(f"Failed to start server: {e}")
            self._update_status()

        eventloop.run(start())

    def _stop(self):
        runner = self._runner()
        if runner is not None:
            eventloop.run(runner.stop())
            self._update_status()

    def _update_status(self):
        runner = self._runner()
        if runner is None:
            self._status.setText("")
            return
        text = _status_text[runner.state]
        if runner.error:
            text += f" - {runner.error}"
        active = self._servers.active
        if self._current is not None and active is not None and self._current.id == active.id:
            text += " " + _("(active)")
        self._status.setText(text)
        self._status.setStyleSheet("color: #c0392b" if runner.state is RunnerState.error else "")
