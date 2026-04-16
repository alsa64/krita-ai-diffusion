from __future__ import annotations

from typing import cast

from krita import Krita
from PyQt5.QtCore import QMetaObject, QSize, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QCursor,
    QDesktopServices,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QPainter,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, eventloop, util
from ..backend import resources
from ..backend.client import Client, MissingResources, User
from ..backend.cloud_client import CloudClient
from ..backend.resources import Arch, ControlMode, ResourceId
from ..backend.server import Server, ServerState
from ..model.custom_workflow import CustomGenerationMode
from ..defaults import defaults
from ..localization import Localization
from ..localization import translate as _
from ..model.connection import ConnectionState, apply_performance_preset
from ..model.model import AnimationTargetLayerDefault, QueueMode, Workspace
from ..model.properties import Binding
from ..model.root import collect_diagnostics, root
from ..model.updates import UpdateState
from ..persistence import (
    animation_defaults_schema,
    custom_defaults_schema,
    document_defaults_schema,
    generation_defaults_schema,
    live_defaults_schema,
    load_document_defaults,
    load_workspace_defaults,
    save_document_defaults,
    save_workspace_defaults,
    upscaling_defaults_schema,
)

from ..settings import ImageFileFormat, PerformancePreset, ServerMode, Settings, settings
from ..style import Style, Styles, style_defaults, style_defaults_schema
from .server import ServerWidget
from .settings_widgets import (
    ComboBoxSetting,
    DoubleSpinBoxSetting,
    FileListSetting,
    SettingsTab,
    SliderSetting,
    SpinBoxSetting,
    SwitchSetting,
    TextSetting,
)
from .style import StylePresets, StyleSettingsEditor
from .theme import add_header, green, grey, logo, prompt_max_line_count, red, yellow


class InitialSetupWidget(QWidget):
    finished = pyqtSignal(ServerMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 0)
        self.setLayout(layout)

        label_title = QLabel("<b>" + _("Welcome to Image Generation in Krita") + "</b>", self)
        label_sub = QLabel(
            _(
                "To create images, the plugin needs to connect to a backend server. Please choose one of the options below (you can always switch later)."
            ),
            self,
        )
        label_sub.setWordWrap(True)
        layout.addWidget(label_title)
        layout.addWidget(label_sub)
        layout.addSpacing(20)

        def add_option(title: str, desc_text: str, button_text: str, mode: ServerMode):
            header = QLabel("<b>" + title + "</b>", self)
            desc = QLabel(desc_text, self)
            desc.setMaximumWidth(600)
            desc.setWordWrap(True)
            button = QPushButton(button_text, self)
            button.setMinimumHeight(int(1.3 * button.sizeHint().height()))
            button.setMaximumWidth(300)
            button.clicked.connect(self._choose(mode))
            layout.addWidget(header)
            layout.addWidget(desc)
            layout.addWidget(button)
            layout.addSpacing(16)

        add_option(
            _("Option {number}", number=1) + ": " + _("Online Service"),
            _(
                "Generate images via {link}. Create an account to get started. No local installation or powerful hardware needed.",
                link=f"<a href='{settings.cloud_web_url}'>interstice.cloud</a>",
            ),
            _("Login or Sign up"),
            ServerMode.cloud,
        )
        add_option(
            _("Option {number}", number=2) + ": " + _("Local Managed Server"),
            _(
                "Install and run a local ComfyUI server on your machine. Installation and updates are performed automatically by the plugin. Requires a compatible GPU (NVIDIA with at least 6GB VRAM recommended)."
            ),
            _("Start Installation"),
            ServerMode.managed,
        )
        add_option(
            _("Option {number}", number=3) + ": " + _("Custom ComfyUI"),
            _(
                "Connect to an existing installation of ComfyUI. It can be on the same machine, or a remote machine over the network. You are responsible to setup ComfyUI and install required custom nodes and models."
            )
            + "<br><a href='https://docs.interstice.cloud/comfyui-setup'>ComfyUI Setup Guide</a>",
            _("Connect via URL"),
            ServerMode.external,
        )
        layout.addStretch()

    def _choose(self, mode: ServerMode):
        def handler():
            settings.server_mode = mode
            settings.save()
            self.finished.emit(mode)

        return handler


class UserWidget(QFrame):
    _user: User | None = None
    _connections: list[QMetaObject.Connection | Binding]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections = []

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setVisible(False)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._user_name = QLabel("", self)
        self._user_name.setStyleSheet("font-weight:bold")
        user_name_layout = QHBoxLayout()
        user_name_layout.addWidget(QLabel(_("Account:"), self), 0)
        user_name_layout.addWidget(self._user_name, 1)
        layout.addLayout(user_name_layout)

        self._images_generated = QLabel("", self)
        image_count_layout = QHBoxLayout()
        image_count_layout.addWidget(QLabel(_("Total generated:"), self), 0)
        image_count_layout.addWidget(self._images_generated, 1)
        layout.addLayout(image_count_layout)

        self._tokens_remaining = QLabel("", self)
        self._tokens_remaining.setStyleSheet("font-weight:bold")
        image_remaining_layout = QHBoxLayout()
        image_remaining_layout.addWidget(QLabel(_("Image tokens remaining:"), self), 0)
        image_remaining_layout.addWidget(self._tokens_remaining, 1)
        layout.addLayout(image_remaining_layout)
        layout.addSpacing(8)

        buy_layout = QHBoxLayout()
        layout.addLayout(buy_layout)

        self._buy_tokens5000_button = QPushButton(_("Buy Tokens") + " (5000)", self)
        self._buy_tokens5000_button.clicked.connect(lambda: self._buy_tokens("5000"))
        buy_layout.addWidget(self._buy_tokens5000_button, 1)

        self._buy_tokens15000_button = QPushButton(_("Buy Tokens") + " (15000)", self)
        self._buy_tokens15000_button.clicked.connect(lambda: self._buy_tokens("15000"))
        buy_layout.addWidget(self._buy_tokens15000_button, 1)

        self._account_button = QPushButton(_("View Account"), self)
        self._account_button.setMinimumWidth(200)
        self._account_button.clicked.connect(self._view_account)
        layout.addWidget(self._account_button)

        self._logout_button = QPushButton(_("Sign out"), self)
        self._logout_button.setMinimumWidth(200)
        self._logout_button.clicked.connect(self._logout)
        layout.addWidget(self._logout_button)

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user: User | None):
        if self._user is not user:
            Binding.disconnect_all(self._connections)
            self.setVisible(user is not None)

            self._user = user
            if user is not None:
                self._user_name.setText(user.name)
                self._connections = [
                    user.images_generated_changed.connect(self._update_counts),
                    user.credits_changed.connect(self._update_counts),
                ]
                self._update_counts()

    def _update_counts(self):
        user = util.ensure(self.user)
        self._images_generated.setText(str(user.images_generated))
        self._tokens_remaining.setText(str(user.credits))

    def _view_account(self):
        QDesktopServices.openUrl(QUrl(f"{settings.cloud_web_url}/user"))

    def _buy_tokens(self, amount: str):
        QDesktopServices.openUrl(QUrl(f"{settings.cloud_web_url}/checkout/tokens{amount}"))

    def _logout(self):
        eventloop.run(self._disconnect_and_logout())

    async def _disconnect_and_logout(self):
        await root.connection.disconnect()
        settings.access_token = ""
        settings.save()


class CloudWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 12, 4, 4)
        self.setLayout(layout)

        service_url = settings.cloud_web_url
        service_url_text = (
            service_url.removeprefix("https://").removeprefix("www.").removesuffix("/")
        )
        header = QLabel(f"<b>{service_url_text}</b>", self)
        service_label = QLabel(f"<a href='{service_url}'>Visit Website</a>", self)
        service_label.setOpenExternalLinks(True)
        layout.addWidget(header)
        layout.addWidget(service_label)

        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._connection_status)

        self.connect_button = QPushButton(_("Login"), self)
        self.connect_button.setMinimumWidth(200)
        self.connect_button.setMinimumHeight(int(1.3 * self.connect_button.sizeHint().height()))
        self.connect_button.clicked.connect(self._connect)

        self._sign_out_button = QPushButton(_("Sign out"), self)
        self._sign_out_button.setVisible(False)
        self._sign_out_button.setMinimumWidth(200)
        self._sign_out_button.clicked.connect(self._sign_out)

        self._user_widget = UserWidget(self)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.connect_button)
        buttons_layout.addWidget(self._sign_out_button)

        connect_layout = QHBoxLayout()
        connect_layout.addLayout(buttons_layout)
        connect_layout.addWidget(self._user_widget)
        connect_layout.addStretch()
        layout.addLayout(connect_layout)

        layout.addStretch()

    def update_connection_state(self, state: ConnectionState):
        is_connected = state == ConnectionState.connected
        self.connect_button.setVisible(not is_connected)
        self._sign_out_button.setVisible(False)
        self._user_widget.user = root.connection.user

        if state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            self.connect_button.setText(_("Sign in"))
            self.connect_button.setEnabled(True)
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif state is ConnectionState.auth_pending:
            self.connect_button.setText(_("Sign in"))
            self.connect_button.setEnabled(False)
            self._connection_status.setText(_("Waiting for sign-in to complete..."))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connection_status.setVisible(True)
        elif state is ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
            self._user_widget.user = root.connection.user
        else:
            can_connect = state in [ConnectionState.disconnected, ConnectionState.error]
            self.connect_button.setEnabled(can_connect)
            self.connect_button.setText(_("Connect") if can_connect else _("Connected"))
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")

        if state in [ConnectionState.error, ConnectionState.auth_error]:
            error = root.connection.error or "Unknown error"
            self._connection_status.setText(f"<b>Error</b>: {error.removeprefix('Error: ')}")
            self._connection_status.setStyleSheet(f"color: {red}; font-weight:bold")
            self._connection_status.setVisible(True)
            if settings.access_token:
                self._sign_out_button.setVisible(True)

    def _connect(self):
        connection = root.connection
        if connection.state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            connection.sign_in()
        else:
            if client := connection.create_client(settings):
                connection.connect(client)

    def _sign_out(self):
        settings.access_token = ""
        settings.save()


_server_mode_text = {
    ServerMode.undefined: "Undefined",
    ServerMode.cloud: _("Online Service"),
    ServerMode.managed: _("Local Managed Server"),
    ServerMode.external: _("Custom Server"),
}
_server_mode_status = {
    "signed_out": (_("Signed out"), grey),
    "not_installed": (_("Not installed"), grey),
    "not_running": (_("Not running"), grey),
    "not_connected": (_("Not connected"), grey),
    "connecting": (_("Connecting"), yellow),
    "connected": (_("Connected"), green),
    "error": (_("Error"), red),
}


class ServerModeButton(QPushButton):
    toggled = pyqtSignal(ServerMode)

    def __init__(self, mode: ServerMode, status: str, parent=None):
        self._text = _server_mode_text[mode]
        super().__init__(self._text, parent)
        self.mode = mode
        self._status = status
        self._is_checked = False

        font = QFontMetrics(self.font())
        self._text_width = font.horizontalAdvance(self._text)
        max_width = max(font.horizontalAdvance(s[0]) for s in _server_mode_status.values())
        self.setMinimumWidth(self._text_width + max_width + 32)
        self.setFixedHeight(int(1.3 * self.sizeHint().height()))

        self.clicked.connect(self._toggle)

    def _toggle(self):
        self.toggled.emit(self.mode)

    def setChecked(self, a0: bool):
        self._is_checked = a0
        self.update()

    def isChecked(self) -> bool:
        return self._is_checked

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, status: str):
        self._status = status
        self.update()

    def paintEvent(self, a0):
        status_text, color = _server_mode_status.get(self._status, (_("Unknown"), red))
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        style = util.ensure(self.style())
        if self.isChecked():
            opt.state |= QStyle.StateFlag.State_Sunken
        style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelButtonCommand, opt, painter, self)

        rect = self.rect().adjusted(8, 0, -8, 0)
        bold = self.font()
        bold.setBold(True)
        painter.setFont(bold)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._text
        )
        painter.setPen(QColor(color))
        painter.setFont(self.font())
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, status_text
        )
        painter.end()


class ServerModeSelect(QWidget):
    changed = pyqtSignal(ServerMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._cloud_button = ServerModeButton(ServerMode.cloud, "signed_out", self)
        self._managed_button = ServerModeButton(ServerMode.managed, "not_installed", self)
        self._external_button = ServerModeButton(ServerMode.external, "not_connected", self)

        for button in (self._cloud_button, self._managed_button, self._external_button):
            button.toggled.connect(self._change_mode)

        layout.addWidget(self._cloud_button)
        layout.addWidget(self._managed_button)
        layout.addWidget(self._external_button)
        layout.addStretch()

    def _change_mode(self, mode: ServerMode):
        self.mode = mode
        self.changed.emit(mode)

    @property
    def mode(self):
        if self._cloud_button.isChecked():
            return ServerMode.cloud
        elif self._managed_button.isChecked():
            return ServerMode.managed
        elif self._external_button.isChecked():
            return ServerMode.external
        return ServerMode.undefined

    @mode.setter
    def mode(self, mode: ServerMode):
        self._cloud_button.setChecked(mode is ServerMode.cloud)
        self._managed_button.setChecked(mode is ServerMode.managed)
        self._external_button.setChecked(mode is ServerMode.external)

    def update_status(self, state: ConnectionState, server_state: ServerState):
        self._cloud_button.status = "signed_out"
        self._external_button.status = "not_connected"
        match server_state:
            case ServerState.not_installed:
                self._managed_button.status = "not_installed"
            case ServerState.stopped:
                self._managed_button.status = "not_running"
            case _:
                self._managed_button.status = "not_connected"

        match self.mode, state, server_state:
            case ServerMode.cloud, ConnectionState.auth_missing | ConnectionState.auth_error, _:
                self._cloud_button.status = "signed_out"
            case ServerMode.cloud, ConnectionState.auth_pending, _:
                self._cloud_button.status = "connecting"
            case ServerMode.cloud, ConnectionState.connected, _:
                self._cloud_button.status = "connected"
            case ServerMode.cloud, ConnectionState.error, _:
                self._cloud_button.status = "error"
            case ServerMode.managed, _, ServerState.starting:
                self._managed_button.status = "connecting"
            case ServerMode.managed, ConnectionState.connecting, _:
                self._managed_button.status = "connecting"
            case ServerMode.managed, ConnectionState.connected, ServerState.running:
                self._managed_button.status = "connected"
            case ServerMode.managed, ConnectionState.error, _:
                self._managed_button.status = "error"
            case ServerMode.external, ConnectionState.disconnected, _:
                self._external_button.status = "disconnected"
            case ServerMode.external, ConnectionState.connecting, _:
                self._external_button.status = "connecting"
            case ServerMode.external, ConnectionState.connected, _:
                self._external_button.status = "connected"
            case ServerMode.external, ConnectionState.error, _:
                self._external_button.status = "error"


class ConnectionSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__(_("Server Configuration"))
        self._server = server

        self._server_mode = ServerModeSelect(self)
        self._server_mode.changed.connect(self._change_server_mode)

        self._setup_widget = InitialSetupWidget(self)
        self._cloud_widget = CloudWidget(self)
        self._server_widget = ServerWidget(server, self)
        self._connection_widget = QWidget(self)
        self._server_stack = QStackedWidget(self)
        self._server_stack.addWidget(self._setup_widget)
        self._server_stack.addWidget(self._cloud_widget)
        self._server_stack.addWidget(self._server_widget)
        self._server_stack.addWidget(self._connection_widget)

        connection_layout = QVBoxLayout()
        connection_layout.setContentsMargins(0, 0, 0, 0)
        self._connection_widget.setLayout(connection_layout)

        add_header(connection_layout, Settings._server_url)
        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self._connection_widget)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton(_("Connect"), self._connection_widget)
        self._connect_button.clicked.connect(self._connect)
        server_layout.addWidget(self._connect_button)
        connection_layout.addLayout(server_layout)

        self._server_authorization = TextSetting(
            Settings._server_authorization, self._connection_widget
        )
        self._server_authorization.value_changed.connect(self.write)
        connection_layout.addWidget(self._server_authorization)

        self._check_server_resources = SwitchSetting(
            Settings._check_server_resources, parent=self._connection_widget
        )
        self._check_server_resources.value_changed.connect(self.write)
        connection_layout.addWidget(self._check_server_resources)

        self._connection_status = QLabel(self._connection_widget)
        self._supported_workloads = QLabel(self._connection_widget)
        self._supported_workloads.setWordWrap(True)
        self._supported_workloads.setTextFormat(Qt.TextFormat.RichText)
        self._supported_workloads.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._supported_workloads.setOpenExternalLinks(True)

        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._connection_status)
        status_layout.addWidget(open_log_button, alignment=Qt.AlignmentFlag.AlignRight)

        connection_layout.addLayout(status_layout)
        connection_layout.addWidget(self._supported_workloads)
        connection_layout.addStretch()

        self._layout.addWidget(self._server_mode)
        self._layout.addWidget(self._server_stack)

        self.update_server_status()
        self._update_server_mode(settings.server_mode)

        root.connection.state_changed.connect(self.update_server_status)
        root.connection.error_changed.connect(self.update_server_status)
        root.connection.progress_changed.connect(self.update_server_status)
        self._setup_widget.finished.connect(self._setup_finished)
        self._server_widget.state_changed.connect(self.update_server_status)

    def _setup_finished(self, mode: ServerMode):
        self._server_mode.mode = mode
        self._update_server_mode(mode)

    def _update_server_mode(self, mode: ServerMode):
        self._server_mode.setVisible(mode is not ServerMode.undefined)
        widget = {
            ServerMode.cloud: self._cloud_widget,
            ServerMode.managed: self._server_widget,
            ServerMode.external: self._connection_widget,
            ServerMode.undefined: self._setup_widget,
        }[mode]
        self._server_stack.setCurrentWidget(widget)

    def update_ui(self):
        self._server_widget.update_ui()

    def _read(self):
        self._server_mode.mode = settings.server_mode
        self._server_mode.update_status(root.connection.state, self._server.state)
        self._update_server_mode(settings.server_mode)
        self._server_url.setText(settings.server_url)
        self._server_authorization.value = settings.server_authorization
        self._check_server_resources.value = settings.check_server_resources

    def _write(self):
        settings.server_mode = self._server_mode.mode
        settings.server_url = self._server_url.text()
        settings.server_authorization = self._server_authorization.value
        settings.check_server_resources = self._check_server_resources.value

    def _change_server_mode(self):
        self._update_server_mode(self._server_mode.mode)
        self.write()

    def _connect(self):
        if client := root.connection.create_client(settings):
            root.connection.connect(client)

    def update_server_status(self):
        connection = root.connection
        self._server_mode.update_status(connection.state, self._server.state)
        self._cloud_widget.update_connection_state(connection.state)
        self._connect_button.setEnabled(True)
        if connection.state == ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif connection.state == ConnectionState.connecting:
            self._connection_status.setText(_("Connecting"))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif connection.state == ConnectionState.discover_models:
            progress = f" ({connection.progress[0]}/{connection.progress[1]})"
            self._connection_status.setText(_("Discovering models") + progress)
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif connection.state == ConnectionState.disconnected:
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif connection.state == ConnectionState.error:
            msg = connection.error.removeprefix("Error: ") if connection.error else "Unknown error"
            self._connection_status.setText("<b>" + _("Error") + f"</b>: {msg}")
            self._connection_status.setStyleSheet(f"color: {red};")

        self._supported_workloads.clear()
        if connection.state in [ConnectionState.connected, ConnectionState.error]:
            if connection.missing_resources is not None:
                self._show_missing_resources(connection.missing_resources, connection.state)

    def _show_missing_resources(self, res: MissingResources, state: ConnectionState):
        def model_name(id: ResourceId, with_file=False):
            if res := resources.find_resource(id):
                if with_file:
                    return f"{res.name} ({', '.join(f.name for f in res.files)})"
                return res.name
            if isinstance(id.identifier, str):
                return f"{id.kind.value} {id.identifier}"
            return f"{id.kind.value} {id.identifier.value}"

        text = ""
        if isinstance(res.missing, list):
            text = (
                _("The following ComfyUI custom nodes are missing or too old")
                + ":<ul>"
                + "\n".join(f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in res.missing)
                + "</ul>"
                + _(
                    "Please install or update the custom node package, then restart the server and try again."
                )
                + _("If nodes are still missing, check the ComfyUI output at startup for errors.")
                + "<br>"
            )
        else:
            basic = [m for lst in res.missing.values() for m in lst if m.arch is Arch.all]
            basic = util.unique(basic, key=lambda m: m.string)
            if len(basic) > 0:
                text = _("Missing common models (required)") + ":\n<ul>"
                text += "\n".join(f"<li>{model_name(m, True)}</li>" for m in basic)
                text += "</ul>"
            text += _("Detected workloads for the following base models:") + "\n<ul>"
            for arch, missing in res.missing.items():
                if arch in [Arch.all, Arch.illu_v]:
                    continue
                text += f"<li><b>{arch.value}</b>: "
                if len(missing) == 0:
                    text += _("supported")
                else:
                    names = [model_name(m) for m in missing if m.arch is arch]
                    if len(names) > 0:
                        text += _("missing") + " " + ", ".join(names)
                    else:
                        text += _("models found")
                text += "</li>"
            text += "</ul>"

        link = "<a href='https://docs.interstice.cloud/comfyui-setup'>Custom ComfyUI Setup</a>"
        text += _(
            "See {link} for required models.<br>Check the client.log file for more details.",
            link=link,
        )
        style = "" if state is ConnectionState.error else f"color: {grey};"
        self._supported_workloads.setStyleSheet(style)
        self._supported_workloads.setText(text)

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


class DiffusionSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Diffusion Settings"))

        S = Settings
        self.add("selection_feather", SliderSetting(S._selection_feather, self, 0, 25, "{} %"))
        self.add("selection_blend", SliderSetting(S._selection_blend, self, 0, 100, "{} px"))
        self.add("selection_padding", SliderSetting(S._selection_padding, self, 0, 25, "{} %"))
        self.add(
            "color_match_generation",
            SwitchSetting(S._color_match_generation, parent=self),
        )
        self.add(
            "color_match_edit",
            SwitchSetting(S._color_match_edit, parent=self),
        )
        self.add("control_layer_mode", ComboBoxSetting(S._control_layer_mode, parent=self))
        self.add(
            "control_layer_preset_value",
            SliderSetting(S._control_layer_preset_value, self, 0, 4, "{}"),
        )
        self.add(
            "control_layer_use_custom_strength",
            SwitchSetting(S._control_layer_use_custom_strength, parent=self),
        )
        self.add(
            "control_layer_strength",
            SliderSetting(S._control_layer_strength, self, 0.0, 1.5, "{}"),
        )
        self.add(
            "control_layer_start",
            SliderSetting(S._control_layer_start, self, 0.0, 1.0, "{}"),
        )
        self.add(
            "control_layer_end",
            SliderSetting(S._control_layer_end, self, 0.0, 1.0, "{}"),
        )
        self.add("upscale_model", ComboBoxSetting(S._upscale_model, parent=self))
        self.add("upscale_model_small", ComboBoxSetting(S._upscale_model_small, parent=self))
        self.add(
            "upscale_highres_refine_strength",
            SliderSetting(S._upscale_highres_refine_strength, self, 0.0, 1.0, "{}"),
        )
        self.add(
            "upscale_tile_overlap_auto_base",
            SpinBoxSetting(S._upscale_tile_overlap_auto_base, self, 0, 512, 8, " px"),
        )
        self.add(
            "upscale_tile_overlap_auto_denoise",
            SpinBoxSetting(S._upscale_tile_overlap_auto_denoise, self, 0, 512, 8, " px"),
        )
        self.add("nsfw_filter", ComboBoxSetting(S._nsfw_filter, parent=self))

        nsfw_settings = [(_("Disabled"), 0.0), (_("Basic"), 0.65), (_("Strict"), 0.8)]
        self._widgets["nsfw_filter"].set_items(nsfw_settings)
        self._widgets["control_layer_mode"].set_items([
            (mode.text, mode) for mode in ControlMode if not mode.is_internal
        ])
        self._widgets["control_layer_use_custom_strength"].value_changed.connect(
            self._update_control_layer_default_widgets
        )
        root.connection.models_changed.connect(self.update_upscalers)
        self.update_upscalers()
        self._update_control_layer_default_widgets()
        DiffusionSettings._warning_shown = self._warning_shown or settings.nsfw_filter > 0

        self._layout.addStretch()

    _warning_shown = False

    def update_upscalers(self):
        upscalers = []
        if client := root.connection.client_if_connected:
            upscalers = sorted(client.models.upscalers, key=str.lower)
        for value in [settings.upscale_model, settings.upscale_model_small]:
            if value and value not in upscalers:
                upscalers.append(value)

        items = [(model.rsplit(".", 1)[0], model) for model in upscalers]
        for name in ["upscale_model", "upscale_model_small"]:
            widget: ComboBoxSetting = self._widgets[name]
            widget.set_items(items)
            widget.value = getattr(settings, name)

    def _read(self):
        self._update_control_layer_default_widgets()

    def _write(self):
        self._update_control_layer_default_widgets()
        if self._widgets["control_layer_start"].value > self._widgets["control_layer_end"].value:
            self._widgets["control_layer_end"].value = self._widgets["control_layer_start"].value
            settings.control_layer_end = self._widgets["control_layer_end"].value

        if self._widgets["nsfw_filter"].value > 0 and not self._warning_shown:
            DiffusionSettings._warning_shown = True
            QMessageBox.warning(
                self,
                _("NSFW Filter Warning"),
                _(
                    "The NSFW filter is a basic tool to exclude explicit content from generated images. It is NOT a guarantee and may not catch all inappropriate content. Please use responsibly and always review the generated images."
                ),
            )

    def _update_control_layer_default_widgets(self):
        use_custom = self._widgets["control_layer_use_custom_strength"].value
        self._widgets["control_layer_strength"].enabled = use_custom
        self._widgets["control_layer_start"].enabled = use_custom
        self._widgets["control_layer_end"].enabled = use_custom


class InterfaceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Interface Settings"))

        S = Settings
        self.add("language", ComboBoxSetting(S._language, parent=self))
        self.add("prompt_translation", ComboBoxSetting(S._prompt_translation, parent=self))
        self.add(
            "prompt_line_count",
            SpinBoxSetting(S._prompt_line_count, self, 1, prompt_max_line_count),
        )
        self.add(
            "negative_prompt_line_count",
            SpinBoxSetting(S._negative_prompt_line_count, self, 1, 10),
        )
        self.add(
            "show_negative_prompt",
            SwitchSetting(S._show_negative_prompt, (_("Show"), _("Hide")), self),
        )
        self.add("show_steps", SwitchSetting(S._show_steps, parent=self))
        self.add("recent_styles_count", SpinBoxSetting(S._recent_styles_count, self, 0, 10))
        self.add("new_region_name", TextSetting(S._new_region_name, parent=self))
        self.add("new_region_layer_name", TextSetting(S._new_region_layer_name, parent=self))
        self.add("new_style_name", TextSetting(S._new_style_name, parent=self))
        self.add("new_style_copy_name", TextSetting(S._new_style_copy_name, parent=self))

        self.add("tag_files", FileListSetting(S._tag_files, files=self._tag_files(), parent=self))
        self._layout.addWidget(self._widgets["tag_files"].list_widget)
        self._widgets["tag_files"].add_button(
            Krita.instance().icon("reload-preset"),
            _("Look for new tag files"),
            self._update_tag_files,
        )
        self._widgets["tag_files"].add_button(
            Krita.instance().icon("document-open"),
            _("Open folder where custom tag files can be placed"),
            self._open_tag_folder,
        )

        self.add(
            "generation_finished_action",
            ComboBoxSetting(S._generation_finished_action, parent=self),
        )
        self.add("apply_behavior", ComboBoxSetting(S._apply_behavior, parent=self))
        self.add("apply_region_behavior", ComboBoxSetting(S._apply_region_behavior, parent=self))
        self.add("apply_behavior_live", ComboBoxSetting(S._apply_behavior_live, parent=self))
        self.add(
            "apply_region_behavior_live",
            ComboBoxSetting(S._apply_region_behavior_live, parent=self),
        )
        self.add("new_seed_after_apply", SwitchSetting(S._new_seed_after_apply, parent=self))
        self.add("save_image_format", ComboBoxSetting(S._save_image_format, parent=self))
        self.add("save_image_metadata", SwitchSetting(S._save_image_metadata, parent=self))
        self.add(
            "save_image_file_name_format",
            TextSetting(S._save_image_file_name_format, parent=self),
        )
        self.add(
            "preview_layer_name_format",
            TextSetting(S._preview_layer_name_format, parent=self),
        )
        self.add("apply_layer_name_format", TextSetting(S._apply_layer_name_format, parent=self))
        self.add(
            "generated_layer_name_prefix",
            TextSetting(S._generated_layer_name_prefix, parent=self),
        )
        self.add(
            "layered_batch_prefix_format",
            TextSetting(S._layered_batch_prefix_format, parent=self),
        )
        self.add(
            "animation_layer_name_format",
            TextSetting(S._animation_layer_name_format, parent=self),
        )
        self.add(
            "animation_import_layer_name_format",
            TextSetting(S._animation_import_layer_name_format, parent=self),
        )
        self.add(
            "live_recording_layer_name_format",
            TextSetting(S._live_recording_layer_name_format, parent=self),
        )
        self.add("debug_dump_workflow", SwitchSetting(S._debug_dump_workflow, parent=self))

        self._widgets["save_image_format"].value_changed.connect(self._update_image_format_widgets)

        languages = [(lang.name, lang.id) for lang in Localization.available]
        self._widgets["language"].set_items(languages)
        self.update_translation(root.connection.client_if_connected)

        for w in ["apply_region_behavior", "apply_region_behavior_live"]:
            self._widgets[w].show_label = False

        self._layout.addStretch()

    def read(self):
        super().read()
        self._update_image_format_widgets()

    def _tag_files(self) -> list[str]:
        plugin_tags_path = util.plugin_dir / "tags"
        user_tags_path = util.user_data_dir / "tags"
        files = set()
        for path in plugin_tags_path.glob("*.csv"):
            files.add(path.stem)
        for path in user_tags_path.glob("*.csv"):
            files.add(path.stem)

        return list(files)

    def _update_tag_files(self):
        self._widgets["tag_files"].reset_files(self._tag_files())

    def _open_tag_folder(self):
        user_tag_folder = util.user_data_dir / "tags"
        user_tag_folder.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_tag_folder)))

    def update_translation(self, client: Client | None):
        translation: ComboBoxSetting = self._widgets["prompt_translation"]
        languages = [("Disabled", "")]
        if client:
            languages += [(lang.name, lang.code) for lang in client.features.languages]
        translation.enabled = client is not None
        translation.set_items(languages)
        self.read()

    def _update_image_format_widgets(self):
        fmt: ImageFileFormat = self._widgets["save_image_format"].value
        self._widgets["save_image_metadata"].enabled = fmt.extension == "png"


class StyleDefaultsSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__(_("Style Defaults"))
        self._style = Style(util.user_data_dir / "defaults-style.json")
        self._editor = StyleSettingsEditor(server, include_name=False, parent=self)
        self._editor.value_changed.connect(self._save)
        self._layout.addWidget(self._editor)

    def read(self):
        self._style = Style(util.user_data_dir / "defaults-style.json")
        for name, value in style_defaults().items():
            setattr(self._style, name, value)
        self._editor.read(self._style)

    def restore_defaults(self):
        defaults.clear_section("style")
        self.read()

    def _save(self):
        values = {name: getattr(self._style, name) for name in style_defaults_schema}
        defaults.write_section("style", values, style_defaults_schema)


class WorkspaceDefaultsPage(SettingsTab):
    def __init__(self, title: str, workspace: Workspace):
        super().__init__(title)
        self.workspace = workspace

    def read(self):
        with self._write_guard:
            values = load_workspace_defaults(self.workspace)
            for name, widget in self._widgets.items():
                widget.value = values[name]
            self._read()

    def write(self, *ignored):
        if not self._write_guard:
            values = {
                name: widget.value for name, widget in self._widgets.items() if widget.enabled
            }
            save_workspace_defaults(self.workspace, values)
            self._write()


class DocumentDefaultsPage(SettingsTab):
    def __init__(self):
        super().__init__(_("General"))

    def read(self):
        with self._write_guard:
            values = load_document_defaults()
            for name, widget in self._widgets.items():
                widget.value = values[name]
            self._read()

    def write(self, *ignored):
        if not self._write_guard:
            values = {
                name: widget.value for name, widget in self._widgets.items() if widget.enabled
            }
            save_document_defaults(values)
            self._write()


class WorkspaceDefaultsSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Workspace Defaults"))

        tabs = QTabWidget(self)
        self._layout.addWidget(tabs)
        self._layout.addStretch()

        self.document = DocumentDefaultsPage()
        self.document.add(
            "workspace",
            ComboBoxSetting(
                document_defaults_schema["workspace"],
                parent=self,
            ),
        )
        cast(ComboBoxSetting, self.document._widgets["workspace"]).set_items([
            (_("Generate"), Workspace.generation),
            (_("Upscale"), Workspace.upscaling),
            (_("Live"), Workspace.live),
            (_("Animation"), Workspace.animation),
            (_("Custom"), Workspace.custom),
        ])

        self.generation = WorkspaceDefaultsPage(_("Generation"), Workspace.generation)
        self.generation.add(
            "style", ComboBoxSetting(generation_defaults_schema["style"], parent=self)
        )
        self.generation.add(
            "strength", SliderSetting(generation_defaults_schema["strength"], self, 0.0, 1.0, "{}")
        )
        self.generation.add(
            "region_only",
            SwitchSetting(generation_defaults_schema["region_only"], parent=self),
        )
        self.generation.add(
            "edit_mode",
            SwitchSetting(generation_defaults_schema["edit_mode"], parent=self),
        )
        self.generation.add(
            "batch_count", SpinBoxSetting(generation_defaults_schema["batch_count"], self, 1, 16)
        )
        self.generation.add(
            "fixed_seed",
            SwitchSetting(generation_defaults_schema["fixed_seed"], parent=self),
        )
        self.generation.add(
            "resolution_multiplier",
            SliderSetting(
                generation_defaults_schema["resolution_multiplier"], self, 0.1, 4.0, "{:.1f}x"
            ),
        )
        self.generation.add(
            "use_smart_resolution",
            SwitchSetting(generation_defaults_schema["use_smart_resolution"], parent=self),
        )
        self.generation.add(
            "smart_rotate",
            SwitchSetting(generation_defaults_schema["smart_rotate"], parent=self),
        )
        self.generation.add(
            "queue_mode", ComboBoxSetting(generation_defaults_schema["queue_mode"], parent=self)
        )
        self.generation.add(
            "translation_enabled",
            SwitchSetting(generation_defaults_schema["translation_enabled"], parent=self),
        )
        self.generation.add(
            "layer_count",
            SpinBoxSetting(generation_defaults_schema["layer_count"], self, 1, 16),
        )
        self.generation.add(
            "inpaint_mode", ComboBoxSetting(generation_defaults_schema["inpaint_mode"], parent=self)
        )
        self.generation.add(
            "inpaint_fill", ComboBoxSetting(generation_defaults_schema["inpaint_fill"], parent=self)
        )
        self.generation.add(
            "inpaint_use_model",
            SwitchSetting(generation_defaults_schema["inpaint_use_model"], parent=self),
        )
        self.generation.add(
            "inpaint_use_prompt_focus",
            SwitchSetting(generation_defaults_schema["inpaint_use_prompt_focus"], parent=self),
        )
        self.generation.add(
            "inpaint_context",
            ComboBoxSetting(generation_defaults_schema["inpaint_context"], parent=self),
        )

        self.upscaling = WorkspaceDefaultsPage(_("Upscaling"), Workspace.upscaling)
        self.upscaling.add(
            "upscale_model",
            ComboBoxSetting(upscaling_defaults_schema["upscale_model"], parent=self),
        )
        self.upscaling.add(
            "factor", SliderSetting(upscaling_defaults_schema["factor"], self, 1.0, 4.0, "{}x")
        )
        self.upscaling.add(
            "use_diffusion",
            SwitchSetting(upscaling_defaults_schema["use_diffusion"], parent=self),
        )
        self.upscaling.add(
            "strength", SliderSetting(upscaling_defaults_schema["strength"], self, 0.0, 1.0, "{}")
        )
        self.upscaling.add(
            "unblur_strength",
            SliderSetting(upscaling_defaults_schema["unblur_strength"], self, 0.0, 1.0, "{}"),
        )
        self.upscaling.add(
            "tile_overlap_mode",
            ComboBoxSetting(upscaling_defaults_schema["tile_overlap_mode"], parent=self),
        )
        self.upscaling.add(
            "tile_overlap",
            SpinBoxSetting(upscaling_defaults_schema["tile_overlap"], self, 0, 512, 8, " px"),
        )
        self.upscaling.add(
            "use_prompt", SwitchSetting(upscaling_defaults_schema["use_prompt"], parent=self)
        )

        self.live = WorkspaceDefaultsPage(_("Live"), Workspace.live)
        self.live.add(
            "strength", SliderSetting(live_defaults_schema["strength"], self, 0.0, 1.0, "{}")
        )
        self.live.add(
            "recording_format",
            ComboBoxSetting(live_defaults_schema["recording_format"], parent=self),
        )
        self.live.add(
            "recording_folder_name_format",
            TextSetting(live_defaults_schema["recording_folder_name_format"], parent=self),
        )
        self.live.add(
            "recording_frame_name_format",
            TextSetting(live_defaults_schema["recording_frame_name_format"], parent=self),
        )

        self.animation = WorkspaceDefaultsPage(_("Animation"), Workspace.animation)
        self.animation.add(
            "sampling_quality",
            ComboBoxSetting(animation_defaults_schema["sampling_quality"], parent=self),
        )
        self.animation.add(
            "target_layer_default",
            ComboBoxSetting(animation_defaults_schema["target_layer_default"], parent=self),
        )
        self.animation.add(
            "batch_mode",
            SwitchSetting(animation_defaults_schema["batch_mode"], parent=self),
        )
        self.animation.add(
            "batch_folder_name_format",
            TextSetting(animation_defaults_schema["batch_folder_name_format"], parent=self),
        )
        self.animation.add(
            "batch_frame_name_format",
            TextSetting(animation_defaults_schema["batch_frame_name_format"], parent=self),
        )

        self.custom = WorkspaceDefaultsPage(_("Custom"), Workspace.custom)
        self.custom.add(
            "workflow_id", ComboBoxSetting(custom_defaults_schema["workflow_id"], parent=self)
        )
        self.custom.add("mode", ComboBoxSetting(custom_defaults_schema["mode"], parent=self))
        self.custom.add(
            "params_ui_height",
            SpinBoxSetting(custom_defaults_schema["params_ui_height"], self, 0, 2000, 4, " px"),
        )
        cast(ComboBoxSetting, self.custom._widgets["mode"]).set_items([
            (_("Generate"), CustomGenerationMode.regular),
            (_("Generate Live"), CustomGenerationMode.live),
            (_("Generate Animation"), CustomGenerationMode.animation),
        ])
        cast(ComboBoxSetting, self.generation._widgets["queue_mode"]).set_items([
            (_("At the Back"), QueueMode.back),
            (_("In Front"), QueueMode.front),
            (_("Replace Queue"), QueueMode.replace),
        ])
        cast(ComboBoxSetting, self.animation._widgets["target_layer_default"]).set_items([
            (_("Active layer"), AnimationTargetLayerDefault.active),
            (_("First image layer"), AnimationTargetLayerDefault.first),
        ])

        tabs.addTab(self.document, _("General"))
        tabs.addTab(self.generation, _("Generation"))
        tabs.addTab(self.upscaling, _("Upscaling"))
        tabs.addTab(self.live, _("Live"))
        tabs.addTab(self.animation, _("Animation"))
        tabs.addTab(self.custom, _("Custom"))

        Styles.list().changed.connect(self._update_styles)
        Styles.list().name_changed.connect(self._update_styles)
        root.connection.models_changed.connect(self._update_upscalers)
        root.workflows.loaded.connect(self._update_workflows)
        self._update_styles()
        self._update_upscalers()
        self._update_workflows()

    def read(self):
        self._update_styles()
        self._update_upscalers()
        self._update_workflows()
        self.document.read()
        self.generation.read()
        self.upscaling.read()
        self.live.read()
        self.animation.read()
        self.custom.read()

    def restore_defaults(self):
        defaults.clear_section("document")
        defaults.clear_section("workspaces")
        self.read()

    def _update_styles(self):
        styles = [(_("Current default style"), "")]
        styles.extend((style.name, style.filename) for style in Styles.list())
        widget: ComboBoxSetting = self.generation._widgets["style"]
        widget.set_items(styles)

    def _update_workflows(self):
        items = [(_("First available workflow"), "")]
        items.extend((workflow.name, workflow.id) for workflow in root.workflows)
        widget: ComboBoxSetting = self.custom._widgets["workflow_id"]
        widget.set_items(items)

    def _update_upscalers(self):
        upscalers = []
        if client := root.connection.client_if_connected:
            upscalers = sorted(client.models.upscalers, key=str.lower)
        for value in [settings.upscale_model, settings.upscale_model_small]:
            if value and value not in upscalers:
                upscalers.append(value)
        items = [(_("Current global default"), "")]
        items.extend((model.rsplit(".", 1)[0], model) for model in upscalers)
        widget: ComboBoxSetting = self.upscaling._widgets["upscale_model"]
        widget.set_items(items)


class HistorySizeWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, maximum: int, step: int, parent=None):
        super().__init__(parent)

        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(5)
        self._history_size.setMaximum(maximum)
        self._history_size.setSingleStep(step)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self._change_value)

        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet(f"font-style:italic; color: {green};")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._history_size)
        layout.addWidget(self._history_usage)
        self.setLayout(layout)

    def _change_value(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._history_size.value()

    @value.setter
    def value(self, v):
        self._history_size.setValue(v)

    def update_usage(self, usage: float):
        self._history_usage.setText(_("Currently using") + f" {usage:.1f} MB")


class PerformanceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Performance Settings"))

        add_header(self._layout, Settings._history_size)
        self._history_size = HistorySizeWidget(maximum=20000, step=100, parent=self)
        self._history_size.value_changed.connect(self.write)
        self._layout.addWidget(self._history_size)

        add_header(self._layout, Settings._history_storage)
        self._history_storage = HistorySizeWidget(maximum=2000, step=5, parent=self)
        self._history_storage.value_changed.connect(self.write)
        self._layout.addWidget(self._history_storage)

        add_header(self._layout, Settings._performance_preset)
        self._device_info = QLabel(self)
        self._device_info.setStyleSheet("font-style:italic")
        self._layout.addWidget(self._device_info)

        self._performance_preset = QComboBox(self)
        for preset in PerformancePreset:
            self._performance_preset.addItem(preset.value)
        self._performance_preset.currentIndexChanged.connect(self._change_performance_preset)
        self._layout.addWidget(self._performance_preset, alignment=Qt.AlignmentFlag.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.performance_preset is PerformancePreset.custom)
        self._layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(8, 0, 0, 4)
        self._advanced.setLayout(advanced_layout)

        self._batch_size = SliderSetting(Settings._batch_size, self._advanced, 1, 16)
        self._batch_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size)

        self._resolution_multiplier = SliderSetting(
            Settings._resolution_multiplier, self._advanced, 0.1, 4.0, "{:.1f}x"
        )
        self._resolution_multiplier.value_changed.connect(self.write)
        advanced_layout.addWidget(self._resolution_multiplier)

        self._max_pixel_count = SpinBoxSetting(
            Settings._max_pixel_count, self._advanced, 1, 99, 1, " MP"
        )
        self._max_pixel_count.value_changed.connect(self.write)
        advanced_layout.addWidget(self._max_pixel_count)

        self._tiled_vae = SwitchSetting(
            Settings._tiled_vae, text=(_("Always"), _("Automatic")), parent=self._advanced
        )
        self._tiled_vae.value_changed.connect(self.write)
        advanced_layout.addWidget(self._tiled_vae)

        self._dynamic_caching = SwitchSetting(Settings._dynamic_caching, parent=self)
        self._dynamic_caching.value_changed.connect(self.write)
        self._layout.addWidget(self._dynamic_caching)

        self._multi_threading = SwitchSetting(Settings._multi_threading, parent=self)
        self._multi_threading.value_changed.connect(self.write)
        self._layout.addWidget(self._multi_threading)

        self._live_poll_rate = DoubleSpinBoxSetting(
            Settings._live_poll_rate, self, 0.01, 5.0, 0.01, 2, " s"
        )
        self._live_poll_rate.value_changed.connect(self.write)
        self._layout.addWidget(self._live_poll_rate)

        self._live_default_grace_period = DoubleSpinBoxSetting(
            Settings._live_default_grace_period, self, 0.0, 5.0, 0.01, 2, " s"
        )
        self._live_default_grace_period.value_changed.connect(self.write)
        self._layout.addWidget(self._live_default_grace_period)

        self._live_max_wait_time = DoubleSpinBoxSetting(
            Settings._live_max_wait_time, self, 0.1, 10.0, 0.1, 2, " s"
        )
        self._live_max_wait_time.value_changed.connect(self.write)
        self._layout.addWidget(self._live_max_wait_time)

        self._live_delay_threshold = DoubleSpinBoxSetting(
            Settings._live_delay_threshold, self, 0.0, 10.0, 0.1, 2, " s"
        )
        self._live_delay_threshold.value_changed.connect(self.write)
        self._layout.addWidget(self._live_delay_threshold)

        self._layout.addStretch()

    def _change_performance_preset(self, index):
        self.write()
        is_custom = settings.performance_preset is PerformancePreset.custom
        self._advanced.setEnabled(is_custom)
        if (
            settings.performance_preset is PerformancePreset.auto
            and root.connection.state is ConnectionState.connected
        ):
            apply_performance_preset(settings, root.connection.client.device_info)
        if not is_custom:
            self.read()

    def update_client_info(self):
        if root.connection.state is ConnectionState.connected:
            client = root.connection.client
            self._device_info.setText(
                _("Device")
                + f": [{client.device_info.type.upper()}] {client.device_info.name} ({client.device_info.vram} GB)"
            )

    def _read(self):
        self._history_size.value = settings.history_size
        self._history_size.update_usage(root.active_model.jobs.memory_usage)
        self._history_storage.value = settings.history_storage
        self._history_storage.update_usage(root.get_active_model_used_storage() / (1024**2))
        self._multi_threading.value = settings.multi_threading
        self._batch_size.value = settings.batch_size
        self._performance_preset.setCurrentIndex(
            list(PerformancePreset).index(settings.performance_preset)
        )
        self._resolution_multiplier.value = settings.resolution_multiplier
        self._max_pixel_count.value = settings.max_pixel_count
        self._tiled_vae.value = settings.tiled_vae
        self._dynamic_caching.value = settings.dynamic_caching
        self._live_poll_rate.value = settings.live_poll_rate
        self._live_default_grace_period.value = settings.live_default_grace_period
        self._live_max_wait_time.value = settings.live_max_wait_time
        self._live_delay_threshold.value = settings.live_delay_threshold
        self.update_client_info()

    def _write(self):
        settings.history_size = self._history_size.value
        settings.history_storage = self._history_storage.value
        settings.multi_threading = self._multi_threading.value
        settings.batch_size = int(self._batch_size.value)
        settings.resolution_multiplier = self._resolution_multiplier.value
        settings.max_pixel_count = self._max_pixel_count.value
        settings.tiled_vae = self._tiled_vae.value
        settings.performance_preset = list(PerformancePreset)[
            self._performance_preset.currentIndex()
        ]
        settings.dynamic_caching = self._dynamic_caching.value
        settings.live_poll_rate = self._live_poll_rate.value
        settings.live_default_grace_period = self._live_default_grace_period.value
        settings.live_max_wait_time = self._live_max_wait_time.value
        settings.live_delay_threshold = self._live_delay_threshold.value


class AdvancedSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Advanced Settings"))

        S = Settings
        self.add("save_image_quality_png", SpinBoxSetting(S._save_image_quality_png, self, 0, 100))
        self.add(
            "save_image_quality_png_small",
            SpinBoxSetting(S._save_image_quality_png_small, self, 0, 100),
        )
        self.add(
            "save_image_quality_webp", SpinBoxSetting(S._save_image_quality_webp, self, 0, 100)
        )
        self.add(
            "save_image_quality_webp_lossless",
            SpinBoxSetting(S._save_image_quality_webp_lossless, self, 0, 100),
        )
        self.add(
            "save_image_quality_jpeg", SpinBoxSetting(S._save_image_quality_jpeg, self, 0, 100)
        )
        self.add(
            "selection_min_transition",
            SpinBoxSetting(S._selection_min_transition, self, 0, 512, suffix=" px"),
        )
        self.add(
            "selection_grow_offset",
            SpinBoxSetting(S._selection_grow_offset, self, 0, 128, suffix=" px"),
        )
        self.add(
            "flux_inpaint_cfg_scale",
            SliderSetting(S._flux_inpaint_cfg_scale, self, 1.0, 50.0, "{}"),
        )
        self.add(
            "server_connect_retry_attempts",
            SpinBoxSetting(S._server_connect_retry_attempts, self, 0, 20),
        )
        self.add(
            "server_connect_retry_delay",
            SpinBoxSetting(S._server_connect_retry_delay, self, 0, 120, suffix=" s"),
        )
        self.add(
            "download_retry_attempts",
            SpinBoxSetting(S._download_retry_attempts, self, 1, 20),
        )
        self.add(
            "download_retry_delay",
            SpinBoxSetting(S._download_retry_delay, self, 0, 120, suffix=" s"),
        )
        self.add(
            "download_inactivity_timeout",
            SpinBoxSetting(S._download_inactivity_timeout, self, 5, 600, suffix=" s"),
        )
        self.add(
            "comfy_get_timeout",
            SpinBoxSetting(S._comfy_get_timeout, self, 1, 600, suffix=" s"),
        )
        self.add(
            "comfy_result_image_timeout",
            SpinBoxSetting(S._comfy_result_image_timeout, self, 1, 1800, suffix=" s"),
        )
        self.add(
            "comfy_model_inspection_timeout",
            SpinBoxSetting(S._comfy_model_inspection_timeout, self, 1, 3600, suffix=" s"),
        )
        self.add(
            "websocket_ping_timeout",
            SpinBoxSetting(S._websocket_ping_timeout, self, 1, 600, suffix=" s"),
        )
        self.add(
            "cloud_sign_in_timeout",
            SpinBoxSetting(S._cloud_sign_in_timeout, self, 5, 1800, suffix=" s"),
        )
        self.add(
            "cloud_auth_poll_interval",
            DoubleSpinBoxSetting(S._cloud_auth_poll_interval, self, 0.1, 60.0, 0.1, 1, " s"),
        )
        self.add(
            "cloud_job_poll_interval",
            DoubleSpinBoxSetting(S._cloud_job_poll_interval, self, 0.1, 60.0, 0.1, 1, " s"),
        )
        self.add(
            "auto_update_check_timeout",
            SpinBoxSetting(S._auto_update_check_timeout, self, 1, 600, suffix=" s"),
        )
        self._layout.addStretch()


class AboutSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Plugin Information and Updates"))

        large = self.font()
        large.setPointSize(large.pointSize() + 2)

        extra_large = self.font()
        extra_large.setPointSize(extra_large.pointSize() + 4)

        bold = self.font()
        bold.setBold(True)

        italic = self.font()
        italic.setItalic(True)

        header_layout = QHBoxLayout()
        header_logo = QLabel(self)
        font_height = QFontMetrics(extra_large).height() + 4
        header_logo.setPixmap(logo().scaled(font_height * 2, font_height * 2))
        header_logo.setMaximumSize(font_height * 2, font_height * 2)
        header_text = QLabel("Generative AI\nfor Krita", self)
        header_text.setFont(extra_large)
        header_layout.addWidget(header_logo)
        header_layout.addWidget(header_text)

        current_version_name = QLabel(_("Current version") + ":", self)
        current_version_value = QLabel(__version__, self)

        latest_version_name = QLabel(_("Latest version") + ":", self)
        self._latest_version_value = QLabel(self)
        self._latest_version_value.setFont(bold)

        self._update_error = QLabel(self)
        self._update_error.setFont(italic)

        self._update_checkbox = QCheckBox(_("Check for updates on startup"), self)
        self._update_checkbox.setChecked(settings.auto_update)
        self._update_checkbox.stateChanged.connect(self.write)

        self._check_button = QPushButton(_("Check for Updates"), self)
        self._check_button.setMinimumWidth(font_height * 6)
        self._check_button.clicked.connect(self._check_updates)

        self._update_button = QPushButton(_("Download and Install"), self)
        self._update_button.setMinimumWidth(font_height * 6)
        self._update_button.clicked.connect(self._run_update)

        sys_header = QLabel(_("System Information"), self)
        sys_header.setFont(large)
        sys_desc = QLabel(_("Please attach this information when reporting issues!"), self)
        sys_desc.setFont(italic)
        sys_button = QPushButton(_("Collect Diagnostics"), self)
        sys_button.setMinimumWidth(font_height * 6)
        sys_button.clicked.connect(self._collect_diagnostics)
        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        doc_header = QLabel(_("Documentation and Support"), self)
        doc_header.setFont(large)

        doc_links = QLabel(_links_text, self)
        doc_links.setOpenExternalLinks(True)
        doc_contact = QLabel(_contact_text, self)
        doc_contact.setOpenExternalLinks(True)

        self._layout.addLayout(header_layout)
        self._layout.addSpacing(10)
        current_version_layout = QHBoxLayout()
        current_version_layout.addWidget(current_version_name)
        current_version_layout.addWidget(current_version_value)
        current_version_layout.addStretch()
        self._layout.addLayout(current_version_layout)
        latest_version_layout = QHBoxLayout()
        latest_version_layout.addWidget(latest_version_name)
        latest_version_layout.addWidget(self._latest_version_value)
        latest_version_layout.addStretch()
        self._layout.addLayout(latest_version_layout)
        self._layout.addWidget(self._update_error)
        self._layout.addWidget(self._update_checkbox)
        update_layout = QHBoxLayout()
        update_layout.addWidget(self._check_button)
        update_layout.addWidget(self._update_button)
        update_layout.addStretch()
        self._layout.addLayout(update_layout)
        self._layout.addSpacing(20)
        self._layout.addWidget(sys_header)
        self._layout.addWidget(sys_desc)
        self._layout.addWidget(sys_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(open_log_button)
        self._layout.addSpacing(20)
        self._layout.addWidget(doc_header)
        self._layout.addSpacing(5)
        doc_layout = QHBoxLayout()
        doc_layout.addWidget(doc_links)
        doc_layout.addSpacing(40)
        doc_layout.addWidget(doc_contact)
        doc_layout.addStretch()
        self._layout.addLayout(doc_layout)
        self._layout.addStretch()

        root.auto_update.state_changed.connect(self._update_content)
        self._update_content()

    def _update_content(self):
        self._check_button.setEnabled(False)
        self._update_button.setEnabled(False)
        self._update_error.clear()

        au = root.auto_update
        match au.state:
            case UpdateState.unknown:
                self._latest_version_value.setText(_("Not checked"))
                self._check_button.setEnabled(True)
            case UpdateState.checking:
                self._latest_version_value.setText(_("Checking for updates..."))
            case UpdateState.latest:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
            case UpdateState.available:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.downloading:
                self._latest_version_value.setText(_("Downloading package..."))
            case UpdateState.installing:
                self._latest_version_value.setText(_("Installing new version..."))
            case UpdateState.failed_check:
                self._latest_version_value.setText(_("Unknown"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
            case UpdateState.failed_update:
                self._latest_version_value.setText(_("Update failed"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.restart_required:
                self._latest_version_value.setText(
                    _("Please restart Krita to complete the update!")
                )

    def _check_updates(self):
        root.auto_update.check()

    def _run_update(self):
        root.auto_update.run()

    def _read(self):
        self._update_checkbox.setChecked(settings.auto_update)

    def _write(self):
        settings.auto_update = self._update_checkbox.isChecked()

    def _collect_diagnostics(self):
        diagnostics = collect_diagnostics()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(diagnostics)

        window = QDialog(self)
        window.setWindowTitle(_("Diagnostics Information"))
        layout = QVBoxLayout()
        text = QTextEdit(window)
        text.setReadOnly(True)
        text.setText(diagnostics)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.SmallestReadableFont))
        if clipboard is not None:
            msg = _("System information has been copied to the clipboard.")
            layout.addWidget(QLabel("✔️ " + msg))
        layout.addSpacing(6)
        layout.addWidget(text)
        window.setLayout(layout)
        window.resize(min(self.width(), 800), 640)
        window.exec_()

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


_links_text = """
<a href='https://www.interstice.cloud'>Website</a><br><br>
<a href='https://docs.interstice.cloud'>Handbook: Guides and Tips</a><br><br>
<a href='https://github.com/Acly/krita-ai-diffusion'>GitHub</a>
"""

_contact_text = """
<a href='https://github.com/Acly/krita-ai-diffusion/issues'>Issues</a><br><br>
<a href='https://github.com/Acly/krita-ai-diffusion/discussions'>Discussions</a><br><br>
<a href='https://discord.gg/pWyzHfHHhU'>Discord</a>
"""


class SettingsDialog(QDialog):
    _instance = None

    @classmethod
    def instance(cls) -> SettingsDialog:
        assert cls._instance is not None
        return cls._instance

    def __init__(self, server: Server):
        super().__init__()
        type(self)._instance = self

        self.setWindowTitle(_("Configure Image Diffusion"))
        self.setMinimumSize(QSize(960, 480))
        if screen := QGuiApplication.screenAt(QCursor.pos()):
            size = screen.availableSize()
            min_w = min(size.width(), QFontMetrics(self.font()).width("M") * 100)
            self.resize(QSize(min_w, int(size.height() * 0.8)))

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.connection = ConnectionSettings(server)
        self.styles = StylePresets(server)
        self.style_defaults = StyleDefaultsSettings(server)
        self.workspace_defaults = WorkspaceDefaultsSettings()
        self.diffusion = DiffusionSettings()
        self.interface = InterfaceSettings()
        self.performance = PerformanceSettings()
        self.advanced = AdvancedSettings()
        self.about = AboutSettings()

        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setFixedWidth(120)

        def create_list_item(text: str, widget: QWidget):
            item = QListWidgetItem(text, self._list)
            item.setSizeHint(QSize(112, 24))
            self._stack.addWidget(widget)

        create_list_item(_("Connection"), self.connection)
        create_list_item(_("Styles"), self.styles)
        create_list_item(_("Style Defaults"), self.style_defaults)
        create_list_item(_("Workspace Defaults"), self.workspace_defaults)
        create_list_item(_("Diffusion"), self.diffusion)
        create_list_item(_("Interface"), self.interface)
        create_list_item(_("Performance"), self.performance)
        create_list_item(_("Advanced"), self.advanced)
        create_list_item(_("Plugin"), self.about)

        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._change_page)
        layout.addWidget(self._list)

        inner = QVBoxLayout()
        layout.addLayout(inner)
        inner.addWidget(self._stack)
        inner.addSpacing(6)

        self._restore_button = QPushButton(_("Restore Defaults"), self)
        self._restore_button.clicked.connect(self.restore_defaults)

        version_label = QLabel(_("Plugin version") + f": {__version__}", self)
        version_label.setStyleSheet(f"font-style:italic; color: {grey};")

        anchor = _("Open Settings folder")
        self._open_folder_link = QLabel(f"<a href='file://{util.user_data_dir}'>{anchor}</a>", self)
        self._open_folder_link.linkActivated.connect(self._open_settings_folder)
        self._open_folder_link.setToolTip(str(util.user_data_dir))

        self._close_button = QPushButton(_("Ok"), self)
        self._close_button.clicked.connect(self._close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(version_label)
        button_layout.addStretch()
        button_layout.addWidget(self._open_folder_link)
        button_layout.addSpacing(8)
        button_layout.addWidget(self._close_button)
        inner.addLayout(button_layout)

        root.connection.state_changed.connect(self._update_connection)
        root.connection.models_changed.connect(self.styles.update_model_lists)

    def read(self):
        self.connection.read()
        self.styles.read()
        self.style_defaults.read()
        self.workspace_defaults.read()
        self.diffusion.read()
        self.interface.read()
        self.performance.read()
        self.advanced.read()
        self.about.read()

    def restore_defaults(self):
        settings.restore()
        settings.save()
        self.style_defaults.restore_defaults()
        self.workspace_defaults.restore_defaults()
        self.read()

    def show(self, style: Style | None = None):
        self.read()
        self.connection.update_ui()
        super().show()

        if style:
            self._list.setCurrentRow(1)
            self.styles.current_style = style
        self._close_button.setFocus()

    def _change_page(self, index):
        self._stack.setCurrentIndex(index)

    def _update_connection(self):
        self.connection.update_server_status()
        if root.connection.state is ConnectionState.connected:
            self.interface.update_translation(root.connection.client)
            self.performance.update_client_info()

    def _open_settings_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.user_data_dir)))

    def _close(self):
        _ = self.close()
