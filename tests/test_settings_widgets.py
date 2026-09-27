from enum import Enum

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from ai_diffusion.settings import Setting
from ai_diffusion.ui.settings_widgets import ComboBoxSetting

_qt_app = QCoreApplication.instance()
if not isinstance(_qt_app, QApplication):
    _qt_app = QApplication([])


class Mode(Enum):
    first = 1
    second = 2


def test_combo_box_settings_remain_readable_after_selecting_seedvr2():
    mode_widget = ComboBoxSetting(Setting("Mode", Mode.first))
    mode_widget.set_items([("First mode", Mode.first), ("Second mode", Mode.second)])
    mode_widget.value = Mode.second
    upscale_widget = ComboBoxSetting(Setting("Upscale model", "4x.pth"))
    upscale_widget.set_items([("4x", "4x.pth"), ("SeedVR2", "SeedVR2")])

    upscale_widget.value = "SeedVR2"

    assert mode_widget.value is Mode.second
    assert mode_widget._combo.currentData() == Mode.second.name
    assert upscale_widget.value == "SeedVR2"


def test_combo_box_enum_without_selection_uses_default():
    widget = ComboBoxSetting(Setting("Mode", Mode.first))
    widget.set_items([("First mode", Mode.first), ("Second mode", Mode.second)])
    widget._combo.setCurrentIndex(-1)

    assert widget.value is Mode.first
