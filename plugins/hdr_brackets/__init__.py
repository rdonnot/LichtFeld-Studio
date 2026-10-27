"""HDR Brackets: align, calibrate and merge exposure brackets to linear EXR for HDR splat training.

Nothing heavy is imported here; the core package is imported when a merge starts.
"""

import lichtfeld as lf
from lfs_plugins.layouts.menus import menu_operator
from lfs_plugins.types import Menu

from .operators import MergeOperator
from .panels.bracket_panel import BracketPanel


class HdrBracketsMenu(Menu):
    label = "HDR Brackets"
    location = "MENU_BAR"
    order = 210

    def menu_items(self):
        return [menu_operator(MergeOperator)]


_classes = [MergeOperator, BracketPanel, HdrBracketsMenu]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("hdr_brackets loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("hdr_brackets unloaded")
