#!/usr/bin/env python3
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import subprocess
import sys
from xasygui import xasyOptions as xo


class SvgObject:
    def __init__(self, file: str):
        self.file = file

    def _create_call_arguments(self, dpi: int):
        settings = xo.BasicConfigs.defaultOpt
        return [
            settings.get("rsvgConverterPath") or "rsvg-convert",
            f"--dpi-x={dpi}",
            f"--dpi-y={dpi}",
            "--format=png",
            self.file,
        ]

    def render(self, dpi: int) -> QtGui.QImage:
        callArgs = self._create_call_arguments(dpi)
        try:
            rawDataProc = subprocess.run(
                callArgs,
                stdout=subprocess.PIPE,
            )
        except OSError:
            QtWidgets.QMessageBox.about(
                None,
                "rsvg-convert missing",
                "Please install rsvg-convert version >= 2.40 is available.",
            )
            sys.exit(-1)

        return QtGui.QImage.fromData(rawDataProc.stdout, "PNG")
