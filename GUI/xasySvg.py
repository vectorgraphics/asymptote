#!/usr/bin/env python3

import PyQt5.QtGui as Qg
import PyQt5.QtWidgets as Qw
import io
import subprocess
import sys

class SvgObject():
    def __init__(self, file: str):
        self.file=file

    def render(self, dpi:int) -> Qg.QImage:
        try:
            rawDataProc = subprocess.Popen(['rsvg-convert', '--dpi-x', str(dpi),
                                            '--dpi-y', str(dpi), self.file],
                                           stdout=subprocess.PIPE)
        except:
            Qw.QMessageBox.about(None,'rsvg-convert missing','Please install rsvg-convert version >= 2.40 in your path.')
            sys.exit(-1)

        return Qg.QImage.fromData(rawDataProc.stdout.read(), 'PNG')
