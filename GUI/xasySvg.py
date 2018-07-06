import PyQt5.QtGui as Qg
import io
import subprocess

class SvgObject():
    def __init__(self, file: str):
        self.file=file

    def render(self, dpi:int) -> Qg.QImage:
        rawDataProc = subprocess.Popen(['rsvg-convert', '--dpi-x', str(dpi),
                                        '--dpi-y', str(dpi), self.file],
                                       stdout=subprocess.PIPE)
        return Qg.QImage.fromData(rawDataProc.stdout.read(), 'PNG')
