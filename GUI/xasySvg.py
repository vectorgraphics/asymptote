import PyQt5.QtGui as Qg
import io
import subprocess

class SvgObject():
    def __init__(self, file: str):
        with io.open(file) as f:
            self._data = f.read().encode('utf-8')

    def render(self, dpi:int) -> Qg.QImage:
        rawDataProc = subprocess.Popen(['rsvg-convert', '--dpi-x', str(dpi), '--dpi-y', str(dpi)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        outData, *args = rawDataProc.communicate(self._data)
        return Qg.QImage.fromData(outData, 'PNG')
