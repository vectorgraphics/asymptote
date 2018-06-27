import cairosvg as csvg
import PyQt5.QtGui as Qg
import io

class SvgObject():
    def __init__(self, file: str):
        with io.open(file) as f:
            self._data = f.read().encode('utf-8')

    def render(self, dpi:int) -> Qg.QImage:
        raw_data = csvg.svg2png(bytestring=self._data, dpi=dpi)
        return Qg.QImage.fromData(raw_data, 'PNG')
