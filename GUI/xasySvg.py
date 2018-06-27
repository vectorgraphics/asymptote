import xml
import xml.etree.ElementTree as xet
import cairosvg as csvg
import PyQt5.QtGui as Qg
import io

class SvgObject():
    xmlns = 'http://www.w3.org/2000/svg'
    def __init__(self, file: str):
        self._data_tree = xet.parse(file)
        self._root = self._data_tree.getroot()
        self._cachedDump = None
        self._changed = False

    def scale(self, sx: float, sy: float):
        # <root> <g><g transf=..> ... 
        self._changed = True
        mainImg = self._root.find('{{{0}}}g'.format(self.xmlns))

        if 'transform' not in mainImg[0].attrib:
            mainImg[0].attrib['transform'] = ''

        oldstr = mainImg[0].attrib['transform']
        height = self._root.attrib['height']
        mainImg[0].attrib['transform'] = (' scale({0} {1}) '.format(
            sx, sy) + 'translate(0 {0})'.format(height[:-2])) + oldstr

    def dump(self) -> bytearray:
        # xet.dump(self._root)
        if self._changed or self._cachedDump is None:
            self._cachedDump = xet.tostring(
                self._root, encoding='unicode').encode('utf-8')
            self._changed = False
        return self._cachedDump

    def render(self, dpi:int) -> Qg.QImage:
        raw_data = csvg.svg2png(bytestring=self.dump(), dpi=dpi)
        return Qg.QImage.fromData(raw_data, 'PNG')
