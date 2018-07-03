import cairosvg as csvg
import xml.etree as xet 
import PyQt5.QtGui as Qg
import io

class SvgObject():
    def __init__(self, file: str):
        with io.open(file) as f:
            self._data = f.read().encode('utf-8')
        self.xmlRoot = xet.ElementTree.fromstring(self._data.decode('utf-8'))

        # once we use pdflatex, uncomment this line to remove any clip-path='url(#clip1)'
        # self.cleanclip()
        self.cached = False

    def cleanclip(self):
        # remove any clip-path url1 as cairo bug
        self.cached = False 

        # see xpath for info. 
        for elem in self.xmlRoot.findall(".//*[@clip-path='url(#clip1)']"):
            elem.attrib.pop('clip-path')
    
    def render(self, dpi:int) -> Qg.QImage:
        if not self.cached:
            self.outData = xet.ElementTree.tostring(self.xmlRoot, encoding='utf-8')
            self.cached = True 
            xet.ElementTree.dump(self.xmlRoot)
        raw_data = csvg.svg2png(bytestring=self.outData, dpi=dpi)
        return Qg.QImage.fromData(raw_data, 'PNG')
