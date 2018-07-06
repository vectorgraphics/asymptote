import xml.etree as xet 
import PyQt5.QtGui as Qg
import io
import subprocess

class SvgObject():
    def __init__(self, file: str):
        with io.open(file) as f:
            self._data = f.read().encode('utf-8')
        # self._data = None
        self.fileName = file
        # self.xmlRoot = xet.ElementTree.fromstring(self._data.decode('utf-8'))

        # self.cleanclip()
        self.cached = False

    def cleanclip(self):
        # For pdf tex engines, remove spurious clip-path='url(#clip1)'
        # to work around cairo bug
        self.cached = False 

        # see xpath for info. 
        for elem in self.xmlRoot.findall(".//*[@clip-path='url(#clip1)']"):
            if 'id' in elem.attrib:
                elem.attrib.pop('clip-path')
    
    def render(self, dpi:int) -> Qg.QImage:
        #if not self.cached:
        #    self.outData = xet.ElementTree.tostring(self.xmlRoot, encoding='utf-8')
        #    self.cached = True 
            # xet.ElementTree.dump(self.xmlRoot)
        # raw_data = csvg.svg2png(bytestring=self.outData, dpi=dpi)

        rawDataProc = subprocess.Popen(['rsvg-convert', '--dpi-x', str(dpi), '--dpi-y', str(dpi)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        outData, *args = rawDataProc.communicate(self._data)

        return Qg.QImage.fromData(outData, 'PNG')
