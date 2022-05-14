import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import xasyVersion

import xasyUtils as xu
import xasy2asy as x2a
import xasyFile as xf
import xasyOptions as xo
import UndoRedoStack as Urs
import xasyArgs as xa
import xasyBezierInterface as xbi
from xasyTransform import xasyTransform as xT
import xasyStrings as xs

import PrimitiveShape
import InplaceAddObj

import CustMatTransform
import SetCustomAnchor
import GuidesManager

#https://www.pythonguis.com/tutorials/creating-multiple-windows/
class AnotherWindow(Qw.QWidget): #Fill, Arrowhead
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        layout = Qw.QVBoxLayout()
        self.label = Qw.QLabel("Another Window")
        layout.addWidget(self.label)

        self.fillButton = Qw.QComboBox()
        self.fillButton.addItem("unfill")
        self.fillButton.addItem("fill")
        self.fillButton.currentIndexChanged.connect(self.fillChange)
        layout.addWidget(self.fillButton)

        self.setLayout(layout)

    def fillChange(self, i):
        self.shape.fill = (i == 1)