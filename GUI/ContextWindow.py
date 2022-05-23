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
import time

class AnotherWindow(Qw.QWidget): #Fill, Arrowhead
    def __init__(self, shape, parent):
        super().__init__()
        self.shape = shape
        self.parent = parent
        layout = Qw.QVBoxLayout()

        self.label = Qw.QLabel("Fill:")
        layout.addWidget(self.label)
        self.fillButton = Qw.QComboBox()
        self.fillButton.addItem("Unfilled")
        self.fillButton.addItem("Filled")
        self.fillButton.setCurrentIndex(int(self.shape.path.fill))
        self.fillButton.currentIndexChanged.connect(self.fillChange)
        layout.addWidget(self.fillButton)

        self.label = Qw.QLabel("Reflection:")
        layout.addWidget(self.label)
        self.reflectionButton = Qw.QComboBox()
        self.reflectionButton.addItem("None")
        self.reflectionButton.addItem("Horizontal")
        self.reflectionButton.addItem("Vertical")
        self.reflectionButton.currentIndexChanged.connect(self.reflectionChange)
        layout.addWidget(self.reflectionButton)

        self.label = Qw.QLabel("Arrowhead:")
        layout.addWidget(self.label)
        self.arrowheadButton = Qw.QComboBox()
        self.arrowheadButton.addItem("Not currently implemented")
        layout.addWidget(self.arrowheadButton)

        self.setLayout(layout)
        self.setWindowTitle("Shape Options Window")

    def fillChange(self, i):
        if self.shape.path.fill != bool(i):
            self.shape.swapFill()
            self.parent.quickUpdate()

    def reflectionChange(self, i):
        currentAnchor = Qc.QPointF(0, 0)
        if i == 0:
            self.parent.newTransform = xT.makeScaleTransform(1, 1, currentAnchor).toQTransform()
        if i == 1:
            self.parent.newTransform = xT.makeScaleTransform(1, -1, currentAnchor).toQTransform()
        if i == 2:
            self.parent.newTransform = xT.makeScaleTransform(-1, 1, currentAnchor).toQTransform()
        self.parent.currentlySelectedObj['selectedIndex'] = self.parent.mostRecentObject
        self.parent.releaseTransform()
        self.parent.newTransform = Qg.QTransform()
