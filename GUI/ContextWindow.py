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

class AnotherWindow(Qw.QWidget):
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
        self.arrowheadButton.addItem("None")
        self.arrowheadButton.addItem("Arrow")
        self.arrowheadButton.currentIndexChanged.connect(self.arrowheadChange)
        layout.addWidget(self.arrowheadButton)

        if not isinstance(self.shape, x2a.xasyShape):
            self.fillButton.setDisabled(True)
            if isinstance(self.shape, x2a.asyArrow):
                self.arrowheadButton.setCurrentIndex(int(self.shape.arrowActive))
            else:
                self.arrowheadButton.setDisabled(True)
        else:
            self.fillButton.setCurrentIndex(int(self.shape.path.fill))

        self.setLayout(layout)
        self.setWindowTitle("Shape Options Window")

    def arrowheadChange(self, i):
        if isinstance(self.shape, x2a.xasyShape):
            if i == 1:
                self.parent.replaceObject(self.parent.mostRecentObject,self.shape.arrowify())
                self.parent.terminateContextWindow()
        else:
            if i != self.shape.arrowActive:
                self.parent.replaceObject(self.parent.mostRecentObject,self.shape.setArrow(i!=0))
                self.parent.terminateContextWindow()

    def fillChange(self, i):
        if self.shape.path.fill != bool(i):
            self.shape.swapFill()
            self.parent.terminateContextWindow()

    def reflectionChange(self, i):
        if i == 0:
            self.parent.newTransform = xT.makeScaleTransform(1, 1, self.parent.currentAnchor).toQTransform()
        if i == 1:
            self.parent.newTransform = xT.makeScaleTransform(1, -1, self.parent.currentAnchor).toQTransform()
        if i == 2:
            self.parent.newTransform = xT.makeScaleTransform(-1, 1, self.parent.currentAnchor).toQTransform()
        self.parent.currentlySelectedObj['selectedIndex'] = self.parent.mostRecentObject
        self.parent.releaseTransform()
        self.parent.newTransform = Qg.QTransform()
