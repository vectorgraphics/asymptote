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
        self.newShape = self.shape
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
        self.arrowList = ["None","Arrow","ArcArrow"]
        for arrowMode in self.arrowList:
            self.arrowheadButton.addItem(arrowMode)
        self.arrowheadButton.currentIndexChanged.connect(self.arrowheadChange)
        layout.addWidget(self.arrowheadButton)

        self.label = Qw.QLabel("Arrow Style:")
        layout.addWidget(self.label)
        self.arrowstyleButton = Qw.QComboBox()
        self.arrowstyleList = ["()","(SimpleHead)","(HookHead)","(TeXHead)"] #Pull this from the arrow if you can. 
        for arrowStyle in self.arrowstyleList:
            self.arrowstyleButton.addItem(arrowStyle)
        self.arrowstyleButton.currentIndexChanged.connect(self.arrowstyleChange)
        layout.addWidget(self.arrowstyleButton)

        self.label = Qw.QLabel("Arrow Size:")
        layout.addWidget(self.label)
        self.arrowSizeBox = Qw.QLineEdit()
        self.arrowSizeBox.returnPressed.connect(self.sizeChange)
        layout.addWidget(self.arrowSizeBox)

        self.label = Qw.QLabel("Arrow Angle:")
        layout.addWidget(self.label)
        self.arrowAngleBox = Qw.QLineEdit()
        self.arrowAngleBox.returnPressed.connect(self.angleChange) #Bug: have to hit enter to change.
        layout.addWidget(self.arrowAngleBox)

        self.label = Qw.QLabel("Arrow Fill:")
        layout.addWidget(self.label)
        self.arrowFillButton = Qw.QComboBox()
        self.arrowFillList = ["","FillDraw","Fill","NoFill","UnFill","Draw"] #Pull this from the arrow if you can. 
        for arrowFillStyle in self.arrowFillList:
            self.arrowFillButton.addItem(arrowFillStyle)
        self.arrowFillButton.currentIndexChanged.connect(self.arrowFillChange)
        layout.addWidget(self.arrowFillButton)

        self.confirmButton = Qw.QPushButton("OK")
        self.confirmButton.clicked.connect(self.renderChanges)
        layout.addWidget(self.confirmButton)

        #TODO: Make this a function. 
        if not isinstance(self.shape, x2a.xasyShape):
            self.fillButton.setDisabled(True)
            if isinstance(self.shape, x2a.asyArrow):
                self.arrowheadButton.setCurrentIndex(int(self.shape.arrowActive))
                self.arrowstyleButton.setCurrentIndex(int(self.shape.arrowStyle))
                self.arrowFillButton.setCurrentIndex(int(self.shape.arrowFill))
            else:
                self.arrowheadButton.setDisabled(True)
        else:
            self.fillButton.setCurrentIndex(int(self.shape.path.fill))
        if not isinstance(self.shape, x2a.asyArrow): #Make these all a list or something.
            self.arrowstyleButton.setDisabled(True)
            self.arrowSizeBox.setDisabled(True)
            self.arrowAngleBox.setDisabled(True)
            self.arrowFillButton.setDisabled(True)

        self.setLayout(layout)
        self.setWindowTitle("Shape Options Window")

    def arrowheadChange(self, i):
        #None, {Arrow, ArcArrow} x {(),(SimpleHead),(HookHead),(TeXHead)}
        if isinstance(self.shape, x2a.xasyShape):
            if i != 0:
                self.newShape = self.newShape.arrowify()
        else:
            if i != self.shape.arrowActive:
                self.newShape = self.newShape.setArrow(i) #Simplify the logic

    def arrowstyleChange(self, i):
        #None, {Arrow, ArcArrow} x {(),(SimpleHead),(HookHead),(TeXHead)}
        if isinstance(self.shape, x2a.xasyShape): #Is this redunant now?
            if i != 0:
                self.newShape = self.newShape.arrowify()
        else:
            if i != self.shape.arrowStyle:
                self.newShape = self.newShape.setStyle(i)

    def fillChange(self, i):
        if self.shape.path.fill != bool(i):
            self.newShape = self.newShape.swapFill()

    def reflectionChange(self, i): #TODO: Modernize this.
        if i == 0:
            self.parent.newTransform = xT.makeScaleTransform(1, 1, self.parent.currentAnchor).toQTransform()
        if i == 1:
            self.parent.newTransform = xT.makeScaleTransform(1, -1, self.parent.currentAnchor).toQTransform()
        if i == 2:
            self.parent.newTransform = xT.makeScaleTransform(-1, 1, self.parent.currentAnchor).toQTransform()
        self.parent.currentlySelectedObj['selectedIndex'] = self.parent.mostRecentObject
        self.parent.releaseTransform()
        self.parent.newTransform = Qg.QTransform()

    def sizeChange(self):
        newSize = self.arrowSizeBox.text()
        try:
            newSize = float(newSize)
            if self.shape.arrowSize != newSize:
                self.newShape = self.newShape.setSize(newSize)
        except:
            return #TODO: Show error message.

    def angleChange(self): #Refactor this with the above. 
        newAngle = self.arrowAngleBox.text()
        try:
            newAngle = float(newAngle)
            if self.shape.arrowAngle != newAngle:
                self.newShape = self.newShape.setAngle(newAngle)
        except:
            return #TODO: Show error message.
            
    def arrowFillChange(self, i):
        if isinstance(self.shape, x2a.xasyShape):
            if i != 0:
                self.newShape = self.newShape.arrowify() #Can this ever be reached?
        else:
            if i != self.shape.arrowFill:
                self.newShape = self.newShape.setFill(i)
        
    def renderChanges(self):
        self.parent.replaceObject(self.parent.mostRecentObject,self.newShape)
        self.parent.terminateContextWindow()        