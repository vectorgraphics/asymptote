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

        self.colorButton = Qw.QPushButton("Set Line Colour")
        self.colorButton.clicked.connect(self.pickColor)
        #layout.addWidget(self.colorButton)

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

        #TODO: Make this a function. 
        if not isinstance(self.shape, x2a.xasyShape):
            #self.fillButton.setDisabled(True)
            if isinstance(self.shape, x2a.asyArrow):
                self.arrowheadButton.setCurrentIndex(int(self.shape.arrowSettings["active"]))
            else:
                self.arrowheadButton.setDisabled(True)
        else:
            self.fillButton.setCurrentIndex(int(self.shape.path.fill))

        if isinstance(self.shape, x2a.asyArrow) and self.shape.arrowSettings["active"]: #Make these all a list or something.
            self.label = Qw.QLabel("Arrow Style:")
            layout.addWidget(self.label)
            self.arrowstyleButton = Qw.QComboBox()
            for arrowStyle in self.shape.arrowStyleList:
                self.arrowstyleButton.addItem(arrowStyle if arrowStyle else "(default)")
            self.arrowstyleButton.currentIndexChanged.connect(self.arrowstyleChange)
            layout.addWidget(self.arrowstyleButton)

            self.label = Qw.QLabel("Arrow Size:")
            layout.addWidget(self.label)
            self.arrowSizeBox = Qw.QLineEdit()
            layout.addWidget(self.arrowSizeBox)
            self.arrowSizeBox.setPlaceholderText(self.getInfo("DefaultHead.size(currentpen)"))

            self.label = Qw.QLabel("Arrow Angle:")
            layout.addWidget(self.label)
            self.arrowAngleBox = Qw.QLineEdit()
            layout.addWidget(self.arrowAngleBox)
            self.arrowAngleBox.setPlaceholderText(self.getInfo("arrowangle"))

            self.label = Qw.QLabel("Arrow Fill:")
            layout.addWidget(self.label)
            self.arrowFillButton = Qw.QComboBox()
            for arrowFillStyle in self.shape.arrowFillList:
                self.arrowFillButton.addItem(arrowFillStyle if arrowFillStyle else "(default)")
            self.arrowFillButton.currentIndexChanged.connect(self.arrowFillChange)
            layout.addWidget(self.arrowFillButton)

            self.arrowstyleButton.setCurrentIndex(int(self.shape.arrowSettings["style"]))
            self.arrowFillButton.setCurrentIndex(int(self.shape.arrowSettings["fill"]))

        self.confirmButton = Qw.QPushButton("Render")
        self.confirmButton.clicked.connect(self.renderChanges)
        layout.addWidget(self.confirmButton)

        self.setLayout(layout)
        self.setWindowTitle("Shape Options Window")

    def arrowheadChange(self, i):
        #None, {Arrow, ArcArrow} x {(),(SimpleHead),(HookHead),(TeXHead)}
        if isinstance(self.shape, x2a.xasyShape):
            if i != 0:
                self.newShape = self.newShape.arrowify(arrowhead=i)
        else:
            self.newShape.arrowSettings["active"] = i #Simplify the logic

    def arrowstyleChange(self, i):
        self.newShape.arrowSettings["style"] = i

    def fillChange(self, i):
        if isinstance(self.shape, x2a.asyArrow):
            self.shape.arrowSettings["fill"] = not self.shape.arrowSettings["fill"]
        elif self.shape.path.fill != bool(i):
            self.newShape = self.newShape.swapFill()

    def reflectionChange(self, i): #TODO: Modernize this.
        reflectionList = [[1,1],[1,-1],[-1,1]]
        self.parent.newTransform = xT.makeScaleTransform(*reflectionList[i], self.parent.currentAnchor).toQTransform()
        self.parent.currentlySelectedObj['selectedIndex'] = self.parent.mostRecentObject
        self.parent.releaseTransform()
        self.parent.newTransform = Qg.QTransform()

    def sizeChange(self):
        newSize = self.arrowSizeBox.text()
        try:
            self.newShape.arrowSettings["size"] = float(newSize)
        except:
            return #TODO: Show error message.

    def angleChange(self): #Refactor this with the above. 
        newAngle = self.arrowAngleBox.text()
        try:
            self.newShape.arrowSettings["angle"] = float(newAngle)
        except:
            return #TODO: Show error message.
            
    def arrowFillChange(self, i): #Can I lambda this? 
        self.newShape.arrowSettings["fill"] = i
        
    def renderChanges(self): #Pull from text boxes here.
        if isinstance(self.shape, x2a.asyArrow) and self.shape.arrowSettings["active"]:
            self.sizeChange()
            self.angleChange()
        if self.newShape:
            self.parent.replaceObject(self.parent.contextWindowObject,self.newShape)
        self.parent.terminateContextWindow()        

    def getInfo(self,value):
        """ Find out the size of an arbitrary Asymptote pen """
        self.asyEngine = self.parent.asyEngine
        assert isinstance(self.asyEngine, x2a.AsymptoteEngine)
        assert self.asyEngine.active

        fout = self.asyEngine.ostream
        fin = self.asyEngine.istream

        fout.write("write(_outpipe,{},endl);\n".format(value))
        fout.write(self.asyEngine.xasy)
        fout.flush()

        return fin.readline()

    def pickColor(self):
        self.colorDialog = Qw.QColorDialog(x2a.asyPen.convertToQColor(self.shape.color['line']), self)
        self.colorDialog.show()
        result = self.colorDialog.exec()
        if result == Qw.QDialog.Accepted:
            self.shape.color['line'] = self.colorDialog.selectedColor()
