from pyUIClass.window1 import Ui_MainWindow

import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc

import numpy as np
import os
import json
import io
import pathlib
import webbrowser
import copy

import xasyUtils

import xasy2asy as x2a
import xasyFile as xf
import xasyOptions as xo
import UndoRedoStack as Urs
import xasyArgs as xa
from xasyTransform import xasyTransform as xT

import PrimitiveShape
import InplaceAddObj

import CustMatTransform
import SetCustomAnchor
import BezierCurveEditor
import GuidesManager


class ActionChanges:
    pass


class TransformationChanges(ActionChanges):
    def __init__(self, objKey, transformation, isLocal=False):
        self.objKey = objKey
        self.transformation = transformation
        self.isLocal = isLocal


class ObjCreationChanges(ActionChanges):
    def __init__(self, obj):
        self.object = obj


class AnchorMode:
    origin = 0
    topLeft = 1
    topRight = 2
    bottomRight = 3
    bottomLeft = 4
    customAnchor = 5
    center = 6


class GridMode:
    cartesian = 0
    polar = 1


class SelectionMode:
    select = 0
    pan = 1
    translate = 2
    rotate = 3
    scale = 4


class AddObjectMode:
    Circle = 0
    Arc = 1
    Polygon = 2


class DefaultSettings:
    defaultKeymap = {
        'commandPalette': 'F1',
        'quit': 'Ctrl+Q'
    }


class MainWindow1(Qw.QMainWindow):
    defaultFrameStyle = """
    QFrame{{ 
        padding: 4.0;
        border-radius: 3.0; 
        background: rgb({0}, {1}, {2})
    }}
    """

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.settings = xo.xasyOptions()
        self.settings.load()
        self.keyMaps = DefaultSettings.defaultKeymap

        self.raw_args = Qc.QCoreApplication.arguments()
        self.args = xa.parseArgs(self.raw_args)

        # For initialization purposes
        self.canvSize = Qc.QSize()
        self.filename = None
        self.mainCanvas = None
        self.canvasPixmap = None

        # Actions
        # <editor-fold> Connecting Actions
        self.ui.txtLineWidth.setValidator(Qg.QDoubleValidator())

        self.connectActions()
        self.connectButtons()
        # </editor-fold>

        # Base Transformations

        self.mainTransformation = Qg.QTransform()
        self.mainTransformation.scale(1, -1)
        self.localTransform = Qg.QTransform()
        self.screenTransformation = Qg.QTransform()

        # Internal Settings
        self.magnification = 1
        self.inMidTransformation = False
        self.addMode = None
        self.currentlySelectedObj = {'type': 'xasyPicture', 'selectedKey': None}
        self.savedMousePosition = None
        self.currentBoundingBox = None
        self.selectionDelta = None
        self.newTransform = None
        self.origBboxTransform = None
        self.deltaAngle = 0
        self.scaleFactor = 1
        self.panOffset = 0, 0

        self.undoRedoStack = Urs.actionStack()

        self.lockX = False
        self.lockY = False
        self.anchorMode = AnchorMode.origin
        self.currentAnchor = Qc.QPointF(0, 0)
        self.customAnchor = None
        self.useGlobalCoords = True
        self.drawAxes = True
        self.drawGrid = False
        self.gridSnap = False  # TODO: for now. turn it on later

        self.finalPixmap = None
        self.preCanvasPixmap = None
        self.postCanvasPixmap = None

        self.previewCurve = None

        self.drawObjects = {}
        self.xasyDrawObj = {'drawDict': self.drawObjects}

        self.modeButtons = {self.ui.btnTranslate, self.ui.btnRotate, self.ui.btnScale, self.ui.btnSelect,
                            self.ui.btnPan}
        self.objButtons = {self.ui.btnCustTransform, self.ui.actionTransform}
        self.globalTransformOnlyButtons = (self.ui.comboAnchor, self.ui.btnAnchor)

        self.currAddOptionsWgt = None
        self.currAddOptions = {
            'inscribed': True,
            'sides': 3,
            'centermode': True
        }

        self.currentMode = SelectionMode.translate
        self.drawGridMode = GridMode.cartesian
        self.setAllInSetEnabled(self.objButtons, False)
        self._currentPen = x2a.asyPen()
        self.currentGuides = []

        # commands switchboard
        self.commandsFunc = {
            'quit': Qc.QCoreApplication.quit,
            'undo': self.undoRedoStack.undo,
            'redo': self.undoRedoStack.redo,
            'manual': self.actionManual,
            'loadFile': self.btnLoadFileonClick,
            'save': self.btnSaveOnClick,
            'saveAs': self.actionSaveAs,
            'transform': self.btnCustTransformOnClick,
            'commandPalette': self.enterCustomCommand,
            'clearGuide': self.clearGuides,
        }

        # Settings Initialization
        # from xasyoptions config file
        self.setupXasyOptions()
        self.loadKeyMaps()

        self.colorDialog = Qw.QColorDialog(x2a.asyPen.convertToQColor(self._currentPen.color), self)
        self.initPenInterface()

    def handleArguments(self):
        if self.args.file is not None:
            self.loadFile(self.args.file)
        else:
            self.initializeEmptyFile()

    def initPenInterface(self):
        self.ui.txtLineWidth.setText(str(self._currentPen.width))
        self.updateFrameDispColor()

    def updateFrameDispColor(self):
        r, g, b = [int(x * 255) for x in self._currentPen.color]
        self.ui.frameCurrColor.setStyleSheet(MainWindow1.defaultFrameStyle.format(r, g, b))

    def initDebug(self):
        debugFunc = {
            'debug:addLineGuide': self.debugAddLineGuide,
            'debug:addArcGuide': self.debugAddArcGuide,
            'debug:pause': self.debug,
            'debug:execPythonCmd': self.execPythonCmd,
            'debug:setPolarGrid': self.debugSetPolarGrid,
            'debug:addUnitCircle': self.dbgAddUnitCircle,
            'debug:addCircle': self.dbgAddCircle,
            'debug:addPoly': self.dbgAddPoly,
            'debug:addLabel': self.debugAddLabel
        }
        self.commandsFunc = {**self.commandsFunc, **debugFunc}

    def connectActions(self):
        self.ui.actionQuit.triggered.connect(lambda: self.execCustomCommand('quit'))
        self.ui.actionUndo.triggered.connect(lambda: self.execCustomCommand('undo'))
        self.ui.actionRedo.triggered.connect(lambda: self.execCustomCommand('redo'))
        self.ui.actionTransform.triggered.connect(lambda: self.execCustomCommand('transform'))

        self.ui.actionSaveAs.triggered.connect(self.actionSaveAs)
        self.ui.actionManual.triggered.connect(self.actionManual)
        self.ui.actionEnterCommand.triggered.connect(self.enterCustomCommand)

    def setupXasyOptions(self):
        if self.settings['debugMode']:
            self.initDebug()

        terminalFont = Qg.QFont(self.settings['terminalFont'], self.settings['terminalFontSize'])
        self.ui.plainTextEdit.setFont(terminalFont)

        newColor = Qg.QColor(self.settings['defaultPenColor'])
        newWidth = self.settings['defaultPenWidth']

        self._currentPen.setColorFromQColor(newColor)
        self._currentPen.setWidth(newWidth)

    def connectButtons(self):
        # Button initialization
        self.ui.btnUndo.clicked.connect(self.btnUndoOnClick)
        self.ui.btnRedo.clicked.connect(self.btnRedoOnClick)
        self.ui.btnLoadFile.clicked.connect(self.btnLoadFileonClick)
        self.ui.btnSave.clicked.connect(self.btnSaveOnClick)
        self.ui.btnQuickScreenshot.clicked.connect(self.btnQuickScreenshotOnClick)

        self.ui.btnDrawAxes.clicked.connect(self.btnDrawAxesOnClick)
        self.ui.btnAsyfy.clicked.connect(self.asyfyCanvas)

        self.ui.btnTranslate.clicked.connect(self.btnTranslateonClick)
        self.ui.btnRotate.clicked.connect(self.btnRotateOnClick)
        self.ui.btnScale.clicked.connect(self.btnScaleOnClick)
        self.ui.btnSelect.clicked.connect(self.btnSelectOnClick)
        self.ui.btnPan.clicked.connect(self.btnPanOnClick)

        # self.ui.btnDebug.clicked.connect(self.pauseBtnOnClick)
        self.ui.btnAlignX.clicked.connect(self.btnAlignXOnClick)
        self.ui.btnAlignY.clicked.connect(self.btnAlignYOnClick)
        self.ui.comboAnchor.currentTextChanged.connect(self.handleAnchorCombo)
        self.ui.btnWorldCoords.clicked.connect(self.btnWorldCoordsOnClick)

        self.ui.btnCustTransform.clicked.connect(self.btnCustTransformOnClick)
        self.ui.btnViewCode.clicked.connect(self.btnLoadEditorOnClick)
        self.ui.btnAnchor.clicked.connect(self.btnCustomAnchorOnClick)

        self.ui.btnSelectColor.clicked.connect(self.btnColorSelectOnClick)
        self.ui.txtLineWidth.textEdited.connect(self.txtLineWithEdited)

        self.ui.btnCreateCurve.clicked.connect(self.btnCreateCurveOnClick)
        self.ui.btnDrawGrid.clicked.connect(self.btnDrawGridOnClick)

        self.ui.btnAddCircle.clicked.connect(self.btnAddCircleOnClick)
        self.ui.btnAddPoly.clicked.connect(self.btnAddPolyOnClick)
        self.ui.btnAddLabel.clicked.connect(self.btnAddLabelOnClick)

    @property
    def currentPen(self):
        return copy.deepcopy(self._currentPen)

    def dbgAddUnitCircle(self):
        newCirclePath = PrimitiveShape.PrimitiveShape.circle((0, 0), 1)
        newCircle = x2a.xasyShape(newCirclePath)
        self.fileItems.append(newCircle)
        self.asyfyCanvas()

    def dbgAddCircle(self):
        commandText, result = Qw.QInputDialog.getText(self, '', 'enter python cmd')
        if result:
            rawArray = [float(rawResult) for rawResult in commandText.split()]
            x, y, rad = rawArray
            newCirclePath = PrimitiveShape.PrimitiveShape.circle((x, y), rad)
            newCircle = x2a.xasyShape(newCirclePath, pen=self.currentPen)
            self.fileItems.append(newCircle)
            self.asyfyCanvas()

    def dbgAddPoly(self):
        newSquarePath = PrimitiveShape.PrimitiveShape.exscribedRegPolygon(6, (0, 0), 100, 0)
        newSquare = x2a.xasyShape(newSquarePath, pen=self.currentPen)
        self.fileItems.append(newSquare)
        self.asyfyCanvas()

    def debugAddLabel(self):
        testText = '$\\displaystyle{\\int_{\\varphi(F)} f = \\int_F (f \\circ \\varphi) \\left| \\det J_{\\varphi} \\right|}$'
        newPath = x2a.xasyText(testText, (0, 0))
        self.fileItems.append(newPath)
        self.asyfyCanvas()

    def debug(self):
        print('Put a breakpoint here.')

    def debugSetPolarGrid(self):
        self.drawGridMode = GridMode.polar

    def execPythonCmd(self):
        commandText, result = Qw.QInputDialog.getText(self, '', 'enter python cmd')
        if result:
            exec(commandText)

    def updateOptionWidget(self):
        if self.currAddOptionsWgt is not None:
            self.currAddOptionsWgt.hide()
            self.ui.addOptionLayout.removeWidget(self.currAddOptionsWgt)
            self.currAddOptionsWgt = None
        self.currAddOptionsWgt = self.addMode.createOptWidget(self.currAddOptions)
        if self.currAddOptionsWgt is not None:
            self.ui.addOptionLayout.addWidget(self.currAddOptionsWgt)

    def debugAddLineGuide(self):
        commandText, result = Qw.QInputDialog.getText(self, '', 'enter <originx> <originy> <angle>')
        if result:
            px, py, ang = [float(val) for val in commandText.split()]
            newLineGuide = GuidesManager.LineGuide(Qc.QPointF(px, py), ang, Qg.QPen(Qg.QColor('red')))
            self.currentGuides.append(newLineGuide)
        self.quickUpdate()

    def debugAddArcGuide(self):
        commandText, result = Qw.QInputDialog.getText(self, '', 'enter <originx> <originy> <rad> <sang> <eang>')
        if result:
            px, py, rad, sang, eang = [float(val) for val in commandText.split()]
            newArcGuide = GuidesManager.ArcGuide(Qc.QPoint(px, py), rad, sang, eang, Qg.QPen(Qg.QColor('red')))
            self.currentGuides.append(newArcGuide)
        self.quickUpdate()

    def clearGuides(self):
        self.currentGuides.clear()
        self.quickUpdate()

    def btnCreateCurveOnClick(self):
        self.inCurveCreationMode = True
        curveDialog = BezierCurveEditor.BezierCurveEditor(useDegrees=self.settings['useDegrees'])
        curveDialog.curveChanged.connect(self.updateCurve)
        curveDialog.show()
        result = curveDialog.exec_()

        if result == Qw.QDialog.Accepted:
            asyCurve = x2a.asyPath.fromBezierPoints(curveDialog.createPointList())
            newXasyObjCurve = x2a.xasyShape(asyCurve)
            # print(newXasyObjCurve.getCode())
            self.fileItems.append(newXasyObjCurve)

        self.inCurveCreationMode = False
        self.previewCurve = None
        self.asyfyCanvas()

    def btnAddCircleOnClick(self):
        self.addMode = InplaceAddObj.AddCircle()
        self.updateOptionWidget()

    def btnAddPolyOnClick(self):
        self.addMode = InplaceAddObj.AddPoly()
        self.updateOptionWidget()

    def btnAddLabelOnClick(self):
        self.addMode = InplaceAddObj.AddLabel()
        self.updateOptionWidget()

    def updateCurve(self, valid, newCurve):
        self.previewCurve = newCurve
        self.quickUpdate()

    def addTransformationChanges(self, objKey, transform, isLocal=False):
        self.undoRedoStack.add(self.createAction(TransformationChanges(objKey,
                            transform, isLocal)))
        self.checkUndoRedoButtons()

    def btnUndoOnClick(self):
        self.undoRedoStack.undo()
        self.checkUndoRedoButtons()

    def btnRedoOnClick(self):
        self.undoRedoStack.redo()
        self.checkUndoRedoButtons()

    def checkUndoRedoButtons(self):
        if self.undoRedoStack.changesMade():
            self.ui.btnUndo.setEnabled(True)
            self.ui.actionUndo.setEnabled(True)
        else:
            self.ui.btnUndo.setEnabled(False)
            self.ui.actionUndo.setEnabled(False)

        if len(self.undoRedoStack.redoStack) > 0:
            self.ui.btnRedo.setEnabled(True)
            self.ui.actionRedo.setEnabled(True)
        else:
            self.ui.btnRedo.setEnabled(False)
            self.ui.actionRedo.setEnabled(False)

    def handleUndoChanges(self, change):
        assert isinstance(change, ActionChanges)
        if isinstance(change, TransformationChanges):
            self.transformObject(change.objKey, change.transformation.inverted(), change.isLocal)
        elif isinstance(change, ObjCreationChanges):
            pass  # for now, until we implement a remove object/add object. This will be trivial
        self.quickUpdate()

    def handleRedoChanges(self, change):
        assert isinstance(change, ActionChanges)
        if isinstance(change, TransformationChanges):
            self.transformObject(change.objKey, change.transformation, change.isLocal)
        elif isinstance(change, ObjCreationChanges):
            pass  # for now, until we implement a remove/add method. By then, this will be trivial.
        self.quickUpdate()

    #  is this a "pythonic" way?
    def createAction(self, changes):
        def _change():
            return self.handleRedoChanges(changes)

        def _undoChange():
            return self.handleUndoChanges(changes)

        return Urs.action((_change, _undoChange))

    def execCustomCommand(self, command):
        if command in self.commandsFunc:
            self.commandsFunc[command]()
        else:
            self.ui.statusbar.showMessage('Command {0} not found'.format(command))

    def enterCustomCommand(self):
        commandText, result = Qw.QInputDialog.getText(self, 'Enter Custom Command', 'Enter Custom Command')
        if result:
            self.execCustomCommand(commandText)

    def addItemFromPath(self, path):
        newItem = x2a.xasyShape(path, pen=self.currentPen)
        self.fileItems.append(newItem)
        self.asyfyCanvas()

    def actionManual(self):
        asyManualURL = 'http://asymptote.sourceforge.net/asymptote.pdf'
        webbrowser.open_new(asyManualURL)

    def loadKeyMapFile(self):
        defaultKeyMap = '.asy/xasy2KeyMapDefault.json'
        fullDefaultKeyMap = pathlib.Path.home().joinpath(pathlib.Path(defaultKeyMap))
        if not fullDefaultKeyMap.exists():
            defaultConfFile = io.open(fullDefaultKeyMap, 'w')
            defaultConfFile.write(json.dumps(DefaultSettings.defaultKeymap, indent=4))

        keymapFile = '.asy/xasy2KeyMap.json'
        keymapPath = pathlib.Path.home().joinpath(pathlib.Path(keymapFile))

        if keymapPath.exists():
            usrKeymapFile = io.open(keymapPath)
            usrKeyMap = json.loads(usrKeymapFile.read())
            self.keyMaps.update(usrKeyMap)
        else:
            usrKeymapFile = io.open(keymapPath, 'w')
            usrKeymapFile.write(json.dumps({}, indent=4))

            usrKeymapFile.close()

    def loadKeyMaps(self):
        """Inverts the mapping of the key
           Input map is in format 'Action' : 'Key Sequence' """
        self.loadKeyMapFile()
        for action, key in self.keyMaps.items():
            shortcut = Qw.QShortcut(self)
            shortcut.setKey(Qg.QKeySequence(key))

            # hate doing this, but python doesn't have explicit way to pass a
            # string to a lambda without an identifier
            # attached to it.
            exec('shortcut.activated.connect(lambda: self.execCustomCommand("{0}"))'.format(action),
                 {'self': self, 'shortcut': shortcut})

    def initializeButtons(self):
        self.ui.btnDrawAxes.setChecked(self.settings['defaultShowAxes'])
        self.btnDrawAxesOnClick(self.settings['defaultShowAxes'])

        self.ui.btnDrawGrid.setChecked(self.settings['defaultShowGrid'])
        self.btnDrawGridOnClick(self.settings['defaultShowGrid'])

    def btnSaveOnClick(self):
        if self.filename is None:
            self.actionSaveAs()
        else:
            saveFile = io.open(self.filename, 'w')
            xf.saveFile(saveFile, self.fileItems)
            saveFile.close()

    def actionSaveAs(self):
        saveLocation = Qw.QFileDialog.getSaveFileName(self, 'Save File', Qc.QDir.homePath())[0]
        if saveLocation[1]:
            saveFile = io.open(saveLocation, 'w')
            xf.saveFile(saveFile, self.fileItems)
            saveFile.close()
            self.filename = saveLocation

    def btnQuickScreenshotOnClick(self):
        saveLocation = Qw.QFileDialog.getSaveFileName(self, 'Save Screenshot', Qc.QDir.homePath())
        if saveLocation[0]:
            self.ui.imgLabel.pixmap().save(saveLocation[0])

    def btnLoadFileonClick(self):
        fileName = Qw.QFileDialog.getOpenFileName(self, 'Open Asymptote File', Qc.QDir.homePath(), '*.asy')
        if fileName[0]:
            self.loadFile(fileName[0])

    def handleAnchorCombo(self, text):
        if text == 'Origin':
            self.anchorMode = AnchorMode.origin
        elif text == 'Center':
            self.anchorMode = AnchorMode.center
        elif text == 'Top Left':
            self.anchorMode = AnchorMode.topLeft
        elif text == 'Bottom Left':
            self.anchorMode = AnchorMode.bottomLeft
        elif text == 'Bottom Right':
            self.anchorMode = AnchorMode.bottomRight
        elif text == 'Custom Anchor':
            if self.customAnchor is not None:
                self.anchorMode = AnchorMode.customAnchor
            else:
                self.btnCustomAnchorOnClick()

    def btnCustomAnchorOnClick(self, text=''):
        custAnchorDialog = SetCustomAnchor.CustomAnchorDialog()
        custAnchorDialog.show()
        result = custAnchorDialog.exec()
        if result == Qw.QDialog.Accepted:
            self.customAnchor = custAnchorDialog.getPoint()
            self.ui.comboAnchor.setCurrentText('Custom Anchor')

    def btnColorSelectOnClick(self):
        self.colorDialog.show()
        result = self.colorDialog.exec()
        if result == Qw.QDialog.Accepted:
            self._currentPen.setColorFromQColor(self.colorDialog.selectedColor())
            self.updateFrameDispColor()

    def txtLineWithEdited(self, text):
        new_val = xasyUtils.tryParse(text, float)
        if new_val is not None:
            if new_val > 0:
                self._currentPen.setWidth(new_val)

    def isReady(self):
        return self.mainCanvas is not None

    def resizeEvent(self, resizeEvent):
        assert isinstance(resizeEvent, Qg.QResizeEvent)
        # newRect = Qc.QRect(Qc.QPoint(0, 0), resizeEvent.size())
        # self.ui.centralFrame.setFrameRect(newRect)

    def show(self):
        super().show()
        self.createMainCanvas()  # somehow, the coordinates doesn't get updated until after showing.
        self.initializeButtons()
        self.postShow()

    def postShow(self):
        self.handleArguments()

    def roundPositionSnap(self, oldPoint):
        minorGridSize = self.settings['gridMajorAxesSpacing'] / (self.settings['gridMinorAxesCount'] + 1)
        if isinstance(oldPoint, list) or isinstance(oldPoint, tuple):
            return [round(val / minorGridSize) * minorGridSize for val in oldPoint]
        elif isinstance(oldPoint, Qc.QPoint) or isinstance(oldPoint, Qc.QPointF):
            x, y = oldPoint.x(), oldPoint.y()
            x = round(x / minorGridSize) * minorGridSize
            y = round(y / minorGridSize) * minorGridSize
            return Qc.QPointF(x, y)
        else:
            raise Exception

    def mouseMoveEvent(self, mouseEvent):  # TODO: Actually refine grid snapping...
        assert isinstance(mouseEvent, Qg.QMouseEvent)
        if not self.ui.imgLabel.underMouse():
            return

        canvasPos = self.getCanvasCoordinates()

        if self.addMode is not None:
            if self.addMode.active:
                self.addMode.mouseMove(canvasPos)
                self.quickUpdate()
            return

        if self.currentMode == SelectionMode.pan:
            mousePos = self.getWindowCoordinates()
            newPos = mousePos - self.savedWindowMousePos
            tx, ty = newPos.x(), newPos.y()
            if self.lockX:
                tx = 0
            if self.lockY:
                ty = 0
            self.screenTransformation = self.currScreenTransform * Qg.QTransform.fromTranslate(tx, ty)
            self.quickUpdate()
            return

        if self.inMidTransformation:
            if self.currentMode == SelectionMode.translate:
                newPos = canvasPos - self.savedMousePosition
                if self.gridSnap:
                    newPos = self.roundPositionSnap(newPos)  # actually round to the nearest minor grid afterwards...

                self.tx, self.ty = newPos.x(), newPos.y()

                if self.lockX:
                    self.tx = 0
                if self.lockY:
                    self.ty = 0
                self.newTransform = Qg.QTransform.fromTranslate(self.tx, self.ty)

            elif self.currentMode == SelectionMode.rotate:
                if self.gridSnap:
                    canvasPos = self.roundPositionSnap(canvasPos)

                adjustedSavedMousePos = self.savedMousePosition - self.currentAnchor
                adjustedCanvasCoords = canvasPos - self.currentAnchor

                origAngle = np.arctan2(adjustedSavedMousePos.y(), adjustedSavedMousePos.x())
                newAng = np.arctan2(adjustedCanvasCoords.y(), adjustedCanvasCoords.x())
                self.deltaAngle = newAng - origAngle
                self.newTransform = xT.makeRotTransform(self.deltaAngle, self.currentAnchor).toQTransform()

            elif self.currentMode == SelectionMode.scale:
                if self.gridSnap:
                    canvasPos = self.roundPositionSnap(canvasPos)
                    x, y = int(round(canvasPos.x())), int(round(canvasPos.y()))  # otherwise it crashes...
                    canvasPos = Qc.QPoint(x, y)

                scaleFactor = Qc.QPoint.dotProduct(canvasPos, self.savedMousePosition) /\
                                   (self.savedMousePosition.manhattanLength() ** 2)
                if not self.lockX:
                    self.scaleFactorX = scaleFactor
                else:
                    self.scaleFactorX = 1

                if not self.lockY:
                    self.scaleFactorY = scaleFactor
                else:
                    self.scaleFactorY = 1

                self.newTransform = xT.makeScaleTransform(self.scaleFactorX, self.scaleFactorY, self.currentAnchor).\
                    toQTransform()

            self.quickUpdate()

    def mouseReleaseEvent(self, mouseEvent):
        assert isinstance(mouseEvent, Qg.QMouseEvent)
        if self.addMode is not None:
            self.addMode.mouseRelease()
            if isinstance(self.addMode, InplaceAddObj.AddLabel):
                self.createLabel(self.addMode.getObject())
            else:
                self.addItemFromPath(self.addMode.getObject())
            self.selectionMode = SelectionMode.select
            self.updateChecks()
        self.addMode = None
        if self.inMidTransformation:
            self.clearSelection()
        self.inMidTransformation = False
        self.quickUpdate()

    def createLabel(self, labelInfo):
        text = labelInfo['txt']
        align = labelInfo['align']
        anchor = labelInfo['anchor']
        newLabel = x2a.xasyText(text=text, location=anchor, pen=self.currentPen, align=align)
        self.fileItems.append(newLabel)

        self.asyfyCanvas()

    def clearSelection(self):
        if self.currentlySelectedObj['selectedKey'] is not None:
            self.releaseTransform()
        self.setAllInSetEnabled(self.objButtons, False)
        self.currentlySelectedObj['selectedKey'] = None
        self.newTransform = Qg.QTransform()
        self.currentBoundingBox = None
        self.quickUpdate()

    def mousePressEvent(self, mouseEvent):
        if not self.ui.imgLabel.underMouse():
            return

        self.savedMousePosition = self.getCanvasCoordinates()

        if self.addMode is not None:
            self.addMode.mouseDown(self.savedMousePosition, self.currAddOptions)
            return

        if self.currentMode == SelectionMode.pan:
            self.savedWindowMousePos = self.getWindowCoordinates()
            self.currScreenTransform = self.screenTransformation * Qg.QTransform()
            return

        if self.inMidTransformation:
            return

        selectedKey = self.selectObject()
        if selectedKey is not None:
            if self.currentMode in {SelectionMode.translate, SelectionMode.rotate, SelectionMode.scale}:
                self.setAllInSetEnabled(self.objButtons, False)
                self.inMidTransformation = True
            else:
                self.setAllInSetEnabled(self.objButtons, True)
                self.inMidTransformation = False

            self.currentlySelectedObj['selectedKey'] = selectedKey

            self.currentBoundingBox = self.drawObjects[selectedKey].boundingBox
            self.origBboxTransform = self.drawObjects[selectedKey].transform.toQTransform()
            self.newTransform = Qg.QTransform()

            if self.anchorMode == AnchorMode.center:
                self.currentAnchor = self.currentBoundingBox.center()
            elif self.anchorMode == AnchorMode.topLeft:
                self.currentAnchor = self.currentBoundingBox.bottomLeft()  # due to internal image being flipped
            elif self.anchorMode == AnchorMode.topRight:
                self.currentAnchor = self.currentBoundingBox.bottomRight()
            elif self.anchorMode == AnchorMode.customAnchor:
                self.currentAnchor = self.customAnchor
            else:
                self.currentAnchor = Qc.QPointF(0, 0)

            if self.anchorMode != AnchorMode.origin:
                pass
                # TODO: Record base points/bbox before hand and use that for
                # anchor?
                # adjTransform =
                # self.drawObjects[selectedKey].transform.toQTransform()
                # self.currentAnchor = adjTransform.map(self.currentAnchor)

        else:
            self.setAllInSetEnabled(self.objButtons, False)
            self.currentBoundingBox = None
            self.inMidTransformation = False
            self.clearSelection()
        self.quickUpdate()

    def releaseTransform(self):
        newTransform = x2a.asyTransform.fromQTransform(self.newTransform)
        objKey = self.currentlySelectedObj['selectedKey']
        self.addTransformationChanges(objKey, newTransform, not self.useGlobalCoords)
        self.transformObject(objKey, newTransform, not self.useGlobalCoords)

    def adjustTransform(self, appendTransform):
        self.screenTransformation = self.screenTransformation * appendTransform

    def createMainCanvas(self):
        self.canvSize = self.ui.imgFrame.size()
        x, y = self.canvSize.width() / 2, self.canvSize.height() / 2

        self.canvasPixmap = Qg.QPixmap(self.canvSize)
        self.canvasPixmap.fill()

        self.finalPixmap = Qg.QPixmap(self.canvSize)

        self.preCanvasPixmap = Qg.QPixmap(self.canvSize)
        self.postCanvasPixmap = Qg.QPixmap(self.canvSize)

        self.mainCanvas = Qg.QPainter(self.canvasPixmap)
        self.xasyDrawObj['canvas'] = self.mainCanvas

        self.mainTransformation = Qg.QTransform()
        self.mainTransformation.scale(1, -1)
        self.mainTransformation.translate(x, -y)

        self.screenTransformation = self.mainTransformation * Qg.QTransform()

        self.mainCanvas.setTransform(self.screenTransformation, True)

        self.ui.imgLabel.setPixmap(self.canvasPixmap)

    def selectObject(self):
        if not self.ui.imgLabel.underMouse():
            return
        canvasCoords = self.getCanvasCoordinates()
        highestDrawPriority = -np.inf
        collidedObjKey = None
        for objKey in self.drawObjects:
            obj = self.drawObjects[objKey]
            if obj.collide(canvasCoords):
                if obj.drawOrder > highestDrawPriority:
                    collidedObjKey = objKey
        if collidedObjKey is not None:
            self.ui.statusbar.showMessage(str('Collide with' + collidedObjKey), 2500)
            return collidedObjKey

    def getCanvasCoordinates(self):
        assert self.ui.imgLabel.underMouse()
        uiPos = self.mapFromGlobal(Qg.QCursor.pos())
        canvasPos = self.ui.imgLabel.mapFrom(self, uiPos)
        return canvasPos * self.screenTransformation.inverted()[0]

    def getWindowCoordinates(self):
        assert self.ui.imgLabel.underMouse()
        return self.mapFromGlobal(Qg.QCursor.pos())
    # def rotateBtnOnClick(self):
    #     theta = float(self.ui.txtTheta.toPlainText())
    #     objectID = int(self.ui.txtObjectID.toPlainText())
    #     self.rotateObject(0, objectID, theta, (0, 0))
    #     self.populateCanvasWithItems()
    #     self.ui.imgLabel.setPixmap(self.canvasPixmap)

    # def custTransformBtnOnClick(self):
    #     xx = float(self.ui.lineEditMatXX.text())
    #     xy = float(self.ui.lineEditMatXY.text())
    #     yx = float(self.ui.lineEditMatYX.text())
    #     yy = float(self.ui.lineEditMatYY.text())
    #     tx = float(self.ui.lineEditTX.text())
    #     ty = float(self.ui.lineEditTY.text())
    #     objectID = int(self.ui.txtObjectID.toPlainText())
    #     self.transformObject(0, objectID, x2a.asyTransform((tx, ty, xx, xy,
    #                          yx, yy)))

    def refreshCanvas(self):
        self.mainCanvas.begin(self.canvasPixmap)
        self.mainCanvas.setTransform(self.screenTransformation)

    def asyfyCanvas(self):
        self.refreshCanvas()
        self.drawObjects.clear()

        self.preDraw(self.mainCanvas)
        self.populateCanvasWithItems()

        self.mainCanvas.end()

        self.postDraw()
        self.updateScreen()

    def quickUpdate(self):
        self.refreshCanvas()

        self.preDraw(self.mainCanvas)
        self.quickDraw()
        self.mainCanvas.end()

        self.postDraw()
        self.updateScreen()

    def quickDraw(self):
        assert self.isReady()
        drawList = sorted(self.drawObjects.values(), key=lambda drawObj: drawObj.drawOrder)
        if self.currentlySelectedObj['selectedKey'] in self.drawObjects:
            selectedObj = self.drawObjects[self.currentlySelectedObj['selectedKey']]
        else:
            selectedObj = None

        for item in drawList:
            if selectedObj is item and self.settings['enableImmediatePreview']:
                if self.useGlobalCoords:
                    item.draw(self.newTransform)
                else:
                    item.draw(self.newTransform, applyReverse=True)
            else:
                item.draw()

    def updateScreen(self):
        self.finalPixmap = Qg.QPixmap(self.canvSize)
        self.finalPixmap.fill(Qc.Qt.black)
        with Qg.QPainter(self.finalPixmap) as finalPainter:
            drawPoint = Qc.QPoint(0, 0)
            # finalPainter.drawPixmap(drawPoint, self.preCanvasPixmap)
            finalPainter.drawPixmap(drawPoint, self.canvasPixmap)
            finalPainter.drawPixmap(drawPoint, self.postCanvasPixmap)
        self.ui.imgLabel.setPixmap(self.finalPixmap)

    def drawCartesianGrid(self, preCanvas):
        majorGrid = self.settings['gridMajorAxesSpacing']
        minorGridCount = self.settings['gridMinorAxesCount']

        majorGridCol = Qg.QColor(self.settings['gridMajorAxesColor'])
        minorGridCol = Qg.QColor(self.settings['gridMinorAxesColor'])

        panX, panY = self.screenTransformation.dx(), self.screenTransformation.dy()

        x_range = self.canvSize.width() / 2 + (2 * abs(panX))
        y_range = self.canvSize.height() / 2 + (2 * abs(panY))

        for x in range(0, 2 * round(x_range) + 1, majorGrid):  # have to do
            # this in two stages...
            preCanvas.setPen(minorGridCol)
            for xMinor in range(1, minorGridCount + 1):
                xCoord = round(x + ((xMinor / (minorGridCount + 1)) * majorGrid))
                preCanvas.drawLine(Qc.QLine(xCoord, -9999, xCoord, 9999))
                preCanvas.drawLine(Qc.QLine(-xCoord, -9999, -xCoord, 9999))

        for y in range(0, 2 * round(y_range) + 1, majorGrid):
            preCanvas.setPen(minorGridCol)
            for yMinor in range(1, minorGridCount + 1):
                yCoord = round(y + ((yMinor / (minorGridCount + 1)) * majorGrid))
                preCanvas.drawLine(Qc.QLine(-9999, yCoord, 9999, yCoord))
                preCanvas.drawLine(Qc.QLine(-9999, -yCoord, 9999, -yCoord))

            preCanvas.setPen(majorGridCol)
            preCanvas.drawLine(Qc.QLine(-9999, y, 9999, y))
            preCanvas.drawLine(Qc.QLine(-9999, -y, 9999, -y))

        for x in range(0, 2 * round(x_range) + 1, majorGrid):
            preCanvas.setPen(majorGridCol)
            preCanvas.drawLine(Qc.QLine(x, -9999, x, 9999))
            preCanvas.drawLine(Qc.QLine(-x, -9999, -x, 9999))

    def drawPolarGrid(self, preCanvas):
        center = Qc.QPointF(0, 0)
        majorGridCol = Qg.QColor(self.settings['gridMajorAxesColor'])
        minorGridCol = Qg.QColor(self.settings['gridMinorAxesColor'])
        majorGrid = self.settings['gridMajorAxesSpacing']
        minorGridCount = self.settings['gridMinorAxesCount']

        majorAxisAng = (np.pi/4)  # 45 degrees - for now.
        minorAxisCount = 2  # 15 degrees each

        subRadiusSize = int(round((majorGrid / (minorGridCount + 1))))
        subAngleSize = majorAxisAng / (minorAxisCount + 1)

        for radius in range(majorGrid, 9999 + 1, majorGrid):
            preCanvas.setPen(majorGridCol)
            preCanvas.drawEllipse(center, radius, radius)

            preCanvas.setPen(minorGridCol)

            for minorRing in range(minorGridCount):
                subRadius = round(radius - (subRadiusSize * (minorRing + 1)))
                preCanvas.drawEllipse(center, subRadius, subRadius)


        currAng = majorAxisAng
        while currAng <= (2 * np.pi):
            preCanvas.setPen(majorGridCol)
            p1 = center + (9999 * Qc.QPointF(np.cos(currAng), np.sin(currAng)))
            preCanvas.drawLine(Qc.QLineF(center, p1))

            preCanvas.setPen(minorGridCol)
            for minorAngLine in range(minorAxisCount):
                newAng = currAng - (subAngleSize * (minorAngLine + 1))
                p1 = center + (9999 * Qc.QPointF(np.cos(newAng), np.sin(newAng)))
                preCanvas.drawLine(Qc.QLineF(center, p1))

            currAng = currAng + majorAxisAng

    def preDraw(self, painter):
        # self.preCanvasPixmap.fill(Qc.Qt.white)
        self.canvasPixmap.fill()
        preCanvas = painter

        # preCanvas = Qg.QPainter(self.preCanvasPixmap)
        preCanvas.setTransform(self.screenTransformation)

        if self.drawAxes:
            preCanvas.setPen(Qc.Qt.gray)
            preCanvas.drawLine(Qc.QLine(-9999, 0, 9999, 0))
            preCanvas.drawLine(Qc.QLine(0, -9999, 0, 9999))

        if self.drawGrid:
            if self.drawGridMode == GridMode.cartesian:
                self.drawCartesianGrid(painter)
            elif self.drawGridMode == GridMode.polar:
                self.drawPolarGrid(painter)

        if self.currentGuides:
            for guide in self.currentGuides:
                guide.drawShape(preCanvas)
        # preCanvas.end()

    def postDraw(self):
        self.postCanvasPixmap.fill(Qc.Qt.transparent)
        with Qg.QPainter(self.postCanvasPixmap) as postCanvas:
            postCanvas.setTransform(self.screenTransformation)
            if self.currentBoundingBox is not None:
                postCanvas.save()
                selObj = self.drawObjects[self.currentlySelectedObj['selectedKey']]
                if not self.useGlobalCoords:
                    postCanvas.save()
                    postCanvas.setTransform(selObj.transform.toQTransform(), True)
                    # postCanvas.setTransform(selObj.baseTransform.toQTransform(), True)
                    postCanvas.setPen(Qc.Qt.gray)
                    postCanvas.drawLine(Qc.QLine(-9999, 0, 9999, 0))
                    postCanvas.drawLine(Qc.QLine(0, -9999, 0, 9999))
                    postCanvas.setPen(Qc.Qt.black)
                    postCanvas.restore()

                    postCanvas.setTransform(selObj.getInteriorScrTransform(self.newTransform).toQTransform(), True)
                    postCanvas.drawRect(selObj.localBoundingBox)
                else:
                    postCanvas.setTransform(self.newTransform, True)
                    postCanvas.drawRect(self.currentBoundingBox)
                postCanvas.restore()
            if self.previewCurve is not None:
                postCanvas.drawPath(self.previewCurve)
            if self.addMode is not None:
                if self.addMode.active:
                    postCanvas.setPen(self.currentPen.toQPen())
                    postCanvas.drawPath(self.addMode.getPreview())

    def updateChecks(self):
        self.addMode = None
        if self.currentMode == SelectionMode.translate:
            activeBtn = self.ui.btnTranslate
        elif self.currentMode == SelectionMode.rotate:
            activeBtn = self.ui.btnRotate
        elif self.currentMode == SelectionMode.scale:
            activeBtn = self.ui.btnScale
        elif self.currentMode == SelectionMode.pan:
            activeBtn = self.ui.btnPan
        elif self.currentMode == SelectionMode.select:
            activeBtn = self.ui.btnSelect
        else:
            activeBtn = None

        for button in self.modeButtons:
            if button is not activeBtn:
                button.setChecked(False)
            else:
                button.setChecked(True)

    def btnAlignXOnClick(self, checked):
        self.lockY = checked
        if self.lockX:
            self.lockX = False
            self.ui.btnAlignY.setChecked(False)

    def btnAlignYOnClick(self, checked):
        self.lockX = checked
        if self.lockY:
            self.lockY = False
            self.ui.btnAlignX.setChecked(False)

    def btnTranslateonClick(self):
        self.currentMode = SelectionMode.translate
        self.ui.statusbar.showMessage('Translate Mode')
        self.clearSelection()
        self.updateChecks()

    def btnRotateOnClick(self):
        self.currentMode = SelectionMode.rotate
        self.ui.statusbar.showMessage('Rotate Mode')
        self.clearSelection()
        self.updateChecks()

    def btnScaleOnClick(self):
        self.currentMode = SelectionMode.scale
        self.ui.statusbar.showMessage('Scale Mode')
        self.clearSelection()
        self.updateChecks()

    def btnPanOnClick(self):
        self.currentMode = SelectionMode.pan
        self.clearSelection()
        self.updateChecks()

    def btnSelectOnClick(self):
        self.currentMode = SelectionMode.select
        self.updateChecks()

    def btnWorldCoordsOnClick(self, checked):
        self.useGlobalCoords = checked
        if not self.useGlobalCoords:
            self.ui.comboAnchor.setCurrentIndex(AnchorMode.origin)
        self.setAllInSetEnabled(self.globalTransformOnlyButtons, checked)

    def setAllInSetEnabled(self, widgetSet, enabled):
        for widget in widgetSet:
            widget.setEnabled(enabled)

    def btnDrawAxesOnClick(self, checked):
        self.drawAxes = checked
        self.quickUpdate()

    def btnDrawGridOnClick(self, checked):
        self.drawGrid = checked
        self.quickUpdate()

    def btnCustTransformOnClick(self):
        matrixDialog = CustMatTransform.CustMatTransform()
        matrixDialog.show()
        result = matrixDialog.exec_()
        if result == Qw.QDialog.Accepted:
            objKey = self.currentlySelectedObj['selectedKey']
            self.transformObject(objKey,
                matrixDialog.getTransformationMatrix(), not
                self.useGlobalCoords)

        # for now, unless we update the bouding box transformation.
        self.clearSelection()
        self.quickUpdate()

    def btnLoadEditorOnClick(self):
        rawExternalEditor = self.settings['externalEditor']
        rawExecEditor = rawExternalEditor.split(' ')
        execEditor = []
        for word in rawExecEditor:
            if word.startswith('*'):
                if word[1:] == 'ASYPATH':
                    execEditor.append('"' + self.filename + '"')
            else:
                execEditor.append(word)
        os.system(' '.join(execEditor))

    def transformObject(self, objKey, transform, applyFirst=False):
        drawObj = self.drawObjects[objKey]
        item, transfIndex = drawObj.originalObj
        key = drawObj.key

        if isinstance(transform, np.ndarray):
            obj_transform = x2a.asyTransform.fromNumpyMatrix(transform)
        elif isinstance(transform, Qg.QTransform):
            assert transform.isAffine()
            obj_transform = x2a.asyTransform.fromQTransform(transform)
        else:
            obj_transform = transform

        # oldTransf = item.transform[transfIndex]
        oldTransf = item.transfKeymap[key]

        if not applyFirst:
            # item.transform[transfIndex] = obj_transform * oldTransf
            # drawObj.transform = item.transform[transfIndex]
            item.transfKeymap[key] = obj_transform * oldTransf
            drawObj.transform = item.transfKeymap[key]

        else:
            item.transfKeymap[key] = oldTransf * obj_transform
            # item.transform[transfIndex] = oldTransf * obj_transform

        drawObj.transform = item.transfKeymap[key]
        self.quickUpdate()

    def initializeEmptyFile(self):
        pass

    def loadFile(self, name):
        self.ui.statusbar.showMessage(name)
        self.filename = os.path.abspath(name)
        x2a.startQuickAsy()
        # self.retitle()
        try:
            try:
                f = open(self.filename, 'rt')
            except:
                if self.filename[-4:] == ".asy":
                    raise
                else:
                    f = open(self.filename + ".asy", 'rt')
                    self.filename += ".asy"
                    self.retitle()
            self.fileItems = xf.parseFile(f)
            f.close()
        except IOError:
            Qw.QMessageBox.critical(self, "File Opening Failed.", "File could"
                                    "not be opened.")
            # messagebox.showerror("File Opening Failed.", "File could not be opened.")
            self.fileItems = []
        except Exception:
            self.fileItems = []
            self.autoMakeScript = True
            if self.autoMakeScript or Qw.QMessageBox.question(self, "Error Opening File", "File was not recognized as an xasy file.\n"
                "Load as a script item?") == Qw.QMessageBox.Yes:
                # try:
                item = x2a.xasyScript(canvas=self.xasyDrawObj)
                f.seek(0)
                item.setScript(f.read())
                # item.setKey()
                self.fileItems.append(item)
                # except:
                #     Qw.QMessageBox.critical(self, "File Opening Failed.",
                # "File could not be opened.")
                #     # messagebox.showerror("File Opening Failed.", "Could not
                # load as a script item.")
                #     self.fileItems = []
        # self.populateCanvasWithItems()
        # self.populatePropertyList()
        # self.updateCanvasSize()
        self.asyfyCanvas()

    def populateCanvasWithItems(self):
        # if (not self.testOrAcquireLock()):
        #     return
        self.itemCount = 0
        for item in self.fileItems:
            item.drawOnCanvas(self.xasyDrawObj, self.magnification, forceAddition=True)
            # self.bindItemEvents(item)
        # self.releaseLock()
