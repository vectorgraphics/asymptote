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
import subprocess
import tempfile
import datetime
import string
import uuid

import xasyUtils
import xasy2asy as x2a
import xasyFile as xf
import xasyOptions as xo
import UndoRedoStack as Urs
import xasyArgs as xa
from xasyTransform import xasyTransform as xT
import xasyStrings as xs

import PrimitiveShape
import InplaceAddObj

import CustMatTransform
import SetCustomAnchor
import BezierCurveEditor
import GuidesManager


class ActionChanges:
    pass


# State Invariance: When ActionChanges is at the top, all state of the program & file
# is exactly like what it was the event right after that ActionChanges was created.

class TransformationChanges(ActionChanges):
    def __init__(self, objIndex, transformation, isLocal=False):
        self.objIndex = objIndex
        self.transformation = transformation
        self.isLocal = isLocal


class ObjCreationChanges(ActionChanges):
    def __init__(self, obj):
        self.object = obj

class HardDeletionChanges(ActionChanges):
    def __init__(self, obj, pos):
        self.item = obj
        self.objIndex = pos

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
    delete = 5
    setAnchor = 6

class AddObjectMode:
    Circle = 0
    Arc = 1
    Polygon = 2

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

        self.settings = xo.BasicConfigs.defaultOpt
        self.keyMaps = xo.BasicConfigs.keymaps

        self.raw_args = Qc.QCoreApplication.arguments()
        self.args = xa.parseArgs(self.raw_args)

        self.strings = None

        if self.settings['asyBaseLocation'] is not None:
            os.environ['ASYMPTOTE_DIR'] = self.settings['asyBaseLocation']

        if self.args.asypath is not None:
            asyPath = self.args.asypath
        else:
            asyPath = self.settings['asyPath']

        self.asyPath = asyPath
        self.asyEngine = x2a.AsymptoteEngine(self.asyPath)
        self.asyEngine.start()

        # For initialization purposes
        self.canvSize = Qc.QSize()
        self.filename = None
        self.currDir = None
        self.mainCanvas = None
        self.canvasPixmap = None

        # Actions
        # <editor-fold> Connecting Actions
        self.ui.txtLineWidth.setValidator(Qg.QDoubleValidator())

        self.connectActions()
        self.connectButtons()

        self.ui.txtLineWidth.returnPressed.connect(self.btnTerminalCommandOnClick)
        # </editor-fold>

        # Base Transformations

        self.mainTransformation = Qg.QTransform()
        self.mainTransformation.scale(1, -1)
        self.localTransform = Qg.QTransform()
        self.screenTransformation = Qg.QTransform()
        self.panTranslation = Qg.QTransform()

        # Internal Settings
        self.magnification = self.args.mag
        self.inMidTransformation = False
        self.addMode = None
        self.currentlySelectedObj = {'key': None, 'allSameKey': set(), 'selectedIndex': None, 'keyIndex': None}
        self.pendingSelectedObjList = [] 
        self.pendingSelectedObjIndex = -1
        
        self.savedMousePosition = None
        self.currentBoundingBox = None
        self.selectionDelta = None
        self.newTransform = None
        self.origBboxTransform = None
        self.deltaAngle = 0
        self.scaleFactor = 1
        self.panOffset = 0, 0

        super().setMouseTracking(True)
        # setMouseTracking(True)
        
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

        self.terminalPythonMode = self.ui.btnTogglePython.isChecked()

        self.savedWindowMousePos = None

        self.finalPixmap = None
        self.preCanvasPixmap = None
        self.postCanvasPixmap = None
        self.previewCurve = None
        self.mouseDown = False

        self.globalObjectCounter = 0

        self.fileItems = []
        self.drawObjects = []
        self.xasyDrawObj = {'drawDict': self.drawObjects}

        self.modeButtons = {self.ui.btnTranslate, self.ui.btnRotate, self.ui.btnScale, # self.ui.btnSelect,
                            self.ui.btnPan, self.ui.btnDeleteMode, self.ui.btnAnchor}
        self.objButtons = {self.ui.btnCustTransform, self.ui.actionTransform, self.ui.btnSendForwards,
                           self.ui.btnSendBackwards, self.ui.btnToggleVisible
                           }

        self.globalTransformOnlyButtons = (self.ui.comboAnchor, self.ui.btnAnchor)

        self.ui.txtTerminalPrompt.setFont(Qg.QFont(self.settings['terminalFont']))

        self.currAddOptionsWgt = None
        self.currAddOptions = {
            'inscribed': True,
            'sides': 3,
            'centermode': True,
            'asyengine': self.asyEngine,
            'fill': self.ui.btnFill.isChecked(),
            'closedPath': False,
            'useBezier': True
        }
        self.currentModeStack = [SelectionMode.translate]
        self.drawGridMode = GridMode.cartesian
        self.setAllInSetEnabled(self.objButtons, False)
        self._currentPen = x2a.asyPen()
        self.currentGuides = []
        self.selectAsGroup = self.settings['groupObjDefault']

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
            'finalizeAddObj': self.finalizeAddObj,
            'finalizeCurve': self.finalizeCurve, 
            'finalizeCurveClosed': self.finalizeCurveClosed, 
            'setMag': self.setMagPrompt,
            'deleteObject': self.btnSelectiveDeleteOnClick, 
            'anchorMode': self.switchToAnchorMode
        }

        self.hiddenKeys = set()

        # Coordinates Label

        self.coordLabel = Qw.QLabel(self.ui.statusbar)
        self.ui.statusbar.addPermanentWidget(self.coordLabel)

        # Settings Initialization
        # from xasyoptions config file
        self.loadKeyMaps()
        self.setupXasyOptions()

        self.colorDialog = Qw.QColorDialog(x2a.asyPen.convertToQColor(self._currentPen.color), self)
        self.initPenInterface()

    def finalizeCurve(self):
        if self.addMode is not None:
            if self.addMode.active and isinstance(self.addMode, InplaceAddObj.AddBezierShape):
                self.addMode.forceFinalize()

    def finalizeCurveClosed(self):
        if self.addMode is not None:
            if self.addMode.active and isinstance(self.addMode, InplaceAddObj.AddBezierShape):
                self.addMode.finalizeClosure()

    def cancelCurve(self):
        if self.addMode is not None:
            if self.addMode.active and isinstance(self.addMode, InplaceAddObj.AddBezierShape):
                self.addMode.active = False


    def finalizeAddObj(self):
        if self.addMode is not None:
            if self.addMode.active:
                self.addMode.forceFinalize()

    def setMagPrompt(self):
        commandText, result = Qw.QInputDialog.getText(self, '', 'Enter magnification:')
        if result:
            self.magnification = float(commandText)
            self.asyfyCanvas(True)

    def btnTogglePythonOnClick(self, checked):
        self.terminalPythonMode = checked

    def internationalize(self):
        self.ui.btnRotate.setToolTip(self.strings.rotate)

    def handleArguments(self):
        if self.args.filename is not None:
            self.loadFile(self.args.filename)
        else:
            self.initializeEmptyFile()

        if self.args.language is not None:
            self.strings = xs.xasyString(self.args.language)
            self.internationalize()

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
            'debug:addLabel': self.debugAddLabel,
            'debug:addFillCircle': self.dbgAddFillCirc
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

        self.ui.btnExportAsy.clicked.connect(self.btnExportAsyOnClick)

        self.ui.btnDrawAxes.clicked.connect(self.btnDrawAxesOnClick)
        self.ui.btnAsyfy.clicked.connect(lambda: self.asyfyCanvas(True))
        self.ui.btnSetZoom.clicked.connect(self.setMagPrompt)

        self.ui.btnTranslate.clicked.connect(self.btnTranslateonClick)
        self.ui.btnRotate.clicked.connect(self.btnRotateOnClick)
        self.ui.btnScale.clicked.connect(self.btnScaleOnClick)
        # self.ui.btnSelect.clicked.connect(self.btnSelectOnClick)
        self.ui.btnPan.clicked.connect(self.btnPanOnClick)

        # self.ui.btnDebug.clicked.connect(self.pauseBtnOnClick)
        self.ui.btnAlignX.clicked.connect(self.btnAlignXOnClick)
        self.ui.btnAlignY.clicked.connect(self.btnAlignYOnClick)
        self.ui.comboAnchor.currentTextChanged.connect(self.handleAnchorCombo)
        self.ui.btnWorldCoords.clicked.connect(self.btnWorldCoordsOnClick)

        self.ui.btnCustTransform.clicked.connect(self.btnCustTransformOnClick)
        self.ui.btnViewCode.clicked.connect(self.btnLoadEditorOnClick)

        self.ui.btnAnchor.clicked.connect(self.btnAnchorModeOnClick)

        self.ui.btnSelectColor.clicked.connect(self.btnColorSelectOnClick)
        self.ui.txtLineWidth.textEdited.connect(self.txtLineWithEdited)

        # self.ui.btnCreateCurve.clicked.connect(self.btnCreateCurveOnClick)
        self.ui.btnDrawGrid.clicked.connect(self.btnDrawGridOnClick)

        self.ui.btnAddCircle.clicked.connect(self.btnAddCircleOnClick)
        self.ui.btnAddPoly.clicked.connect(self.btnAddPolyOnClick)
        self.ui.btnAddLabel.clicked.connect(self.btnAddLabelOnClick)
        self.ui.btnAddBezierInplace.clicked.connect(self.btnAddBezierInplaceOnClick)
        self.ui.btnAddCode.clicked.connect(self.btnAddCodeOnClick)

        self.ui.btnFill.clicked.connect(self.btnFillOnClick)

        self.ui.btnSendBackwards.clicked.connect(self.btnSendBackwardsOnClick)
        self.ui.btnSendForwards.clicked.connect(self.btnSendForwardsOnClick)
        # self.ui.btnDelete.clicked.connect(self.btnSelectiveDeleteOnClick)
        self.ui.btnDeleteMode.clicked.connect(self.btnDeleteModeOnClick)
        # self.ui.btnSoftDelete.clicked.connect(self.btnSoftDeleteOnClick)
        self.ui.btnToggleVisible.clicked.connect(self.btnSetVisibilityOnClick)
        
        self.ui.btnEnterCommand.clicked.connect(self.btnTerminalCommandOnClick)
        self.ui.btnTogglePython.clicked.connect(self.btnTogglePythonOnClick)

    def btnDeleteModeOnClick(self):
        self.currentModeStack = [SelectionMode.delete]
        self.ui.statusbar.showMessage('Delete Mode')
        self.clearSelection()
        self.updateChecks()

    def btnTerminalCommandOnClick(self):
        if self.terminalPythonMode:
            exec(self.ui.txtTerminalPrompt.text())
        else:
            print('What to do?')
            # TODO: How to handle this case?
            # Like AutoCAD? 
        self.ui.txtTerminalPrompt.clear()

    def btnFillOnClick(self, checked):
        self.currAddOptions['fill'] = checked

    @property
    def currentPen(self):
        return x2a.asyPen.fromAsyPen(self._currentPen)

    def dbgAddUnitCircle(self):
        newCirclePath = PrimitiveShape.PrimitiveShape.circle((0, 0), 1)
        newCircle = x2a.xasyShape(newCirclePath, asyengine=self.asyEngine)
        self.fileItems.append(newCircle)
        self.asyfyCanvas()

    def dbgAddFillCirc(self):
        newCirclePath = PrimitiveShape.PrimitiveShape.circle((0, 0), 100)
        newCircle = x2a.xasyFilledShape(newCirclePath, pen=self.currentPen, asyengine=self.asyEngine)
        self.fileItems.append(newCircle)
        self.asyfyCanvas()

    def dbgAddCircle(self):
        commandText, result = Qw.QInputDialog.getText(self, '', 'enter python cmd')
        if result:
            rawArray = [float(rawResult) for rawResult in commandText.split()]
            x, y, rad = rawArray
            newCirclePath = PrimitiveShape.PrimitiveShape.circle((x, y), rad)
            newCircle = x2a.xasyShape(newCirclePath, pen=self.currentPen, asyengine=self.asyEngine)
            self.fileItems.append(newCircle)
            self.asyfyCanvas()

    def dbgAddPoly(self):
        newSquarePath = PrimitiveShape.PrimitiveShape.exscribedRegPolygon(6, (0, 0), 100, 0)
        newSquare = x2a.xasyShape(newSquarePath, pen=self.currentPen, asyengine=self.asyEngine)
        self.fileItems.append(newSquare)
        self.asyfyCanvas()

    def debugAddLabel(self):
        testText = '$\\displaystyle{\\int_{\\varphi(F)} f = \\int_F (f \\circ \\varphi) \\left| \\det J_{\\varphi} \\right|}$'
        newPath = x2a.xasyText(testText, (0, 0), asyengine=self.asyEngine)
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
        try:
            self.addMode.objectCreated.disconnect()
        except Exception:
            pass

        self.addMode.objectCreated.connect(self.addInPlace)
        if self.currAddOptionsWgt is not None:
            self.currAddOptionsWgt.hide()
            self.ui.addOptionLayout.removeWidget(self.currAddOptionsWgt)
            self.currAddOptionsWgt = None

        self.currAddOptionsWgt = self.addMode.createOptWidget(self.currAddOptions)
        if self.currAddOptionsWgt is not None:
            self.ui.addOptionLayout.addWidget(self.currAddOptionsWgt)

    def addInPlace(self, obj):
        obj.asyengine = self.asyEngine
        obj.pen = self.currentPen
        obj.onCanvas = self.xasyDrawObj
        obj.setKey('x' + str(self.globalObjectCounter))
        self.globalObjectCounter = self.globalObjectCounter + 1

        if self.magnification != 1:
            assert self.magnification != 0
            # Convert screen level to asy coordinate level
            obj.asyfied = False

        self.fileItems.append(obj)
        self.asyfyCanvas()

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
            asyCurve = x2a.asyPath.fromBezierPoints(curveDialog.createPointList(), engine=self.asyEngine)
            newXasyObjCurve = x2a.xasyShape(asyCurve, asyengine=self.asyEngine)
            # print(newXasyObjCurve.getCode())
            self.fileItems.append(newXasyObjCurve)
        self.inCurveCreationMode = False
        self.previewCurve = None
        self.asyfyCanvas()

    def btnAddCircleOnClick(self):
        self.addMode = InplaceAddObj.AddCircle(self)
        self.updateOptionWidget()

    def btnAddBezierInplaceOnClick(self):
        self.addMode = InplaceAddObj.AddBezierShape(self)
        self.updateOptionWidget()

    def btnAddPolyOnClick(self):
        self.addMode = InplaceAddObj.AddPoly(self)
        self.updateOptionWidget()

    def btnAddLabelOnClick(self):
        self.addMode = InplaceAddObj.AddLabel(self)
        self.updateOptionWidget()

    def updateCurve(self, valid, newCurve):
        self.previewCurve = newCurve
        self.quickUpdate()

    def addTransformationChanges(self, objIndex, transform, isLocal=False):
        self.undoRedoStack.add(self.createAction(TransformationChanges(objIndex, 
                            transform, isLocal)))
        self.checkUndoRedoButtons()

    def btnSendForwardsOnClick(self):
        if self.currentlySelectedObj['selectedIndex'] is not None:
            maj, minor = self.currentlySelectedObj['selectedIndex']
            selectedObj = self.drawObjects[maj][minor]
            index = self.fileItems.index(selectedObj.parent())

            self.clearSelection()
            if index == len(self.fileItems) - 1:
                return
            else:
                self.fileItems[index], self.fileItems[index + 1] = self.fileItems[index + 1], self.fileItems[index]
                self.asyfyCanvas()

    def btnSelectiveDeleteOnClick(self):
        if self.currentlySelectedObj['selectedIndex'] is not None:
            maj, minor = self.currentlySelectedObj['selectedIndex']
            selectedObj = self.drawObjects[maj][minor]

            parent = selectedObj.parent()

            if isinstance(parent, x2a.xasyScript):
                self.hiddenKeys.add((selectedObj.key, selectedObj.keyIndex))
                self.softDeleteObj((maj, minor))
            else:
                index = self.fileItems.index(selectedObj.parent())

                self.undoRedoStack.add(self.createAction(
                    HardDeletionChanges(selectedObj.parent(), index)
                ))
                self.checkUndoRedoButtons()
                
                self.fileItems.remove(selectedObj.parent())

            self.clearSelection()
            self.asyfyCanvas()
        else:
            result = self.selectOnHover()
            if result:
                self.btnSelectiveDeleteOnClick()

    def btnSetVisibilityOnClick(self):
        if self.currentlySelectedObj['selectedIndex'] is not None:
            maj, minor = self.currentlySelectedObj['selectedIndex']
            selectedObj = self.drawObjects[maj][minor]

            self.hiddenKeys.symmetric_difference_update({(selectedObj.key, selectedObj.keyIndex)})
            self.clearSelection()
            self.quickUpdate()

    def btnSendBackwardsOnClick(self):
        if self.currentlySelectedObj['selectedIndex'] is not None:
            maj, minor = self.currentlySelectedObj['selectedIndex']
            selectedObj = self.drawObjects[maj][minor]
            index = self.fileItems.index(selectedObj.parent())

            self.clearSelection()
            if index == 0:
                return
            else:
                self.fileItems[index], self.fileItems[index - 1] = self.fileItems[index - 1], self.fileItems[index]
                self.asyfyCanvas()


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
            self.transformObject(change.objIndex, change.transformation.inverted(), change.isLocal)
        elif isinstance(change, ObjCreationChanges):
            pass  # for now, until we implement a remove object/add object. This will be trivial
        elif isinstance(change, HardDeletionChanges):
            self.fileItems.insert(change.objIndex, change.item)
        self.asyfyCanvas()

    def handleRedoChanges(self, change):
        assert isinstance(change, ActionChanges)
        if isinstance(change, TransformationChanges):
            self.transformObject(
                 change.objIndex, change.transformation, change.isLocal)
        elif isinstance(change, ObjCreationChanges):
            pass  # for now, until we implement a remove/add method. By then, this will be trivial.
        elif isinstance(change, HardDeletionChanges):
            self.fileItems.remove(change.item)
        self.asyfyCanvas()

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
        newItem = x2a.xasyShape(path, self.asyEngine, pen=self.currentPen)
        self.fileItems.append(newItem)
        self.asyfyCanvas()

    def actionManual(self):
        asyManualURL = 'http://asymptote.sourceforge.net/asymptote.pdf'
        webbrowser.open_new(asyManualURL)

    def btnExportAsyOnClick(self):
        diag = Qw.QFileDialog(self)
        diag.setAcceptMode(Qw.QFileDialog.AcceptSave)

        formatId = {
            'pdf': {
                'name': 'PDF Files',
                'ext': ['*.pdf']
            },
            'svg': {
                'name': 'Scalable Vector Graphics',
                'ext': ['*.svg']
            },
            'eps': {
                'name': 'Postscript Files',
                'ext': ['*.eps']
            },
            'png': {
                'name': 'Portable Network Graphics',
                'ext': ['*.png']
            },
            '*': {
                'name': 'Any Files',
                'ext': ['*.*']
            }
        }

        formats = ['pdf', 'svg', 'eps', 'png', '*']

        formatText = ';;'.join('{0:s} ({1:s})'.format(formatId[form]['name'], ' '.join(formatId[form]['ext']))
                               for form in formats)

        if self.currDir is not None:
            diag.setDirectory(self.currDir)
            rawFile = os.path.splitext(os.path.basename(self.filename))[0] + '.pdf'
            diag.selectFile(rawFile)

        diag.setNameFilter(formatText)
        diag.show()
        result = diag.exec_()

        if result != diag.Accepted:
            return

        finalFiles = diag.selectedFiles()

        with io.StringIO() as finalCode:
            xf.saveFile(finalCode, self.fileItems)
            finalString = finalCode.getvalue()

        for file in finalFiles:
            ext = os.path.splitext(file)
            if len(ext) < 2:
                ext = 'pdf'
            else:
                ext = ext[1][1:]

            with subprocess.Popen(args=[self.asyPath, '-f{0}'.format(ext), '-o{0}'.format(file), '-'], encoding='utf-8',
                                  stdin=subprocess.PIPE) as asy:
                print('test:', finalString)
                asy.stdin.write(finalString)
                asy.stdin.close()
                asy.wait(timeout=35)


    def loadKeyMaps(self):
        """Inverts the mapping of the key
           Input map is in format 'Action' : 'Key Sequence' """
        for action, key in self.keyMaps.options.items():
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
                result = self.btnCustomAnchorOnClick()
                if not result:
                    self.ui.comboAnchor.setCurrentIndex(0)
                    self.anchorMode = AnchorMode.origin
                else:
                    self.anchorMode = AnchorMode.customAnchor

    def btnCustomAnchorOnClick(self, text=''):
        custAnchorDialog = SetCustomAnchor.CustomAnchorDialog()
        custAnchorDialog.show()
        result = custAnchorDialog.exec()
        if result == Qw.QDialog.Accepted:
            self.customAnchor = custAnchorDialog.getPoint()
            self.ui.comboAnchor.setCurrentText('Custom Anchor')
            return True
        else:
            return False

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
        # super().resizeEvent(resizeEvent)
        assert isinstance(resizeEvent, Qg.QResizeEvent)

        if self.isReady():
            self.canvSize = self.ui.imgFrame.size()
            self.ui.imgFrame.setSizePolicy(Qw.QSizePolicy.Ignored, Qw.QSizePolicy.Ignored)
            x, y = self.canvSize.width() / 2, self.canvSize.height() / 2

            self.canvasPixmap = Qg.QPixmap(self.canvSize)
            self.postCanvasPixmap = Qg.QPixmap(self.canvSize)

            self.mainTransformation.reset()
            self.mainTransformation.scale(1, -1)
            self.mainTransformation.translate(x, -y)

            self.screenTransformation = self.mainTransformation * self.panTranslation

            self.mainCanvas.setTransform(self.screenTransformation, True)
            # self.createMainCanvas()
            self.quickUpdate()

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

    def getAsyCoordinates(self):
        canvasPosOrig = self.getCanvasCoordinates()
        if self.magnification != 1:
            assert self.magnification != 0
            invmag = 1 / self.magnification
            canvasPos = canvasPosOrig * Qg.QTransform.fromScale(invmag, invmag)
            return canvasPos, canvasPosOrig
        else:
            return canvasPosOrig, canvasPosOrig

    def mouseMoveEvent(self, mouseEvent):  # TODO: Actually refine grid snapping...
        assert isinstance(mouseEvent, Qg.QMouseEvent)

        if not self.ui.imgLabel.underMouse() and not self.mouseDown:
            return 

        asyPos, canvasPos = self.getAsyCoordinates()

        self.coordLabel.setText('{0:d}, {1:d}    '.format(
            canvasPos.x(), canvasPos.y()))

        # add mode 
        if self.addMode is not None:
            if self.addMode.active:
                self.addMode.mouseMove(asyPos, mouseEvent)
                self.quickUpdate()
            return

        # pan mode
        if self.currentModeStack[-1] == SelectionMode.pan and int(mouseEvent.buttons()) and self.savedWindowMousePos is not None:
            mousePos = self.getWindowCoordinates()
            newPos = mousePos - self.savedWindowMousePos
            tx, ty = newPos.x(), newPos.y()
            if self.lockX:
                tx = 0
            if self.lockY:
                ty = 0
            self.panTranslation = Qg.QTransform.fromTranslate(tx, ty)
            self.screenTransformation = self.currScreenTransform * self.panTranslation
            self.quickUpdate()
            return

        # otherwise, in transformation 
        if self.inMidTransformation:
            if self.currentModeStack[-1] == SelectionMode.translate:
                newPos = canvasPos - self.savedMousePosition
                if self.gridSnap:
                    newPos = self.roundPositionSnap(newPos)  # actually round to the nearest minor grid afterwards...

                self.tx, self.ty = newPos.x(), newPos.y()

                if self.lockX:
                    self.tx = 0
                if self.lockY:
                    self.ty = 0
                self.newTransform = Qg.QTransform.fromTranslate(self.tx, self.ty)

            elif self.currentModeStack[-1] == SelectionMode.rotate:
                if self.gridSnap:
                    canvasPos = self.roundPositionSnap(canvasPos)

                adjustedSavedMousePos = self.savedMousePosition - self.currentAnchor
                adjustedCanvasCoords = canvasPos - self.currentAnchor

                origAngle = np.arctan2(adjustedSavedMousePos.y(), adjustedSavedMousePos.x())
                newAng = np.arctan2(adjustedCanvasCoords.y(), adjustedCanvasCoords.x())
                self.deltaAngle = newAng - origAngle
                self.newTransform = xT.makeRotTransform(self.deltaAngle, self.currentAnchor).toQTransform()

            elif self.currentModeStack[-1] == SelectionMode.scale:
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
            return

        # otherwise, select a candinate for selection

        if self.currentlySelectedObj['selectedIndex'] is None:
            selectedIndex, selKeyList = self.selectObject()
            if selectedIndex is not None:
                if self.pendingSelectedObjList != selKeyList:
                    self.pendingSelectedObjList = selKeyList
                    self.pendingSelectedObjIndex = -1
            else:
                self.pendingSelectedObjList.clear()
                self.pendingSelectedObjIndex = -1
            self.quickUpdate()
            return 


    def mouseReleaseEvent(self, mouseEvent):
        assert isinstance(mouseEvent, Qg.QMouseEvent) 
        if not self.mouseDown:
            return

        self.mouseDown = False
        if self.addMode is not None:
            self.addMode.mouseRelease()
        if self.inMidTransformation:
            self.clearSelection()
        self.inMidTransformation = False
        self.quickUpdate()

    def clearSelection(self):
        if self.currentlySelectedObj['selectedIndex'] is not None:
            self.releaseTransform()
        self.setAllInSetEnabled(self.objButtons, False)
        self.currentlySelectedObj['selectedIndex'] = None
        self.currentlySelectedObj['key'] = None

        self.currentlySelectedObj['allSameKey'].clear()
        self.newTransform = Qg.QTransform()
        self.currentBoundingBox = None
        self.quickUpdate()

    def changeSelection(self, offset):
        if self.pendingSelectedObjList:
            if offset > 0:
                if self.pendingSelectedObjIndex + offset <= -1:
                    self.pendingSelectedObjIndex = self.pendingSelectedObjIndex + offset
            else:
                if self.pendingSelectedObjIndex + offset >= -len(self.pendingSelectedObjList):
                    self.pendingSelectedObjIndex = self.pendingSelectedObjIndex + offset
            self.quickUpdate()
    
    def wheelEvent(self, event):
        assert isinstance(event, Qg.QWheelEvent)
        rawAngle = event.angleDelta().y() / 8

        if rawAngle >= 15:
            self.changeSelection(1)
        elif rawAngle <= -15:
            self.changeSelection(-1)

    def selectOnHover(self):
        """Returns True if selection happened, False otherwise.
        """
        if self.pendingSelectedObjList:
            selectedIndex = self.pendingSelectedObjList[self.pendingSelectedObjIndex]
            self.pendingSelectedObjList.clear()

            maj, minor = selectedIndex

            self.currentlySelectedObj['selectedIndex'] = selectedIndex
            self.currentlySelectedObj['key'],  self.currentlySelectedObj['allSameKey'] = self.selectObjectSet(
            )

            self.currentBoundingBox = self.drawObjects[maj][minor].boundingBox

            if self.selectAsGroup:
                for selItems in self.currentlySelectedObj['allSameKey']:
                    obj = self.drawObjects[selItems[0]][selItems[1]]
                    self.currentBoundingBox = self.currentBoundingBox.united(obj.boundingBox)

            self.origBboxTransform = self.drawObjects[maj][minor].transform.toQTransform()
            self.newTransform = Qg.QTransform()
            return True
        else:
            return False

    def mousePressEvent(self, mouseEvent):
        # we make an exception for bezier curve
        bezierException = False
        if self.addMode is not None:
            if self.addMode.active and isinstance(self.addMode, InplaceAddObj.AddBezierShape):
                bezierException = True
                
        if not self.ui.imgLabel.underMouse() and not bezierException:
            return

        self.mouseDown = True
        asyPos, self.savedMousePosition = self.getAsyCoordinates()

        if self.addMode is not None:
            self.addMode.mouseDown(asyPos, self.currAddOptions)
            return

        if self.currentModeStack[-1] == SelectionMode.pan:
            self.savedWindowMousePos = self.getWindowCoordinates()
            self.currScreenTransform = self.screenTransformation * Qg.QTransform()
            return

        if self.currentModeStack[-1] == SelectionMode.setAnchor:
            self.customAnchor = self.savedMousePosition
            self.currentModeStack.pop()
            self.updateChecks()
            self.quickUpdate()
            return

        if self.inMidTransformation:
            return

        if self.pendingSelectedObjList:
            self.selectOnHover()

            if self.currentModeStack[-1] in {SelectionMode.translate, SelectionMode.rotate, SelectionMode.scale}:
                self.setAllInSetEnabled(self.objButtons, False)
                self.inMidTransformation = True
            elif self.currentModeStack[-1] == SelectionMode.delete:
                self.btnSelectiveDeleteOnClick()
                return
            else:
                self.setAllInSetEnabled(self.objButtons, True)
                self.inMidTransformation = False

            if self.anchorMode == AnchorMode.center:
                self.currentAnchor = self.currentBoundingBox.center()
            elif self.anchorMode == AnchorMode.topLeft:
                self.currentAnchor = self.currentBoundingBox.bottomLeft()  # due to internal image being flipped
            elif self.anchorMode == AnchorMode.topRight:
                self.currentAnchor = self.currentBoundingBox.bottomRight()
            elif self.anchorMode == AnchorMode.bottomLeft:
                self.currentAnchor = self.currentBoundingBox.topLeft()
            elif self.anchorMode == AnchorMode.bottomRight:
                self.currentAnchor = self.currentBoundingBox.topRight()
            elif self.anchorMode == AnchorMode.customAnchor:
                self.currentAnchor = self.customAnchor
            else:
                self.currentAnchor = Qc.QPointF(0, 0)

            if self.anchorMode != AnchorMode.origin:
                pass
                # TODO: Record base points/bbox before hand and use that for
                # anchor?
                # adjTransform =
                # self.drawObjects[selectedIndex].transform.toQTransform()
                # self.currentAnchor = adjTransform.map(self.currentAnchor)

        else:
            self.setAllInSetEnabled(self.objButtons, False)
            self.currentBoundingBox = None
            self.inMidTransformation = False
            self.clearSelection()
        self.quickUpdate()

    def releaseTransform(self):
        if self.newTransform.isIdentity():
            return
        newTransform = x2a.asyTransform.fromQTransform(self.newTransform)
        objKey = self.currentlySelectedObj['selectedIndex']
        self.addTransformationChanges(objKey, newTransform, not self.useGlobalCoords)
        self.transformObject(objKey, newTransform, not self.useGlobalCoords)

    def adjustTransform(self, appendTransform):
        self.screenTransformation = self.screenTransformation * appendTransform

    def createMainCanvas(self):
        self.canvSize = self.ui.imgFrame.size()
        self.ui.imgFrame.setSizePolicy(Qw.QSizePolicy.Ignored, Qw.QSizePolicy.Ignored)
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
            return None, []
        canvasCoords = self.getCanvasCoordinates()
        highestDrawPriority = -np.inf
        collidedObjKey = None
        rawObjNumList = []
        for objKeyMaj in range(len(self.drawObjects)):
            for objKeyMin in range(len(self.drawObjects[objKeyMaj])):
                obj = self.drawObjects[objKeyMaj][objKeyMin]
                if obj.collide(canvasCoords) and (obj.key, obj.keyIndex) not in self.hiddenKeys:
                    rawObjNumList.append(((objKeyMaj, objKeyMin), obj.drawOrder))
                    if obj.drawOrder > highestDrawPriority:
                        collidedObjKey = (objKeyMaj, objKeyMin)
        if collidedObjKey is not None:
            rawKey = self.drawObjects[collidedObjKey[0]][collidedObjKey[1]].key
            self.ui.statusbar.showMessage('Collide with {0}, Key is {1}.'.format(str(collidedObjKey), rawKey), 2500)
            return collidedObjKey, [rawObj[0] for rawObj in sorted(rawObjNumList, key=lambda ordobj: ordobj[1])]
        else:
            return None, []

    def selectObjectSet(self):
        objKey = self.currentlySelectedObj['selectedIndex']
        if objKey is None:
            return set()
        assert isinstance(objKey, (tuple, list)) and len(objKey) == 2
        rawObj = self.drawObjects[objKey[0]][objKey[1]]
        rawKey = rawObj.key
        rawSet = {objKey}
        for objKeyMaj in range(len(self.drawObjects)):
            for objKeyMin in range(len(self.drawObjects[objKeyMaj])):
                obj = self.drawObjects[objKeyMaj][objKeyMin]
                if obj.key == rawKey:
                    rawSet.add((objKeyMaj, objKeyMin))
        return rawKey, rawSet

    def getCanvasCoordinates(self):
        # assert self.ui.imgLabel.underMouse()
        uiPos = self.mapFromGlobal(Qg.QCursor.pos())
        canvasPos = self.ui.imgLabel.mapFrom(self, uiPos)

        # Issue: For magnification, should xasy treats this at xasy level, or asy level?
        return canvasPos * self.screenTransformation.inverted()[0]

    def getWindowCoordinates(self):
        # assert self.ui.imgLabel.underMouse()
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

    def asyfyCanvas(self, force=False):
        self.drawObjects = []
        self.populateCanvasWithItems(force)
        self.quickUpdate()

    def quickUpdate(self):
        self.refreshCanvas()

        self.preDraw(self.mainCanvas)
        self.quickDraw()
        self.mainCanvas.end()

        self.postDraw()
        self.updateScreen()

    def quickDraw(self):
        assert self.isReady()

        activeItem = None
        for majorItem in self.drawObjects:
            for item in majorItem:
                # hidden objects - toggleable
                if (item.key, item.keyIndex) in self.hiddenKeys:
                    continue
                isSelected = item.key == self.currentlySelectedObj['key']
                if not self.selectAsGroup and isSelected and self.currentlySelectedObj['selectedIndex'] is not None:
                    maj, min_ = self.currentlySelectedObj['selectedIndex']
                    isSelected = isSelected and item is self.drawObjects[maj][min_]
                if isSelected and self.settings['enableImmediatePreview']:
                    activeItem = item
                    if self.useGlobalCoords:
                        item.draw(self.newTransform, canvas=self.mainCanvas)
                    else:
                        item.draw(self.newTransform, applyReverse=True, canvas=self.mainCanvas)
                else:
                    item.draw(canvas=self.mainCanvas)

        if self.settings['drawSelectedOnTop']:
            if self.pendingSelectedObjList:
                maj, minor = self.pendingSelectedObjList[self.pendingSelectedObjIndex]
                self.drawObjects[maj][minor].draw(canvas=self.mainCanvas)
            # and apply the preview too... 
            elif activeItem is not None:
                if self.useGlobalCoords:
                    activeItem.draw(self.newTransform, canvas=self.mainCanvas)
                else:
                    activeItem.draw(self.newTransform, applyReverse=True, canvas=self.mainCanvas)
                activeItem = None

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

    def drawAddModePreview(self, painter):
        if self.addMode is not None:
            if self.addMode.active and self.addMode.getPreview() is not None:
                if self.magnification != 1:
                    assert self.magnification != 0
                    painter.save()
                    painter.scale(self.magnification, self.magnification)
                painter.setPen(self.currentPen.toQPen())
                painter.drawPath(self.addMode.getPreview())
                if self.magnification != 1:
                    painter.restore()

    def drawTransformPreview(self, painter):
        if self.currentBoundingBox is not None and self.currentlySelectedObj['selectedIndex'] is not None:
            painter.save()
            maj, minor = self.currentlySelectedObj['selectedIndex']
            selObj = self.drawObjects[maj][minor]
            if not self.useGlobalCoords:
                painter.save()
                painter.setTransform(
                    selObj.transform.toQTransform(), True)
                # painter.setTransform(selObj.baseTransform.toQTransform(), True)
                painter.setPen(Qc.Qt.gray)
                painter.drawLine(Qc.QLine(-9999, 0, 9999, 0))
                painter.drawLine(Qc.QLine(0, -9999, 0, 9999))
                painter.setPen(Qc.Qt.black)
                painter.restore()

                painter.setTransform(selObj.getInteriorScrTransform(
                    self.newTransform).toQTransform(), True)
                painter.drawRect(selObj.localBoundingBox)
            else:
                painter.setTransform(self.newTransform, True)
                painter.drawRect(self.currentBoundingBox)
            painter.restore()

    def postDraw(self):
        self.postCanvasPixmap.fill(Qc.Qt.transparent)
        with Qg.QPainter(self.postCanvasPixmap) as postCanvas:
            postCanvas.setTransform(self.screenTransformation)

            self.drawTransformPreview(postCanvas)

            if self.pendingSelectedObjList:
                maj, minor = self.pendingSelectedObjList[self.pendingSelectedObjIndex]
                postCanvas.drawRect(self.drawObjects[maj][minor].boundingBox)
            if self.previewCurve is not None:
                postCanvas.drawPath(self.previewCurve)
            self.drawAddModePreview(postCanvas)

            if self.customAnchor is not None:
                self.drawAnchorCursor(postCanvas)

    def drawAnchorCursor(self, painter):
        painter.drawEllipse(self.customAnchor, 6, 6)
        newCirclePath = Qg.QPainterPath()
        newCirclePath.addEllipse(self.customAnchor, 2, 2)

        painter.fillPath(newCirclePath, Qg.QColor.fromRgb(0, 0, 0))

    def updateChecks(self):
        self.addMode = None
        if self.currentModeStack[-1] == SelectionMode.translate:
            activeBtn = self.ui.btnTranslate
        elif self.currentModeStack[-1] == SelectionMode.rotate:
            activeBtn = self.ui.btnRotate
        elif self.currentModeStack[-1] == SelectionMode.scale:
            activeBtn = self.ui.btnScale
        elif self.currentModeStack[-1] == SelectionMode.pan:
            activeBtn = self.ui.btnPan
        elif self.currentModeStack[-1] == SelectionMode.setAnchor:
            activeBtn = self.ui.btnAnchor
        elif self.currentModeStack[-1] == SelectionMode.delete:
            activeBtn = self.ui.btnDeleteMode
        # elif self.currentModeStack[-1] == SelectionMode.select:
            # activeBtn = self.ui.btnSelect
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

    def btnAnchorModeOnClick(self):
        if self.currentModeStack[-1] != SelectionMode.setAnchor:
            self.currentModeStack.append(SelectionMode.setAnchor)
            self.updateChecks()

    def switchToAnchorMode(self):
        if self.currentModeStack[-1] != SelectionMode.setAnchor:
            self.currentModeStack.append(SelectionMode.setAnchor)
            self.updateChecks()

    def btnTranslateonClick(self):
        self.currentModeStack = [SelectionMode.translate]
        self.ui.statusbar.showMessage('Translate Mode')
        self.clearSelection()
        self.updateChecks()

    def btnRotateOnClick(self):
        self.currentModeStack = [SelectionMode.rotate]
        self.ui.statusbar.showMessage('Rotate Mode')
        self.clearSelection()
        self.updateChecks()

    def btnScaleOnClick(self):
        self.currentModeStack = [SelectionMode.scale]
        self.ui.statusbar.showMessage('Scale Mode')
        self.clearSelection()
        self.updateChecks()

    def btnPanOnClick(self):
        self.currentModeStack = [SelectionMode.pan]
        self.clearSelection()
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
            objKey = self.currentlySelectedObj['selectedIndex']
            self.transformObject(objKey,
                matrixDialog.getTransformationMatrix(), not
                self.useGlobalCoords)

        # for now, unless we update the bouding box transformation.
        self.clearSelection()
        self.quickUpdate()

    def btnLoadEditorOnClick(self):
        rawExternalEditor = self.settings['externalEditor']
        rawExtEditorArgs = self.settings['externalEditorArgs']
        execEditor = [rawExternalEditor]

        for arg in rawExtEditorArgs:
            execEditor.append(string.Template(
                arg).substitute(asypath=(self.filename)))

        subprocess.Popen(args=execEditor)

    def btnAddCodeOnClick(self):
        header = """
// xasy object created at $time 
// Object Number: $uid
// This header is automatically generated by xasy. 
// Your code here
"""
        header = string.Template(header).substitute(time=str(datetime.datetime.now()), uid=str(self.globalObjectCounter))

        rawExternalEditor = self.settings['externalEditor']
        rawExtEditorArgs = self.settings['externalEditorArgs']
        execEditor = [rawExternalEditor]

        with tempfile.TemporaryDirectory() as tmpdir:
            newPath = os.path.join(tmpdir, 'tmpcode.asy')

            for arg in rawExtEditorArgs:
                execEditor.append(string.Template(arg).substitute(asypath=newPath))

            f = io.open(newPath, 'w')
            f.write(header)
            f.close()

            subprocess.run(args=execEditor)

            f = io.open(newPath, 'r')
            newItem = x2a.xasyScript(engine=self.asyEngine, canvas=self.xasyDrawObj)
            newItem.setScript(f.read())
            f.close()

        newItem.setKey(str(self.globalObjectCounter) + ':')
        self.fileItems.append(newItem)
        self.asyfyCanvas()

        self.globalObjectCounter = self.globalObjectCounter + 1
    def softDeleteObj(self, objKey):
        maj, minor = objKey
        drawObj = self.drawObjects[maj][minor]
        item = drawObj.originalObj
        key = drawObj.key
        keyIndex = drawObj.keyIndex


        item.transfKeymap[key][keyIndex].deleted = True
        # item.asyfied = False

    def getSelectedObjInfo(self, objIndex):
        maj, minor = objIndex
        drawObj = self.drawObjects[maj][minor]
        item = drawObj.originalObj
        key = drawObj.key
        keyIndex = drawObj.keyIndex

        return item, key, keyIndex

    def transformObjKey(self, item, key, keyIndex, transform, applyFirst=False, drawObj=None):
        if isinstance(transform, np.ndarray):
            obj_transform = x2a.asyTransform.fromNumpyMatrix(transform)
        elif isinstance(transform, Qg.QTransform):
            assert transform.isAffine()
            obj_transform = x2a.asyTransform.fromQTransform(transform)
        else:
            obj_transform = transform

        scr_transform = obj_transform
        if self.magnification != 1.0:
            assert self.magnification != 0
            invmag = 1/self.magnification
            mag = self.magnification
            obj_transform = x2a.asyTransform((0, 0, invmag, 0, 0, invmag)) * obj_transform * \
                x2a.asyTransform((0, 0, mag, 0, 0, mag))

        if not applyFirst:
            item.transfKeymap[key][keyIndex] = obj_transform * \
                item.transfKeymap[key][keyIndex]
            if drawObj is not None:
                drawObj.transform = scr_transform * drawObj.transform
        else:
            item.transfKeymap[key][keyIndex] = item.transfKeymap[key][keyIndex] * obj_transform
            if drawObj is not None:
                drawObj.transform = drawObj.transform * scr_transform

        if self.selectAsGroup:
            for (maj2, min2) in self.currentlySelectedObj['allSameKey']:
                if (maj2, min2) == (maj, minor):
                    continue
                obj = self.drawObjects[maj2][min2]
                newIndex = obj.keyIndex
                if not applyFirst:
                    item.transfKeymap[key][newIndex] = obj_transform * \
                        item.transfKeymap[key][newIndex]
                    obj.transform = scr_transform * obj.transform
                else:
                    item.transfKeymap[key][newIndex] = item.transfKeymap[key][newIndex] * obj_transform
                    obj.transform = obj.transform * scr_transform

        self.quickUpdate()

    def transformObject(self, objKey, transform, applyFirst=False):
        maj, minor = objKey
        drawObj = self.drawObjects[maj][minor]
        item, key, keyIndex = self.getSelectedObjInfo(objKey)
        self.transformObjKey(item, key, keyIndex, transform, applyFirst, drawObj)

    def initializeEmptyFile(self):
        pass

    def loadFile(self, name):
        self.ui.statusbar.showMessage('Load {0}'.format(name))
        self.filename = os.path.abspath(name)
        self.currDir = os.path.dirname(self.filename)

        if not os.path.isfile(self.filename):
            self.filename = self.filename + '.asy'

        f = open(self.filename, 'rt')
        try:
            rawFileStr = f.read()
        except IOError:
            Qw.QMessageBox.critical(self, self.strings.fileOpenFailed, self.strings.fileOpenFailedText)
        else:
            rawText, transfDict, maxKey = xf.extractTransformsFromFile(rawFileStr)
            item = x2a.xasyScript(canvas=self.xasyDrawObj, engine=self.asyEngine, transfKeyMap=transfDict)

            
            item.setScript(rawText)
            item.setKey()
            self.fileItems.append(item)
            self.asyfyCanvas(True)

            maxKey2 = item.getMaxKeyCounter()
            self.globalObjectCounter = max(maxKey + 1, maxKey2)
        finally:
            f.close()

    def populateCanvasWithItems(self, forceUpdate=False):
        self.itemCount = 0
        for item in self.fileItems:
            self.drawObjects.append(item.generateDrawObjects(self.magnification, forceUpdate))
