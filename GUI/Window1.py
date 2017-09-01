import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import numpy as np
import os
import xasy2asy as x2a
import xasyFile as xf
import UndoRedoStack as Urs
import json
import io
import pathlib
from xasyTransform import xasyTransform as xT
from pyUIClass.window1 import Ui_MainWindow

import webbrowser

import CustMatTransform
import SetCustomAnchor
import BezierCurveEditor

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


class SelectionMode:
    select = 0
    pan = 1
    translate = 2
    rotate = 3
    scale = 4


class DefaultSettings:
    defaultSettings = {
        'externalEditor': 'gedit *ASYPATH',
        'enableImmediatePreview': True
    }


class MainWindow1(Qw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.settings = DefaultSettings.defaultSettings

        # For initialization purposes
        self.canvSize = Qc.QSize()
        self.filename = None
        self.mainCanvas = None
        self.canvasPixmap = None

        self.ui.actionSaveAs.triggered.connect(self.actionSaveAs)
        self.ui.actionManual.triggered.connect(self.actionManual)
        self.ui.actionEnterCommand.triggered.connect(self.enterCustomCommand)

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

        self.ui.btnCreateCurve.clicked.connect(self.btnCreateCurveOnClick)

        self.mainTransformation = Qg.QTransform()
        self.mainTransformation.scale(1, -1)

        self.localTransform = Qg.QTransform()

        self.screenTransformation = Qg.QTransform()

        self.magnification = 1
        self.inMidTransformation = False
        self.currentlySelectedObj = {'type': 'xasyPicture', 'selectedKey': None}
        self.savedMousePosition = None
        self.currentBoundingBox = None
        self.selectionDelta = None
        self.newTransform = None
        self.origBboxTransform = None
        self.deltaAngle = 0
        self.scaleFactor = 1

        self.undoRedoStack = Urs.actionStack()

        self.lockX = False
        self.lockY = False
        self.anchorMode = AnchorMode.origin
        self.currentAnchor = Qc.QPointF(0, 0)
        self.useGlobalCoords = True
        self.drawAxes = True

        self.finalPixmap = None
        self.preCanvasPixmap = None
        self.postCanvasPixmap = None

        self.previewCurve = None

        self.drawObjects = {}
        self.xasyDrawObj = {'drawDict': self.drawObjects}

        self.modeButtons = {self.ui.btnTranslate, self.ui.btnRotate, self.ui.btnScale, self.ui.btnSelect,
                            self.ui.btnPan}
        self.objButtons = {self.ui.btnCustTransform}
        self.globalTransformOnlyButtons = (self.ui.comboAnchor, self.ui.btnAnchor)

        self.currentMode = SelectionMode.translate
        self.setAllInSetEnabled(self.objButtons, False)

        self.loadSettings()

    def btnCreateCurveOnClick(self):
        self.inCurveCreationMode = True
        curveDialog = BezierCurveEditor.BezierCurveEditor()
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

    def updateCurve(self, valid, newCurve):
        self.previewCurve = newCurve
        self.quickUpdate()

    def addTransformationChanges(self, objKey, transform, isLocal=False):
        self.undoRedoStack.add(self.createAction(TransformationChanges(objKey, transform, isLocal)))
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
        else:
            self.ui.btnUndo.setEnabled(False)

        if len(self.undoRedoStack.redoStack) > 0:
            self.ui.btnRedo.setEnabled(True)
        else:
            self.ui.btnRedo.setEnabled(False)

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
        if command == 'xasy:testPan':
            self.adjustTransform(Qg.QTransform.fromTranslate(100, 100))
            self.quickUpdate()
        elif command == 'xasy:undo':
            self.undoRedoStack.undo()
        elif command == 'xasy:redo':
            self.undoRedoStack.redo()
<<<<<<< HEAD
        elif command == 'xasy:showBeizerEditor':
=======
        elif command == 'xasy:showBezierEditor':
>>>>>>> 093568e25e61c860b896497264b19ea4f078c0ac
            editor = BezierCurveEditor.BezierCurveEditor()
            editor.show()
            editor.exec_()
        elif command == 'xasy:quit' or command == 'xasy:exit':
            Qc.QCoreApplication.exit()
        elif command == 'xasy:pause':
            pass
        else:
            self.ui.statusbar.showMessage('Command not found')

    def enterCustomCommand(self):
        commandText, result = Qw.QInputDialog.getText(self, 'Enter Custom Command', 'Enter Custom Command')
        if result:
            self.execCustomCommand(commandText)

    def actionManual(self):
        asyManualURL = 'http://asymptote.sourceforge.net/asymptote.pdf'
        webbrowser.open_new(asyManualURL)

    def loadSettings(self):
        configFile = '.asy/xasy2conf.json'
        keyList = DefaultSettings.defaultSettings.keys()
        fullConfigFile = pathlib.Path.home().joinpath(pathlib.Path(configFile))

        if fullConfigFile.exists():
            settingsFile = io.open(fullConfigFile)
            self.settings = json.loads(settingsFile.read())
            addedKey = False
            for key in keyList:
                addedKey = True
                if key not in self.settings:
                    self.settings[key] = DefaultSettings.defaultSettings[key]

            if addedKey:
                settingsFile.close()
                settingsFile = io.open(fullConfigFile, 'w')
                settingsFile.write(json.dumps(self.settings))
        else:
            settingsFile = io.open(fullConfigFile, 'w')
            settingsFile.write(json.dumps(self.settings))
        settingsFile.close()

    def btnSaveOnClick(self):
        saveFile = io.open(self.filename, 'w')
        xf.saveFile(saveFile, self.fileItems)
        saveFile.close()

    def actionSaveAs(self):
        saveLocation = Qw.QFileDialog.getSaveFileName(self, 'Save File', Qc.QDir.homePath())[0]
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
            self.anchorMode = AnchorMode.customAnchor

    def btnCustomAnchorOnClick(self, text):
        custAnchorDialog = SetCustomAnchor.CustomAnchorDialog()
        custAnchorDialog.show()
        result = custAnchorDialog.exec()
        if result == Qw.QDialog.Accepted:
            self.customAnchor = custAnchorDialog.getPoint()
            self.ui.comboAnchor.setCurrentText('Custom Anchor')

    def btnColorSelectOnClick(self):
        colorDialog = Qw.QColorDialog.getColor(Qc.Qt.black, self)

    def isReady(self):
        return self.mainCanvas is not None

    def resizeEvent(self, resizeEvent):
        assert isinstance(resizeEvent, Qg.QResizeEvent)
        newRect = Qc.QRect(Qc.QPoint(0, 0), resizeEvent.size())
        # self.ui.centralFrame.setFrameRect(newRect)

    def show(self):
        super().show()
        self.createMainCanvas()  # somehow, the coordinates doesn't get updated until after showing.

    def mouseMoveEvent(self, mouseEvent):
        assert isinstance(mouseEvent, Qg.QMouseEvent)
        if not self.ui.imgLabel.underMouse():
            return

        canvasPos = self.getCanvasCoordinates()
        if self.currentMode == SelectionMode.pan:
            mousePos = self.getWindowCoordinates()
            newPos = mousePos - self.savedWindowMousePos
            tx, ty = newPos.x(), newPos.y()
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
                self.tx, self.ty = newPos.x(), newPos.y()
                if self.lockX:
                    self.tx = 0
                if self.lockY:
                    self.ty = 0
                self.newTransform = Qg.QTransform.fromTranslate(self.tx, self.ty)

            elif self.currentMode == SelectionMode.rotate:
                adjustedSavedMousePos = self.savedMousePosition - self.currentAnchor
                adjustedCanvasCoords = canvasPos - self.currentAnchor
                origAngle = np.arctan2(adjustedSavedMousePos.y(), adjustedSavedMousePos.x())
                newAng = np.arctan2(adjustedCanvasCoords.y(), adjustedCanvasCoords.x())
                self.deltaAngle = newAng - origAngle
                self.newTransform = xT.makeRotTransform(self.deltaAngle, self.currentAnchor).toQTransform()

            elif self.currentMode == SelectionMode.scale:
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
        if self.inMidTransformation:
            self.clearSelection()
        self.inMidTransformation = False
        self.quickUpdate()

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
                # TODO: Record base points/bbox before hand and use that for anchor?
                # adjTransform = self.drawObjects[selectedKey].transform.toQTransform()
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

    def keyPressEvent(self, keyEvent):
        assert isinstance(keyEvent, Qg.QKeyEvent)
        if keyEvent.key() == Qc.Qt.Key_S:
            self.selectObject()
        elif keyEvent.key() == Qc.Qt.Key_F1:
            self.enterCustomCommand()

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
    #     self.transformObject(0, objectID, x2a.asyTransform((tx, ty, xx, xy, yx, yy)))

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
        finalPainter = Qg.QPainter(self.finalPixmap)
        drawPoint = Qc.QPoint(0, 0)
        # finalPainter.drawPixmap(drawPoint, self.preCanvasPixmap)
        finalPainter.drawPixmap(drawPoint, self.canvasPixmap)
        finalPainter.drawPixmap(drawPoint, self.postCanvasPixmap)
        finalPainter.end()
        self.ui.imgLabel.setPixmap(self.finalPixmap)

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

        # preCanvas.end()

    def postDraw(self):
        self.postCanvasPixmap.fill(Qc.Qt.transparent)
        postCanvas = Qg.QPainter(self.postCanvasPixmap)
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
        postCanvas.end()

    def updateChecks(self):
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

    def btnCustTransformOnClick(self):
        matrixDialog = CustMatTransform.CustMatTransform()
        matrixDialog.show()
        result = matrixDialog.exec_()
        if result == Qw.QDialog.Accepted:
            objKey = self.currentlySelectedObj['selectedKey']
            self.transformObject(objKey, matrixDialog.getTransformationMatrix(), not self.useGlobalCoords)

        self.clearSelection()  # for now, unless we update the bouding box transformation.
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

        if isinstance(transform, np.ndarray):
            obj_transform = x2a.asyTransform.fromNumpyMatrix(transform)
        elif isinstance(transform, Qg.QTransform):
            assert transform.isAffine()
            obj_transform = x2a.asyTransform.fromQTransform(transform)
        else:
            obj_transform = transform

        oldTransf = item.transform[transfIndex]

        if not applyFirst:
            item.transform[transfIndex] = obj_transform * oldTransf
            drawObj.transform = item.transform[transfIndex]
        else:
            item.transform[transfIndex] = oldTransf * obj_transform

        drawObj.transform = item.transform[transfIndex]

        self.quickUpdate()

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
            Qw.QMessageBox.critical(self, "File Opening Failed.", "File could not be opened.")
            # messagebox.showerror("File Opening Failed.", "File could not be opened.")
            self.fileItems = []
        except Exception:
            self.fileItems = []
            self.autoMakeScript = True
            if self.autoMakeScript or Qw.QMessageBox.question(self, "Error Opening File",
                                                              "File was not recognized as an xasy file.\nLoad as a script item?") == \
                    Qw.QMessageBox.Yes:
                # try:
                item = x2a.xasyScript(self.xasyDrawObj)
                f.seek(0)
                item.setScript(f.read())
                self.fileItems.append(item)
                # except:
                #     Qw.QMessageBox.critical(self, "File Opening Failed.", "File could not be opened.")
                #     # messagebox.showerror("File Opening Failed.", "Could not load as a script item.")
                #     self.fileItems = []
        # self.populateCanvasWithItems()
        # self.populatePropertyList()
        # self.updateCanvasSize()
        self.asyfyCanvas()

    def populateCanvasWithItems(self):
        # if (not self.testOrAcquireLock()):
        #     return
        self.itemCount = 0
        for itemIndex in range(len(self.fileItems)):
            item = self.fileItems[itemIndex]
            item.drawOnCanvas(self.xasyDrawObj, self.magnification, forceAddition=True)
            # self.bindItemEvents(item)
        # self.releaseLock()


