import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import numpy as np
import os
import xasy2asy as x2a
import xasyFile as xf
from xasyTransform import xasyTransform as xT
from pyUIClass.window1 import Ui_MainWindow

import CustMatTransform


class AnchorMode:
    origin = 0
    topLeft = 1
    topRight = 2
    bottomRight = 3
    bottomLeft = 4
    customAnchor = 5
    center = 6


class MainWindow1(Qw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # For initialization purposes
        self.canvSize = Qc.QSize()
        self.filename = None
        self.mainCanvas = None
        self.canvasPixmap = None

        # Button initialization
        self.ui.btnTranslate.clicked.connect(self.btnTranslateonClick)
        self.ui.btnRotate.clicked.connect(self.btnRotateOnClick)
        self.ui.btnScale.clicked.connect(self.btnScaleOnClick)
        self.ui.btnDebug.clicked.connect(self.pauseBtnOnClick)
        self.ui.btnAlignX.clicked.connect(self.btnAlignXOnClick)
        self.ui.btnAlignY.clicked.connect(self.btnAlignYOnClick)
        self.ui.comboAnchor.currentTextChanged.connect(self.handleAnchorCombo)
        self.ui.btnLoadFile.clicked.connect(self.btnLoadFileonClick)
        self.ui.btnWorldCoords.clicked.connect(self.btnWorldCoordsOnClick)
        self.ui.btnCustTransform.clicked.connect(self.btnCustTransformOnClick)

        self.mainTransformation = Qg.QTransform()
        self.mainTransformation.scale(1, -1)

        self.localTransform = Qg.QTransform()

        self.magnification = 1
        self.inMidTransformation = False
        self.currentlySelectedObj = {'type': 'xasyPicture', 'ord': -1}
        self.savedMousePosition = None
        self.currentBoundingBox = None
        self.selectionDelta = None
        self.tmpboundingBoxTransform = None
        self.deltaAngle = 0
        self.scaleFactor = 1

        self.lockX = False
        self.lockY = False
        self.anchorMode = AnchorMode.origin
        self.currentAnchor = Qc.QPointF(0, 0)
        self.useGlobalCoords = True

        self.finalPixmap = None
        self.preCanvasPixmap = None
        self.postCanvasPixmap = None

        self.drawObjects = {}
        self.xasyDrawObj = {'drawDict': self.drawObjects}

        self.asyModes = {'Select', 'Pan', 'Translate', 'Rotate', 'Scale'}
        self.modeButtons = {self.ui.btnTranslate, self.ui.btnRotate, self.ui.btnScale}
        self.currentMode = 'Translate'

    def btnLoadFileonClick(self):
        fileName = Qw.QFileDialog.getOpenFileName(self, 'Open Asymptote File', Qc.QDir.homePath(), '*.asy')
        if fileName[0]:
            self.loadFile(fileName[0])

    def handleAnchorCombo(self, text):
        if text == 'Origin':
            self.anchorMode = AnchorMode.origin
        elif text == 'Center':
            self.anchorMode = AnchorMode.center

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
        if self.inMidTransformation:
            canvasPos = self.getCanvasCoordinates()
            if self.currentMode == 'Translate':
                newPos = canvasPos - self.savedMousePosition
                self.tx, self.ty = newPos.x(), newPos.y()
                if self.lockX:
                    self.tx = 0
                if self.lockY:
                    self.ty = 0
                self.tmpboundingBoxTransform = Qg.QTransform.fromTranslate(self.tx, self.ty)

            elif self.currentMode == 'Rotate':
                adjustedSavedMousePos = self.savedMousePosition - self.currentAnchor
                adjustedCanvasCoords = canvasPos - self.currentAnchor
                origAngle = np.arctan2(adjustedSavedMousePos.y(), adjustedSavedMousePos.x())
                newAng = np.arctan2(adjustedCanvasCoords.y(), adjustedCanvasCoords.x())
                self.deltaAngle = newAng - origAngle
                self.tmpboundingBoxTransform = xT.makeRotTransform(self.deltaAngle, self.currentAnchor).toQTransform()

            elif self.currentMode == 'Scale':
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

                self.tmpboundingBoxTransform = xT.makeScaleTransform(self.scaleFactorX, self.scaleFactorY,
                                                                     self.currentAnchor).toQTransform()
            self.quickUpdate()

    def mouseReleaseEvent(self, mouseEvent):
        assert isinstance(mouseEvent, Qg.QMouseEvent)
        if self.inMidTransformation:
            self.releaseTransform()
            self.currentBoundingBox = None
            self.tmpboundingBoxTransform = None
        self.inMidTransformation = False
        self.quickUpdate()

    def mousePressEvent(self, mouseEvent):
        if self.inMidTransformation:
            return
        selectedKey = self.selectObject()
        if selectedKey is not None:
            self.localTransform = Qg.QTransform()
            self.inMidTransformation = True
            obj, ID = self.drawObjects[selectedKey].originalObj
            self.currentlySelectedObj['ord'] = ID
            self.savedMousePosition = self.getCanvasCoordinates()
            self.currentBoundingBox = self.drawObjects[selectedKey].boundingBox

            if self.anchorMode == AnchorMode.center:
                self.currentAnchor = self.currentBoundingBox.center()
            else:
                self.currentAnchor = Qc.QPointF(0, 0)

            if not self.useGlobalCoords:
                self.localTransform = self.fileItems[0].transform[ID].toQTransform()

                if self.anchorMode != AnchorMode.origin:
                    self.currentAnchor = self.localTransform.inverted()[0].map(self.currentAnchor)

        self.totalUpdate()

    def releaseTransform(self):
        newTransform = x2a.asyTransform.fromQTransform(self.tmpboundingBoxTransform)
        self.transformObject(0, self.currentlySelectedObj['ord'], newTransform, not self.useGlobalCoords)

    def createMainCanvas(self):
        self.canvSize = self.ui.imgFrame.size()
        x, y = self.canvSize.width() / 2, self.canvSize.height() / 2

        self.canvasPixmap = Qg.QPixmap(self.canvSize)
        self.canvasPixmap.fill()

        self.finalPixmap = Qg.QPixmap(self.canvSize)

        self.preCanvasPixmap = Qg.QPixmap(self.canvSize)
        self.postCanvasPixmap = Qg.QPixmap(self.canvSize)

        self.mainCanvas = Qg.QPainter(self.canvasPixmap)

        self.ui.imgLabel.setPixmap(self.canvasPixmap)
        self.mainTransformation.translate(x, -y)
        self.mainCanvas.setTransform(self.mainTransformation, True)

        self.xasyDrawObj['canvas'] = self.mainCanvas

    def keyPressEvent(self, keyEvent):
        assert isinstance(keyEvent, Qg.QKeyEvent)
        if keyEvent.key() == Qc.Qt.Key_S:
            self.selectObject()

    def selectObject(self):
        if not self.ui.imgLabel.underMouse():
            return
        canvasCoords = self.getCanvasCoordinates()
        highestDrawPriority = -1
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
        return canvasPos * self.mainTransformation.inverted()[0]

    def rotateBtnOnClick(self):
        theta = float(self.ui.txtTheta.toPlainText())
        objectID = int(self.ui.txtObjectID.toPlainText())
        self.rotateObject(0, objectID, theta, (0, 0))
        self.populateCanvasWithItems()
        self.ui.imgLabel.setPixmap(self.canvasPixmap)

    def custTransformBtnOnClick(self):
        xx = float(self.ui.lineEditMatXX.text())
        xy = float(self.ui.lineEditMatXY.text())
        yx = float(self.ui.lineEditMatYX.text())
        yy = float(self.ui.lineEditMatYY.text())
        tx = float(self.ui.lineEditTX.text())
        ty = float(self.ui.lineEditTY.text())
        objectID = int(self.ui.txtObjectID.toPlainText())
        self.transformObject(0, objectID, x2a.asyTransform((tx, ty, xx, xy, yx, yy)))

    def totalUpdate(self):
        self.preDraw(self.mainCanvas)
        self.updateCanvas()
        self.postDraw()
        self.updateScreen()

    def quickUpdate(self):
        self.postDraw()
        self.updateScreen()

    def updateCanvas(self, clear=True):
        # self.canvasPixmap.fill(Qc.Qt.transparent)
        self.populateCanvasWithItems()

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
        preCanvas.setTransform(self.mainTransformation)

        preCanvas.setPen(Qc.Qt.gray)
        preCanvas.drawLine(Qc.QLine(-9999, 0, 9999, 0))
        preCanvas.drawLine(Qc.QLine(0, -9999, 0, 9999))

        # preCanvas.end()

    def postDraw(self):
        self.postCanvasPixmap.fill(Qc.Qt.transparent)
        postCanvas = Qg.QPainter(self.postCanvasPixmap)
        postCanvas.setTransform(self.mainTransformation)
        if self.currentBoundingBox is not None:
            postCanvas.save()

            if not self.useGlobalCoords:
                postCanvas.setTransform(self.localTransform, True)
                postCanvas.setPen(Qc.Qt.gray)
                postCanvas.drawLine(Qc.QLine(-9999, 0, 9999, 0))
                postCanvas.drawLine(Qc.QLine(0, -9999, 0, 9999))
                postCanvas.setPen(Qc.Qt.black)

            if self.tmpboundingBoxTransform is not None:
                postCanvas.setTransform(self.tmpboundingBoxTransform, True)
                postCanvas.setTransform(self.localTransform.inverted()[0], True)

            postCanvas.drawRect(self.currentBoundingBox)
            postCanvas.restore()
        postCanvas.end()

    def pauseBtnOnClick(self):
        print('Supposed to pause execution. Set breakpoint here.')

    def updateChecks(self):
        if self.currentMode == 'Translate':
            activeBtn = self.ui.btnTranslate
        elif self.currentMode == 'Rotate':
            activeBtn = self.ui.btnRotate
        elif self.currentMode == 'Scale':
            activeBtn = self.ui.btnScale
        for button in self.modeButtons:
            if button is not activeBtn:
                button.setChecked(False)
            else:
                button.setChecked(True)

    def btnAlignXOnClick(self, checked):
        self.lockX = checked
        if checked:
            self.ui.statusbar.showMessage('Enabled Lock on X')
        else:
            self.ui.statusbar.showMessage('Disabled Lock on X')

    def btnAlignYOnClick(self, checked):
        self.lockY = checked
        if checked:
            self.ui.statusbar.showMessage('Enabled Lock on Y')
        else:
            self.ui.statusbar.showMessage('Disabled Lock on Y')

    def btnTranslateonClick(self):
        self.currentMode = 'Translate'
        self.ui.statusbar.showMessage('Translate Mode')
        self.updateChecks()

    def btnRotateOnClick(self):
        self.currentMode = 'Rotate'
        self.ui.statusbar.showMessage('Rotate Mode')
        self.updateChecks()

    def btnScaleOnClick(self):
        self.currentMode = 'Scale'
        self.ui.statusbar.showMessage('Scale Mode')
        self.updateChecks()

    def btnWorldCoordsOnClick(self, checked):
        self.useGlobalCoords = checked

    def btnCustTransformOnClick(self):
        matrixDialog = CustMatTransform.CustMatTransform()
        matrixDialog.show()
        result = matrixDialog.exec_()
        if result == Qw.QDialog.Accepted:
            print(matrixDialog.getTransformationMatrix())

    def transformObject(self, itemIndex, objIndex, transform, applyFirst=False):
        item = self.fileItems[itemIndex]
        if isinstance(transform, np.ndarray):
            obj_transform = x2a.asyTransform.fromNumpyMatrix(transform)
        elif isinstance(transform, Qg.QTransform):
            assert transform.isAffine()
            obj_transform = x2a.asyTransform.fromQTransform(transform)
        else:
            obj_transform = transform

        oldTransf = item.transform[objIndex]
        if not applyFirst:
            item.transform[objIndex] = obj_transform * oldTransf
        else:
            item.transform[objIndex] = oldTransf * obj_transform
        # TODO: Fix bounding box
        # self.drawObjects[item.imageList[objIndex].IDTag].useCanvasTransform = True
        self.totalUpdate()

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
        self.totalUpdate()

    def populateCanvasWithItems(self):
        # if (not self.testOrAcquireLock()):
        #     return
        self.itemCount = 0
        for itemIndex in range(len(self.fileItems)):
            item = self.fileItems[itemIndex]
            item.drawOnCanvas(self.xasyDrawObj, self.magnification, forceAddition=True)
            # self.bindItemEvents(item)
        # self.releaseLock()


