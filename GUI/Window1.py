import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import numpy as np
import numpy.linalg as npl
import time
import os
import xasy2asy as x2a
import xasyFile as xf
from xasyTransform import xasyTransform as xT
from pyUIClass.window1 import Ui_MainWindow


class MainWindow1(Qw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btnPause.clicked.connect(self.pauseBtnOnClick)
        self.ui.btnTranslate.clicked.connect(self.transformBtnOnClick)
        self.ui.btnRotate.clicked.connect(self.rotateBtnOnClick)
        self.ui.btnCustomTransform.clicked.connect(self.custTransformBtnOnClick)

        # For initialization purposes
        self.canvSize = Qc.QSize()
        self.filename = None
        self.mainCanvas = None
        self.canvasPixmap = None

        self.mainTransformation = Qg.QTransform()
        self.mainTransformation.scale(1, -1)

        self.magnification = 1
        self.inMidTransformation = False
        self.currentlySelectedObj = {'type':'xasyPicture', 'ord': -1}
        self.savedMousePosition = None
        self.currentBoundingBox = None
        self.selectionDelta = None

        self.drawObjects = {}
        self.xasyDrawObj = {'drawDict': self.drawObjects}

    def isReady(self):
        return self.mainCanvas is not None

    def show(self):
        super().show()
        self.createMainCanvas()  # somehow, the coordinates doesn't get updated until after showing.

    def mouseMoveEvent(self, mouseEvent):
        assert isinstance(mouseEvent, Qg.QMouseEvent)
        if self.inMidTransformation:
            self.quickUpdate()
            canvasPos = self.getCanvasCoordinates()
            self.currentBoundingBox.moveTo(canvasPos + self.selectionDelta)

    def mouseReleaseEvent(self, mouseEvent):
        assert isinstance(mouseEvent, Qg.QMouseEvent)
        if self.inMidTransformation:
            self.releaseTransform()

    def mousePressEvent(self, mouseEvent):
        if self.inMidTransformation:
            return
        selectedKey = self.selectObject()
        if selectedKey is not None:
            self.inMidTransformation = True
            obj, ID = self.drawObjects[selectedKey].originalObj
            self.currentlySelectedObj['ord'] = ID
            self.savedMousePosition = self.getCanvasCoordinates()
            self.currentBoundingBox = self.drawObjects[selectedKey].boundingBox
            self.selectionDelta = self.currentBoundingBox.topLeft() - self.getCanvasCoordinates()
        self.totalUpdate()

    def releaseTransform(self):
        if not self.inMidTransformation:
            return
        self.currentBoundingBox = None
        currMouseCoords = self.getCanvasCoordinates()
        dx = currMouseCoords.x() - self.savedMousePosition.x()
        dy = currMouseCoords.y() - self.savedMousePosition.y()
        newTransform = x2a.asyTransform((dx, dy, 1, 0, 0, 1))
        self.transformObject(0, self.currentlySelectedObj['ord'], newTransform)
        self.inMidTransformation = False

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
        self.preDraw()
        self.updateCanvas()
        self.postDraw()
        self.updateScreen()

    def quickUpdate(self):
        self.preDraw()
        self.postDraw()
        self.updateScreen()

    def updateCanvas(self, clear=True):
        self.canvasPixmap.fill(Qc.Qt.white)
        self.populateCanvasWithItems()

    def updateScreen(self):
        self.finalPixmap = Qg.QPixmap(self.canvSize)
        self.finalPixmap.fill(Qc.Qt.black)
        finalPainter = Qg.QPainter(self.finalPixmap)
        drawPoint = Qc.QPoint(0, 0)
        finalPainter.drawPixmap(drawPoint, self.preCanvasPixmap)
        finalPainter.drawPixmap(drawPoint, self.canvasPixmap)
        finalPainter.drawPixmap(drawPoint, self.postCanvasPixmap)
        finalPainter.end()
        self.ui.imgLabel.setPixmap(self.finalPixmap)

    def preDraw(self):
        self.preCanvasPixmap.fill(Qc.Qt.white)

    def postDraw(self):
        self.postCanvasPixmap.fill(Qc.Qt.transparent)
        postCanvas = Qg.QPainter(self.postCanvasPixmap)
        postCanvas.setTransform(self.mainTransformation)
        if self.currentBoundingBox is not None:
            postCanvas.drawRect(self.currentBoundingBox)
        postCanvas.end()

    def pauseBtnOnClick(self):
        print('Test')

    def transformBtnOnClick(self):
        x_transform = int(self.ui.txtXTransform.toPlainText())
        y_transform = int(self.ui.txtYTransform.toPlainText())
        objectID = int(self.ui.txtObjectID.toPlainText())
        self.translateObject(0, objectID, (x_transform, y_transform))
        self.totalUpdate()

    def translateObject(self, itemIndex, objIndex, translation):
        item = self.fileItems[itemIndex]
        transform = x2a.asyTransform((translation[0], translation[1], 1, 0, 0, 1))
        if isinstance(item, x2a.xasyText) or isinstance(item, x2a.xasyScript):
            item.transform[objIndex] = transform * item.transform[objIndex]
            bbox = item.imageList[objIndex].originalImage.bbox
            item.imageList[objIndex].originalImage.bbox = bbox[0] + translation[0], bbox[1] + translation[1], \
                                                          bbox[2] + translation[0], bbox[3] + translation[1]
        else:
            item.transform = [transform * item.transform[0]]

    def transformObject(self, itemIndex, objIndex, transform):
        item = self.fileItems[itemIndex]
        if isinstance(transform, np.ndarray):
            obj_transform = x2a.asyTransform.fromNumpyMatrix(transform)
        elif isinstance(transform, Qg.QTransform):
            assert transform.isAffine()
            obj_transform = x2a.asyTransform.fromQTransform(transform)
        else:
            obj_transform = transform

        item.transform[objIndex] = obj_transform * item.transform[objIndex]
        # TODO: Fix bounding box
        # self.drawObjects[item.imageList[objIndex].IDTag].useCanvasTransform = True
        self.totalUpdate()

    def rotateObject(self, itemIndex, objIndex, theta, origin=None):
        # print ("Rotating by {} around {}".format(theta*180.0/math.pi,origin))
        item = self.fileItems[itemIndex]
        if origin is None:
            origin = (item.imageList[objIndex].originalImage.bbox[0], -item.imageList[objIndex].originalImage.bbox[3])
        rotMat = xT.makeRotTransform(theta, (origin[0] / self.magnification, origin[1] / self.magnification))
        if isinstance(item, x2a.xasyText) or isinstance(item, x2a.xasyScript):
            # transform the image
            oldBbox = item.imageList[objIndex].originalImage.bbox
            oldBbox = (oldBbox[0], -oldBbox[1], oldBbox[2], -oldBbox[3])
            item.transform[objIndex] = rotMat * item.transform[objIndex]
            # item.transform[objIndex] = rotMat * original
            item.imageList[objIndex].originalImage.theta += theta
            item.imageList[objIndex].image = item.imageList[objIndex].originalImage.rotate(
                np.degrees(item.imageList[objIndex].originalImage.theta), expand=True, resample=Image.BICUBIC)
            # item.imageList[objIndex].iqt = ImageTk.PhotoImage(item.imageList[index].image)
            item.imageList[objIndex].iqt = Qg.QImage(piq.ImageQt(item.imageList[objIndex].image))
            # self.mainCanvas.itemconfigure(ID, image=item.imageList[objIndex].itk)
            # the image has been rotated in place
            # now, compensate for any resizing and shift to the correct location
            #
            #  p0 --- p1               p1
            #  |      |     --->      /  \
            #  p2 --- p3             p0  p3
            #                         \ /
            #                          p2
            #
            rotMat2 = xT.makeRotTransform(theta, origin)
            p0 = rotMat2 * (oldBbox[0], oldBbox[3])  # switch to usual coordinates
            p1 = rotMat2 * (oldBbox[2], oldBbox[3])
            p2 = rotMat2 * (oldBbox[0], oldBbox[1])
            p3 = rotMat2 * (oldBbox[2], oldBbox[1])
            newTopLeft = (min(p0[0], p1[0], p2[0], p3[0]), max(p0[1], p1[1], p2[1], p3[1]))
            # newBottomRight = (max(p0[0], p1[0], p2[0], p3[0]), -min(p0[1], p1[1], p2[1], p3[1]))
            # switch back to screen coords
            shift = (newTopLeft[0] - oldBbox[0], newTopLeft[1] - oldBbox[2])
            # print (theta*180.0/math.pi,origin,oldBbox,newTopLeft,shift)
            # print (item.imageList[index].originalImage.size)
            # print (item.imageList[index].image.size)
            # print
            # new_coord =
            # self.translateObject(itemIndex, objIndex, shift)
            # item.imageList[objIndex].originalImage.bbox[0] = newTopLeft[0]
            # item.imageList[objIndex].originalImage.bbox[3] = newTopLeft[1]
            # TODO: Fix Rotation bounding box
            item.imageList[objIndex].originalImage.btmLeftPoint += Qc.QPointF(shift[0], shift[1])
            # item.imageList[objIndex].originalImage.bbox[1] = newTopLeft[0]
            # item.imageList[objIndex].originalImage.bbox[2] = newTopLeft[1]

            # self.mainCanvas.coords(ID, oldBbox[0] + shift[0], oldBbox[3] + shift[1])

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
        self.populateCanvasWithItems()
        #self.populatePropertyList()
        #self.updateCanvasSize()
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


