#!/usr/bin/env python3

import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import numpy as numpy
import xasy2asy as xasy2asy
from pyUIClass.custMatTransform import Ui_Dialog


class CustMatTransform(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.btnAccept.clicked.connect(self.accept)
        self.ui.btnCancel.clicked.connect(self.reject)
        self.ui.btnReset.clicked.connect(self.resetDialog)

        self.mainTransformation = QtGui.QTransform()
        self.mainTransformation.scale(1, -1)

        self.matrixLineInputs = [
            self.ui.lineMat00, self.ui.lineMat01, self.ui.lineMatTx,
            self.ui.lineMat10, self.ui.lineMat11, self.ui.lineMatTy]

        validator = QtGui.QDoubleValidator()
        for lineInput in self.matrixLineInputs:
            lineInput.setValidator(validator)
            lineInput.textChanged.connect(self.handleUpdateText)

    def show(self):
        super().show()
        self.createCanvas()
        self.updatePreview()

    def createCanvas(self):
        self.canvSize = self.ui.imgPreview.size()
        self.previewPixmap = QtGui.QPixmap(self.canvSize)
        tx, ty = self.canvSize.width() / 2, self.canvSize.height() / 2
        self.mainTransformation.translate(tx, -ty)

    def handleUpdateText(self, text):
        if str(text) not in {'.', '-', '.-', '-.'} and str(text):
            self.updatePreview()
            self.ui.btnAccept.setEnabled(True)
        else:
            self.previewPixmap.fill()
            self.ui.imgPreview.setPixmap(self.previewPixmap)
            self.ui.btnAccept.setEnabled(False)

    def updatePreview(self):
        self.previewPixmap.fill()
        canvas = QtGui.QPainter(self.previewPixmap)
        if not canvas.isActive():
            return
        canvas.setTransform(self.mainTransformation)

        canvas.save()
        canvas.setPen(QtCore.Qt.lightGray)
        self.drawBasicGrid(canvas)
        transform = xasy2asy.asyTransform.fromNumpyMatrix(self.getTransformationMatrix())
        canvTransform = transform.toQTransform()
        canvas.setTransform(canvTransform, True)

        canvas.setPen(QtCore.Qt.black)

        if canvTransform.isInvertible():
            self.drawBasicGrid(canvas, False)

            if canvTransform.determinant() <= 0:
                canvas.setPen(QtCore.Qt.red)

            canvas.drawRect(QtCore.QRect(QtCore.QPoint(0, 0), QtCore.QSize(20, 20)))

        self.ui.imgPreview.setPixmap(self.previewPixmap)

    def resetDialog(self):
        self.ui.lineMatTx.setText('0')
        self.ui.lineMatTx.setText('0')

        self.ui.lineMat00.setText('1')
        self.ui.lineMat01.setText('0')
        self.ui.lineMat10.setText('0')
        self.ui.lineMat11.setText('1')

    def drawBasicGrid(self, canvas, grid=True):
        canvas.drawLine(QtCore.QLine(-9999, 0, 9999, 0))
        canvas.drawLine(QtCore.QLine(0, -9999, 0, 9999))

        fromIter, toIter = -7, 7
        gridSize = 20
        if grid:
            for iterIndex in range(fromIter, toIter + 1):
                canvas.drawLine(QtCore.QLine(-9999, iterIndex * gridSize, 9999, iterIndex * gridSize))
                canvas.drawLine(QtCore.QLine(iterIndex * gridSize, -9999, iterIndex * gridSize, 9999))

    def getTransformationMatrix(self):
        rawMatrixNum = [float(lineInput.text()) for lineInput in self.matrixLineInputs]
        rawMatrixNum.extend([0, 0, 1])
        return numpy.matrix(rawMatrixNum).reshape((3, 3))
