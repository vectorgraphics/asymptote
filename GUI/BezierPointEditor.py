import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
from pyUIClass.bezierPointEditor import Ui_Form
import xasyValidator as xV
import BezierCurveEditor
import numpy as np


class BezierPointEditor(Qw.QDialog):
    def __init__(self, basePointInformation=None, useDegrees=False):
        assert isinstance(basePointInformation, BezierCurveEditor.BezierPoint) or basePointInformation is None
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        self.numLineEdits = {self.ui.linePointYorM, self.ui.linePointXorA,
                 self.ui.lineLCtYorM, self.ui.lineLCtXorA,
                 self.ui.lineRCtYorM, self.ui.lineRCtXorA,
                 self.ui.lineMagnitude, self.ui.lineAngle}

        self.numLinePoints = {self.ui.linePointYorM, self.ui.linePointXorA}
        txtValidator = Qg.QDoubleValidator()

        for lineEdit in self.numLineEdits:
            lineEdit.setValidator(txtValidator)
            lineEdit.textEdited.connect(self.handleTextChange)

        self.ui.btnPointUsePolar.clicked.connect(lambda checked: self.createHandlebtnChange(
            self.ui.linePointXorA, self.ui.linePointYorM, checked, self.lastSavedPoint))

        self.ui.btnLCtrlManualAdj.clicked.connect(lambda checked: self.createBtnToggleEditable(
            self.createLCtrlPoint, self.ui.btnLCtrlRelative, self.ui.btnLCtrlUsePolar, self.ui.lineLCtXorA,
            self.ui.lineLCtYorM, checked))

        self.ui.btnRCtrlManualAdj.clicked.connect(lambda checked: self.createBtnToggleEditable(
            self.createRCtrlPoint, self.ui.btnRCtrlRelative, self.ui.btnRCtrlUsePolar, self.ui.lineRCtXorA,
            self.ui.lineRCtYorM, checked))

        self.ui.btnRCtrlUsePolar.clicked.connect(lambda checked: self.createHandlebtnChange(
            self.ui.lineRCtXorA, self.ui.lineRCtYorM, checked, self.lastRCtrlPoint, self.ui.btnRCtrlRelative))

        self.ui.btnLCtrlUsePolar.clicked.connect(lambda checked: self.createHandlebtnChange(
            self.ui.lineLCtXorA, self.ui.lineLCtYorM, checked, self.lastLCtrlPoint, self.ui.btnLCtrlRelative))

        self.ui.btnLCtrlRelative.clicked.connect(lambda checked: self.createBtnSetRelative(
            self.ui.lineLCtXorA, self.ui.lineLCtYorM, self.lastLCtrlPoint, checked))

        self.ui.btnRCtrlRelative.clicked.connect(lambda checked: self.createBtnSetRelative(
            self.ui.lineRCtXorA, self.ui.lineRCtYorM, self.lastRCtrlPoint, checked))

        self.basePoint = basePointInformation
        if self.basePoint is not None:
            self.lastSavedPoint = self.basePoint.point
            self.lastLCtrlPoint = self.basePoint.lCtrlPoint
            self.lastRCtrlPoint = self.basePoint.rCtrlPoint
            self.ui.linePointXorA.setText(str(self.lastSavedPoint.x()))
            self.ui.linePointYorM.setText(str(self.lastSavedPoint.y()))

            self.ui.lineLCtXorA.setText(str(self.lastLCtrlPoint.x()))
            self.ui.lineLCtYorM.setText(str(self.lastLCtrlPoint.y()))

            self.ui.lineRCtXorA.setText(str(self.lastRCtrlPoint.x()))
            self.ui.lineRCtYorM.setText(str(self.lastRCtrlPoint.y()))
        else:
            self.lastSavedPoint = Qc.QPointF()
            self.lastLCtrlPoint = Qc.QPointF()
            self.lastRCtrlPoint = Qc.QPointF()

        self.selectionAngle = 0.0
        self.selectionMagnitude = 20.0

        self.useDegrees = useDegrees

        self.drawAngleSelection()

    def drawAngleSelection(self):
        canvPixmap = Qg.QPixmap(self.ui.lblPreview.size())
        canvPixmap.fill()
        canvas = Qg.QPainter(canvPixmap)
        canvas.setPen(Qc.Qt.gray)

        tx, ty = self.ui.lblPreview.size().width() / 2, self.ui.lblPreview.size().height() / 2
        canvas.scale(1, -1)
        canvas.translate(tx, -ty)

        # axes
        canvas.drawLine(Qc.QLine(-9999, 0, 9999, 0))
        canvas.drawLine(Qc.QLine(0, -9999, 0, 9999))

        # unit (or 50-unit) circle
        rad = 50
        canvas.drawEllipse(Qc.QRectF(-rad, -rad, 2*rad, 2*rad))

        # angle
        canvas.setPen(Qc.Qt.black)
        x, y = np.cos(self.selectionAngle), np.sin(self.selectionAngle)
        angleLine = Qc.QLineF(Qc.QPointF(0, 0), rad * Qc.QPointF(x, y))
        canvas.drawLine(angleLine)

        # dot
        dotPen = Qg.QPen()
        dotPen.setColor(Qc.Qt.black)
        dotPen.setWidthF(5.0)

        canvas.setPen(dotPen)
        canvas.drawPoint(rad * Qc.QPointF(x, y))

        canvas.end()
        self.ui.lblPreview.setPixmap(canvPixmap)

    @classmethod
    def cartesian2polar(cls, x, y):
        ang = round(np.arctan2(y, x), 2)
        r = round(np.linalg.norm([x, y]), 2)
        return ang, r

    @classmethod
    def polar2cartesian(cls, ang, r):
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        return x, y

    def createHandlebtnChange(self, leXorA, leYorM, usePolar, lastCtrlPoint, relativeButton=None):
        if usePolar:
            leXorA.setPlaceholderText('Angle')
            leYorM.setPlaceholderText('Magnitude')
            if relativeButton is not None:
                relativeButton.setChecked(False)
        else:
            leXorA.setPlaceholderText('X')
            leYorM.setPlaceholderText('Y')

        if relativeButton is not None:
            relativeButton.setEnabled(not usePolar)

        if not (xV.validateFloat(leXorA.text()) and xV.validateFloat(leYorM.text()) and self.mainPointValid()):
            return

        x, y = lastCtrlPoint.x(), lastCtrlPoint.y()
        if usePolar:  # (x, y) -> (ang, r)
            ang, r = BezierPointEditor.cartesian2polar(x - self.lastSavedPoint.x(), y - self.lastSavedPoint.y())
            leXorA.setText(str(ang))
            leYorM.setText(str(r))
        else:  # (ang, r) -> (x, y)
            leXorA.setText(str(x))
            leYorM.setText(str(y))

    def mainPointValid(self):
        validPoint = True
        for line in self.numLinePoints:
            validPoint = validPoint and xV.validateFloat(line.text())
        return validPoint

    def createBtnToggleEditable(self, pointFunc, btnRel, btnPol, leXorA, leYorM, checked):
        leXorA.setEnabled(checked)
        leYorM.setEnabled(checked)
        btnRel.setEnabled(checked)
        btnPol.setEnabled(checked)
        btnRel.setChecked(False)
        btnPol.setChecked(False)

        if not checked and self.mainPointValid():
            x, y = pointFunc()
            leXorA.setText(str(x))
            leYorM.setText(str(y))

    def getUsrAngle(self, inputAngle):
        if self.useDegrees:
            inputAngle = np.deg2rad(inputAngle)
        return inputAngle

    def createBtnSetRelative(self, leXorA , leYorM, lastPoint, checked):
        if checked:  # assuming cartesian mode here
            leXorA.setText(str(lastPoint.x() - self.lastSavedPoint.x()))
            leYorM.setText(str(lastPoint.y() - self.lastSavedPoint.y()))
        else:
            leXorA.setText(str(lastPoint.x()))
            leYorM.setText(str(lastPoint.y()))

    def handleTextChange(self):
        if xV.validateFloat(self.ui.lineAngle.text()):
            self.selectionAngle = self.getUsrAngle(float(self.ui.lineAngle.text()))

        if xV.validateFloat(self.ui.lineMagnitude.text()):
            self.selectionMagnitude = float(self.ui.lineMagnitude.text())

        if self.mainPointValid():
            x, y = self.createPoint()
            self.lastSavedPoint = Qc.QPointF(x, y)

            x, y = self.createLCtrlPoint()
            if not self.ui.btnLCtrlManualAdj.isChecked():  # automatic
                self.ui.lineLCtXorA.setText(str(x))
                self.ui.lineLCtYorM.setText(str(y))
            self.lastLCtrlPoint = Qc.QPointF(x, y)

            x, y = self.createRCtrlPoint()
            if not self.ui.btnRCtrlManualAdj.isChecked():
                self.ui.lineRCtXorA.setText(str(x))
                self.ui.lineRCtYorM.setText(str(y))
            self.lastRCtrlPoint = Qc.QPointF(x, y)

        self.drawAngleSelection()

    # TODO: Allow arbitrary adjustment of rounding cutoff (in settings)

    def createPoint(self, forcePolar=None):
        if forcePolar is None:
            forcePolar = self.ui.btnPointUsePolar.isChecked()
        if forcePolar:
            ang = self.getUsrAngle(float(self.ui.linePointXorA.text()))
            mag = float(self.ui.linePointYorM.text())
            x = mag * np.cos(ang)
            y = mag * np.sin(ang)
        else:
            x = float(self.ui.linePointXorA.text())
            y = float(self.ui.linePointYorM.text())
        return round(x, 2), round(y, 2)

    def createCtrlPoint(self, usePolar, useRelative, lineXOrA, lineYorM, isLeft=False):
        if isLeft:
            multFactor = -1
        else:
            multFactor = 1
        if not self.ui.btnLCtrlManualAdj.isChecked():  # automatic
            x = multFactor * self.selectionMagnitude * np.cos(self.selectionAngle) + self.lastSavedPoint.x()
            y = multFactor * self.selectionMagnitude * np.sin(self.selectionAngle) + self.lastSavedPoint.y()
        else:  # manual
            xOrA = float(lineXOrA.text())
            yOrM = float(lineYorM.text())
            if usePolar:
                x, y = BezierPointEditor.polar2cartesian(self.getUsrAngle(xOrA), yOrM)
                x = x + self.lastSavedPoint.x()
                y = y + self.lastSavedPoint.y()
            else:
                x = xOrA
                y = yOrM
                if useRelative:
                    x = x + self.lastSavedPoint.x()
                    y = y + self.lastSavedPoint.y()
        return round(x, 2), round(y, 2)

    def createLCtrlPoint(self, forcePolar=None):
        if forcePolar is None:
            forcePolar = self.ui.btnLCtrlUsePolar.isChecked()
        useRelative = self.ui.btnLCtrlRelative.isChecked()
        x, y = self.createCtrlPoint(forcePolar, useRelative, self.ui.lineLCtXorA, self.ui.lineLCtYorM, True)
        return x, y

    def createRCtrlPoint(self, forcePolar=None):
        if forcePolar is None:
            forcePolar = self.ui.btnRCtrlUsePolar.isChecked()
        useRelative = self.ui.btnRCtrlRelative.isChecked()
        x, y = self.createCtrlPoint(forcePolar, useRelative, self.ui.lineRCtXorA, self.ui.lineRCtYorM)
        return x, y

    def createBeizerPoint(self, label='Created Point'):
        x, y = self.createPoint()
        lx, ly = self.createLCtrlPoint()
        rx, ry = self.createRCtrlPoint()
        return BezierCurveEditor.BezierPoint(Qc.QPointF(x, y), Qc.QPointF(lx, ly), Qc.QPointF(rx, ry), label)

