import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
from pyUIClass.beizerCurveEditor import Ui_Form


class BezierPoint:
    @classmethod
    def fromQStandardItem(cls, pointItem):
        """
        Creates a Bezier point from the following data structure:
        ---------------------------
        Point             | X | Y |
        LeftControlPoint  | X | Y |
        RightControlPoint | X | Y |
        ---------------------------
        """
        x, y = pointItem.child(0, 1).text(), pointItem.child(0, 2).text()
        lCx, lCy = pointItem.child(1, 1).text(), pointItem.child(1, 2).text()
        rCx, rCy = pointItem.child(2, 1).text(), pointItem.child(2, 2).text()

        mainPoint = Qc.QPointF(float(x), float(y))
        lCtrlPoint = Qc.QPointF(float(lCx), float(lCy))
        rCtrlPoint = Qc.QPointF(float(rCx), float(lCy))
        return BezierPoint(mainPoint, lCtrlPoint, rCtrlPoint, pointItem.text())

    def __init__(self, point, lCtrlPoint=None, rCtrlPoint=None, pointName=''):
        self.point = point
        self.pointName = pointName
        if lCtrlPoint is None:
            self.lCtrlPoint = point
        else:
            self.lCtrlPoint = lCtrlPoint

        if rCtrlPoint is None:
            self.rCtrlPoint = point
        else:
            self.rCtrlPoint = rCtrlPoint

    def __str__(self):
        return str.format("Bezier Point: {0}, Ctrl Points: ({1}, {2})", str(self.point), str(self.lCtrlPoint),
                          str(self.lCtrlPoint))

    def drawCurve(self, otherBezierPoint):
        painterPath = Qg.QPainterPath(self.point)
        painterPath.cubicTo(self.rCtrlPoint, otherBezierPoint.lCtrlPoint, otherBezierPoint.point)
        return painterPath

    def createQStandardItem(self):
        baseItem = Qg.QStandardItem(self.pointName)
        baseItem.setEditable(False)
        baseItem.appendRow(self.createDataRow(self.point, 'Point'))
        baseItem.appendRow(self.createDataRow(self.lCtrlPoint, 'Left Control Point'))
        baseItem.appendRow(self.createDataRow(self.rCtrlPoint, 'Right Control Point'))
        return baseItem

    def createDataRow(self, point, label=''):
        lblItem = Qg.QStandardItem(label)
        lblItem.setEditable(False)
        return [lblItem, Qg.QStandardItem(str(point.x())), Qg.QStandardItem(str(point.y()))]


class BezierCurveEditor(Qw.QDialog):
    curveChanged = Qc.pyqtSignal(bool, Qg.QPainterPath)

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.setFixedSize(self.size())

        self.ui.btnAddPoint.clicked.connect(self.addPoint)

        self.ui.btnAccept.clicked.connect(self.accept)
        self.ui.btnCancel.clicked.connect(self.reject)

        self.pointList = []

        self.model = Qg.QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Item', 'X', 'Y'])
        self.model.setColumnCount(3)
        self.modelRoot = self.model.invisibleRootItem()

        self.ui.treeViewPoints.setModel(self.model)

        self.ui.treeViewPoints.setColumnWidth(0, 180)
        self.ui.treeViewPoints.setColumnWidth(1, 50)
        self.ui.treeViewPoints.setColumnWidth(2, 50)

        self.pointCounter = 1

        self.updateGeometry()

    def resetModel(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Item', 'X', 'Y'])
        self.model.setColumnCount(3)
        self.modelRoot = self.model.invisibleRootItem()

    def updateCurve(self):
        self.createPointList()
        if len(self.pointList) >= 2:
            self.curveChanged.emit(True, self.generateCurve())
        else:
            self.curveChanged.emit(False, Qg.QPainterPath())

    def generateCurve(self):
        if len(self.pointList) <= 1:
            return None
        painterPath = Qg.QPainterPath()
        for pointIndex in range(len(self.pointList) - 1):
            basePoint, nextPoint = self.pointList[pointIndex], self.pointList[pointIndex + 1]
            painterPath.addPath(basePoint.drawCurve(nextPoint))
        return painterPath

    def createPointList(self):
        self.pointList.clear()
        for pointIndex in range(self.model.rowCount()):
            pointItem = self.model.item(pointIndex)
            self.pointList.append(BezierPoint.fromQStandardItem(pointItem))

    def addPoint(self):
        inputPoint = Qw.QInputDialog.getText(self, '', '')
        if not inputPoint[1]:
            return
        x, y, px, py, qx, qy = [float(inputText) for inputText in inputPoint[0].split(' ')]
        point = BezierPoint(Qc.QPointF(x, y), Qc.QPointF(px, py), Qc.QPointF(qx, qy),
                                          str.format('Point {0}', self.pointCounter)).createQStandardItem()
        self.model.appendRow(point)
        self.ui.treeViewPoints.expand(point.index())
        self.pointCounter = self.pointCounter + 1
        self.updateCurve()



