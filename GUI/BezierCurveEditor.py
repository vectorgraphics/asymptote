import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
from pyUIClass.bezierCurveEditor import Ui_Form
import BezierPointEditor


def QPoint2Tuple(point):
    assert isinstance(point, Qc.QPoint) or isinstance(point, Qc.QPointF)
    return point.x(), point.y()


class BezierPoint:
    @classmethod
    def fromQStandardItem(cls, pointItem):
        """
        Creates a Bezier point from the following data structure:
        -----------------------------
        | Point             | X | Y |
        | LeftControlPoint  | X | Y |
        | RightControlPoint | X | Y |
        -----------------------------
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

    def setLastPoint(self):
        self.rCtrlPoint = None

    def setFirstPoint(self):
        self.lCtrlPoint = None

    def __str__(self):
        return str.format("Beizer Point: {0}, Ctrl Points: ({1}, {2})", str(self.point), str(self.lCtrlPoint),
                          str(self.lCtrlPoint))

    def drawCurve(self, otherBeizerPoint):
        painterPath = Qg.QPainterPath(self.point)
        painterPath.cubicTo(self.rCtrlPoint, otherBeizerPoint.lCtrlPoint, otherBeizerPoint.point)
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

    def __init__(self, useDegrees=False):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.setFixedSize(self.size())

        self.ui.btnAddPoint.clicked.connect(self.addPoint)
        self.ui.btnEditPoint.clicked.connect(self.btnEditPointOnClick)
        self.ui.btnRemovePoint.clicked.connect(self.deletePoint)
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
        self.ui.treeViewPoints.setSelectionMode(Qw.QAbstractItemView.SingleSelection)

        self.pointCounter = 1
        self.useDegrees = useDegrees

        self.updateGeometry()

    def resetModel(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Item', 'X', 'Y'])
        self.model.setColumnCount(3)
        self.modelRoot = self.model.invisibleRootItem()

    def btnEditPointOnClick(self):
        selectedIndex = self.ui.treeViewPoints.selectionModel().currentIndex()
        if selectedIndex.parent().isValid():  # selected one of child items
            selectedIndex = selectedIndex.parent()
        editDialog = BezierPointEditor.BezierPointEditor(BezierPoint.fromQStandardItem(self.model.itemFromIndex(
            selectedIndex)))
        editDialog.show()
        result = editDialog.exec_()
        if result == Qw.QDialog.Accepted:
            currItem = self.model.itemFromIndex(selectedIndex)
            x, y = [Qg.QStandardItem(str(num)) for num in editDialog.createPoint()]

            currItem.setChild(0, 1, x)
            currItem.setChild(0, 2, y)

            lcx, lcy = [Qg.QStandardItem(str(num)) for num in editDialog.createLCtrlPoint()]
            rcx, rcy = [Qg.QStandardItem(str(num)) for num in editDialog.createRCtrlPoint()]

            currItem.setChild(1, 1, lcx)
            currItem.setChild(2, 2, lcy)
            currItem.setChild(2, 1, rcx)
            currItem.setChild(2, 2, rcy)

            self.updateCurve()

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
        if len(self.pointList) >= 2:
            self.pointList[0].setFirstPoint()
            self.pointList[-1].setLastPoint()
        return self.pointList

    def addPoint(self):
        addPointDialog = BezierPointEditor.BezierPointEditor(useDegrees=self.useDegrees)
        addPointDialog.show()
        result = addPointDialog.exec_()

        if result == Qw.QDialog.Rejected:
            return
        point = addPointDialog.createBeizerPoint(str.format('Point {0}', self.pointCounter)).createQStandardItem()
        self.model.appendRow(point)
        self.ui.treeViewPoints.expand(point.index())
        self.pointCounter = self.pointCounter + 1
        self.updateCurve()

    def deletePoint(self):
        selectedIndex = self.ui.treeViewPoints.selectionModel().currentIndex()
        if selectedIndex.parent().isValid():  # selected one of child items
            selectedIndex = selectedIndex.parent()
        self.model.removeRow(selectedIndex.row())
        self.createPointList()
        self.updateCurve()



