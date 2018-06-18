import xasy2asy as x2a
import PyQt5.QtCore as Qc
import PyQt5.QtGui as Qg
import PyQt5.QtWidgets as Qw
import InplaceAddObj

class InteractiveBezierEditor(InplaceAddObj.InplaceObjProcess):
    def __init__(self, parent: Qc.QObject, obj: x2a.xasyDrawnItem):
        super().__init__(parent)
        self.asyPath = obj.path
        self.transf = obj.transfKeymap[obj.transfKey][0]
        self._active = True

    def postDrawPreview(self, canvas: Qg.QPainter):
        assert canvas.isActive()

        # draw the base points
        canvas.save()

        canvas.setWorldTransform(self.transf.toQTransform(), True)
        for index in range(len(self.asyPath.nodeSet)):
            point = self.asyPath.nodeSet[index]
            basePoint = Qc.QPointF(point[0], point[1])
            if point != 'cycle':
                canvas.setPen(Qg.QColor('blue'))
                canvas.drawEllipse(basePoint, 5, 5)

            dashedPen = Qg.QPen(Qc.Qt.DashLine)

            if index != 0:
                canvas.setPen(Qg.QColor('red'))
                postCtrolSet = self.asyPath.controlSet[index - 1][1]
                postCtrlPoint = Qc.QPointF(postCtrolSet[0], postCtrolSet[1])
                canvas.drawEllipse(postCtrlPoint, 5, 5)

                canvas.setPen(dashedPen)
                canvas.drawLine(basePoint, postCtrlPoint)

            if index != len(self.asyPath.nodeSet) - 1:
                canvas.setPen(Qg.QColor('red'))
                preCtrlSet = self.asyPath.controlSet[index][0]
                preCtrlPoint = Qc.QPointF(preCtrlSet[0], preCtrlSet[1])
                canvas.drawEllipse(preCtrlPoint, 5, 5)

                canvas.setPen(dashedPen)
                canvas.drawLine(basePoint, preCtrlPoint)


        canvas.restore()



        

    def mouseDown(self, pos, info):
        pass

    def mouseMove(self, pos, event: Qg.QMouseEvent):
        pass

    def mouseRelease(self):
        pass

    def forceFinalize(self):
        pass

    def getObject(self):
        pass

    def getXasyObject(self):
        pass
