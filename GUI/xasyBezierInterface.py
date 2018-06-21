import xasy2asy as x2a
import PyQt5.QtCore as Qc
import PyQt5.QtGui as Qg
import PyQt5.QtWidgets as Qw
import InplaceAddObj

class CurrentlySelctedType:
    none = -1
    node = 0
    ctrlPoint = 1

class InteractiveBezierEditor(InplaceAddObj.InplaceObjProcess):
    def __init__(self, parent: Qc.QObject, obj: x2a.xasyDrawnItem, info: dict={}):
        super().__init__(parent)
        self.info = info
        self.asyPath = obj.path
        assert isinstance(self.asyPath, x2a.asyPath)
        self.transf = obj.transfKeymap[obj.transfKey][0]
        self._active = True

        self.currentSelMode = None
        # (Node index, Node subindex for )
        self.currentSelIndex = (None, 0)

        nodeselRects, ctrlSelRects = self.getSelectionBoundaries()
        self.nodeSelRects = nodeselRects
        self.ctrlSelRects = ctrlSelRects
        

        self.lastSelPoint = None
        self.preCtrlOffset = None
        self.postCtrlOffset = None
        self.inTransformMode = False

        self.prosectiveNodes = []

    def getSelectionBoundaries(self):
        nodeSelectionBounaries = []

        for node in self.asyPath.nodeSet:
            if node == 'cycle':
                nodeSelectionBounaries.append(None)
                continue

            selEpsilon = 6
            newRect = Qc.QRect(0, 0, 2 * selEpsilon, 2 * selEpsilon)
            x, y = self.transf * node
            x = int(round(x))
            y = int(round(y))
            newRect.moveCenter(Qc.QPoint(x, y))

            nodeSelectionBounaries.append(newRect)

        ctrlPointSelBoundaries = []

        for nodes in self.asyPath.controlSet:
            nodea, nodeb = nodes

            selEpsilon = 6
            newRect = Qc.QRect(0, 0, 2 * selEpsilon, 2 * selEpsilon)
            newRectb = Qc.QRect(0, 0, 2 * selEpsilon, 2 * selEpsilon)

            x, y = self.transf * nodea
            x2, y2 = self.transf * nodeb 

            x = int(round(x))
            y = int(round(y))

            x2 = int(round(x2))
            y2 = int(round(y2))

            newRect.moveCenter(Qc.QPoint(x, y))
            newRectb.moveCenter(Qc.QPoint(x, y))

            ctrlPointSelBoundaries.append((newRect, newRectb))

        return nodeSelectionBounaries, ctrlPointSelBoundaries

    def postDrawPreview(self, canvas: Qg.QPainter):
        assert canvas.isActive()

        dashedPen = Qg.QPen(Qc.Qt.DashLine)
        # draw the base points
        canvas.save()
        canvas.setWorldTransform(self.transf.toQTransform(), True)

        if self.info['magnification'] != 1:
            canvas.setWorldTransform(Qg.QTransform.fromScale(self.info['magnification'], self.info['magnification']), True) 

        canvas.setPen(dashedPen)

        canvas.drawPath(self.asyPath.toQPainterPath())
        for index in range(len(self.asyPath.nodeSet)):
            point = self.asyPath.nodeSet[index]
            
            if point == 'cycle':
                continue

            basePoint = Qc.QPointF(point[0], point[1])
            canvas.setPen(Qg.QColor('blue'))
            canvas.drawEllipse(basePoint, 5, 5)

            

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
        if self.prosectiveNodes and not self.inTransformMode:
            self.currentSelMode = CurrentlySelctedType.node
            self.currentSelIndex = (self.prosectiveNodes[0], 0)
            self.inTransformMode = True
            self.lastSelPoint = pos

            # find the offset of each control point to the node

    def mouseMove(self, pos, event: Qg.QMouseEvent):
        if self.currentSelMode is None and not self.inTransformMode:
            # in this case, search for prosective nodes. 
            prospectiveNodes = []

            for i in range(len(self.nodeSelRects)):
                rect = self.nodeSelRects[i]
                if rect is None:
                    continue
                if rect.contains(pos):
                    prospectiveNodes.append(i)

            self.prosectiveNodes = prospectiveNodes

        if self.inTransformMode:
            index, subindex = self.currentSelIndex
            if self.currentSelMode == CurrentlySelctedType.node:
                deltaPos = pos - self.lastSelPoint
                # static throughout the moving

                if self.asyPath.nodeSet[index] == 'cycle':
                    return

                self.asyPath.setNode(index, (pos.x(), pos.y()))

                # if also move node: 


    def mouseRelease(self):
        if self.inTransformMode:
            self.inTransformMode = False
            self.currentSelMode = None
            nodeselRects, ctrlSelRects = self.getSelectionBoundaries()
            self.nodeSelRects = nodeselRects
            self.ctrlSelRects = ctrlSelRects
            
    def forceFinalize(self):
        self.objectUpdated.emit()

    def getObject(self):
        pass

    def getXasyObject(self):
        pass
