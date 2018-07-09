#!/usr/bin/env python3

import xasy2asy as x2a
import xasyUtils as xu

import PyQt5.QtCore as Qc
import PyQt5.QtGui as Qg
import PyQt5.QtWidgets as Qw

import Widg_editBezier as Web

import InplaceAddObj

import math

class CurrentlySelctedType:
    none = -1
    node = 0
    ctrlPoint = 1

class InteractiveBezierEditor(InplaceAddObj.InplaceObjProcess):
    editAccepted = Qc.pyqtSignal()
    editRejected = Qc.pyqtSignal()

    def __init__(self, parent: Qc.QObject, obj: x2a.xasyDrawnItem, info: dict={}):
        super().__init__(parent)
        self.info = info
        self.asyPathBackup = x2a.asyPath.fromPath(obj.path)
        self.asyPath = obj.path
        assert isinstance(self.asyPath, x2a.asyPath)
        self.transf = obj.transfKeymap[obj.transfKey][0]
        self._active = True

        self.currentSelMode = None
        # (Node index, Node subindex for )
        self.currentSelIndex = (None, 0)

        self.nodeSelRects = []
        self.ctrlSelRects = []

        self.setSelectionBoundaries()

        self.lastSelPoint = None
        self.preCtrlOffset = None
        self.postCtrlOffset = None
        self.inTransformMode = False

        self.info['autoRecompute'] = not self.containsCurvedLinkage()
        self.opt = None

        self.prosectiveNodes = []
        self.prospectiveCtrlPts = []

    def containsCurvedLinkage(self):
        return '..' in self.asyPath.linkSet

    def setSelectionBoundaries(self):
        self.nodeSelRects = self.handleNodeSelectionBounds()
        self.ctrlSelRects = self.handleCtrlSelectionBoundaries()

    def handleNodeSelectionBounds(self):
        nodeSelectionBoundaries = []

        for node in self.asyPath.nodeSet:
            if node == 'cycle':
                nodeSelectionBoundaries.append(None)
                continue

            selEpsilon = 6
            newRect = Qc.QRect(0, 0, 2 * selEpsilon, 2 * selEpsilon)
            x, y = self.transf * node
            x = int(round(x))
            y = int(round(y))
            newRect.moveCenter(Qc.QPoint(x, y))

            nodeSelectionBoundaries.append(newRect)

        return nodeSelectionBoundaries

    def handleCtrlSelectionBoundaries(self):
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
            newRectb.moveCenter(Qc.QPoint(x2, y2))

            ctrlPointSelBoundaries.append((newRect, newRectb))

        return ctrlPointSelBoundaries

    def postDrawPreview(self, canvas: Qg.QPainter):
        assert canvas.isActive()

        dashedPen = Qg.QPen(Qc.Qt.DashLine)
        # draw the base points
        canvas.save()
        canvas.setWorldTransform(self.transf.toQTransform(), True)

        if self.info['autoRecompute']:
            ctrlPtsColor = 'gray'
        else:
            ctrlPtsColor = 'red'

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
                canvas.setPen(Qg.QColor(ctrlPtsColor))
                postCtrolSet = self.asyPath.controlSet[index - 1][1]
                postCtrlPoint = Qc.QPointF(postCtrolSet[0], postCtrolSet[1])
                canvas.drawEllipse(postCtrlPoint, 5, 5)

                canvas.setPen(dashedPen)
                canvas.drawLine(basePoint, postCtrlPoint)

            if index != len(self.asyPath.nodeSet) - 1:
                canvas.setPen(Qg.QColor(ctrlPtsColor))
                preCtrlSet = self.asyPath.controlSet[index][0]
                preCtrlPoint = Qc.QPointF(preCtrlSet[0], preCtrlSet[1])
                canvas.drawEllipse(preCtrlPoint, 5, 5)

                canvas.setPen(dashedPen)
                canvas.drawLine(basePoint, preCtrlPoint)


        canvas.restore()

    def getPreAndPostCtrlPts(self, index):
        if index == 0:
            preCtrl = None
        else:
            preCtrl = self.asyPath.controlSet[index - 1][1]

        if index == len(self.asyPath.nodeSet) - 1:
            postCtrl = None
        else:
            postCtrl = self.asyPath.controlSet[index][0]

        return preCtrl, postCtrl

    def findLinkingNode(self, index, subindex):
        """index and subindex are of the control points list."""
        if subindex == 0:
            return index
        else:
            return index + 1

    def resetObj(self):
        self.asyPath.setInfo(self.asyPathBackup)
        self.setSelectionBoundaries()

    def mouseDown(self, pos, info, mouseEvent: Qg.QMouseEvent=None):
        self.lastSelPoint = pos
        if self.inTransformMode:
            return

        if self.prosectiveNodes and not self.inTransformMode:
            self.currentSelMode = CurrentlySelctedType.node
            self.currentSelIndex = (self.prosectiveNodes[0], 0)
            self.inTransformMode = True
            self.parentNodeIndex = self.currentSelIndex[0]

        if self.prospectiveCtrlPts and not self.inTransformMode:
            self.currentSelMode = CurrentlySelctedType.ctrlPoint
            self.currentSelIndex = self.prospectiveCtrlPts[0]
            self.inTransformMode = True
            self.parentNodeIndex = self.findLinkingNode(*self.currentSelIndex)
        
        if self.inTransformMode:
            parentNode = self.asyPath.nodeSet[self.parentNodeIndex]

            # find the offset of each control point to the node
            preCtrl, postCtrl = self.getPreAndPostCtrlPts(self.parentNodeIndex)
            
            if preCtrl is not None:
                self.preCtrlOffset = xu.funcOnList(
                    preCtrl, parentNode, lambda a, b: a - b)
            else:
                self.preCtrlOffset = None

            if postCtrl is not None:
                self.postCtrlOffset = xu.funcOnList(
                    postCtrl, parentNode, lambda a, b: a - b)
            else:
                self.postCtrlOffset = None

    def mouseMove(self, pos, event: Qg.QMouseEvent):
        if self.currentSelMode is None and not self.inTransformMode:
            # in this case, search for prosective nodes. 
            prospectiveNodes = []
            prospectiveCtrlpts = []

            for i in range(len(self.nodeSelRects)):
                rect = self.nodeSelRects[i]
                if rect is None:
                    continue
                if rect.contains(pos):
                    prospectiveNodes.append(i)

            self.prosectiveNodes = prospectiveNodes

            if not self.info['autoRecompute']:
                for i in range(len(self.ctrlSelRects)):
                    recta, rectb = self.ctrlSelRects[i]

                    if recta.contains(pos):
                        prospectiveCtrlpts.append((i, 0))

                    if rectb.contains(pos):
                        prospectiveCtrlpts.append((i, 1))

                self.prospectiveCtrlPts = prospectiveCtrlpts
            else:
                self.prospectiveCtrlPts = []


        if self.inTransformMode:
            index, subindex = self.currentSelIndex
            deltaPos = pos - self.lastSelPoint
            newNode = (pos.x(), pos.y())
            if self.currentSelMode == CurrentlySelctedType.node:
                # static throughout the moving
                if self.asyPath.nodeSet[index] == 'cycle':
                    return

                self.asyPath.setNode(index, newNode)
                # if also move node: 

                checkPre, checkPost = self.getPreAndPostCtrlPts(index)

                if 1 == 1: # TODO: Replace this with an option to also move control pts. 
                    if checkPre is not None:
                        self.asyPath.controlSet[index - 1][1] = xu.funcOnList(
                            newNode, self.preCtrlOffset, lambda a, b: a + b
                        )
                    if checkPost is not None:
                        self.asyPath.controlSet[index][0] = xu.funcOnList(
                            newNode, self.postCtrlOffset, lambda a, b: a + b
                        )

                if self.info['autoRecompute']:
                    self.quickRecalculateCtrls()
                    

            elif self.currentSelMode == CurrentlySelctedType.ctrlPoint:
                self.asyPath.controlSet[index][subindex] = newNode

                parentNode = self.asyPath.nodeSet[self.parentNodeIndex]

                rawNewNode = xu.funcOnList(newNode, parentNode, lambda a, b: a - b)
                rawAngle = math.atan2(rawNewNode[1], rawNewNode[0])
                newNorm = xu.twonorm(rawNewNode)

                if self.info['editBezierlockMode'] >= Web.LockMode.angleLock:
                    otherIndex = 1 - subindex       # 1 if 0, 0 otherwise. 
                    if otherIndex == 0:
                        if index < len(self.asyPath.controlSet) - 1:
                            oldOtherCtrlPnt = xu.funcOnList(
                                self.asyPath.controlSet[index + 1][0] , parentNode, lambda a, b: a - b)
                        
                            if self.info['editBezierlockMode'] >= Web.LockMode.angleAndScaleLock:
                                rawNorm = newNorm
                            else:
                                rawNorm = xu.twonorm(oldOtherCtrlPnt)

                            newPnt = (rawNorm * math.cos(rawAngle + math.pi), 
                                rawNorm * math.sin(rawAngle + math.pi))
                                
                            self.asyPath.controlSet[index + 1][0] = xu.funcOnList(newPnt, parentNode, lambda a, b: a + b)
                    else:
                        if index > 0:
                            oldOtherCtrlPnt = xu.funcOnList(
                                self.asyPath.controlSet[index-1][1], parentNode, lambda a, b: a - b)

                            if self.info['editBezierlockMode'] >= Web.LockMode.angleAndScaleLock:
                                rawNorm = newNorm
                            else:
                                rawNorm = xu.twonorm(oldOtherCtrlPnt)

                            newPnt = (rawNorm * math.cos(rawAngle + math.pi),
                                      rawNorm * math.sin(rawAngle + math.pi))
                            self.asyPath.controlSet[index - 1][1] = xu.funcOnList(
                                newPnt, parentNode, lambda a, b: a + b)
        
    def recalculateCtrls(self):
        self.quickRecalculateCtrls()
        self.setSelectionBoundaries()

    def quickRecalculateCtrls(self):
        self.asyPath.controlSet.clear()
        self.asyPath.computeControls()

    def mouseRelease(self):
        if self.inTransformMode:
            self.inTransformMode = False
            self.currentSelMode = None

            self.setSelectionBoundaries()
            
    def forceFinalize(self):
        self.objectUpdated.emit()

    def createOptWidget(self, info):
        self.opt = Web.Widg_editBezier(self.info)
        self.opt.ui.btnOk.clicked.connect(self.editAccepted)
        self.opt.ui.btnCancel.clicked.connect(self.editRejected)
        self.opt.ui.btnForceRecompute.clicked.connect(self.recalculateCtrls)

        return self.opt

    def getObject(self):
        pass

    def getXasyObject(self):
        pass
