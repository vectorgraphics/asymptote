#!/usr/bin/env python3

import PyQt5.QtCore as Qc
import PyQt5.QtGui as Qg
import xasy2asy as x2a

import PrimitiveShape
import math

import Widg_addPolyOpt
import Widg_addLabel


class InplaceObjProcess(Qc.QObject):
    objectCreated = Qc.pyqtSignal(Qc.QObject)
    objectUpdated = Qc.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        pass

    @property
    def active(self):
        return self._active

    def mouseDown(self, pos, info, mouseEvent: Qg.QMouseEvent=None):
        raise NotImplementedError

    def mouseMove(self, pos, event: Qg.QMouseEvent):
        raise NotImplementedError

    def mouseRelease(self):
        raise NotImplementedError

    def forceFinalize(self):
        raise NotImplementedError

    def getPreview(self):
        return None

    def getObject(self):
        raise NotImplementedError

    def getXasyObject(self):
        raise NotImplementedError

    def postDrawPreview(self, canvas: Qg.QPainter):
        pass

    def createOptWidget(self, info):
        return None


class AddCircle(InplaceObjProcess):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.center = Qc.QPointF(0, 0)
        self.radius = 0

    def mouseDown(self, pos, info, mouseEvent: Qg.QMouseEvent=None):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.radius = 0
        self.center.setX(x)
        self.center.setY(y)
        self.fill = info['fill']
        self._active = True

    def mouseMove(self, pos, event):
        self.radius = PrimitiveShape.PrimitiveShape.euclideanNorm(pos, self.center)

    def mouseRelease(self):
        self.objectCreated.emit(self.getXasyObject())
        self._active = False

    def getPreview(self):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(self.center)
        boundRect = Qc.QRectF(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)
        # because the internal image is flipped...
        newPath = Qg.QPainterPath()
        newPath.addEllipse(boundRect)
        # newPath.addRect(boundRect)
        return newPath

    def getObject(self):
        return PrimitiveShape.PrimitiveShape.circle(self.center, self.radius)

    def getXasyObject(self):
        if self.fill:
            newObj = x2a.xasyFilledShape(self.getObject(), None)
        else:
            newObj = x2a.xasyShape(self.getObject(), None)
        return newObj

    def forceFinalize(self):
        self.mouseRelease()


class AddLabel(InplaceObjProcess):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.alignMode = None
        self.opt = None
        self.text = None
        self.anchor = Qc.QPointF(0, 0)
        self._active = False

    def createOptWidget(self, info):
        self.opt = Widg_addLabel.Widg_addLabel(info)
        return self.opt

    def getPreview(self):
        return None

    def mouseRelease(self):
        self.objectCreated.emit(self.getXasyObject())
        self._active = False

    def mouseMove(self, pos, event):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.anchor.setX(x)
        self.anchor.setY(y)

    def mouseDown(self, pos, info, mouseEvent: Qg.QMouseEvent=None):
        if self.opt is not None:
            self.text = self.opt.labelText
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.anchor.setX(x)
        self.anchor.setY(y)

        self.alignMode = info['align']
        self.fontSize = info['fontSize']
        self._active = True

    def getObject(self):
        finalTuple = PrimitiveShape.PrimitiveShape.pos_to_tuple(self.anchor)
        return {'txt': self.text, 'align': str(self.alignMode), 'anchor': finalTuple}

    def getXasyObject(self):
        text = self.text
        align = str(self.alignMode)
        anchor = PrimitiveShape.PrimitiveShape.pos_to_tuple(self.anchor)
        newLabel = x2a.xasyText(text=text, location=anchor, pen=None,
                                align=align, asyengine=None, fontsize=self.fontSize)
        newLabel.asyfied = False
        return newLabel

    def forceFinalize(self):
        self.mouseRelease()


class AddBezierShape(InplaceObjProcess):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.asyengine = None
        self.basePath = None
        self.basePathPreview = None
        self.closedPath = None
        self.info = None
        self.fill = False
        self.opt = None

        # list of "committed" points with Linkage information.
        # Linkmode should be to the last point.
        # (x, y, linkmode), (u, v, lm2) <==> (x, y) <=lm2=> (u, v)
        self.pointsList = []
        self.currentPoint = Qc.QPointF(0, 0)
        self.pendingPoint = None
        self.useLegacy = False

    def mouseDown(self, pos, info, mouseEvent: Qg.QMouseEvent=None):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.currentPoint.setX(x)
        self.currentPoint.setY(y)
        self.info = info

        if not self._active:
            self._active = True
            self.fill = info['fill']
            self.asyengine = info['asyengine']
            self.closedPath = info['closedPath']
            self.useBezierBase = info['useBezier']
            self.useLegacy = self.info['options']['useLegacyDrawMode']
            self.pointsList.clear()
            self.pointsList.append((x, y, None))
        else:
            # see http://doc.qt.io/archives/qt-4.8/qt.html#MouseButton-enum
            if (int(mouseEvent.buttons()) if mouseEvent is not None else 0) & 0x2 and self.useLegacy:
                self.forceFinalize()

    def _getLinkType(self):
        if self.info['useBezier']:
            return '..'
        else:
            return '--'

    def mouseMove(self, pos, event):
        # in postscript coords.
        if self._active:
            x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)

            if self.useLegacy or int(event.buttons()) != 0:
                self.currentPoint.setX(x)
                self.currentPoint.setY(y)
            else:
                self.forceFinalize()


    def createOptWidget(self, info):
        return None
        # self.opt = Widg_addBezierInPlace.Widg_addBezierInplace(info)
        # return self.opt

    def finalizeClosure(self):
        if self.active:
            self.closedPath = True
            self.forceFinalize()

    def mouseRelease(self):
        x, y = self.currentPoint.x(), self.currentPoint.y()
        self.pointsList.append((x, y, self._getLinkType()))
        # self.updateBasePath()

    def updateBasePath(self):
        self.basePath = x2a.asyPath(asyengine=self.asyengine, forceCurve=self.useBezierBase)
        newNode = [(x, y) for x, y, _ in self.pointsList]
        newLink = [lnk for *args, lnk in self.pointsList[1:]]
        if self.useLegacy:
            newNode += [(self.currentPoint.x(), self.currentPoint.y())]
            newLink += [self._getLinkType()]
        if self.closedPath:
            newNode.append('cycle')
            newLink.append(self._getLinkType())
        self.basePath.initFromNodeList(newNode, newLink)

        if self.useBezierBase:
            self.basePath.computeControls()

    def updateBasePathPreview(self):
        self.basePathPreview = x2a.asyPath(
            asyengine=self.asyengine, forceCurve=self.useBezierBase)
        newNode = [(x, y) for x, y, _ in self.pointsList] + [(self.currentPoint.x(), self.currentPoint.y())]
        newLink = [lnk for *args, lnk in self.pointsList[1:]] + [self._getLinkType()]
        if self.closedPath:
            newNode.append('cycle')
            newLink.append(self._getLinkType())
        self.basePathPreview.initFromNodeList(newNode, newLink)

        if self.useBezierBase:
            self.basePathPreview.computeControls()

    def forceFinalize(self):
        self.updateBasePath()
        self._active = False
        self.pointsList.clear()
        self.objectCreated.emit(self.getXasyObject())
        self.basePath = None

    def getObject(self):
        if self.basePath is None:
            raise RuntimeError('BasePath is None')
        self.basePath.asyengine = self.asyengine
        return self.basePath

    def getPreview(self):
        if self._active:
            if self.pointsList:
                self.updateBasePathPreview()
                newPath = self.basePathPreview.toQPainterPath()
                return newPath

    def getXasyObject(self):
        if self.fill:
            return x2a.xasyFilledShape(self.getObject(), None)
        else:
            return x2a.xasyShape(self.getObject(), None)


class AddPoly(InplaceObjProcess):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.center = Qc.QPointF(0, 0)
        self.currPos = Qc.QPointF(0, 0)
        self.sides = None
        self.inscribed = None
        self.centermode = None
        self.asyengine = None
        self.fill = None
        self.opt = None

    def mouseDown(self, pos, info, mouseEvent: Qg.QMouseEvent=None):
        self._active = True
        self.sides = info['sides']
        self.inscribed = info['inscribed']
        self.centermode = info['centermode']
        self.fill = info['fill']

        
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.center.setX(x)
        self.center.setY(y)
        self.currPos = Qc.QPointF(self.center)

    def mouseMove(self, pos, event):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.currPos.setX(x)
        self.currPos.setY(y)

    def mouseRelease(self):
        if self.active:
            self.objectCreated.emit(self.getXasyObject())
            self._active = False

    def forceFinalize(self):
        self.mouseRelease()

    def getObject(self):
        if self.inscribed:
            return PrimitiveShape.PrimitiveShape.inscribedRegPolygon(self.sides, self.center, self._rad(),
                                                                     self._angle())
        else:
            return PrimitiveShape.PrimitiveShape.exscribedRegPolygon(self.sides, self.center, self._rad(),
                                                                     self._angle())

    def getPreview(self):
        if self.inscribed:
            poly = PrimitiveShape.PrimitiveShape.inscribedRegPolygon(self.sides, self.center, self._rad(),
                                                                     self._angle(), qpoly=True)
        else:
            poly = PrimitiveShape.PrimitiveShape.exscribedRegPolygon(self.sides, self.center, self._rad(),
                                                                     self._angle(), qpoly=True)
        newPath = Qg.QPainterPath()
        newPath.addPolygon(poly)
        return newPath

    def createOptWidget(self, info):
        self.opt = Widg_addPolyOpt.Widg_addPolyOpt(info)
        return self.opt

    def _rad(self):
        return PrimitiveShape.PrimitiveShape.euclideanNorm(self.currPos, self.center)

    def _angle(self):
        dist_x = self.currPos.x() - self.center.x()
        dist_y = self.currPos.y() - self.center.y()
        if dist_x == 0 and dist_y == 0:
            return 0
        else:
            return math.atan2(dist_y, dist_x)

    def getXasyObject(self):
        if self.fill:
            newObj = x2a.xasyFilledShape(self.getObject(), None)
        else:
            newObj = x2a.xasyShape(self.getObject(), None)
        return newObj
