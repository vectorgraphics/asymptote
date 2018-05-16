import PyQt5.QtCore as Qc
import PyQt5.QtGui as Qg
import PrimitiveShape
import math

import Widg_addPolyOpt
import Widg_addLabel


class InplaceObjProcess:
    def __init__(self):
        self._active = False
        pass

    @property
    def active(self):
        return self._active

    def mouseDown(self, pos, info):
        raise NotImplementedError

    def mouseMove(self, pos):
        raise NotImplementedError

    def mouseRelease(self):
        raise NotImplementedError

    def getPreview(self):
        raise NotImplementedError

    def getObject(self):
        raise NotImplementedError

    def createOptWidget(self, info):
        return None


class AddCircle(InplaceObjProcess):
    def __init__(self):
        super().__init__()
        self.center = Qc.QPointF(0, 0)
        self.radius = 0

    def mouseDown(self, pos, info):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.center.setX(x)
        self.center.setY(y)
        self._active = True

    def mouseMove(self, pos):
        self.radius = PrimitiveShape.PrimitiveShape.euclideanNorm(pos, self.center)

    def mouseRelease(self):
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


class AddLabel(InplaceObjProcess):
    def __init__(self):
        super().__init__()
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
        self._active = False

    def mouseMove(self, pos):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.anchor.setX(x)
        self.anchor.setY(y)

    def mouseDown(self, pos, info):
        if self.opt is not None:
            self.text = self.opt.labelText
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.anchor.setX(x)
        self.anchor.setY(y)

        self.alignMode = info['align']
        self._active = True

    def getObject(self):
        finalTuple = PrimitiveShape.PrimitiveShape.pos_to_tuple(self.anchor)
        return {'txt': self.text, 'align': str(self.alignMode), 'anchor': finalTuple}


class AddPoly(InplaceObjProcess):
    def __init__(self):
        super().__init__()
        self.center = Qc.QPointF(0, 0)
        self.currPos = Qc.QPointF(0, 0)
        self.sides = None
        self.inscribed = None
        self.centermode = None

    def mouseDown(self, pos, info):
        self._active = True
        self.sides = info['sides']
        self.inscribed = info['inscribed']
        self.centermode = info['centermode']

        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.center.setX(x)
        self.center.setY(y)

    def mouseMove(self, pos):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.currPos.setX(x)
        self.currPos.setY(y)

    def mouseRelease(self):
        self._active = False

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
        return Widg_addPolyOpt.Widg_addPolyOpt(info)

    def _rad(self):
        return PrimitiveShape.PrimitiveShape.euclideanNorm(self.currPos, self.center)

    def _angle(self):
        dist_x = self.currPos.x() - self.center.x()
        dist_y = self.currPos.y() - self.center.y()
        if dist_x == 0 and dist_y == 0:
            return 0
        else:
            return math.atan2(dist_y, dist_x)
