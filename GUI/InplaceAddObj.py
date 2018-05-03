import PyQt5.QtCore as Qc
import PyQt5.QtGui as Qg
import PrimitiveShape


class InplaceObjProcess:
    def __init__(self):
        self.__active = False
        pass

    def mouseDown(self, pos):
        raise NotImplementedError

    def mouseMove(self, pos):
        raise NotImplementedError

    def mouseRelease(self):
        raise NotImplementedError

    def getPreview(self):
        raise NotImplementedError

    def getObject(self):
        raise NotImplementedError


class AddCircle(InplaceObjProcess):
    def __init__(self):
        super().__init__()
        self.center = Qc.QPointF(0, 0)
        self.radius = 0

    @property
    def active(self):
        return self.__active

    def mouseDown(self, pos):
        x, y = PrimitiveShape.PrimitiveShape.pos_to_tuple(pos)
        self.center.setX(x)
        self.center.setY(y)
        self.__active = True

    def mouseMove(self, pos):
        self.radius = PrimitiveShape.PrimitiveShape.euclideanNorm(pos, self.center)

    def mouseRelease(self):
        self.__active = False

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
