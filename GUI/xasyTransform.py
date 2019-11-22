#!/usr/bin/env python3
import xasy2asy as x2a
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import numpy as np
import math


class xasyTransform:
    @classmethod
    def makeRotTransform(cls, theta, origin):
        if isinstance(origin, Qc.QPointF) or isinstance(origin, Qc.QPoint):
            origin = (origin.x(), origin.y())
        rotMat = (math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta))
        shift = x2a.asyTransform((0, 0, 1 - rotMat[0], -rotMat[1], -rotMat[2], 1 - rotMat[3])) * origin
        return x2a.asyTransform((shift[0], shift[1], rotMat[0], rotMat[1], rotMat[2], rotMat[3]))

    @classmethod
    def makeScaleTransform(cls, sx, sy, origin):
        if isinstance(origin, Qc.QPointF) or isinstance(origin, Qc.QPoint):
            origin = (origin.x(), origin.y())
        shiftMat = x2a.asyTransform((0, 0, 1 - sx, 0, 0, 1 - sy)) * origin
        return x2a.asyTransform((shiftMat[0], shiftMat[1], sx, 0, 0, sy))

