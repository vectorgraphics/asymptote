#!/usr/bin/env python3
import xasy2asy as xasy2asy
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import numpy as numpy
import math


class xasyTransform:
    @classmethod
    def makeRotTransform(cls, theta, origin):
        if isinstance(origin, QtCore.QPointF) or isinstance(origin, QtCore.QPoint):
            origin = (origin.x(), origin.y())
        rotMat = (math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta))
        shift = xasy2asy.asyTransform((0, 0, 1 - rotMat[0], -rotMat[1], -rotMat[2], 1 - rotMat[3])) * origin
        return xasy2asy.asyTransform((shift[0], shift[1], rotMat[0], rotMat[1], rotMat[2], rotMat[3]))

    @classmethod
    def makeScaleTransform(cls, sx, sy, origin):
        if isinstance(origin, QtCore.QPointF) or isinstance(origin, QtCore.QPoint):
            origin = (origin.x(), origin.y())
        shiftMat = xasy2asy.asyTransform((0, 0, 1 - sx, 0, 0, 1 - sy)) * origin
        return xasy2asy.asyTransform((shiftMat[0], shiftMat[1], sx, 0, 0, sy))

