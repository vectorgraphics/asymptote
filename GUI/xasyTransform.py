import xasy2asy as x2a
import PyQt5.QtGui as Qg
import numpy as np
import math


class xasyTransform:
    @classmethod
    def makeRotTransform(cls, theta, origin):
        rotMat = (math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta))
        shift = x2a.asyTransform((0, 0, 1 - rotMat[0], -rotMat[1], -rotMat[2], 1 - rotMat[3])) * origin
        return x2a.asyTransform((shift[0], shift[1], rotMat[0], rotMat[1], rotMat[2], rotMat[3]))\

