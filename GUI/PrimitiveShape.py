#!/usr/bin/env python3

import xasy2asy as xasy2asy
import numpy as numpy
import math
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui


class PrimitiveShape:
    # The magic number.
    # see https://www.desmos.com/calculator/lw6j7khikj for unitcircle
    # optimal_ctl_pt = 0.5447

    @staticmethod
    def pos_to_tuple(pos):
        if isinstance(pos, tuple) or isinstance(pos, numpy.ndarray):
            return pos
        elif isinstance(pos, QtCore.QPoint) or isinstance(pos, QtCore.QPointF):
            return pos.x(), pos.y()
        else:
            raise TypeError("Position must be a valid type!")

    @staticmethod
    def euclideanNorm(p1, p2):
        x1, y1 = PrimitiveShape.pos_to_tuple(p1)
        x2, y2 = PrimitiveShape.pos_to_tuple(p2)

        normSq = ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
        return math.sqrt(normSq)

    @classmethod
    def circle(cls, position, radius):
        pos_x, pos_y = PrimitiveShape.pos_to_tuple(position)
        newCircle = xasy2asy.asyPath()
        ptsList = [(pos_x + radius, pos_y), (pos_x, pos_y + radius), (pos_x - radius, pos_y), (pos_x, pos_y - radius),
                   'cycle']
        # cycle doesn't work for now.
        lkList = ['..', '..', '..', '..']
        newCircle.initFromNodeList(ptsList, lkList)
        return newCircle

    @classmethod
    def inscribedRegPolygon(cls, sides, position, radius, starting_rad, qpoly=False):
        pos_x, pos_y = PrimitiveShape.pos_to_tuple(position)
        lkList = ['--'] * sides
        ptsList = []
        for ang in numpy.linspace(starting_rad, starting_rad + math.tau, sides, endpoint=False):
            ptsList.append((pos_x + radius * math.cos(ang), pos_y + radius * math.sin(ang)))

        if qpoly:
            ptsList.append((pos_x + radius * math.cos(starting_rad), pos_y + radius * math.sin(starting_rad)))
            qpoints = [QtCore.QPointF(x, y) for (x, y) in ptsList]
            return QtGui.QPolygonF(qpoints)
        else:
            ptsList.append('cycle')
            newPoly = xasy2asy.asyPath()
            newPoly.initFromNodeList(ptsList, lkList)
            return newPoly

    @classmethod
    def exscribedRegPolygon(cls, sides, position, length, starting_rad, qpoly=False):
        ang = math.tau/sides
        # see notes
        adjusted_radius = length / math.cos(ang/2)
        return cls.inscribedRegPolygon(sides, position, adjusted_radius, starting_rad - ang/2, qpoly)
