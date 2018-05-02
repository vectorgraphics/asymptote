import xasy2asy as x2a
import numpy as np
import math
import PyQt5.QtCore as Qc


class PrimitiveShape:
    # The magic number.
    # see https://www.desmos.com/calculator/lw6j7khikj for unitcircle
    # optimal_ctl_pt = 0.5447

    @staticmethod
    def pos_to_tuple(pos):
        if isinstance(pos, tuple) or isinstance(pos, np.ndarray):
            return pos
        elif isinstance(pos, Qc.QPoint) or isinstance(pos, Qc.QPointF):
            return pos.x(), pos.y()
        else:
            raise TypeError("Position must be a valid type!")

    @classmethod
    def circle(cls, position, radius):
        pos_x, pos_y = PrimitiveShape.pos_to_tuple(position)
        newCircle = x2a.asyPath()
        ptsList = [(pos_x + radius, pos_y), (pos_x, pos_y + radius), (pos_x - radius, pos_y), (pos_x, pos_y - radius),
                   (pos_x + radius, pos_y)]
        # cycle doesn't work for now.
        lkList = ['..', '..', '..', '..']
        newCircle.initFromNodeList(ptsList, lkList)
        return newCircle

    @classmethod
    def inscribedRegPolygon(cls, sides, position, radius, starting_rad):
        pos_x, pos_y = PrimitiveShape.pos_to_tuple(position)
        lkList = ['--'] * sides
        ptsList = []
        for ang in np.linspace(starting_rad, starting_rad + math.tau, sides, endpoint=False):
            ptsList.append((pos_x + radius * np.cos(ang), pos_y + radius * np.sin(ang)))

        ptsList.append((pos_x + radius * np.cos(starting_rad), pos_y + radius * np.sin(starting_rad)))
        newPoly = x2a.asyPath()
        newPoly.initFromNodeList(ptsList, lkList)
        return newPoly

    @classmethod
    def exscribedRegPolygon(cls, sides, position, length, starting_rad):
        ang = math.tau/sides
        # see notes
        adjusted_radius = length / np.cos(ang/2)
        return cls.inscribedRegPolygon(sides, position, adjusted_radius, starting_rad - ang/2)
